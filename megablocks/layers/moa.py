from megablocks.layers import common
from megablocks.layers import moe
from megablocks.layers import dmlp_registry
from megablocks.layers import mpu
from megablocks.layers import router
from megablocks.layers.arguments import Arguments
import megablocks.ops as ops
import numpy as np
import torch

def promote_scalar(x):
    return x.view(1) if not len(x.size()) else x

class ParallelDroplessLinear(moe.ParallelMLP):

    def __init__(self, args : Arguments):
        super(ParallelDroplessLinear, self).__init__(args)
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = mpu.features_per_rank(args)
        self.blocking = 128
        self.mlp = dmlp_registry.get(args)

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = (
            (self.ffn_hidden_size * self.num_experts) // self.blocking)
        self.transpose_sort_end_bit = max(
            int(np.ceil(np.log2(max_column_index))), 1)
    
    def map(self, x, expert_weights, top_experts):
        # x: [sl, bs, hs]
        # expert_weights: [sl * bs, top-k]
        # top_experts: [sl * bs, top-k]
        sl, bs, hs = x.shape
        self.expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            self.indices, self.bin_ids, self.bins, self.tokens_per_expert = (
                self.indices_and_bins(top_experts))

        out = self.grouped_permute_and_compute(
            x,
            self.tokens_per_expert,
            self.indices,
            self.bin_ids,
            None,
            self.bins,
            -1,  # unused
            self.top_k,
            map=True)
        return out.view(sl, bs, self.top_k, self.ffn_hidden_size)
    
    def reduce(self, x):
        # x: [sl, bs, k, hs]
        sl, bs, k, hs = x.shape

        out = self.grouped_permute_and_compute(
            x,
            self.tokens_per_expert,
            self.indices,
            self.bin_ids,
            self.expert_weights,
            self.bins,
            -1,  # unused
            self.args.moe_top_k,
            map=False)
        return out.view(sl, bs, self.hidden_size)

    def grouped_permute_and_compute(
            self,
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            expert_capactiy,  # unused
            top_k,
            map=True):

        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.gather(
            x,
            indices,
            bin_ids,
            bins,
            top_k if map else 1)

        # Perform the expert computation.
        if map:
            x = self.mlp.map(x, tokens_per_expert)
        else:
            x = self.mlp.reduce(x, tokens_per_expert)

        # Un-route the data for the MoE output.
        return ops.scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            top_k if not map else 1,
            self.args.quantize_scatter_num_bits)
    

class dMoA(torch.nn.Module):

    def __init__(self, args : Arguments):
        super(dMoA, self).__init__()

        # Token router.
        self.router = router.LearnedRouter(args)
        
        # Expert computation helper.
        self.experts = ParallelDroplessLinear(args)

    def map(self, x):
        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth.
        x = common.cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments.
        scores, expert_weights, top_experts = self.router(x)

        # Compute the experts.
        return self.experts.map(x, expert_weights, top_experts), self.router.loss
    
    def reduce(self, x):
        x = self.experts.reduce(x)
        return x