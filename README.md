# :robot: MegaBlocks

This is a modified version of the original MegaBlocks repository. The original repository can be found [here](https://github.com/databricks/megablocks).
The modifications include:
- [x] Added support for Mixture of Attention heads (https://arxiv.org/abs/2210.05144)
- [x] Added support for zloss introduced in ST-MoE (https://arxiv.org/pdf/2202.08906.pdf)

Here is an example of implementing the Mixture of Attention heads with this library:
```python
import torch
import torch.nn as nn

from megablocks.layers.arguments import Arguments
from megablocks.layers.moa import dMoA

from flash_attn import flash_attn_func
from rotary_embedding_torch import RotaryEmbedding

class SparseCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.n_head == 0

        self.n_head = config.n_head
        self.top_k = config.k_att
        self.hidden_dim = config.hidden_dim
        self.att_hidden = config.hidden_dim
        self.head_size = config.att_hidden // config.n_head

        args = Arguments(
            hidden_size=config.hidden_dim,
            ffn_hidden_size=self.att_hidden,
            moe_num_experts=config.n_att_experts,
            moe_top_k=config.k_att,
            mlp_type='mlp',
            mlp_impl='grouped',
            memory_optimized_mlp=True,
            bias=False,
            activation_fn=None,
        )
        self.q_proj = dMoA(args)
        self.k_proj = nn.Linear(config.hidden_dim, self.att_hidden)
        self.v_proj = nn.Linear(config.hidden_dim, self.att_hidden)

        self.rotary_embed = RotaryEmbedding(self.head_size // 2)
        rope_freqs = self.rotary_embed.freqs.data
        del self.rotary_embed.freqs
        self.rotary_embed.register_buffer("freqs", rope_freqs)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj.map(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        context_length = k.size(1)
        
        q = q.view(B, T, self.top_k * self.n_head, self.head_size) # (B, T, k * nh, hs)
        k = k.view(B, context_length, self.n_head, self.head_size) # (B, T, nh, hs)
        v = v.view(B, context_length, self.n_head, self.head_size) # (B, T, nh, hs)

        k = k.repeat(1, 1, self.top_k, 1) # (B, T, k * nh, hs)
        v = v.repeat(1, 1, self.top_k, 1) # (B, T, k * nh, hs)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        q = self.rotary_embed.rotate_queries_or_keys(q, seq_dim=-2, offset=context_length - T)
        k = self.rotary_embed.rotate_queries_or_keys(k, seq_dim=-2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        y = flash_attn_func(q, k, v, causal=True)

        # output projection
        y = self.q_proj.reduce(y.reshape(B, T, self.top_k, self.att_hidden).type_as(x))

        y = y.view(B, T, C) # re-assemble all head outputs side by side
        return y
```

# Original Introduction
MegaBlocks is a light-weight library for mixture-of-experts (MoE) training. The core of the system is efficient "dropless-MoE" ([dMoE](megablocks/layers/dmoe.py), [paper](https://arxiv.org/abs/2211.15841)) and standard [MoE](megablocks/layers/moe.py) layers.

MegaBlocks is integrated with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), where we support data, expert and pipeline parallel training of MoEs. Stay tuned for tighter integration with Databricks libraries and tools!

# :rocket: Performance

![MegaBlocks Performance](media/dropping_end_to_end.png)

MegaBlocks dMoEs outperform MoEs trained with [Tutel](https://github.com/microsoft/tutel) by up to **40%** compared to Tutel's best performing `capacity_factor` configuration. MegaBlocks dMoEs use a reformulation of MoEs in terms of block-sparse operations, which allows us to avoid token dropping without sacrificing hardware efficiency. In addition to being faster, MegaBlocks simplifies MoE training by removing the `capacity_factor` hyperparameter altogether. Compared to dense Transformers trained with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), MegaBlocks dMoEs can accelerate training by as much as **2.4x**. Check out our [paper](https://arxiv.org/abs/2211.15841) for more details!

# :building_construction: Installation

NOTE: This assumes you have `numpy` and `torch` installed.

**Training models with Megatron-LM:** We recommend using NGC's [`nvcr.io/nvidia/pytorch:23.09-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) PyTorch container. The [Dockerfile](Dockerfile) builds on this image with additional dependencies. To build the image, run `docker build . -t megablocks-dev` and then `bash docker.sh` to launch the container. Once inside the container, install MegaBlocks with `pip install .`. See [Usage](#steam_locomotive-usage) for instructions on training MoEs with MegaBlocks + Megatron-LM.

**Using MegaBlocks in other packages:** To install the MegaBlocks package for use in other frameworks, run `pip install megablocks`. For example, [Mixtral-8x7B](https://mistral.ai/news/mixtral-of-experts/) can be run with [vLLM](https://github.com/vllm-project/vllm) + MegaBlocks with this installation method.

**Extras:** MegaBlocks has optional dependencies that enable additional features.

Installing `megablocks[gg]` enables dMoE computation with grouped GEMM. This feature is enabled by setting the `mlp_impl` argument to `grouped`. This is currently our recommended path for Hopper-generation GPUs.

MegaBlocks can be installed with all dependencies via the `megablocks[all]` package.

# :steam_locomotive: Usage

We provide scripts for pre-training Transformer MoE and dMoE language models under the [top-level directory](megablocks/). The quickest way to get started is to use one of the [experiment launch scripts](exp/). These scripts require a dataset in Megatron-LM's format, which can be created by following their [instructions](https://github.com/NVIDIA/Megatron-LM#data-preprocessing).

# :writing_hand: Citation

```
@article{megablocks,
  title={{MegaBlocks: Efficient Sparse Training with Mixture-of-Experts}},
  author={Trevor Gale and Deepak Narayanan and Cliff Young and Matei Zaharia},
  journal={Proceedings of Machine Learning and Systems},
  volume={5},
  year={2023}
}
```
