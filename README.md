[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

# Jax Transformer
[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


This repository demonstrates how to build a **Decoder-Only Transformer** with **Multi-Query Attention** in **JAX**. Multi-Query Attention is an efficient variant of the traditional multi-head attention, where all attention heads share the same key-value pairs, but maintain separate query projections.

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Installation](#installation)
- [Usage](#usage)
- [Code Walkthrough](#code-walkthrough)
  - [Multi-Query Attention](#multi-query-attention)
  - [Feed-Forward Layer](#feed-forward-layer)
  - [Transformer Decoder Block](#transformer-decoder-block)
  - [Causal Masking](#causal-masking)
- [Running the Transformer Decoder](#running-the-transformer-decoder)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project is a tutorial for building Transformer models from scratch in **JAX**, with a specific focus on implementing **Decoder-Only Transformers** using **Multi-Query Attention**. Transformers are state-of-the-art models used in various NLP tasks, including language modeling, text generation, and more. Multi-Query Attention (MQA) is an optimized version of multi-head attention, which reduces memory and computational complexity by sharing key and value matrices across all heads.

## Key Concepts

- **Multi-Query Attention**: Shares a single key and value across all attention heads, reducing memory usage and computational overhead compared to traditional multi-head attention.
- **Transformer Decoder Block**: A core component of decoder models, which consists of multi-query attention, a feed-forward network, and residual connections.
- **Causal Masking**: Ensures that each position in the sequence can only attend to itself and previous positions to prevent future token leakage during training.

## Installation

```bash
pip3 install -U jax-transformer
```

### Requirements

- **JAX**: A library for high-performance machine learning research. Install JAX with GPU support (optional) by following the instructions on the [JAX GitHub page](https://github.com/google/jax).

## Usage

After installing the dependencies, you can run the model on random input data to see how the transformer decoder works:

```python
import jax
from jax_transformer.main import transformer_decoder, causal_mask

# Example usage
batch_size = 2
seq_len = 10
dim = 64
heads = 8
d_ff = 256
depth = 6

# Random input tokens
x = jax.random.normal(
    jax.random.PRNGKey(0), (batch_size, seq_len, dim)
)
rng = jax.random.PRNGKey(42)
# Generate causal mask
mask = causal_mask(seq_len)

# Run through transformer decoder
out = transformer_decoder(
    x=x,
    mask=mask,
    depth=depth,
    heads=heads,
    dim=dim,
    d_ff=d_ff,
    dropout_rate=0.1,
    rng=rng,
)


print(out.shape)  # Should be (batch_size, seq_len, dim)

```

## Code Walkthrough

This section explains the key components of the model in detail.

### Multi-Query Attention

The **Multi-Query Attention** mechanism replaces the traditional multi-head attention by sharing the same set of key-value pairs for all heads while keeping separate query projections. This drastically reduces the memory footprint and computation.

```python
def multi_query_attention(query, key, value, mask):
    ...
```

### Feed-Forward Layer

After the attention mechanism, the transformer applies a two-layer feed-forward network with a ReLU activation in between. This allows the model to add depth and capture complex patterns.

```python
def feed_forward(x, d_ff):
    ...
```

### Transformer Decoder Block

The **Transformer Decoder Block** combines the multi-query attention mechanism with the feed-forward network and adds **residual connections** and **layer normalization** to stabilize the learning process. It processes sequences in a causal manner, meaning that tokens can only attend to previous tokens, which is crucial for auto-regressive models (e.g., language models).

```python
def transformer_decoder_block(x, key, value, mask, num_heads, d_model, d_ff):
    ...
```

### Causal Masking

The **Causal Mask** ensures that during training or inference, tokens in the sequence can only attend to themselves or previous tokens. This prevents "future leakage" and is crucial for tasks such as language modeling and text generation.

```python
def causal_mask(seq_len):
    ...
```

## Running the Transformer Decoder

To run the decoder model, execute the following script:

```python
python run_transformer.py
```

The model takes random input and runs it through the Transformer decoder stack with multi-query attention. The output shape will be `(batch_size, seq_len, d_model)`.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request with your improvements. You can also open an issue if you find a bug or want to request a new feature.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Citation


```bibtex
@article{JaxTransformer,
    author={Kye Gomez},
    title={Jax Transformer},
    year={2024},
}
```
