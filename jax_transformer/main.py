import jax
import jax.numpy as jnp
from typing import Optional


def layer_norm(x: jnp.ndarray, epsilon: float = 1e-6) -> jnp.ndarray:
    """
    Applies Layer Normalization to the input tensor.

    Layer Normalization helps in stabilizing the learning process and reducing
    the covariate shift by normalizing the inputs across the features.

    Args:
        x (jnp.ndarray): Input tensor to be normalized.
        epsilon (float, optional): A small float added to the variance to avoid
                                   division by zero. Defaults to 1e-6.

    Returns:
        jnp.ndarray: The normalized tensor with the same shape as the input.

    Note:
        This function computes the mean and variance across the last dimension
        of the input tensor, which is typically the feature dimension.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(variance + epsilon)
    return normalized


def silu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(x)


def dropout(
    x: jnp.ndarray,
    rate: float = 0.1,
    rng: jax.random.PRNGKey = None,
) -> jnp.ndarray:
    keep_prob = 1 - rate
    mask = jax.random.bernoulli(rng, keep_prob, x.shape)
    return jnp.where(mask, x / keep_prob, 0)


def mqa_attn(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Multi-query attention mechanism with shared key-value across heads.

    Args:
        query (jnp.ndarray): The query matrix of shape (batch_size, seq_len, num_heads, d_k).
        key (jnp.ndarray): The key matrix of shape (batch_size, seq_len, d_k).
        value (jnp.ndarray): The value matrix of shape (batch_size, seq_len, d_v).
        mask (jnp.ndarray): The causal mask of shape (batch_size, 1, seq_len, seq_len).

    Returns:
        jnp.ndarray: The attention output.
    """
    d_k = query.shape[-1]
    scaled_query = query / jnp.sqrt(d_k)

    # Compute attention scores
    attn_weights = jnp.einsum("bhqd,bkd->bhqk", scaled_query, key)

    # Expand the mask to match the shape of attn_weights
    mask = jnp.expand_dims(
        mask, axis=2
    )  # Add the num_heads dimension

    # Apply causal mask: where mask is 0, set attention weights to -inf
    attn_weights = jnp.where(mask == 1, attn_weights, -jnp.inf)

    # Apply softmax
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    # Compute weighted sum of values
    attn_output = jnp.einsum("bhqk,bkd->bhqd", attn_weights, value)

    return attn_output


# def ffn(
#     x: jnp.ndarray,
#     d_ff: int,
# ) -> jnp.ndarray:
#     """
#     Feed-forward network for the Transformer decoder block.

#     Args:
#         x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, dim).
#         d_ff (int): Dimension of the feed-forward layer.

#     Returns:
#         jnp.ndarray: Output tensor of shape (batch_size, seq_len, dim).

#     Note:
#         This function applies two linear transformations with a ReLU activation
#         in between, which is a standard component in Transformer architectures.
#     """
#     w1 = jax.random.normal(
#         jax.random.PRNGKey(0),
#         (x.shape[-1], d_ff),
#     )
#     w2 = jax.random.normal(
#         jax.random.PRNGKey(1),
#         (d_ff, x.shape[-1]),
#     )

#     hidden = jax.nn.relu(x @ w1)
#     output = hidden @ w2

#     return output


def ffn(
    x: jnp.ndarray,
    d_ff: int,
    dim: int,
    dropout_rate: float = 0.1,
    rng: jax.random.PRNGKey = None,
) -> jnp.ndarray:
    w1 = jax.random.normal(
        rng,
        (dim, d_ff),
    )
    b1 = jnp.zeros(d_ff)
    w2 = jax.random.normal(
        rng,
        (d_ff, dim),
    )
    b2 = jnp.zeros(dim)

    # Apply first layer transformation
    hidden = jnp.dot(x, w1) + b1
    hidden = silu(hidden)

    # Apply dropout
    dropout_rng = jax.random.split(rng)[1]
    hidden = dropout(hidden, dropout_rate, dropout_rng)

    # Second linear
    output = jnp.dot(hidden, w2) + b2

    # Layer Normalization
    output = layer_norm(output)
    return output


def decoder_block(
    x: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    heads: int = 8,
    dim: int = None,
    d_ff: int = None,
    dropout_rate: float = 0.1,
    ffn_rng: jax.random.PRNGKey = None,
) -> jnp.ndarray:
    """
    Implements a single decoder block of the Transformer architecture with multi-query attention.

    Args:
        x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, dim).
        key (jnp.ndarray): Key tensor of shape (batch_size, seq_len, dim).
        value (jnp.ndarray): Value tensor of shape (batch_size, seq_len, dim).
        mask (Optional[jnp.ndarray]): Optional mask tensor of shape (batch_size, seq_len, seq_len).
        heads (int): Number of attention heads. Default is 8.
        dim (int): Dimension of the model. Should be divisible by heads.
        d_ff (int): Dimension of the feed-forward network.

    Returns:
        jnp.ndarray: Output tensor of shape (batch_size, seq_len, dim).

    Note:
        This function applies multi-query attention, followed by a feed-forward network,
        with residual connections and layer normalization after each sub-layer.
    """
    # Project input to query, key, value
    d_k = dim // heads  # Dimension per head

    # Project input to query, key, value
    q_proj = jax.random.normal(
        jax.random.PRNGKey(0), (x.shape[0], x.shape[1], heads, d_k)
    )
    k_proj = jax.random.normal(
        jax.random.PRNGKey(1), (key.shape[0], key.shape[1], d_k)
    )  # Shared key
    v_proj = jax.random.normal(
        jax.random.PRNGKey(2), (value.shape[0], value.shape[1], d_k)
    )  # Shared value

    # Apply multi-query attention (shared key-value pairs)
    attn_output = mqa_attn(q_proj, k_proj, v_proj, mask)

    # Concatenate attention output and project back to d_model
    attn_output = jnp.reshape(
        attn_output, (x.shape[0], x.shape[1], dim)
    )

    # Add residual connection and apply layer normalization
    x = layer_norm(x + attn_output)

    # Feed-forward network and another residual connection
    ff_output = ffn(x, d_ff, dim, dropout_rate, ffn_rng)
    output = layer_norm(x + ff_output)

    return output


def transformer_decoder(
    x: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    depth: int = 12,
    heads: int = 8,
    dim: int = None,
    d_ff: int = None,
    dropout_rate: float = 0.1,
    rng: jax.random.PRNGKey = None,
) -> jnp.ndarray:
    """
    Applies a stack of Transformer decoder blocks to the input.

    Args:
        x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, dim).
        mask (Optional[jnp.ndarray]): Optional mask tensor of shape (batch_size, seq_len, seq_len).
        depth (int): Number of decoder blocks to stack. Default is 12.
        heads (int): Number of attention heads. Default is 8.
        dim (int): Dimension of the model. Should be divisible by heads.
        d_ff (int): Dimension of the feed-forward network.

    Returns:
        jnp.ndarray: Output tensor of shape (batch_size, seq_len, dim).

    Note:
        This function applies a stack of Transformer decoder blocks, where each block
        consists of multi-query attention followed by a feed-forward network.
    """
    for _ in range(depth):
        x = decoder_block(
            x, x, x, mask, heads, dim, d_ff, dropout_rate, rng
        )

        x = jax.nn.softmax(x, axis=-1)
    return x


def causal_mask(
    seq_len: int,
) -> jnp.ndarray:
    """
    Creates a causal mask for self-attention.

    Args:
        seq_len (int): Length of the sequence.

    Returns:
        jnp.ndarray: Causal mask of shape (1, seq_len, seq_len).

    Note:
        The causal mask ensures that each position in the sequence can only attend
        to itself and previous positions, preventing information leakage from future tokens.
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return mask[None, :, :]
