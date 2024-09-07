import jax
from jax_transformer import ffn

# Example usage of the feed-forward network

batch_size = 2
seq_len = 10
dim = 64
d_ff = 256  # Hidden layer size
dropout_rate = 0.1

# Input tensor
x = jax.random.normal(
    jax.random.PRNGKey(0), (batch_size, seq_len, dim)
)

# Random key for dropout and FFN
rng = jax.random.PRNGKey(42)

# Forward pass through the FFN
ffn_output = ffn(x, d_ff, dim, dropout_rate, rng)

print(ffn_output)  # Output: (batch_size, seq_len, dim)
