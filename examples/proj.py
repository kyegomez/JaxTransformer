import jax
from jax_transformer import linear_projection

# Example usage of linear projection
batch_size = 2
seq_len = 10
in_features = 64
out_features = 128

# Input tensor of shape (batch_size, seq_len, in_features)
x = jax.random.normal(
    jax.random.PRNGKey(0), (batch_size, seq_len, in_features)
)

# Random key for weight initialization
rng = jax.random.PRNGKey(42)

# Apply linear projection
output = linear_projection(x, in_features, out_features, rng)

print(output.shape)  # Output: (batch_size, seq_len, out_features)
