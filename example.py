import jax

from jax_transformer.main import causal_mask, transformer_decoder

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
