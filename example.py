from jax_transformer.mqa_attn import transformer_decoder, causal_mask
import jax

# Example usage
batch_size = 2
seq_len = 10
d_model = 64
num_heads = 8
d_ff = 256
num_layers = 6

# Random input tokens
x = jax.random.normal(
    jax.random.PRNGKey(0), (batch_size, seq_len, d_model)
)

# Generate causal mask
mask = causal_mask(seq_len)

# Run through transformer decoder
out = transformer_decoder(
    x=x,
    mask=mask,
    depth=num_layers,
    heads=num_heads,
    dim=d_model,
    d_ff=d_ff,
)


print(out)  # Should be (batch_size, seq_len, d_model)
