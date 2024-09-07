import jax
import jax.numpy as jnp


def embedding_layer(
    token_ids: jnp.ndarray,
    vocab_size: int,
    d_model: int,
    rng: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Embedding layer that converts token indices to dense embeddings.

    Args:
        token_ids (jnp.ndarray): Input token indices of shape (batch_size, seq_len).
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the embeddings (model dimensionality).
        rng (jax.random.PRNGKey): Random key for initializing the embeddings.

    Returns:
        jnp.ndarray: Embedding vectors of shape (batch_size, seq_len, d_model).
    """
    # Initialize embedding matrix with shape (vocab_size, d_model)
    embedding_matrix = jax.random.normal(rng, (vocab_size, d_model))

    # Look up embeddings for each token in token_ids
    embeddings = jnp.take(embedding_matrix, token_ids, axis=0)

    return embeddings


# # Example inputs
# batch_size = 2
# seq_len = 10
# vocab_size = 1000
# d_model = 64

# # Token IDs (randomly chosen for demonstration)
# token_ids = jnp.array([[5, 10, 7, 23, 45, 67, 89, 34, 56, 12],
#                        [13, 20, 15, 30, 48, 60, 90, 31, 58, 14]])

# # Random key for embedding initialization
# rng = jax.random.PRNGKey(0)

# # Generate embeddings
# embeddings = embedding_layer(token_ids, vocab_size, d_model, rng)

# print(embeddings.shape)  # Output: (batch_size, seq_len, d_model)
