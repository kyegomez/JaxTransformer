import jax
import jax.numpy as jnp
from flax import nnx


class Linear(nnx.Module):
    """
    A linear layer (fully connected layer) implemented as a Flax module.

    This layer performs a linear transformation on the input data.

    Attributes:
        w (nnx.Param): The weight matrix of shape (in_features, out_features).
        b (nnx.Param): The bias vector of shape (out_features,).
        in_features (int): The number of input features.
        out_features (int): The number of output features.
    """

    def __init__(
        self, in_features: int, out_features: int, *, rngs: nnx.Rngs
    ):
        """
        Initialize the Linear layer.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            rngs (nnx.Rngs): Random number generator for parameter initialization.
        """
        key = rngs.params()
        self.w = nnx.Param(
            jax.random.uniform(key, (in_features, out_features))
        )
        self.b = nnx.Param(jnp.zeros((out_features,)))
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Perform the forward pass of the linear layer.

        Args:
            x (jax.Array): Input array of shape (..., in_features).

        Returns:
            jax.Array: Output array of shape (..., out_features).

        Note:
            There appears to be a typo in the original implementation.
            The line `return x & self.w + self.b` should likely be
            `return jnp.dot(x, self.w) + self.b`.
        """
        return x @ self.w + self.b


# model = Linear(2, 5, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)))
# y = model(x=jnp.ones((1, 2)))

# print(y)
# nnx.display(model)


class LoraParam(nnx.Param):
    pass


class LoraLinear(nnx.Module):
    def __init__(self, linear: Linear, rank: int, rngs: nnx.Rngs):
        self.linear = linear
        self.A = LoraParam(
            jax.random.uniform(rngs(), (linear.in_features, rank))
        )
        self.B = LoraParam(
            jax.random.uniform(rngs(), (rank, linear.out_features))
        )

    def __call__(self, x: jax.Array):
        return self.linear(x) + x @ self.A @ self.B


# model = LoraLinear(
#     linear=Linear(2, 5, rngs=nnx.Rngs(params=jax.random.PRNGKey(0))),
# )

# y = model(x=jnp.ones((1, 2)))
# print(y)
# nnx.display(model)
