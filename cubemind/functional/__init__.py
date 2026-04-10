"""CubeMind Functional API — stateless, composable, grilly-accelerated.

Decorator-based functions usable by both production and experiments.
All functions are pure, grilly GPU-accelerated where available.

Usage:
    from cubemind.functional import F
    from cubemind.functional.decorators import layer, gpu_fallback, timed, logged

    # Functional ops
    score = F.cosine_similarity(a, b)
    h = F.gelu(x)
    w = F.oja_update(w, x)
    idx, weights = F.top_k(scores, k=2)

    # Layer factory (resolves from grilly.nn → cubemind.brain)
    linear = layer("Linear", in_features=64, out_features=32)
    gif = layer("GIFNeuron", input_dim=64, hidden_dim=64, L=16)

    # Decorators for custom functions
    @gpu_fallback
    def my_op(x): ...
"""

from . import kernels, activations, similarity, learning, routing
from .decorators import layer, gpu_fallback as gpu_fallback, timed as timed, logged as logged


class F:
    """Functional namespace — cubemind.functional.F.*"""

    # Kernels
    rbf_kernel = staticmethod(kernels.rbf_kernel)
    matern_kernel = staticmethod(kernels.matern_kernel)
    rkhs_distance = staticmethod(kernels.rkhs_distance)
    rff_transform = staticmethod(kernels.rff_transform)
    create_rff_params = staticmethod(kernels.create_rff_params)

    # Activations
    gelu = staticmethod(activations.gelu)
    sign = staticmethod(activations.sign_activation)
    additive_sigmoid = staticmethod(activations.additive_sigmoid)

    # Similarity
    cosine_similarity = staticmethod(similarity.cosine_similarity)
    batch_cosine_similarity = staticmethod(similarity.batch_cosine_similarity)
    l1_distance = staticmethod(similarity.l1_distance)

    # Learning rules
    oja_update = staticmethod(learning.oja_update)
    oja_update_batch = staticmethod(learning.oja_update_batch)
    hebbian_update = staticmethod(learning.hebbian_update)
    anti_hebbian_update = staticmethod(learning.anti_hebbian_update)
    stdp_update = staticmethod(learning.stdp_update)

    # Routing
    top_k = staticmethod(routing.top_k_select)
    softmax = staticmethod(routing.softmax)
    gumbel_softmax = staticmethod(routing.gumbel_softmax)
    entropy = staticmethod(routing.entropy)
    load_balance_loss = staticmethod(routing.load_balance_loss)

    # Layer factory
    layer = staticmethod(layer)
