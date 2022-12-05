import math
import torch


def gradient_redistribution(masking, name, weight, mask):
    '''
    calculate distribution based on gradient
    :param masking: Masking class with state about current
                    layers and the entire sparse network.
    :param name:    The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.
    :param weight:  The weight of the respective sparse layer.
                    This is a torch parameter.
    :param mask:    The binary mask. 1s indicated active weights.
    :return: Layer importance:
                    The unnormalized layer importance statistic
                    for the layer "name". A higher value indicates
                    that more pruned parameters are redistributed
                    to this layer compared to layers with lower value.
                    The values will be automatically sum-normalized
                    after this step.
    '''
    grad = masking.get_gradient_for_weight(weight)
    layer_importance = torch.abs(grad[mask.bool()]).mean().item()
    return layer_importance


def gradient_growth(masking, name, new_mask, total_regrowth, weight):
    """Grows weights in places where the momentum is largest.

    Growth function in the sparse learning library work by
    changing 0s to 1s in a binary mask which will enable
    gradient flow. Weights default value are 0 and it can
    be changed in this function. The number of parameters
    to be regrown is determined by the total_regrowth
    parameter. The masking object in conjunction with the name
    of the layer enables the access to further statistics
    and objects that allow more flexibility to implement
    custom growth functions.

    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.

        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.

        new_mask    The binary mask. 1s indicated active weights.
                    This binary mask has already been pruned in the
                    pruning step that preceeds the growth step.

        total_regrowth    This variable determines the number of
                    parameters to regrowtn in this function.
                    It is automatically determined by the
                    redistribution function and algorithms
                    internal to the sparselearning library.

        weight      The weight of the respective sparse layer.
                    This is a torch parameter.

    Returns:
        mask        Binary mask with newly grown weights.
                    1s indicated active weights in the binary mask.

    Access to optimizer:
        masking.optimizer

    Access to momentum/Adam update:
        masking.get_momentum_for_weight(weight)

    Accessable global statistics:

    Layer statistics:
        Non-zero count of layer:
            masking.name2nonzeros[name]
        Zero count of layer:
            masking.name2zeros[name]
        Redistribution proportion:
            masking.name2variance[name]
        Number of items removed through pruning:
            masking.name2removed[name]

    Network statistics:
        Total number of nonzero parameter in the network:
            masking.total_nonzero = 0
        Total number of zero-valued parameter in the network:
            masking.total_zero = 0
        Total number of parameters removed in pruning:
            masking.total_removed = 0
    """
    grad = weight
    if grad.dtype == torch.float16:  # 转换为半精度降低计算量
        grad = grad * (new_mask == 0).half()
    else:
        grad = grad * (new_mask == 0).float()
    _, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
    new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

    return new_mask

