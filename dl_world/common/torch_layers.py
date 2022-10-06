from torch import nn

from typing import Type, List


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
):
    """
    Create an MLP with the given data
    :param input_dim: Input data dimension
    :param output_dim: Target data dimension
    :param net_arch: List of Neurons in each layer
    :param activation_fn: Activation function between each layer
    :return: NN layers as a List
    """
    modules = []
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_dim, output_dim))

    return modules
