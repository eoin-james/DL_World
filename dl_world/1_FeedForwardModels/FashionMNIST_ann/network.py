from torch import nn

from typing import Type, List

from dl_world.common import create_mlp


class NNModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.net_arch = net_arch
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn

        self.model = self._setup_model()

    def _setup_model(self) -> nn.Module:
        """
        Create the MLP
        :return: Torch NN
        """
        model = create_mlp(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn
        )
        return nn.Sequential(*model)

    def forward(self, x):
        return self.model(nn.Flatten()(x))

