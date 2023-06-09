from torch import nn

from torch.nn.parameter import Parameter
from torch import Tensor
import torch


class ExtraNode(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        # >>> m = nn.Linear(20, 30)
        # >>> input = torch.randn(128, 20)
        # >>> output = m(input)
        # >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(ExtraNode, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.ones((out_features, in_features)).mul(-0.01))

        if bias:
            self.bias = Parameter(torch.ones((1)).mul(200))

        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        print("requires_grad of parameters of Sub_node -> False")
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        x = torch.pow(input,2)
        x = x.mul(self.weight)
        x = torch.sum(x,1)
        x = torch.add(x,self.bias)
        return torch.unsqueeze(x,-1)