import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss as CrossEntropyLossTemp

from typing import Union

class CrossEntropyLoss(CrossEntropyLossTemp):
    def __init__(
            self,
            weight: Union[Tensor, None] = None,
            size_average=None,
            ignore_index: int = -100,
            reduce=None,
            reduction: str = 'mean',
            label_smoothing: float = 0,
            class_index: int = 1) -> None:
        super().__init__(
            weight,
            size_average,
            ignore_index,
            reduce,
            reduction,
            label_smoothing)
        self.class_index = class_index
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.ndim > 1:
            input = torch.movedim(input, self.class_index, 1)
            target = torch.movedim(target, self.class_index, 1)
        return super().forward(input, target)