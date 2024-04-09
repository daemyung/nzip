# Copyright 2024 Daemyung Jang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import Function


class Quantize(Function):
    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        return Quantize.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], outputs: Any):
        Quantize.__setup_context(ctx, outputs[1])

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return Quantize.__backward(*ctx.saved_tensors, grad_outputs[0])

    @staticmethod
    def __forward(input: Tensor, scale: Tensor, bias: Optional[Tensor], lower_bound: int, upper_bound: int) -> Any:
        output = scale * input

        if bias is not None:
            output += bias

        torch.round(output, out=output)
        condition = torch.logical_and(torch.ge(output, lower_bound), torch.le(output, upper_bound))
        torch.clip(output, lower_bound, upper_bound, out=output)
        return output, condition

    @staticmethod
    def __setup_context(ctx: Any, condition: Tensor):
        ctx.save_for_backward(condition)

    @staticmethod
    def __backward(condition: Tensor, grad_output: Tensor) -> Any:
        grad_input = torch.where(condition, 1.0, 0.0)
        return grad_output * grad_input, None, None, None, None


def quantize(input: Tensor, scale: Tensor, bias: Optional[Tensor], lower_bound: int, upper_bound: int) -> Tensor:
    return Quantize.apply(input, scale, bias, lower_bound, upper_bound)[0]


class Dequantize(Function):
    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        return Dequantize.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], outputs: Any):
        Dequantize.__setup_context(ctx, inputs[1])

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return Dequantize.__backward(*ctx.saved_tensors, grad_outputs[0])

    @staticmethod
    def __forward(input: Tensor, scale: Tensor, bias: Optional[Tensor]) -> Any:
        output = input.detach().clone() if bias is None else input - bias
        torch.div(output, scale, out=output)
        return output

    @staticmethod
    def __setup_context(ctx: Any, scale: Tensor):
        ctx.save_for_backward(scale)

    @staticmethod
    def __backward(scale: Tensor, grad_output: Tensor):
        return grad_output / scale, None, None


def dequantize(input: Tensor, scale: Tensor, bias: Optional[Tensor]) -> Tensor:
    return Dequantize.apply(input, scale, bias)
