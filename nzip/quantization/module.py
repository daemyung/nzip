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

from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

import nzip.nn.function as function
from nzip.quantization.analyzer import Analyzer
from nzip.quantization.utils import PatchAttribute


class Quantizer(Module):
    def __init__(self, bits: int, analyzer: Analyzer):
        """
        Constructor.

        :param bits: The number of bits to use for the quantization.
        :param analyzer: An instance of Analyzer, which provides information for the quantization.
        """
        super().__init__()
        self.bits = bits
        self.analyzer = analyzer
        self.min = None
        self.max = None

    @contextmanager
    def calibrate(self):
        """
        Calibrate the quantization parameters.
        """

        @wraps(self.forward)
        def wrapper(input: Tensor) -> Tensor:
            self.analyzer.update_stats(input)

            with ExitStack() as stack:
                stack.enter_context(PatchAttribute(self, 'min', self.analyzer.stats.min))
                stack.enter_context(PatchAttribute(self, 'max', self.analyzer.stats.max))
                return wrapper.forward(input)

        wrapper.forward = self.forward

        try:
            with PatchAttribute(self, 'forward', wrapper):
                yield
        except Exception:
            raise
        else:
            self.min = self.analyzer.stats.min.detach().clone()
            self.max = self.analyzer.stats.max.detach().clone()
        finally:
            self.analyzer.reset_stats()

    def forward(self, input: Tensor) -> Tensor:
        """
        Perform the quantization on the input.

        :param input: The input to be quantized.
        :return: The quantized tensor.
        """
        if self.min is None and self.max is None:
            raise RuntimeError('Quantization parameters are not initialized.')

        return function.quantize(input, self.scale, self.bias, self.lower_bound, self.upper_bound)

    @property
    def scale(self) -> Tensor:
        """
        Return the scale for the quantization.

        :return: The scale for the quantization.
        """
        if self.symmetric:
            return self.upper_bound / self.max
        else:
            return (2 ** self.bits - 1) / (self.max - self.min)


    @property
    def bias(self) -> Optional[Tensor]:
        """
        Return the bias for the quantization.

        :return: The bias for the quantization.
        """
        if self.symmetric:
            return None
        else:
            return -torch.round(self.min * self.scale) - 2 ** (self.bits - 1)

    @property
    def symmetric(self) -> bool:
        """
        Return whether the current context is symmetric.

        :return: True if it is symmetric, False otherwise.
        """
        return self.analyzer.symmetric

    @property
    def lower_bound(self) -> int:
        """
        Return the lower bound of the value range for the given number of bits.

        :return: The lower bound that can be represented.
        """
        return -2 ** (self.bits - 1) + 1 if self.symmetric else -2 ** (self.bits - 1)

    @property
    def upper_bound(self) -> int:
        """
        Return the upper bound of the value range for the given number of bits.

        :return: The upper bound that can be represented.
        """
        return 2 ** (self.bits - 1) - 1
