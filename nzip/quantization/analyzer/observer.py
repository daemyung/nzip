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

import abc
from abc import ABC
from typing import Union

import torch
from torch import Tensor

from .stats import Stats, Range


class Observer(ABC):
    dim: tuple[int] = ()
    stats: Stats = None

    def __init__(self, dim: Union[int, list[int], tuple[int, ...]], stats: Stats):
        self.dim = dim if isinstance(dim, tuple) else tuple(dim)
        self.stats = stats

    @abc.abstractmethod
    def compute_stats(self, input: Tensor) -> Stats:
        pass

    @abc.abstractmethod
    def merge_status(self, stats: Stats):
        pass

    @abc.abstractmethod
    def reset_stats(self):
        pass


class MinMaxObserver(Observer):
    def __init__(self, dim: Union[int, list[int], tuple[int, ...]] = ()):
        super().__init__(dim, Range())

    def compute_stats(self, input: Tensor) -> Range:
        return Range(torch.amin(input, self.dim), torch.amax(input, self.dim))

    def merge_status(self, stats: Range):
        if stats.min is not None:
            if self.stats.min is None:
                self.stats.min = stats.min
            else:
                self.stats.min = torch.minimum(self.stats.min, stats.min)

        if stats.max is not None:
            if self.stats.max is None:
                self.stats.max = stats.max
            else:
                self.stats.max = torch.maximum(self.stats.max, stats.max)

    def reset_stats(self):
        self.stats = Range()

    @property
    def min(self):
        return self.stats.min

    @property
    def max(self):
        return self.stats.max
