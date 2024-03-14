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

from .observer import Observer, MinMaxObserver
from .stats import Stats, Range


class Analyzer(ABC):
    observer: Observer

    def __init__(self, observer: Observer):
        self.observer = observer

    @abc.abstractmethod
    def compute_stats(self, bit_width: int, symmetric: bool) -> Stats:
        pass

    def update_stats(self, input: Tensor):
        stats = self.observer.compute_stats(input)
        self.observer.merge_status(stats)

    def reset_stats(self):
        self.observer.reset_stats()

    @property
    def stats(self):
        return self.observer.stats


class MinMaxAnalyzer(Analyzer):
    def __init__(self, dim: Union[int, list[int], tuple[int, ...]] = ()):
        super().__init__(MinMaxObserver(dim))

    def compute_stats(self, bit_width: int, symmetric: bool) -> Range:
        if self.stats.min is None or self.stats.max is None:
            raise ValueError

        if symmetric:
            return Range(torch.minimum(self.stats.min, -self.stats.max),
                         torch.maximum(-self.stats.max, self.stats.max))
        else:
            return self.stats
