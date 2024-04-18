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
from typing import Callable, List, Tuple, Union

import torch
from torch import Tensor

from .stats import Stats, Range


class Analyzer(ABC):
    def __init__(self, stats: Stats, symmetric: bool, dim: Union[int, Tuple, List] = ()):
        """
        Constructor.
        :param stats: The stats to be computed.
        :param symmetric: Whether symmetric analysis is used or not.
        :param dim: The dimension or dimensions to reduce.
        """
        self.stats = stats
        self.symmetric = symmetric
        self.dim = dim if isinstance(dim, tuple) else tuple(dim)

    @abc.abstractmethod
    def compute_stats(self, input: Tensor) -> Stats:
        """
        Compute the stats from the given input.
        :param input: The input tensor to compute stats.
        :return: The computed stats.
        """

    @abc.abstractmethod
    def merge_stats(self, stats: Stats):
        """
        Merge the stats with the existing stats.
        :param stats: The stats to be merged.
        """

    def update_stats(self, input: Tensor):
        """
        Update the stats from the given input.
        :param input: The input tensor to update the stats.
        """
        stats = self.compute_stats(input)
        self.merge_stats(stats)

    def reset_stats(self):
        """
        Reset the stats.
        """
        self.stats = type(self.stats)()


class MinMaxAnalyzer(Analyzer):
    def __init__(self, symmetric: bool, dim: Union[int, Tuple, List] = ()):
        super().__init__(Range(), symmetric, dim)

    def compute_stats(self, input: Tensor) -> Stats:
        if self.symmetric:
            max = torch.amax(torch.abs(input), self.dim)
            min = -max
        else:
            min = torch.amin(input, self.dim)
            max = torch.amax(input, self.dim)

        return Stats(min, max)

    def merge_stats(self, stats: Stats):
        if stats.min is not None:
            self.__merge_bounds('min', stats.min, torch.minimum)

        if stats.max is not None:
            self.__merge_bounds('max', stats.max, torch.maximum)

    def __merge_bounds(self, name: str, value: Tensor, compare: Callable):
        """
        Merge the value with the existing bound based on the result of the comparison.

        :param name: The name of the attribute to be merged.
        :param value: The value of the attribute to be merged.
        :param compare: A function that compares two values and updates the attribute.
        """
        if (attr := getattr(self.stats, name)) is None:
            setattr(self.stats, name, value)
        else:
            compare(attr, value, out=attr)
