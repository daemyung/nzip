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

import torch

from nzip.quantization.analyzer import MinMaxObserver


class TestObserver:
    def test_min_max_observer(self):
        observer = MinMaxObserver()
        assert observer.min is None
        assert observer.max is None

        input = torch.arange(-4.0, 20.0)
        stats = observer.compute_stats(input)
        assert stats.min == -4.0
        assert stats.max == 19.0

        observer.merge_status(stats)
        assert observer.min == -4.0
        assert observer.max == 19.0

        input = torch.arange(-8.0, 11.0)
        stats = observer.compute_stats(input)
        assert stats.min == -8.0
        assert stats.max == 10.0

        observer.merge_status(stats)
        assert observer.min == -8.0
        assert observer.max == 19.0
