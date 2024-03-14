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

import pytest
import torch

from nzip.quantization.analyzer import MinMaxAnalyzer, Range


class TestAnalyzer:
    @pytest.mark.parametrize('symmetric', [True, False])
    def test_min_max_analyzer(self, bit_width, symmetric):
        analyzer = MinMaxAnalyzer()

        with pytest.raises(ValueError):
            analyzer.compute_stats(bit_width, symmetric)

        input = torch.arange(-1.0, 10.0)
        analyzer.update_stats(input)
        stats = analyzer.compute_stats(bit_width, symmetric)

        if symmetric:
            assert stats == Range(torch.tensor(-9.0), torch.tensor(9.0))
        else:
            assert stats == Range(torch.tensor(-1.0), torch.tensor(9.0))
