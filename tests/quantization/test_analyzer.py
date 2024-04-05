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

from nzip.quantization import MinMaxAnalyzer


class TestMinMaxAnalyzer:
    @pytest.mark.parametrize('symmetric', [True, False])
    def test_compute_stats(self, symmetric):
        stats = MinMaxAnalyzer(symmetric).compute_stats(torch.arange(-9.8, 4.3, 0.1))

        if symmetric:
            assert torch.allclose(stats.min, torch.tensor(-9.8))
            assert torch.allclose(stats.max, torch.tensor(9.8))
        else:
            assert torch.allclose(stats.min, torch.tensor(-9.8))
            assert torch.allclose(stats.max, torch.tensor(4.2))

    @pytest.mark.parametrize('symmetric', [True, False])
    def test_merge_stats(self, symmetric):
        analyzer = MinMaxAnalyzer(symmetric)
        stats = analyzer.compute_stats(torch.arange(0.0, 4.3, 0.1))
        analyzer.merge_stats(stats)

        if symmetric:
            assert torch.allclose(analyzer.stats.min, torch.tensor(-4.2))
            assert torch.allclose(analyzer.stats.max, torch.tensor(4.2))
        else:
            assert torch.allclose(analyzer.stats.min, torch.tensor(0.0))
            assert torch.allclose(analyzer.stats.max, torch.tensor(4.2))

        stats = analyzer.compute_stats(torch.arange(0.2, 2.2, 0.1))
        analyzer.merge_stats(stats)

        if symmetric:
            assert torch.allclose(analyzer.stats.min, torch.tensor(-4.2))
            assert torch.allclose(analyzer.stats.max, torch.tensor(4.2))
        else:
            assert torch.allclose(analyzer.stats.min, torch.tensor(0.0))
            assert torch.allclose(analyzer.stats.max, torch.tensor(4.2))

        stats = analyzer.compute_stats(torch.arange(-9.8, 0.1, 0.1))
        analyzer.merge_stats(stats)

        if symmetric:
            assert torch.allclose(analyzer.stats.min, torch.tensor(-9.8))
            assert torch.allclose(analyzer.stats.max, torch.tensor(9.8))
        else:
            assert torch.allclose(analyzer.stats.min, torch.tensor(-9.8))
            assert torch.allclose(analyzer.stats.max, torch.tensor(4.2))

    @pytest.mark.parametrize("symmetric", [True, False])
    def test_update_stats(self, symmetric):
        analyzer = MinMaxAnalyzer(symmetric)
        analyzer.update_stats(torch.arange(-9.8, 0.1, 0.1))

        if symmetric:
            assert torch.allclose(analyzer.stats.min, torch.tensor(-9.8))
            assert torch.allclose(analyzer.stats.max, torch.tensor(9.8))
        else:
            assert torch.allclose(analyzer.stats.min, torch.tensor(-9.8))
            assert torch.allclose(analyzer.stats.max, torch.tensor(0.0))

    def test_reset_stats(self):
        analyzer = MinMaxAnalyzer(symmetric=False)
        analyzer.update_stats(torch.arange(0.0, 9.0))

        assert analyzer.stats.min is not None
        assert analyzer.stats.max is not None

        analyzer.reset_stats()

        assert analyzer.stats.min is None
        assert analyzer.stats.max is None
