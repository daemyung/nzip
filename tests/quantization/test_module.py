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

from nzip.quantization import MinMaxAnalyzer, Quantizer, Dequantizer


class TestModule:
    @pytest.mark.parametrize('symmetric', [True, False])
    def test_quantizer(self, symmetric):
        quantizer = Quantizer(3, MinMaxAnalyzer(symmetric))
        input = torch.tensor([-9.8, -7.8, -5.8, -3.8, -1.8, 0.2, 2.2, 4.2], requires_grad=True)

        with pytest.raises(RuntimeError):
            quantizer(input)

        assert quantizer.min is None
        assert quantizer.max is None

        with quantizer.calibrate():
            quantizer(input)

        if symmetric:
            assert quantizer.min == torch.tensor(-9.8)
            assert quantizer.max == torch.tensor(9.8)
        else:
            assert quantizer.min == torch.tensor(-9.8)
            assert quantizer.max == torch.tensor(4.2)

        output = quantizer(input)

        if symmetric:
            expectation = torch.tensor([-3.0, -2.0, -2.0, -1.0, -1.0, 0.0, 1.0, 1.0])
            assert torch.allclose(output, expectation)
        else:
            expectation = torch.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
            assert torch.allclose(output, expectation)

        output.backward(torch.ones_like(output))
        assert torch.all(torch.eq(input.grad, 1.0))

    def test_dequantizer(self):
        scale = torch.tensor(0.3061)
        dequantizer = Dequantizer(scale, None)
        input = torch.tensor([-3.0, -2.0, -2.0, -1.0, -1.0, 0.0, 1.0, 1.0])
        output = dequantizer(input)
        expectation = torch.tensor([-9.801, -6.534, -6.534, -3.267, -3.267, 0.0, 3.267, 3.267])
        assert torch.allclose(output, expectation, atol=1e-2)

    def test_dequantizer_with_bias(self):
        scale = torch.tensor(0.5)
        bias = torch.tensor(1.0)
        dequantizer = Dequantizer(scale, bias)
        input = torch.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        output = dequantizer(input)
        expectation = torch.tensor([-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0])
        assert torch.allclose(output, expectation)
