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

import nzip.nn.function as function


class TestFunction:
    def test_quantize(self):
        input = torch.tensor([-9.8, -7.8, -5.8, -3.8, -1.8, 0.2, 2.2, 4.2])
        max = torch.max(torch.abs(input))
        bits = 3
        scale = (2 ** (bits - 1) - 1) / max
        output = function.quantize(input, scale, None, -2 ** (bits - 1) + 1, 2 ** (bits - 1) - 1)
        expectation = torch.tensor([-3.0, -2.0, -2.0, -1.0, -1.0, 0.0, 1.0, 1.0])
        assert torch.allclose(output, expectation)

    def test_quantize_with_bias(self):
        input = torch.tensor([-9.8, -7.8, -5.8, -3.8, -1.8, 0.2, 2.2, 4.2])
        min = torch.min(input)
        max = torch.max(input)
        bits = 3
        scale = (2 ** bits - 1) / (max - min)
        bias = -torch.round(min * scale) - 2 ** (bits - 1)
        output = function.quantize(input, scale, bias, -2 ** (bits - 1), 2 ** (bits - 1) - 1)
        expectation = torch.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        assert torch.allclose(output, expectation)

    def test_quantize_backpropagation(self):
        input = torch.tensor([-9.8, -7.8, -5.8, -3.8, -1.8, 0.2, 2.2, 4.2], requires_grad=True)
        scale = torch.tensor(0.3061)
        output = function.quantize(input, scale, None, -3, 3)
        output.backward(torch.ones_like(output))
        expectation = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert torch.allclose(input.grad, expectation)

    def test_quantize_with_bias_backpropagation(self):
        input = torch.tensor([-9.8, -7.8, -5.8, -3.8, -1.8, 0.2, 2.2, 4.2], requires_grad=True)
        scale = torch.tensor(0.5)
        bias = torch.tensor(1.0)
        output = function.quantize(input, scale, bias, -2, 1)
        output.backward(torch.ones_like(output))
        expectation = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        assert torch.allclose(input.grad, expectation)
