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

from nzip.quantization import PatchAttribute


class TestUtils:
    def test_patch_attribute(self):
        class Foo:
            def __init__(self):
                self.message = None

        foo = Foo()
        assert foo.message is None

        with PatchAttribute(foo, 'message', 'hello!'):
            assert foo.message == 'hello!'
        assert foo.message is None
