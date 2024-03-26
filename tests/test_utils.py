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

from contextlib import ExitStack

from nzip.utils import PatchAttribute


class TestUtils:
    def test_patch_attribute(self):
        class Foo:
            def __init__(self):
                self.name = 'foo'

        foo = Foo()
        assert foo.name == 'foo'
        assert not hasattr(foo, 'id')

        with ExitStack() as stack:
            stack.enter_context(PatchAttribute(foo, 'name', 'bar'))
            assert foo.name == 'bar'
        assert foo.name == 'foo'

        assert foo.name == 'foo'
        with PatchAttribute(foo, 'name', 'baz'):
            assert foo.name == 'baz'
        assert foo.name == 'foo'

        with ExitStack() as stack:
            stack.enter_context(PatchAttribute(foo, 'id', 1))
            assert hasattr(foo, 'id')
            assert getattr(foo, 'id') == 1
        assert not hasattr(foo, 'id')

        with PatchAttribute(foo, 'id', 2):
            assert hasattr(foo, 'id')
            assert getattr(foo, 'id') == 2
        assert not hasattr(foo, 'id')
