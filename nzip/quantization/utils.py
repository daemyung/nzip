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

from contextlib import AbstractContextManager
from typing import Any


class PatchAttribute(AbstractContextManager):
    def __init__(self, target: Any, name: str, value: Any):
        """
        Constructor.

        :param target: The object who's the attribute is to be patched.
        :param name: The name of the attribute to be patched.
        :param value: The value to temporarily assign to the attribute.
        """
        super().__init__()
        self.target = target
        self.name = name
        self.value_to_patch = value
        self.existence = hasattr(target, name)
        self.value = getattr(target, name) if self.existence else None

    def __enter__(self):
        """
        Enter the runtime context related to this object.

        :return: Itself as the context manager.
        """
        setattr(self.target, self.name, self.value_to_patch)
        return self

    def __exit__(self, *exception):
        """
        Exit the runtime context related to this object.

        :param exception: A tuple holding the exception details if any exception is occurred.
        :return: False if allowing any exceptions to propagate.
        """
        if self.existence:
            setattr(self.target, self.name, self.value)
        else:
            delattr(self.target, self.name)
        return False
