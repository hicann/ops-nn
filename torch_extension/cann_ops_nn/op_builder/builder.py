# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
from abc import ABC, abstractmethod
from typing import List, Union
import torch
from torch.utils.cpp_extension import load
from torch.library import Library

ASCEND_HOME_PATH = "ASCEND_HOME_PATH"
_as_library = None


def get_as_library():
    global _as_library
    if _as_library is None:
        _as_library = Library("cann_ops_nn", "DEF")
    return _as_library


class OpBuilder(ABC):
    """
    基于 aclnn 的算子构建基类

    :param name: 算子名称，如 'mat_mul_v3'
    :param category: 算子类别目录名，如 'matmul'。用于解析源码路径：csrc/<category>/<name>.cpp
    """

    _loaded_ops = {}

    def __init__(self, name, category=None):
        self.name = name
        self.category = category
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        import torch_npu
        import cann_ops_nn

        self._torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
        self._package_path = os.path.dirname(os.path.abspath(cann_ops_nn.__file__))
        self._cann_path = self.get_cann_path()
        if not hasattr(torch.ops.cann_ops_nn, self.name):
            self.register_schema(self.schema())
            self.register_meta()
        self._initialized = True

    def get_cann_path(self):
        if ASCEND_HOME_PATH in os.environ and os.path.exists(
            os.environ[ASCEND_HOME_PATH]
        ):
            return os.environ[ASCEND_HOME_PATH]
        return os.path.dirname(os.path.dirname(self._torch_npu_path))

    @property
    def cann_path(self):
        self._ensure_initialized()
        return self._cann_path

    @property
    def torch_npu_path(self):
        self._ensure_initialized()
        return self._torch_npu_path

    def get_absolute_paths(self, paths):
        self._ensure_initialized()
        return [os.path.join(self._package_path, path) for path in paths]

    def resolve_source(self, cpp_filename):
        self._ensure_initialized()
        if self.category is None:
            return cpp_filename
        return f"csrc/{self.category}/{cpp_filename}"

    def register_schema(self, op_schema: Union[str, List[str]]):
        if isinstance(op_schema, str):
            op_schema = [op_schema]
        for schema in op_schema:
            get_as_library().define(schema)

    @abstractmethod
    def sources(self): ...

    @abstractmethod
    def schema(self): ...

    @abstractmethod
    def register_meta(self): ...

    def include_paths(self):
        self._ensure_initialized()
        paths = [
            os.path.join(self._cann_path, "include"),
            os.path.join(self._torch_npu_path, "include"),
            os.path.join(self._torch_npu_path, "include/third_party/hccl/inc"),
            os.path.join(self._torch_npu_path, "include/third_party/acl/inc"),
            os.path.join(self._package_path, "common"),
        ]
        return paths

    def cxx_args(self):
        args = [
            "-O3",
            "-w",
            "-std=c++17",
            "-fPIC",
            "-fstack-protector-all",
            "-Wl,-z,relro,-z,now,-z,noexecstack",
            "-pie",
            "-s",
            "-fvisibility=hidden",
            "-D_FORTIFY_SOURCE=2",
        ]
        if torch._C._GLIBCXX_USE_CXX11_ABI:
            args.append("-D_GLIBCXX_USE_CXX11_ABI=1")
        else:
            args.append("-D_GLIBCXX_USE_CXX11_ABI=0")
        return args

    def extra_ldflags(self):
        self._ensure_initialized()
        flags = [
            "-L" + os.path.join(self._cann_path, "lib64"),
            "-lascendcl",
            "-L" + os.path.join(self._torch_npu_path, "lib"),
            "-ltorch_npu",
        ]
        return flags

    def load(self, verbose=True):
        self._ensure_initialized()
        if self.name in OpBuilder._loaded_ops:
            return OpBuilder._loaded_ops[self.name]

        op_module = load(
            name=self.name,
            sources=self.get_absolute_paths(self.sources()),
            extra_include_paths=self.get_absolute_paths(self.include_paths()),
            extra_cflags=self.cxx_args(),
            extra_ldflags=self.extra_ldflags(),
            verbose=verbose,
        )
        OpBuilder._loaded_ops[self.name] = op_module

        return op_module
