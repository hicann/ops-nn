#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""index 算子多测试路径的 golden 编写（同 ops-test-kit ttk/test_spec/examples/06_golden_multi_path.py 规范）。

Kernel/GEIR 的 golden 收到 numpy.ndarray，需手动转 torch 计算后转回 numpy；
ACLNN 的 golden 直接收到 torch.Tensor（已在设备上），无需转换。
GEIR 复用 Kernel 的 spec（注册名同为 CSV op_name "index"），无需额外编写。
"""

import numpy as np


__spec__ = {
    "index": "IndexKernelSpec",
    "aclnnIndex": "AclnnIndexSpec",
}


def index_golden(x, indexed_sizes, indexed_strides, indices, **kwargs):
    """
    Golden function for index (Kernel / GEIR path).
    All the parameters (names and order) follow @index_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    bf16_mark = False
    if "bfloat16" in str(x.dtype):
        bf16_mark = True
        x = x.astype(np.float32)

    x_torch = torch.from_numpy(x)  # noqa: F841
    indices_list = [torch.from_numpy(arr.astype(np.int64)) for arr in indices]  # noqa: F841

    cmd = "x_torch["
    idx = 0
    for i in range(indexed_sizes.size):
        if indexed_sizes[i]:
            cmd += "indices_list[{}]".format(idx)
            idx += 1
        else:
            cmd += ":"
        cmd += ","
    cmd += "]"
    res = eval(cmd)

    if bf16_mark:
        res.to(torch.bfloat16)
    res = res.numpy()
    return res


def aclnn_index_golden(self, indices, out=None, **kwargs):
    """
    Aclnn golden for aclnnIndex (ACLNN path).
    Parameters follow @aclnnIndexGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    # Convert indices: empty tensors -> None
    if isinstance(indices, (list, tuple)):
        idx_list = []
        for idx in indices:
            if idx is None or idx.numel() == 0:
                idx_list.append(None)
            else:
                idx_list.append(idx.to(torch.int64))
        indices = tuple(idx_list)

    return [torch.ops.aten.index(self, indices)]


class IndexKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，参数按位置对齐 index_def.cpp 形参。"""

    golden = staticmethod(index_golden)


class AclnnIndexSpec:
    """ACLNN 流程 — golden 收到 torch.Tensor（已在设备上），参数按位置对齐 aclnn 头文件形参。"""

    golden = staticmethod(aclnn_index_golden)
