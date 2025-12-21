/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file npu_ops_def.cpp
 * \brief
 */

#define Py_LIMITED_API_VERSION 0x03080000
#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"

#include <vector>

extern "C" {
PyObject* PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL,
    };
    return PyModule_Create(&module_def);
}
}

namespace ascend_ops {

TORCH_LIBRARY(ascend_ops, m)
{
    m.def(
        "conv3d_custom(Tensor input, Tensor weight, int[3] stride, int[3] padding, int[3] dilation, int[5] "
        "oriInputShape, int[5] oriWeightShape, Tensor? bias, bool enable_hf32 = False) -> Tensor");
}

} // namespace ascend_ops
