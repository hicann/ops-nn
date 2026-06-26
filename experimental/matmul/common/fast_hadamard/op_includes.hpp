/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FAST_HADAMARD_COMMON_OP_INCLUDES_HPP
#define FAST_HADAMARD_COMMON_OP_INCLUDES_HPP

#include <cmath>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "kernel_operator.h"

#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif
#include <pto/pto-inst.hpp>

#endif // FAST_HADAMARD_COMMON_OP_INCLUDES_HPP
