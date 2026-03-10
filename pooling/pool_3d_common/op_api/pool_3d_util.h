/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_3D_H_
#define POOL_3D_H_
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "opdev/format_utils.h"

namespace Pool3DCommon {
const aclTensor* View3Das5D(const aclTensor* input, aclOpExecutor* executor);
const aclTensor* View4Das5D(const aclTensor* input, aclOpExecutor* executor);
const aclTensor* View5Das3D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor);
const aclTensor* View5Das4D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor);
const aclTensor* View3Das4D(const aclTensor* input, aclOpExecutor* executor);
const aclTensor* ViewCDHWas5D(const aclTensor* input, aclOpExecutor* executor);
const aclTensor* View5DasCDHW(const aclTensor* input, const op::Format& format, aclOpExecutor* executor);
}


#endif