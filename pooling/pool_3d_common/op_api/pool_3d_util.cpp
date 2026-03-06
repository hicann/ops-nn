/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "level0/unsqueeze.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "pool_3d_util.h"

#ifdef __cplusplus
extern "C" {
#endif
const aclTensor* View3Das5D(const aclTensor* input, aclOpExecutor* executor)
{
    // CHW -> unsqueeze -> reformat -> NCDHW
    // unsqueeze input into 4D
    const aclTensor* unsqueezedInput = l0op::UnsqueezeNd(input, 1, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // unsqueeze input into 5D
    auto unsqueezedInput5D = l0op::UnsqueezeNd(unsqueezedInput, static_cast<int64_t>(0), executor);
    CHECK_RET(unsqueezedInput5D != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput5D, op::Format::FORMAT_NCDHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

const aclTensor* View4Das5D(const aclTensor* input, aclOpExecutor* executor)
{
    // NCHW -> unsqueeze -> reformat -> NCDHW
    // unsqueeze input into 5D
    auto unsqueezedInput = l0op::UnsqueezeNd(input, 2, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

const aclTensor* View5Das3D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor)
{
    // NCDHW -> squeeze -> reformat -> CHW
    // squeeze out into 4D
    const aclTensor* squeezedInput = l0op::SqueezeNd(input, 2, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // squeeze out into 3D
    auto squeezedInput3D = l0op::SqueezeNd(squeezedInput, static_cast<int64_t>(0), executor);
    CHECK_RET(squeezedInput3D != nullptr, nullptr);
    // reformat to NCL
    auto reformatInput = l0op::ReFormat(squeezedInput3D, format);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

const aclTensor* View5Das4D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor)
{
    // NCDHW -> squeeze -> reformat -> NCHW
    // squeeze out into 3D
    auto squeezedInput = l0op::SqueezeNd(input, 2, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(squeezedInput, format);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

const aclTensor* View3Das4D(const aclTensor* input, aclOpExecutor* executor)
{
    // NCL -> unsqueeze -> reformat -> NCHW
    // unsqueeze input into 4D
    const int64_t appendDim[] = {0};
    aclIntArray* dimUnsqueeze = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dimUnsqueeze != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

const aclTensor* ViewCDHWas5D(const aclTensor* input, aclOpExecutor* executor) 
 { 
     // CDHW -> unsqueeze -> reformat -> NCDHW 
     // unsqueeze input into 5D 
     const int64_t dim = 0; // Unsqueeze at dimension 0 
     auto unsqueezedInput = l0op::UnsqueezeNd(input, dim, executor); 
     CHECK_RET(unsqueezedInput != nullptr, nullptr); 
     // reformat to NCDHW 
     auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW); 
     CHECK_RET(reformatInput != nullptr, nullptr); 
 
     return reformatInput; 
 } 
 
 
const aclTensor* View5DasCDHW(const aclTensor* input, const op::Format& format, aclOpExecutor* executor) 
 { 
     // NCDHW -> squeeze -> reformat -> CDHW 
     // squeeze out into 4D 
     const int64_t dim = 0; // Squeeze out dimension 0 
     auto squeezedInput = l0op::SqueezeNd(input, dim, executor); 
     CHECK_RET(squeezedInput != nullptr, nullptr); 
     // reformat to CDHW 
     auto reformatInput = l0op::ReFormat(squeezedInput, format); 
     CHECK_RET(reformatInput != nullptr, nullptr); 
 
     return reformatInput; 
 }

#ifdef __cplusplus
}
#endif
