/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "data_format_dim_map.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(DataFormatDimMap);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT32, DataType::DT_INT64
};

static bool IsAiCoreSupport(const aclTensor* x)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "DataFormatDimMap not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "DataFormatDimMap not supported: dtype=%d. Supported dtypes: INT32, INT64.",
                     static_cast<int>(x->GetDataType())),
             return false);
    return true;
}

static const aclTensor* DataFormatDimMapAiCore(const aclTensor* x, const char* srcFormat,
                                                const char* dstFormat,
                                                const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(DataFormatDimMapAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(DataFormatDimMap,
        OP_INPUT(x), OP_OUTPUT(out), OP_ATTR(srcFormat, dstFormat));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DataFormatDimMapAiCore failed."),
        return nullptr);
    return out;
}

const aclTensor* DataFormatDimMap(const aclTensor* x, const char* srcFormat,
                                   const char* dstFormat, aclOpExecutor* executor)
{
    Shape outShape = x->GetViewShape();
    const aclTensor* out = nullptr;

    OP_CHECK(IsAiCoreSupport(x),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    out = executor->AllocTensor(outShape, x->GetDataType());

    return DataFormatDimMapAiCore(x, srcFormat, dstFormat, out, executor);
}

} // namespace l0op
