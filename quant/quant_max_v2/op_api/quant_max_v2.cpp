/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_max_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantMaxV2);

// AiCore 执行体：将算子连同输入/输出/属性注册进 executor 的下发列表（launcher），
// 本层不做参数校验（校验在 aclnn 层完成），返回输出张量指针数组。
const std::array<aclTensor*, QUANT_MAX_V2_OUT_NUM> QuantMaxV2AiCore(const aclTensor* x, const aclTensor* scale,
                                                                    const aclTensor* groupList, const char* roundMode,
                                                                    int64_t dstType, float dstTypeMax,
                                                                    const aclTensor* y, const aclTensor* amax,
                                                                    const aclTensor* yScaleOutOptional,
                                                                    aclOpExecutor* executor)
{
    L0_DFX(QuantMaxV2AiCore, x, scale, groupList, roundMode, dstType, dstTypeMax, y, amax, yScaleOutOptional);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(QuantMaxV2, OP_INPUT(x, scale, groupList),
                                           OP_OUTPUT(y, amax, yScaleOutOptional),
                                           OP_ATTR(roundMode, dstType, dstTypeMax));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QuantMaxV2AiCore failed."),
             return (std::array<aclTensor*, QUANT_MAX_V2_OUT_NUM>{nullptr, nullptr, nullptr}));
    return std::array<aclTensor*, QUANT_MAX_V2_OUT_NUM>{const_cast<aclTensor*>(y), const_cast<aclTensor*>(amax),
                                                        const_cast<aclTensor*>(yScaleOutOptional)};
}

// L0 对外入口：参数契约见 quant_max_v2.h 的接口注释；当前唯一实现为 AiCore 下发，
// 预留在此分发其它执行体（如future Dvpp/Aicpu）的扩展点。
const std::array<aclTensor*, QUANT_MAX_V2_OUT_NUM> QuantMaxV2(const aclTensor* x, const aclTensor* scale,
                                                              const aclTensor* groupList, const char* roundMode,
                                                              int64_t dstType, float dstTypeMax, const aclTensor* y,
                                                              const aclTensor* amax, const aclTensor* yScaleOutOptional,
                                                              aclOpExecutor* executor)
{
    return QuantMaxV2AiCore(x, scale, groupList, roundMode, dstType, dstTypeMax, y, amax, yScaleOutOptional, executor);
}

} // namespace l0op
