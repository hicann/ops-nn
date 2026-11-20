/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_matmul_activation_quant.h"
#include "quant_matmul_activation_quant_common.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

extern "C" {

aclnnStatus aclnnQuantMatmulActivationQuantGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                            const aclTensor* x1ScaleOptional, const aclTensor* x2Scale,
                                                            const aclTensor* biasOptional, bool transposeX1,
                                                            bool transposeX2, int64_t groupSize, char* activationType,
                                                            char* quantMode, char* roundMode, int64_t scaleAlg,
                                                            double dstTypeMax, aclTensor* yOut, aclTensor* yScaleOut,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantMatmulActivationQuant,
                   DFX_IN(x1, x2, x1ScaleOptional, x2Scale, biasOptional, transposeX1, transposeX2, groupSize,
                          activationType, quantMode, roundMode, scaleAlg, dstTypeMax),
                   DFX_OUT(yOut, yScaleOut));

    return QBMMActivationQuant::GetWorkspaceSize(
        __func__,
        {x1, x2, x1ScaleOptional, x2Scale, biasOptional, nullptr, yOut, yScaleOut, activationType, quantMode, roundMode,
         groupSize, 0, scaleAlg, dstTypeMax, transposeX1, transposeX2},
        workspaceSize, executor);
}

aclnnStatus aclnnQuantMatmulActivationQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMatmulActivationQuant);
    OP_CHECK(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
             OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER, "This is an error in QuantMatmulActivationQuant launch aicore."),
             return ACLNN_ERR_INNER);
    return ACLNN_SUCCESS;
}

} // extern "C"
