
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
 * \file max_pool3d_grad_small_kernel_tiling.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "max_pool3d_grad_tiling.h"

namespace optiling {
static constexpr int64_t KSIZE_THRESHOLD= 4096;

bool MaxPool3DGradNCDHWSmallKernelTiling::IsCapable()
{
    base->InitializationVars(context_, ubSize_, coreNum_);
    if (inputData.inputFormat != ge::Format::FORMAT_NCDHW)
    {
        OP_LOGI("IsCapable", "inputFormat error");
        return false;
    }
    if(inputData.dDilation != 1 || inputData.hDilation != 1 || inputData.wDilation != 1) {
        OP_LOGI("IsCapable", "dDilation:%ld,  hDilation:%ld,  wDilation:%ld",
                inputData.dDilation, inputData.hDilation, inputData.wDilation);
        return false;
    }
    bool ksizeCheck = true;
    if (base->GetBaseData().dProBatchSize == 1 &&
        base->GetBaseData().hProBatchSize == 1 &&
        base->GetBaseData().wProBatchSize == 1) {
        ksizeCheck = inputData.dKernel * inputData.hKernel * inputData.wKernel < KSIZE_THRESHOLD;
    }
    // ub is not enough
    base->GetSplitData().highAxisInner = 1;
    base->GetSplitData().dOutputInner = 1;
    base->GetSplitData().hOutputInner = 1;
    base->GetSplitData().wOutputInner = std::min(inputData.wX, base->GetBaseData().proDataNumInOneBeatT2);
    base->DoBufferCalculate();
    OP_LOGI("IsCapable", "ksizeCheck:%d,  totalBufferSize:%ld,  availableUb:%ld",
            ksizeCheck, base->GetSplitData().totalBufferSize, base->GetBaseData().availableUb);
    return ksizeCheck && base->GetSplitData().totalBufferSize <= base->GetBaseData().availableUb;
}

uint64_t MaxPool3DGradNCDHWSmallKernelTiling::GetTilingKey() const
{
    int64_t outDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    uint32_t idxDtype = outDataCount <= static_cast<int64_t>(MAX_INT32) ? TPL_INT32 : TPL_INT64;
    uint32_t isChannelLast = 0;
    uint32_t isSimt = 0;
    uint32_t useINT64Index = 0;
    return GET_TPL_TILING_KEY(idxDtype, isSimt, isChannelLast, base->GetSplitData().isCheckRange, useINT64Index);
}

ge::graphStatus MaxPool3DGradNCDHWSmallKernelTiling::DoOpTiling()
{
    return base->DoOpTiling(context_);
}

ge::graphStatus MaxPool3DGradNCDHWSmallKernelTiling::PostTiling()
{
    return base->PostTiling(context_, GetTilingKey());
}

REGISTER_TILING_TEMPLATE("MaxPool3DGrad", MaxPool3DGradNCDHWSmallKernelTiling, 0);

} // namespace optiling
