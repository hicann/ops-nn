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
 * \file dynamic_quant_unalign_310p.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_UNALIGN_310P_H
#define DYNAMIC_QUANT_UNALIGN_310P_H

#include "kernel_operator.h"
#include "dynamic_quant_align_310p.h"

namespace DynamicQuantNDOpt {

#if __CCE_AICORE__ < 220
class DynamicQuantUnalign310P : public DynamicQuantAlign310p
{
public:
    __aicore__ inline DynamicQuantUnalign310P()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, const DynamicQuantTilingData* __restrict tilingData)
    {
        DynamicQuantAlign310p::InitParams(tilingData, offset);
        DynamicQuantAlign310p::InitBuffer(x, y, scale, offset);
        DynamicQuantAlign310p::InitQueue();
    }
    __aicore__ inline void Process()
    {
        DynamicQuantAlign310p::ProcessWithHeadDoublePipLineForAlign();
        ProcessWithTailCopyInForUnalign();
        DynamicQuantAlign310p::ProcessWithHeadLastComputeAndCopyOutForAlign();
        ProcessWithTailComputeAndCopyOutForUnalign();
    }

    __aicore__ inline void ProcessWithTailCopyInForUnalign()
    {
        if (isTailLoop_) {
            DynamicQuantAlign310p::CopyIn(loopCount_, lenLastTail_);
        }
    }

    __aicore__ inline void ProcessWithTailComputeAndCopyOutForUnalign()
    {
        if (isTailLoop_) {
            if (asymmetric_) {
                DynamicQuantAlign310p::ComputeAsymmetric(numLastTailRow_, sizeH_);
            } else {
                DynamicQuantAlign310p::Compute(numLastTailRow_, sizeH_);
            }
            DynamicQuantAlign310p::CopyOut(loopCount_, lenLastTail_);
        }
    }
};
#endif
} // namespace DynamicQuantNDOpt

#endif // DYNAMIC_QUANT_UNALIGN_310P_H