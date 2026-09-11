/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_LARGE_GM_OVERFLOW_H_
#define _RENORM_LARGE_GM_OVERFLOW_H_

#include "kernel_operator.h"
#include "renorm_tiling_data.h"

namespace NsRenormLargeGmOverflow {

using namespace AscendC;

template <typename D_T_X>
class RenormLargeGmOverflow {
public:
    __aicore__ inline void Init(GM_ADDR y, const RenormTilingData* tilingData)
    {
        outputAddr_ = y;
        totalElements_ = tilingData->totalElements;
    }

    __aicore__ inline void Process()
    {
        if (totalElements_ <= 0) {
            return;
        }
        int64_t coreCount = GetBlockNum();
        int64_t elementsPerCore = (totalElements_ + coreCount - 1) / coreCount;
        int64_t start = GetBlockIdx() * elementsPerCore;
        if (start >= totalElements_) {
            return;
        }
        int64_t count = totalElements_ - start;
        if (count > elementsPerCore) {
            count = elementsPerCore;
        }
        GlobalTensor<D_T_X> outputGM;
        outputGM.SetGlobalBuffer((__gm__ D_T_X*)outputAddr_ + start, count);
        InitOutput<D_T_X>(outputGM[0], static_cast<uint32_t>(count), static_cast<D_T_X>(0));
    }

private:
    GM_ADDR outputAddr_ = nullptr;
    int64_t totalElements_ = 0;
};

} // namespace NsRenormLargeGmOverflow

#endif // _RENORM_LARGE_GM_OVERFLOW_H_
