/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_mx_quant_reduce_empty.h
 * \brief When numCol=0 and numRow!=0, fill rstd output with NaN.
 */
#ifndef ADD_RMS_NORM_DYNAMIC_MX_QUANT_REDUCE_EMPTY_H_
#define ADD_RMS_NORM_DYNAMIC_MX_QUANT_REDUCE_EMPTY_H_

#include "add_rms_norm_dynamic_mx_quant_common.h"

namespace AddRmsNormDynamicMxQuant {
using namespace AscendC;

class AddRmsNormDynamicMxQuantReduceEmpty {
public:
    __aicore__ inline explicit AddRmsNormDynamicMxQuantReduceEmpty(
        const AddRmsNormDynamicMxQuantReduceEmptyTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR rstd)
    {
        blockIdx_ = GetBlockIdx();
        usedCoreNum_ = GetBlockNum();
        ASSERT(usedCoreNum_ != 0 && "block dim can not be zero!");
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }

        rstdFlag_ = tilingData_->rstdFlag;
        numRow_ = tilingData_->numRow;
        if (rstdFlag_ == 0 || numRow_ == 0) {
            return;
        }

        perCoreElements_ = tilingData_->perCoreElements;
        if (blockIdx_ < usedCoreNum_ - 1) {
            curCoreElements_ = tilingData_->perCoreElements;
            coreLoopsNum_ = tilingData_->perCoreLoops;
            perLoopElements_ = tilingData_->perCorePerLoopElements;
            lastLoopElements_ = tilingData_->perCoreLastLoopElements;
        } else {
            curCoreElements_ = tilingData_->lastCoreElements;
            coreLoopsNum_ = tilingData_->lastCoreLoops;
            perLoopElements_ = tilingData_->lastCorePerLoopElements;
            lastLoopElements_ = tilingData_->lastCoreLastLoopElements;
        }

        rstdGm_.SetGlobalBuffer(
            (__gm__ float*)rstd + perCoreElements_ * blockIdx_, curCoreElements_);

        pipe_.InitBuffer(outNanQueue_, BUFFER_NUM, perLoopElements_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_ || rstdFlag_ == 0 || numRow_ == 0) {
            return;
        }

        LocalTensor<float> nanLocal = outNanQueue_.AllocTensor<float>();

        float nanVal = AscendC::NumericLimits<float>::QuietNaN();
        Duplicate(nanLocal, nanVal, perLoopElements_);

        outNanQueue_.EnQue(nanLocal);
        nanLocal = outNanQueue_.DeQue<float>();

        for (uint64_t i = 0; i < coreLoopsNum_; i++) {
            uint64_t curElements = (i == coreLoopsNum_ - 1) ? lastLoopElements_ : perLoopElements_;
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(curElements * sizeof(float));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(rstdGm_[i * perLoopElements_], nanLocal, copyParams);
        }

        outNanQueue_.FreeTensor(nanLocal);
    }

private:
    constexpr static int64_t BUFFER_NUM = 1;

    TPipe pipe_;
    const AddRmsNormDynamicMxQuantReduceEmptyTilingData* tilingData_;

    GlobalTensor<float> rstdGm_;
    TQue<QuePosition::VECOUT, 1> outNanQueue_;

    uint64_t usedCoreNum_{0};
    uint64_t blockIdx_{0};
    uint32_t rstdFlag_{0};
    uint64_t numRow_{0};
    uint64_t perCoreElements_{0};
    uint64_t curCoreElements_{0};
    uint64_t coreLoopsNum_{0};
    uint64_t perLoopElements_{0};
    uint64_t lastLoopElements_{0};
};

} // namespace AddRmsNormDynamicMxQuant
#endif // ADD_RMS_NORM_DYNAMIC_MX_QUANT_REDUCE_EMPTY_H_
