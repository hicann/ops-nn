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
 * \file add_rms_norm_cast_regbase_reduce_empty.h
 * \brief 空张量(reduce empty)兜底 kernel:归一化维为 0 且行数 > 0 时,rstd=[M,1] 填 NaN,
 *        y1/y2/x=[M,0] 空不写。使用独立的 AddRmsNormCastReduceEmptyTilingData,
 *        host 多核切分+定每循环元素数,kernel Duplicate NaN 一次、循环复用 buffer 搬出。
 */

#ifndef ADD_RMS_NORM_CAST_REGBASE_REDUCE_EMPTY_H
#define ADD_RMS_NORM_CAST_REGBASE_REDUCE_EMPTY_H

#include "add_rms_norm_cast_regbase_common.h"

namespace AddRmsNormCast {
using namespace AscendC;

class RegbaseReduceEmpty {
public:
    __aicore__ inline RegbaseReduceEmpty(TPipe* pipe, const AddRmsNormCastReduceEmptyTilingData* tilingData)
        : pipe_(pipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR rstd)
    {
        blockNum_ = GetBlockNum();
        rowsPerLoop_ = static_cast<int64_t>(tiling_->rowsPerLoop);
        int64_t coreIdx = GetBlockIdx();
        if (coreIdx >= blockNum_) {
            return;
        }
        int64_t rowsPerCore = static_cast<int64_t>(tiling_->mPerCore);
        int64_t rowsPerTailCore = static_cast<int64_t>(tiling_->mLastCore);
        int64_t tailCoreStartIndex = blockNum_ - 1; // only the last core is the tail
        rowsThisCore_ = (coreIdx < tailCoreStartIndex) ? rowsPerCore : rowsPerTailCore;
        rowOffset_ = coreIdx * rowsPerCore;
        loopCount_ = (rowsThisCore_ + rowsPerLoop_ - 1) / rowsPerLoop_;
        tailLen_ = rowsThisCore_ - (loopCount_ - 1) * rowsPerLoop_;
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + rowOffset_, rowsThisCore_);
        pipe_->InitBuffer(outNanQueue_, 1, rowsPerLoop_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t coreIdx = GetBlockIdx();
        if (coreIdx >= blockNum_ || rowsThisCore_ <= 0) {
            return;
        }

        LocalTensor<float> nanLocal = outNanQueue_.AllocTensor<float>();
        float nanVal = AscendC::NumericLimits<float>::QuietNaN();
        Duplicate(nanLocal, nanVal, static_cast<int32_t>(rowsPerLoop_));
        outNanQueue_.EnQue(nanLocal);
        nanLocal = outNanQueue_.DeQue<float>();

        for (int64_t i = 0; i < loopCount_; i++) {
            int64_t curLen = (i == loopCount_ - 1) ? tailLen_ : rowsPerLoop_;
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(float));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(rstdGm_[i * rowsPerLoop_], nanLocal, copyParams);
        }

        outNanQueue_.FreeTensor(nanLocal);
    }

private:
    TPipe* pipe_;
    const AddRmsNormCastReduceEmptyTilingData* tiling_;
    GlobalTensor<float> rstdGm_;
    TQue<QuePosition::VECOUT, 1> outNanQueue_;
    int64_t blockNum_{0};
    int64_t rowsThisCore_{0};
    int64_t rowOffset_{0};
    int64_t rowsPerLoop_{0};
    int64_t loopCount_{0};
    int64_t tailLen_{0};
};
} // namespace AddRmsNormCast

#endif // ADD_RMS_NORM_CAST_REGBASE_REDUCE_EMPTY_H
