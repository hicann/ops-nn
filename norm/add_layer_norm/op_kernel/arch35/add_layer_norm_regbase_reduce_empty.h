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
 * \file add_layer_norm_regbase_reduce_empty.h
 * \brief 空张量(reduce empty)兜底 kernel:合轴后 R==0 且 A>0 时,mean/rstd=[A,1] 填 NaN,
 *        y/x=[A,0] 空不写。把 mean/rstd 当一维 A 个 fp32,host 多核切分+定每循环元素数,
 *        kernel Duplicate NaN 一次、循环复用 buffer 搬出。
 */

#ifndef ADD_LAYER_NORM_REGBASE_REDUCE_EMPTY_H
#define ADD_LAYER_NORM_REGBASE_REDUCE_EMPTY_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AddLayerNorm {
using namespace AscendC;

class RegbaseReduceEmpty {
public:
    __aicore__ inline RegbaseReduceEmpty(const AddLayerNormRegbaseTilingData* tilingData) : tiling_(tilingData) {}

    __aicore__ inline void Init(GM_ADDR mean, GM_ADDR rstd)
    {
        usedCoreNum_ = tiling_->usedCoreNum;
        rowsPerLoop_ = tiling_->rowsPerLoop;
        int64_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        int64_t rowsPerCore = tiling_->rowsPerCore;
        int64_t rowsPerTailCore = tiling_->rowsPerTailCore;
        int64_t tailCoreStartIndex = tiling_->tailCoreStartIndex;
        rowsThisCore_ = (coreIdx < tailCoreStartIndex) ? rowsPerCore : rowsPerTailCore;
        rowOffset_ = coreIdx * rowsPerCore;
        loopCount_ = (rowsThisCore_ + rowsPerLoop_ - 1) / rowsPerLoop_;
        tailLen_ = rowsThisCore_ - (loopCount_ - 1) * rowsPerLoop_;
        meanGm_.SetGlobalBuffer((__gm__ float*)mean + rowOffset_, rowsThisCore_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + rowOffset_, rowsThisCore_);
        pipe_.InitBuffer(outNanQueue_, 1, rowsPerLoop_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_ || rowsThisCore_ <= 0) {
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
            DataCopyPad(meanGm_[i * rowsPerLoop_], nanLocal, copyParams);
            DataCopyPad(rstdGm_[i * rowsPerLoop_], nanLocal, copyParams);
        }

        outNanQueue_.FreeTensor(nanLocal);
    }

private:
    TPipe pipe_;
    const AddLayerNormRegbaseTilingData* tiling_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;
    TQue<QuePosition::VECOUT, 1> outNanQueue_;
    int64_t usedCoreNum_{0};
    int64_t rowsThisCore_{0};
    int64_t rowOffset_{0};
    int64_t rowsPerLoop_{0};
    int64_t loopCount_{0};
    int64_t tailLen_{0};
};
} // namespace AddLayerNorm

#endif // ADD_LAYER_NORM_REGBASE_REDUCE_EMPTY_H
