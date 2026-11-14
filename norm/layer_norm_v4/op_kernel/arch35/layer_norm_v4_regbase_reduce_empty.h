/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LAYER_NORM_V4_REGBASE_REDUCE_EMPTY_H
#define LAYER_NORM_V4_REGBASE_REDUCE_EMPTY_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace LayerNormV4 {
using namespace AscendC;

class LayerNormV4RegbaseReduceEmpty {
public:
    __aicore__ inline LayerNormV4RegbaseReduceEmpty(const LayerNormV4TilingDataRegBaseReduceEmpty* tiling)
    {
        td_ = tiling;
    }

    __aicore__ inline void Init(GM_ADDR mean, GM_ADDR rstd)
    {
        const uint64_t begin = GetBlockIdx() * td_->rowsPerCore;
        rows_ = td_->rows - begin < td_->rowsPerCore ? td_->rows - begin : td_->rowsPerCore;
        rowsPerTile_ = td_->rowsPerTile;
        meanGm_.SetGlobalBuffer((__gm__ float*)mean + begin, rows_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + begin, rows_);
        pipe_.InitBuffer(outQueue_, 1, 2 * rowsPerTile_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> outLocalTensor = outQueue_.AllocTensor<float>();
        Duplicate(outLocalTensor, 0.0f, rowsPerTile_);
        Duplicate(outLocalTensor[rowsPerTile_], NumericLimits<float>::QuietNaN(), rowsPerTile_);
        outQueue_.EnQue(outLocalTensor);
        outLocalTensor = outQueue_.DeQue<float>();
        for (uint64_t offset = 0; offset < rows_; offset += rowsPerTile_) {
            const uint32_t count = static_cast<uint32_t>(rows_ - offset < rowsPerTile_ ? rows_ - offset : rowsPerTile_);
            DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
            DataCopyPad(meanGm_[offset], outLocalTensor, dataCopyParams);
            DataCopyPad(rstdGm_[offset], outLocalTensor[rowsPerTile_], dataCopyParams);
        }
        outQueue_.FreeTensor(outLocalTensor);
    }

private:
    TPipe pipe_;
    const LayerNormV4TilingDataRegBaseReduceEmpty* td_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<float> meanGm_, rstdGm_;
    uint64_t rows_ = 0;
    uint32_t rowsPerTile_ = 0;
};
} // namespace LayerNormV4
#endif // LAYER_NORM_V4_REGBASE_REDUCE_EMPTY_H
