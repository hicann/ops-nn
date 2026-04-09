/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

 #ifndef SOFTPLUSV2_H
#define SOFTPLUSV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "soft_plus_v2_tiling_data.h"
#include "soft_plus_v2_tiling_key.h"

namespace NsSoftPlusV2 {
  
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class SoftPlusV2 {
public:
    __aicore__ inline SoftPlusV2() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const SoftPlusV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;      
    uint32_t tileLength;    
};

__aicore__ inline void SoftPlusV2::Init(GM_ADDR x, GM_ADDR z, const SoftPlusV2TilingData* tilingData) {
    this->blockLength = tilingData->totalLength / GetBlockNum();
    this->tileNum = tilingData->tileNum;
    this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

    xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x +
                            this->blockLength * GetBlockIdx(),
                        this->blockLength);
    zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z +
                            this->blockLength * GetBlockIdx(),
                        this->blockLength);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
}

// 仅保留类成员Process函数
__aicore__ inline void SoftPlusV2::Process() {
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

__aicore__ inline void SoftPlusV2::CopyIn(int32_t progress) {
    LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
    DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
    inQueueX.EnQue(xLocal);
}

__aicore__ inline void SoftPlusV2::Compute(int32_t progress) {
    LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();  // 统一用DTYPE_X而非硬编码half
    LocalTensor<DTYPE_Z> yLocal = outQueueZ.AllocTensor<DTYPE_Z>();
    LocalTensor<DTYPE_X> tmpLocal = inQueueX.AllocTensor<DTYPE_X>();

    // 修正为this->tileLength
    Exp(tmpLocal, xLocal, this->tileLength);

    // yLocal = 1.0
    for (uint32_t i = 0; i < this->tileLength; ++i) {
        yLocal.SetValue(i, static_cast<DTYPE_Z>(1.0f));
    }

    // 修正为this->tileLength
    Add(tmpLocal, tmpLocal, yLocal, this->tileLength);
    Ln(yLocal, tmpLocal, this->tileLength);

    outQueueZ.EnQue<DTYPE_Z>(yLocal);

    inQueueX.FreeTensor(xLocal);
    inQueueX.FreeTensor(tmpLocal);
}

__aicore__ inline void SoftPlusV2::CopyOut(int32_t progress) {
    LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
    DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
    outQueueZ.FreeTensor(zLocal);
}

} // namespace NsSoftPlusV2 
#endif // SOFTPLUSV2_H