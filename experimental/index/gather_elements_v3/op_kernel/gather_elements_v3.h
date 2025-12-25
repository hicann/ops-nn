/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
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
/*!
 * \file gather_elements_v3.h
 * \brief
 */
#ifndef __GATHER_ELEMENTS_V3_H__
#define __GATHER_ELEMENTS_V3_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gather_elements_v3_tiling_data.h"
#include "gather_elements_v3_tiling_key.h"

namespace NsGatherElementsV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALIGN_BYTES = 32;

template <typename T>
class GatherElementsV3 {
public:
    __aicore__ inline GatherElementsV3(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR idx, GM_ADDR y, const GatherElementsV3TilingData* t);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t idxDataOffset, uint32_t curDataNum);
    __aicore__ inline void CopyOut(uint32_t yDataOffset, uint32_t curDataNum);
    __aicore__ inline void Compute(uint32_t xBase, uint32_t postStart, uint32_t len);

private:
    TPipe pipe;
    uint32_t xPreDim_;
    uint32_t xGatherDim_;
    uint32_t xPostDim_;
    uint32_t idxPreDim_;
    uint32_t idxGatherDim_;
    uint32_t idxPostDim_;
    uint32_t tileSize_;
    uint32_t coreId_;
    uint32_t coreNum_;
    GlobalTensor<T> xGm_;
    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<T> yGm_;

    TQue<TPosition::VECIN, BUFFER_NUM> idxInQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> yOutQue_;
};

template <typename T>
__aicore__ inline void GatherElementsV3<T>::Init(GM_ADDR x, GM_ADDR idx, GM_ADDR y, const GatherElementsV3TilingData* t)
{
        xPreDim_ = t->xPreDim;
        xGatherDim_ = t->xGatherDim;
        xPostDim_ = t->xPostDim;
        idxPreDim_ = t->idxPreDim;
        idxGatherDim_ = t->idxGatherDim;
        idxPostDim_ = t->idxPostDim;
        tileSize_ = t->tileSize;

        coreId_ = GetBlockIdx();
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        coreNum_ = GetBlockNum();

        xGm_.SetGlobalBuffer((__gm__ T*)x);
        idxGm_.SetGlobalBuffer((__gm__ int32_t*)idx);
        yGm_.SetGlobalBuffer((__gm__ T*)y);

        uint32_t idxByteSize = tileSize_ * sizeof(int32_t);
        uint32_t yByteSize = tileSize_ * sizeof(T);
        
        uint32_t idxAllocSize = (idxByteSize + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        uint32_t yAllocSize = (yByteSize + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;

        pipe.InitBuffer(idxInQue_, BUFFER_NUM, idxAllocSize);
        pipe.InitBuffer(yOutQue_, BUFFER_NUM, yAllocSize);
}

template <typename T>
__aicore__ inline void GatherElementsV3<T>::CopyIn(uint32_t idxDataOffset, uint32_t curDataNum)
{
        LocalTensor<int32_t> idxLocal = idxInQue_.AllocTensor<int32_t>();

        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(sizeof(int32_t) * curDataNum), 0, 0, 0};
        DataCopyPadExtParams<int32_t> idxPadParams{true, 0, 0, 0};
        
        DataCopyPad(idxLocal, idxGm_[idxDataOffset], idxCopyParams, idxPadParams);
  
        idxInQue_.EnQue(idxLocal);
}

template <typename T>
__aicore__ inline void GatherElementsV3<T>::CopyOut(uint32_t yDataOffset, uint32_t curDataNum)
{
        LocalTensor<T> yLocal = yOutQue_.DeQue<T>();

        DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(sizeof(T) * curDataNum), 0, 0, 0};
        DataCopyPad(yGm_[yDataOffset], yLocal, yCopyParams);

        yOutQue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void GatherElementsV3<T>::Compute(uint32_t xBase, uint32_t postStart, uint32_t len)
{
    {
        LocalTensor<int32_t> idxLocal = idxInQue_.DeQue<int32_t>();
        LocalTensor<T> yLocal = yOutQue_.AllocTensor<T>();

        for (uint32_t i = 0; i < len; ++i) {
            int32_t indexVal = idxLocal.GetValue(i);
            
            // 处理负索引
            if (indexVal < 0) indexVal += xGatherDim_;

            uint32_t xRealOffset = xBase + indexVal * xPostDim_ + (postStart + i);

            T xVal = xGm_.GetValue(xRealOffset);
            yLocal.SetValue(i, xVal);
        }

        idxInQue_.FreeTensor(idxLocal);
        yOutQue_.EnQue(yLocal);
    }
}

template <typename T>
__aicore__ inline void GatherElementsV3<T>::Process()
{
        uint32_t totalRows = idxPreDim_ * idxGatherDim_;

        for (uint32_t rowId = coreId_; rowId < totalRows; rowId += coreNum_) {
            
            uint32_t preIdx = rowId / idxGatherDim_;
            uint32_t gatherIdx = rowId % idxGatherDim_;

            uint32_t idxBase = preIdx * idxGatherDim_ * idxPostDim_ + gatherIdx * idxPostDim_;
            uint32_t xBase = preIdx * xGatherDim_ * xPostDim_;
            uint32_t yBase = idxBase;

            for (uint32_t colOffset = 0; colOffset < idxPostDim_; colOffset += tileSize_) {
                uint32_t len = tileSize_ < (idxPostDim_ - colOffset) ? tileSize_ : (idxPostDim_ - colOffset);
                CopyIn(idxBase + colOffset, len);
                Compute(xBase, colOffset, len);
                CopyOut(yBase + colOffset, len);
            }
        }
    }

} // namespace NsGatherElementsV3
#endif // GATHER_ELEMENTS_V3_H