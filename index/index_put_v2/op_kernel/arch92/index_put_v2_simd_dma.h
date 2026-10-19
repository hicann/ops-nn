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
 * \file index_put_v2_simd_dma.h
 * \brief IndexPutV2 SIMD DMA kernel, inherits IndexPutV2Simd and adapts CopyOut loop with ScatterDataCopy.
 */

#ifndef INDEX_PUT_V2_SIMD_DMA
#define INDEX_PUT_V2_SIMD_DMA

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "../arch35/index_put_v2_simd.h"

namespace IndexPutV2 {
using namespace AscendC;

constexpr uint64_t ALIGN_BYTES = 32;

template <typename TX, typename TIDX, bool IsAtomicAdd>
class IndexPutV2SimdDma : public IndexPutV2Simd<TX, TIDX, IsAtomicAdd> {
public:
    __aicore__ inline IndexPutV2SimdDma(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR value, GM_ADDR indexedSizes, GM_ADDR indexedStrides,
                                GM_ADDR indices, IndexPutV2SimdTilingData tilingData_);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInValue(int64_t blockCount, int64_t len, int64_t offset);

private:
    TQue<QuePosition::VECOUT, 1> scatterOffsetBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;

private:
    __aicore__ inline void ComputeScatterOffset(int64_t curRows, int64_t colOffset);
};

template <typename TX, typename TIDX, bool IsAtomicAdd>
__aicore__ inline void IndexPutV2SimdDma<TX, TIDX, IsAtomicAdd>::Init(GM_ADDR inputX, GM_ADDR value,
                                                                      GM_ADDR indexedSizes, GM_ADDR indexedStrides,
                                                                      GM_ADDR indices,
                                                                      IndexPutV2SimdTilingData tilingData)
{
    IndexPutV2Simd<TX, TIDX, IsAtomicAdd>::Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
    if (this->blockIdx_ >= this->tilingData_.coreNum) {
        return;
    }

    uint64_t scatterOffsetBufSize = Ops::Base::CeilAlign(this->rowsFactor_ * sizeof(int64_t), ALIGN_BYTES);
    uint64_t maskBufSize = Ops::Base::CeilAlign(this->rowsFactor_ * sizeof(uint8_t), ALIGN_BYTES);
    this->pipe_.InitBuffer(scatterOffsetBuf_, 1, scatterOffsetBufSize);
    this->pipe_.InitBuffer(maskBuf_, maskBufSize);
}

template <typename TX, typename TIDX, bool IsAtomicAdd>
__aicore__ inline void IndexPutV2SimdDma<TX, TIDX, IsAtomicAdd>::CopyInValue(int64_t blockCount, int64_t len,
                                                                             int64_t offset)
{
    int64_t gmStride = this->tilingData_.nonIndexedLength - len;
    int64_t rowBytes = len * sizeof(TX);
    int64_t alignLen = Ops::Base::CeilAlign(len, static_cast<int64_t>(ALIGN_BYTES / sizeof(TX)));
    int64_t rightPadElems = alignLen - len;

    LocalTensor<TX> valueLocal = this->valueQue_.template AllocTensor<TX>();
    DataCopyPadExtParams<TX> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = true;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = rightPadElems; // 每块尾部追加的元素数
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = blockCount;
    dataCoptExtParams.blockLen = rowBytes;
    dataCoptExtParams.srcStride = gmStride * sizeof(TX);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(valueLocal, this->valueGM_[offset], dataCoptExtParams, dataCopyPadExtParams);
    this->valueQue_.template EnQue(valueLocal);
}

template <typename TX, typename TIDX, bool IsAtomicAdd>
__aicore__ inline void IndexPutV2SimdDma<TX, TIDX, IsAtomicAdd>::ComputeScatterOffset(int64_t curRows,
                                                                                      int64_t colOffset)
{
    LocalTensor<TIDX> tmpLocal = this->linearIndexBuf_.template Get<TIDX>();
    LocalTensor<int64_t> scatterOffset = scatterOffsetBuf_.AllocTensor<int64_t>();
    LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
    if constexpr (std::is_same<TIDX, int64_t>::value) {
        Adds(scatterOffset, tmpLocal, colOffset, static_cast<int32_t>(curRows));
    } else {
        Cast(scatterOffset, tmpLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(curRows));
        Adds(scatterOffset, scatterOffset, colOffset, static_cast<int32_t>(curRows));
    }
    Muls(scatterOffset, scatterOffset, static_cast<int64_t>(sizeof(TX)), static_cast<int32_t>(curRows));

    Compares(mask, tmpLocal, static_cast<TIDX>(this->invalidIndex), CMPMODE::NE, static_cast<int32_t>(curRows));
    Select(scatterOffset, mask, scatterOffset, static_cast<int64_t>(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           static_cast<uint32_t>(curRows));

    scatterOffsetBuf_.EnQue(scatterOffset);
}

template <typename TX, typename TIDX, bool IsAtomicAdd>
__aicore__ inline void IndexPutV2SimdDma<TX, TIDX, IsAtomicAdd>::Process()
{
    if (this->blockIdx_ >= this->tilingData_.coreNum) {
        return;
    }

    for (int64_t rowId = 0; rowId < this->rowsUbLoop; rowId++) {
        int64_t curRows = (rowId == this->rowsUbLoop - 1) ? (this->actualRowsNum_ - rowId * this->rowsFactor_) :
                                                            static_cast<int64_t>(this->rowsFactor_);

        this->GetIndices(curRows);
        this->HandleIndices(curRows);

        for (int64_t colId = 0; colId < this->colsUbLoop; colId++) {
            int64_t curCols = (colId == this->colsUbLoop - 1) ? (this->actualColsNum_ - colId * this->colsFactor_) :
                                                                static_cast<int64_t>(this->colsFactor_);

            int64_t valueGmOffset = this->blockOffset_ +
                                    rowId * this->rowsFactor_ * this->tilingData_.nonIndexedLength +
                                    colId * this->colsFactor_;

            this->CopyInValue(curRows, curCols, valueGmOffset);
            PipeBarrier<PIPE_MTE2>();
            LocalTensor<TX> valueLocal = this->valueQue_.template DeQue<TX>();

            int64_t colOffset = this->colGmOffset_ + colId * this->colsFactor_;
            ComputeScatterOffset(curRows, colOffset);

            ScatterParams scatterParams;
            scatterParams.offsetNum = static_cast<uint16_t>(curRows);
            scatterParams.burstLen = static_cast<uint32_t>(curCols * sizeof(TX));
            scatterParams.srcStride = static_cast<uint32_t>(Ops::Base::CeilAlign(curCols * sizeof(TX), ALIGN_BYTES));

            LocalTensor<int64_t> scatterOffset = scatterOffsetBuf_.DeQue<int64_t>();
            if constexpr (IsAtomicAdd) {
                SetAtomicAdd<TX>();
                ScatterDataCopy(this->inputGM_, valueLocal, scatterOffset, scatterParams);
                SetAtomicNone();
            } else {
                ScatterDataCopy(this->inputGM_, valueLocal, scatterOffset, scatterParams);
            }

            scatterOffsetBuf_.FreeTensor(scatterOffset);
            this->valueQue_.FreeTensor(valueLocal);
        }
    }
}

} // namespace IndexPutV2

#endif // defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
#endif // INDEX_PUT_V2_SIMD_DMA
