/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file reverse_v2_simd.h
 * \brief reverse_v2
 */

#ifndef ASCENDC_REVERSE_V2_SIMD_H
#define ASCENDC_REVERSE_V2_SIMD_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "platform/platform_info.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

namespace ReverseV2
{

using namespace AscendC;

constexpr int32_t BUFFER_NUM_SIMD = 2;
constexpr int64_t MAX_DIM_SIZE = 8;
constexpr int64_t LOOP_THRESHOLD = 1024;

const int32_t INDEX_0 = 0;
const int32_t INDEX_1 = 1;
const int32_t INDEX_2 = 2;
const int32_t INDEX_3 = 3;
const int32_t INDEX_4 = 4;
const int32_t INDEX_5 = 5;
const int32_t INDEX_6 = 6;
const int32_t INDEX_7 = 7;

constexpr int64_t TWO_DIMS = 2;
constexpr int64_t THREE_DIMS = 3;

class ReverseV2Simd
{
public:
    __aicore__ inline ReverseV2Simd(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReverseV2TilingData4AscendC* tilingData, TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(int64_t srcOffset, int64_t inUbBlockCount, int64_t inUbBlockLen);
    __aicore__ inline void CopyOut(int64_t offset, int64_t nBurst, int64_t copyLen);
    __aicore__ inline void Compute(int64_t outBlockOffset, int64_t dimInNum, int64_t splitIdx);
    __aicore__ inline void AreaReverse(__local_mem__ int8_t* xPtr, __local_mem__ int8_t* yPtr, int32_t dim0,
                                       int32_t dim1, int32_t dim2);
    __aicore__ inline void AreaReverseSmallTail(__local_mem__ int8_t* xPtr, __local_mem__ int8_t* yPtr, int32_t dim0,
                                                int32_t dim1, int32_t dim2);
    __aicore__ inline int64_t CalcDimUbStride(int64_t dim, int64_t tailNum);
    __aicore__ inline void CalcUbStride(int64_t tailDimNum);
    __aicore__ inline int64_t GetOutOffset(int32_t idx, int32_t splitDimRemian, int32_t tailDimNum,
                                           int32_t splitDimInNum);
    __aicore__ inline int64_t GetBlockOffsetAndIdx(int64_t idx, int64_t& inBlockOffset, int64_t& outBlcokOffset);

private:
    GlobalTensor<int8_t> xGm_;
    GlobalTensor<int8_t> yGm_;
    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_SIMD> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_SIMD> outQueue_;
    const ReverseV2TilingData4AscendC* tilingData_;
    int64_t xShape_[MAX_DIM_SIZE] = {1};
    int64_t xStride_[MAX_DIM_SIZE] = {1};
    int64_t blockStride_[MAX_DIM_SIZE] = {1};
    int64_t ubStride_[MAX_DIM_SIZE] = {1};
    int64_t ubBlockSize_ = Ops::Base::GetUbBlockSize();
    int64_t eleBlk_ = 1;
    int64_t blockIdx_ = 0;
};

__aicore__ inline void ReverseV2Simd::Init(GM_ADDR x, GM_ADDR y, const ReverseV2TilingData4AscendC* tilingData,
                                           TPipe* pipe)
{
    pipe_ = pipe;
    tilingData_ = tilingData;
    xGm_.SetGlobalBuffer((__gm__ int8_t*)x);
    yGm_.SetGlobalBuffer((__gm__ int8_t*)y);
    pipe_->InitBuffer(inQueue_, BUFFER_NUM_SIMD, tilingData_->inUbSize * tilingData_->dtypeSize);
    pipe_->InitBuffer(outQueue_, BUFFER_NUM_SIMD, tilingData_->inUbSize * tilingData_->dtypeSize);
    xShape_[INDEX_0] = tilingData_->dim0;
    xShape_[INDEX_1] = tilingData_->dim1;
    xShape_[INDEX_2] = tilingData_->dim2;
    xShape_[INDEX_3] = tilingData_->dim3;
    xShape_[INDEX_4] = tilingData_->dim4;
    xShape_[INDEX_5] = tilingData_->dim5;
    xShape_[INDEX_6] = tilingData_->dim6;
    xShape_[INDEX_7] = tilingData_->dim7;
    xStride_[INDEX_0] = tilingData_->param0;
    xStride_[INDEX_1] = tilingData_->param1;
    xStride_[INDEX_2] = tilingData_->param2;
    xStride_[INDEX_3] = tilingData_->param3;
    xStride_[INDEX_4] = tilingData_->param4;
    xStride_[INDEX_5] = tilingData_->param5;
    xStride_[INDEX_6] = tilingData_->param6;
    for (int64_t i = 0; i < MAX_DIM_SIZE; i++) {
        blockStride_[i] = tilingData->loopStride[i];
    }
    eleBlk_ = ubBlockSize_ / tilingData_->dtypeSize;
}

__aicore__ inline void ReverseV2Simd::CopyIn(int64_t srcOffset, int64_t inUbBlockCount, int64_t inUbBlockLen)
{
    LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
    int64_t alignBlockLen = ops::Aligned(inUbBlockLen, eleBlk_);
    DataCopyPadExtParams<int8_t> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = inUbBlockCount;
    dataCoptExtParams.blockLen = inUbBlockLen * tilingData_->dtypeSize;
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = (alignBlockLen - inUbBlockLen) * tilingData_->dtypeSize / Ops::Base::GetUbBlockSize();
    DataCopyPad(xLocal, xGm_[srcOffset], dataCoptExtParams, dataCopyPadExtParams);
    inQueue_.EnQue<int8_t>(xLocal);
}

__aicore__ inline void ReverseV2Simd::CopyOut(int64_t offset, int64_t outUbBlockOut, int64_t outLen)
{
    DataCopyExtParams dataCoptExtParams;
    int64_t alignBlockLen = ops::Aligned(outLen, eleBlk_);
    dataCoptExtParams.blockCount = outUbBlockOut;
    dataCoptExtParams.blockLen = outLen * tilingData_->dtypeSize;
    dataCoptExtParams.srcStride = (alignBlockLen - outLen) * tilingData_->dtypeSize / Ops::Base::GetUbBlockSize();
    dataCoptExtParams.dstStride = 0;
    LocalTensor<int8_t> yLocal = outQueue_.DeQue<int8_t>();
    DataCopyPad(yGm_[offset], yLocal, dataCoptExtParams);
    outQueue_.FreeTensor(yLocal);
}

__aicore__ inline void ReverseV2Simd::Process()
{
    int64_t startIdx = 0;
    int64_t endIdx = 0;
    blockIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
    if (blockIdx_ < tilingData_->blockTail) {
        startIdx = blockIdx_ * (tilingData_->blockFactor + 1);
        endIdx = startIdx + tilingData_->blockFactor + 1;
    } else {
        startIdx = blockIdx_ * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx = startIdx + tilingData_->blockFactor;
    }

    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        int64_t inBlockOffset = 0;
        int64_t outBlockOffset = 0;
        int64_t splitIdx = GetBlockOffsetAndIdx(idx, inBlockOffset, outBlockOffset);
        int64_t splitDimIn =
            splitIdx == tilingData_->splitDimLoop - 1 ? tilingData_->splitDimTailInNum : tilingData_->splitDimInNum;
        inBlockOffset += splitIdx * tilingData_->splitDimInNum * xStride_[tilingData_->splitDim];
        int64_t inUbBlockCount = tilingData_->splitDim == tilingData_->dimNum - 1 ? 1 : splitDimIn;
        for (int64_t i = tilingData_->splitDim + 1; i < tilingData_->dimNum - 1; i++) {
            inUbBlockCount *= xShape_[i];
        }
        int64_t coreDimInNum = tilingData_->splitDimInNum;
        if ((blockIdx_ == tilingData_->usedCoreNum - 1) && (idx == endIdx - 1)) {
            coreDimInNum = tilingData_->splitDimTailInNum;
        }
        int64_t inUbBlockLen = tilingData_->splitDim == tilingData_->dimNum - 1 ? coreDimInNum
                                                                                : xShape_[tilingData_->dimNum - 1];
        int64_t srcOffset = inBlockOffset * tilingData_->dtypeSize;
        CopyIn(srcOffset, inUbBlockCount, inUbBlockLen);
        Compute(outBlockOffset, splitDimIn, splitIdx);
    }
}

__aicore__ inline void ReverseV2Simd::Compute(int64_t outBlockOffset, int64_t dimInNum, int64_t splitIdx)
{
    int16_t loop = 1;
    int64_t dim0Size = 1;
    int64_t dim1Size = 1;
    int64_t dim2Size = 1;
    int64_t tailUbStride = 0;
    int64_t tailDimNum = 1;
    int64_t dimNum = tilingData_->dimNum;
    int64_t splitDim = tilingData_->splitDim;
    if (dimNum > THREE_DIMS && dimNum - splitDim > THREE_DIMS) {
        for (int16_t i = splitDim + 1; i < dimNum - THREE_DIMS; i++) {
            loop *= xShape_[i];
        }
        loop *= dimInNum;
        dim0Size = xShape_[dimNum - THREE_DIMS];
        dim1Size = xShape_[dimNum - TWO_DIMS];
        dim2Size = xShape_[dimNum - 1];
        tailDimNum = THREE_DIMS;
        int64_t alignDim2 = ops::Aligned(dim2Size, eleBlk_);
        tailUbStride = dim0Size * dim1Size * alignDim2 * tilingData_->dtypeSize;
    } else {
        if (dimNum - splitDim == THREE_DIMS) {
            dim0Size = dimInNum;
            dim1Size = xShape_[dimNum - TWO_DIMS];
            dim2Size = xShape_[dimNum - 1];
            tailDimNum = TWO_DIMS;
        } else if (dimNum - splitDim == TWO_DIMS) {
            dim1Size = dimInNum;
            dim2Size = xShape_[dimNum - 1];
            tailDimNum = 1;
        } else {
            dim2Size = dimInNum;
            tailDimNum = 0;
        }
    }
    CalcUbStride(tailDimNum);
    int64_t splitDimRemain = tilingData_->splitDimInNum * splitIdx;
    LocalTensor<int8_t> xLocal = inQueue_.DeQue<int8_t>();
    __local_mem__ int8_t* srcAddr = (__local_mem__ int8_t*)xLocal.GetPhyAddr();
    for (int16_t i = 0; i < loop; i++) {
        __local_mem__ int8_t* srcAddr1 = srcAddr + tailUbStride * i;
        LocalTensor<int8_t> yLocal = outQueue_.AllocTensor<int8_t>();
        __local_mem__ int8_t* yAddr = (__local_mem__ int8_t*)yLocal.GetPhyAddr();
        if (dim2Size * tilingData_->dtypeSize > LOOP_THRESHOLD) {
            AreaReverse(srcAddr1, yAddr, dim0Size, dim1Size, dim2Size);
        } else {
            AreaReverseSmallTail(srcAddr1, yAddr, dim0Size, dim1Size, dim2Size);
        }
        outQueue_.EnQue<int8_t>(yLocal);
        int64_t outOffset = outBlockOffset + GetOutOffset(i, splitDimRemain, tailDimNum, dimInNum);
        outOffset = outOffset * tilingData_->dtypeSize;
        CopyOut(outOffset, dim0Size * dim1Size, dim2Size);
    }
    inQueue_.FreeTensor<int8_t>(xLocal);
}

__aicore__ inline int64_t ReverseV2Simd::CalcDimUbStride(int64_t dim, int64_t tailDimNum)
{
    if (dim == tilingData_->dimNum - tailDimNum - 1) {
        return 1;
    }
    return ubStride_[dim + 1] * xShape_[dim + 1];
}

__aicore__ inline void ReverseV2Simd::CalcUbStride(int64_t tailDimNum)
{
    for (int64_t i = tilingData_->dimNum - tailDimNum - 1; i >= tilingData_->splitDim; i--) {
        ubStride_[i] = CalcDimUbStride(i, tailDimNum);
    }
}

__aicore__ inline int64_t ReverseV2Simd::GetOutOffset(int32_t idx, int32_t splitDimRemian, int32_t tailDimNum,
                                                      int32_t splitDimInNum)
{
    if (tailDimNum == 0) {
        return splitDimRemian;
    }
    int64_t outUbOffset = 0;
    int64_t splitDim = tilingData_->splitDim;
    int64_t tmpIdx = idx;
    bool needReverse = tilingData_->splitDimReversed;
    int64_t curIdx = tmpIdx / ubStride_[splitDim];
    curIdx += splitDimRemian;
    if (tailDimNum == 1 && needReverse) {
        outUbOffset = (xShape_[splitDim] - curIdx - splitDimInNum) * xStride_[splitDim];
        return outUbOffset;
    }
    if (needReverse) {
        outUbOffset += (xShape_[splitDim] - curIdx - 1) * xStride_[splitDim];
    } else {
        outUbOffset += curIdx * xStride_[splitDim];
    }
    needReverse = !needReverse;
    tmpIdx = tmpIdx % ubStride_[splitDim];
    for (int32_t i = splitDim + 1; i < tilingData_->dimNum - tailDimNum; i++) {
        curIdx = tmpIdx / ubStride_[i];
        if (needReverse) {
            outUbOffset += (xShape_[i] - curIdx - 1) * xStride_[i];
        } else {
            outUbOffset += curIdx * xStride_[i];
        }
        needReverse = !needReverse;
        tmpIdx = tmpIdx % ubStride_[i];
    }
    return outUbOffset;
}

__aicore__ inline int64_t ReverseV2Simd::GetBlockOffsetAndIdx(int64_t idx, int64_t& inBlockOffset,
                                                              int64_t& outBlcokOffset)
{
    int64_t tmpIdx = idx;
    if (tilingData_->splitDim == 0) {
        return tmpIdx;
    }
    bool needReverse = tilingData_->dim0Reversed;
    for (int64_t i = 0; i < tilingData_->splitDim; i++) {
        int64_t curIdx = tmpIdx / blockStride_[i];
        if (needReverse) {
            outBlcokOffset += (xShape_[i] - curIdx - 1) * xStride_[i];
        } else {
            outBlcokOffset += curIdx * xStride_[i];
        }
        inBlockOffset += curIdx * xStride_[i];
        needReverse = !needReverse;
        tmpIdx = tmpIdx % blockStride_[i];
    }
    return tmpIdx;
}

__aicore__ inline void ReverseV2Simd::AreaReverseSmallTail(__local_mem__ int8_t* xPtr, __local_mem__ int8_t* yPtr,
                                                           int32_t dim0, int32_t dim1, int32_t dim2)
{
    int32_t dim2Size = dim2 * tilingData_->dtypeSize;
    int32_t dim2Stride = ops::Aligned(dim2Size, static_cast<int32_t>(ubBlockSize_));
    int32_t repeatNum = Ops::Base::GetVRegSize();
    uint16_t dim2Loop = (dim2Stride + repeatNum - 1) / repeatNum;
    uint16_t dim1Loop = static_cast<uint16_t>(dim1);
    uint16_t dim0Loop = static_cast<uint16_t>(dim0);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int8_t> inReg;
        uint32_t sreg = dim2Stride;
        AscendC::MicroAPI::MaskReg preg;
        for (uint16_t i = 0; i < dim2Loop; i++) {
            preg = AscendC::MicroAPI::UpdateMask<int8_t>(sreg);
            auto srcAddr = xPtr + i * repeatNum;
            auto dstAddr = yPtr + i * repeatNum + (dim1 - 1) * dim2Stride;
            for (uint16_t j = 0; j < dim0Loop; j++) {
                auto srcAddr1 = srcAddr + j * dim1 * dim2Stride;
                auto dstAddr1 = dstAddr + j * dim1 * dim2Stride;
                for (uint16_t k = 0; k < dim1Loop; k++) {
                    auto curSrcAddr = srcAddr1 + k * dim2Stride;
                    auto curDstAddr = dstAddr1 - k * dim2Stride;
                    AscendC::MicroAPI::DataCopy(inReg, curSrcAddr);
                    AscendC::MicroAPI::DataCopy(curDstAddr, inReg, preg);
                }
            }
        }
    }
}

__aicore__ inline void ReverseV2Simd::AreaReverse(__local_mem__ int8_t* xPtr, __local_mem__ int8_t* yPtr, int32_t dim0,
                                                  int32_t dim1, int32_t dim2)
{
    int32_t dim2Size = dim2 * tilingData_->dtypeSize;
    int32_t dim2Stride = ops::Aligned(dim2Size, static_cast<int32_t>(ubBlockSize_));
    int32_t repeatNum = Ops::Base::GetVRegSize();
    uint16_t dim2Loop = (dim2Stride + repeatNum - 1) / repeatNum;
    uint16_t dim1Loop = static_cast<uint16_t>(dim1);
    uint16_t dim0Loop = static_cast<uint16_t>(dim0);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int8_t> inReg;
        for (uint16_t i = 0; i < dim0Loop; i++) {
            auto srcAddr = xPtr + i * dim1 * dim2Stride;
            auto dstAddr = yPtr + i * dim1 * dim2Stride + (dim1 - 1) * dim2Stride;
            for (uint16_t j = 0; j < dim1Loop; j++) {
                auto srcAddr1 = srcAddr + j * dim2Stride;
                auto dstAddr1 = dstAddr - j * dim2Stride;
                uint32_t sreg = dim2Size;
                for (uint16_t k = 0; k < dim2Loop; k++) {
                    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<int8_t>(sreg);
                    auto curSrcAddr = srcAddr1 + k * repeatNum;
                    auto curDstAddr = dstAddr1 + k * repeatNum;
                    AscendC::MicroAPI::DataCopy(inReg, curSrcAddr);
                    AscendC::MicroAPI::DataCopy(curDstAddr, inReg, preg);
                }
            }
        }
    }
}

}  // namespace ReverseV2

#endif  // ASCENDC_REVERSE_V2_SIMD_H