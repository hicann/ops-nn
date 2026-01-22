/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gather_elements.h
 * \brief
 */
#ifndef GATHER_ELEMENTS_FULL_LOAD_H
#define GATHER_ELEMENTS_FULL_LOAD_H

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"

namespace GatherElements {
using namespace AscendC;

constexpr uint16_t BUFFER_NUM = 2;
constexpr uint16_t INT16DTYPESIZE = 2;
constexpr uint16_t INT64DTYPESIZE = 8;

template <typename X_T, typename INDEX_T, int32_t DIM_NUM, int32_t AXIS>
class GatherElementsFullLoadKernel {
public:
    __aicore__ inline GatherElementsFullLoadKernel(const GatherElementsTilingData *tilingData, TPipe &pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInX(int64_t offset, uint16_t rows, int64_t dataLen);
    __aicore__ inline void CopyInIdx(int64_t offset, uint16_t rows, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t offset, uint16_t rows, int64_t dataLen);
    __aicore__ inline void Compute(LocalTensor<X_T> &xLocal, LocalTensor<INDEX_T> &idxLocal, LocalTensor<X_T> &yLocal, int64_t indexUbFactor,
                                    int64_t idxOffset, int64_t xRegOffset, int64_t idxRegOffset, int64_t yRegOffset);

private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<INDEX_T> indexGm_;
    AscendC::GlobalTensor<X_T> yGm_;
    TPipe &pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> indexQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    const GatherElementsTilingData *tilingData_;

    int64_t coreNum_;
    int64_t xUbFactor_;
    int64_t indexUbFactor_;
    int64_t xAfterAxis_;
    int64_t idxAfterAxis_;
};

template <typename X_T, typename INDEX_T, int32_t DIM_NUM, int32_t AXIS>
__aicore__ inline void GatherElementsFullLoadKernel<X_T, INDEX_T, DIM_NUM, AXIS>::Init(GM_ADDR x, GM_ADDR index,
    GM_ADDR y)
{
    coreNum_ = tilingData_->usedCore;
    xUbFactor_ = tilingData_->xUbFactor;
    indexUbFactor_ = tilingData_->indexUbFactor;
    xAfterAxis_ = tilingData_->xAfterAxis;
    idxAfterAxis_ = tilingData_->idxAfterAxis;

    uint32_t blockSize = platform::GetUbBlockSize();

    xGm_.SetGlobalBuffer((__gm__ X_T *)x);
    indexGm_.SetGlobalBuffer((__gm__ INDEX_T *)index);
    yGm_.SetGlobalBuffer((__gm__ X_T *)y);
    pipe_.InitBuffer(xQueue_, BUFFER_NUM, ops::CeilAlign(xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize)) * tilingData_->xLoadInUbNum);
    pipe_.InitBuffer(indexQueue_, BUFFER_NUM, ops::CeilAlign(idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize)) * tilingData_->indexLoadInUbNum);
    pipe_.InitBuffer(yQueue_, BUFFER_NUM, ops::CeilAlign(idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize)) * tilingData_->indexLoadInUbNum);
}


template <typename X_T, typename INDEX_T, int32_t DIM_NUM, int32_t AXIS>
__aicore__ inline void GatherElementsFullLoadKernel<X_T, INDEX_T, DIM_NUM, AXIS>::CopyInX(int64_t offset, uint16_t rows, int64_t dataLen)
{
    DataCopyExtParams inParams = { rows, static_cast<uint32_t>(dataLen * sizeof(X_T)), 0, 0, 0};
    DataCopyPadExtParams<X_T> padParams = { false, 0, 0, 0 };
    LocalTensor<X_T> xLocal = xQueue_.AllocTensor<X_T>();
    DataCopyPad(xLocal, xGm_[offset], inParams, padParams);
    xQueue_.EnQue(xLocal);
}

template <typename X_T, typename INDEX_T, int32_t DIM_NUM, int32_t AXIS>
__aicore__ inline void GatherElementsFullLoadKernel<X_T, INDEX_T, DIM_NUM, AXIS>::CopyInIdx(int64_t offset, uint16_t rows, int64_t dataLen)
{
    DataCopyExtParams inParams = { rows, static_cast<uint32_t>(dataLen * sizeof(INDEX_T)), 0, 0, 0};
    DataCopyPadExtParams<INDEX_T> padParams = { false, 0, 0, 0 };
    LocalTensor<INDEX_T> idxLocal = indexQueue_.AllocTensor<INDEX_T>();
    DataCopyPad(idxLocal, indexGm_[offset], inParams, padParams);
    indexQueue_.EnQue(idxLocal);
}

template <typename X_T, typename INDEX_T, typename OFFSET_T>
__aicore__ inline void GatherValue(MicroAPI::MaskReg& maskRegInput, uint16_t index, uint16_t oneRepeatSize,
    __local_mem__ X_T* xPtr, __local_mem__ X_T* yPtr, MicroAPI::RegTensor<OFFSET_T>& RegGatherIndex)
{
    MicroAPI::RegTensor<X_T> RegY;
    __local_mem__ X_T* yPtrOffset = yPtr + index * oneRepeatSize;
    if constexpr(std::is_same<X_T, int8_t>::value) {
        MicroAPI::RegTensor<int16_t> RegInt16Y;
        MicroAPI::RegTensor<int8_t> RegInt8Y1;
        MicroAPI::RegTensor<int8_t> RegInt8Y2;
        MicroAPI::MaskReg maskRegU16;
        MicroAPI::MaskReg maskRegU8;
        if constexpr (std::is_same<INDEX_T, int32_t>::value) {
            MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU16, maskRegInput);
        } else {
            MicroAPI::MaskReg maskRegU32;
            MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU32, maskRegInput);
            MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU16, maskRegU32);
        }
        MicroAPI::DataCopyGather(RegInt16Y, xPtr, RegGatherIndex, maskRegU16);
        MicroAPI::DeInterleave(
                RegInt8Y1, RegInt8Y2, (MicroAPI::RegTensor<X_T> &)RegInt16Y, (MicroAPI::RegTensor<X_T> &)RegInt16Y);
        MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU8, maskRegU16);
        MicroAPI::DataCopy(yPtrOffset, RegInt8Y1, maskRegU8);
    } else if constexpr(std::is_same<X_T, uint8_t>::value) {
        MicroAPI::RegTensor<uint16_t> RegUint16Y;
        MicroAPI::RegTensor<uint8_t> RegUint8Y1;
        MicroAPI::RegTensor<uint8_t> RegUint8Y2;
        MicroAPI::MaskReg maskRegU16;
        MicroAPI::MaskReg maskRegU8;
        if constexpr (std::is_same<INDEX_T, int32_t>::value) {
            MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU16, maskRegInput);
        } else {
            MicroAPI::MaskReg maskRegU32;
            MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU32, maskRegInput);
            MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU16, maskRegU32);
        }
        MicroAPI::DataCopyGather(RegUint16Y, xPtr, RegGatherIndex, maskRegU16);
        MicroAPI::DeInterleave(
                RegUint8Y1, RegUint8Y2, (MicroAPI::RegTensor<X_T> &)RegUint16Y, (MicroAPI::RegTensor<X_T> &)RegUint16Y);
        MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU8, maskRegU16);
        MicroAPI::DataCopy(yPtrOffset, RegUint8Y1, maskRegU8);
    } else if constexpr (
        (std::is_same<X_T, uint64_t>::value || std::is_same<X_T, int64_t>::value) ||
        ((std::is_same<X_T, uint32_t>::value || std::is_same<X_T, int32_t>::value || std::is_same<X_T, float32_t>::value) &&
        std::is_same<INDEX_T, int32_t>::value)) {
        MicroAPI::DataCopyGather(RegY, xPtr, RegGatherIndex, maskRegInput);
        MicroAPI::DataCopy(yPtrOffset, RegY, maskRegInput);
    } else if constexpr ((std::is_same<X_T, uint32_t>::value || std::is_same<X_T, int32_t>::value || std::is_same<X_T, float32_t>::value) &&
        std::is_same<INDEX_T, int64_t>::value) {
        MicroAPI::MaskReg maskRegU32;
        MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU32, maskRegInput);
        MicroAPI::DataCopyGather(RegY, xPtr, RegGatherIndex, maskRegU32);
        MicroAPI::DataCopy(yPtrOffset, RegY, maskRegU32);
    } else {
        MicroAPI::MaskReg maskRegU16;
        if constexpr (std::is_same<INDEX_T, int32_t>::value) {
            MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU16, maskRegInput);
        } else {
            MicroAPI::MaskReg maskRegU32;
            MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU32, maskRegInput);
            MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskRegU16, maskRegU32);
        }
        MicroAPI::DataCopyGather(RegY, xPtr, RegGatherIndex, maskRegU16);
        MicroAPI::DataCopy(yPtrOffset, RegY, maskRegU16);
    }
    return;
}

template <typename X_T, typename INDEX_T, typename OFFSET_T>
__aicore__ inline void ComputeDim1Vf(LocalTensor<X_T> &xLocal, LocalTensor<INDEX_T> &idxLocal, LocalTensor<X_T> &yLocal, int64_t xRegOffset, int64_t idxRegOffset, int64_t yRegOffset, uint16_t repeatNum,
                                            uint16_t oneRepeatSize, uint32_t indexUbFactor)
{
    __local_mem__ X_T* xPtr = (__local_mem__ X_T*)(xLocal.GetPhyAddr() + xRegOffset);
    __local_mem__ uint32_t* idxPtr = (__local_mem__ uint32_t*)(idxLocal.GetPhyAddr() + idxRegOffset);
    __local_mem__ X_T* yPtr = (__local_mem__ X_T*)(yLocal.GetPhyAddr() + yRegOffset);
    uint16_t dtypeFactor = sizeof(INDEX_T) / sizeof(uint32_t);
    uint32_t inputMask = indexUbFactor;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<OFFSET_T> RegIndex;
        MicroAPI::RegTensor<uint32_t> RegIndextmp;
        MicroAPI::RegTensor<X_T> RegY;
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<OFFSET_T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegInput;
        for (uint16_t i = 0; i < repeatNum; i++) {
            if constexpr ((std::is_same<X_T, uint64_t>::value || std::is_same<X_T, int64_t>::value) && (std::is_same<INDEX_T, int32_t>::value)) {
                maskRegInput = MicroAPI::UpdateMask<X_T>(inputMask);
            } else {
                maskRegInput = MicroAPI::UpdateMask<INDEX_T>(inputMask);
            }
            MicroAPI::DataCopy(RegIndextmp, idxPtr + i * oneRepeatSize * dtypeFactor);
            if constexpr (std::is_same<INDEX_T, int64_t>::value) {
                MicroAPI::RegTensor<uint32_t>RegUint32Index;
                MicroAPI::DeInterleave(
                        RegIndextmp, RegUint32Index, RegIndextmp, RegIndextmp);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::RegTensor<uint16_t>RegUint16Index;
                MicroAPI::DeInterleave(
                        RegIndex, RegUint16Index, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp);
            } else {
                MicroAPI::Copy(RegIndex, RegIndextmp, maskReg);
            }
            GatherValue<X_T, INDEX_T, OFFSET_T>(maskRegInput, i, oneRepeatSize, xPtr, yPtr, RegIndex);
        }
    }
}

template <typename X_T, typename INDEX_T, typename OFFSET_T>
__aicore__ inline void ComputeDim2Vf(LocalTensor<X_T> &xLocal, LocalTensor<INDEX_T> &idxLocal, LocalTensor<X_T> &yLocal, int64_t xRegOffset, int64_t idxRegOffset, int64_t yRegOffset, uint16_t repeatNum,
                                            uint16_t oneRepeatSize, int64_t idxOffset, uint32_t indexUbFactor, int64_t indexStrideArr0, int64_t xStrideArr0)
{
    __local_mem__ X_T* xPtr = (__local_mem__ X_T*)(xLocal.GetPhyAddr() + xRegOffset);
    __local_mem__ uint32_t* idxPtr = (__local_mem__ uint32_t*)(idxLocal.GetPhyAddr() + idxRegOffset);
    __local_mem__ X_T* yPtr = (__local_mem__ X_T*)(yLocal.GetPhyAddr() + yRegOffset);
    uint16_t dtypeFactor = sizeof(INDEX_T) / sizeof(uint32_t);
    uint32_t inputMask = indexUbFactor;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<OFFSET_T> RegIndex;
        MicroAPI::RegTensor<OFFSET_T> RegGatherIndex;
        MicroAPI::RegTensor<OFFSET_T> RegXDim0;
        MicroAPI::RegTensor<OFFSET_T> RegXDim1;
        MicroAPI::RegTensor<OFFSET_T> RegIdxStride;
        MicroAPI::RegTensor<uint32_t> RegIndextmp;
        MicroAPI::RegTensor<OFFSET_T> RegDim0Stride;
        MicroAPI::RegTensor<X_T> RegY;
        MicroAPI::RegTensor<OFFSET_T> RegArange;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg maskRegInput;
        for (uint16_t i = 0; i < repeatNum; i++) {
            maskReg = MicroAPI::CreateMask<OFFSET_T, MicroAPI::MaskPattern::ALL>();
            if constexpr ((std::is_same<X_T, uint64_t>::value || std::is_same<X_T, int64_t>::value) && (std::is_same<INDEX_T, int32_t>::value)) {
                maskRegInput = MicroAPI::UpdateMask<X_T>(inputMask);
            } else {
                maskRegInput = MicroAPI::UpdateMask<INDEX_T>(inputMask);
            }
            MicroAPI::DataCopy(RegIndextmp, idxPtr + i * oneRepeatSize * dtypeFactor);
            if constexpr (std::is_same<INDEX_T, int64_t>::value) {
                MicroAPI::RegTensor<uint32_t>RegUint32Index;
                MicroAPI::DeInterleave(
                        RegIndextmp, RegUint32Index, RegIndextmp, RegIndextmp);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::RegTensor<uint16_t>RegUint16Index;
                MicroAPI::DeInterleave(
                        RegIndex, RegUint16Index, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp);
            } else {
                MicroAPI::Copy(RegIndex, RegIndextmp, maskReg);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::Arange((MicroAPI::RegTensor<int16_t> &)RegArange, (int16_t)idxOffset + i * oneRepeatSize);
            } else {
                MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)RegArange, (int32_t)idxOffset + i * oneRepeatSize);
            }
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr0);
            MicroAPI::Div(RegXDim0, RegArange, RegIdxStride, maskReg);
            MicroAPI::Mul(RegXDim1, RegIdxStride, RegXDim0, maskReg);
            MicroAPI::Sub(RegXDim1, RegArange, RegXDim1, maskReg);
            MicroAPI::Muls(RegDim0Stride, RegIndex, (OFFSET_T)xStrideArr0, maskReg);
            MicroAPI::Add(RegGatherIndex, RegDim0Stride, RegXDim1, maskReg);
            GatherValue<X_T, INDEX_T, OFFSET_T>(maskRegInput, i, oneRepeatSize, xPtr, yPtr, RegGatherIndex);
        }
    }
}

template <typename X_T, typename INDEX_T, typename OFFSET_T>
__aicore__ inline void ComputeDim3Vf(LocalTensor<X_T> &xLocal, LocalTensor<INDEX_T> &idxLocal, LocalTensor<X_T> &yLocal, int64_t xRegOffset, int64_t idxRegOffset, int64_t yRegOffset, uint16_t repeatNum,
                                            uint16_t oneRepeatSize, int64_t idxOffset, uint32_t indexUbFactor, int64_t indexStrideArr0, int64_t xStrideArr0,
                                            int64_t indexStrideArr1, int64_t xStrideArr1)
{
    __local_mem__ X_T* xPtr = (__local_mem__ X_T*)(xLocal.GetPhyAddr() + xRegOffset);
    __local_mem__ uint32_t* idxPtr = (__local_mem__ uint32_t*)(idxLocal.GetPhyAddr() + idxRegOffset);
    __local_mem__ X_T* yPtr = (__local_mem__ X_T*)(yLocal.GetPhyAddr() + yRegOffset);
    uint16_t dtypeFactor = sizeof(INDEX_T) / sizeof(uint32_t);
    uint32_t inputMask = indexUbFactor;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<OFFSET_T> RegIndex;
        MicroAPI::RegTensor<uint32_t> RegIndextmp;
        MicroAPI::RegTensor<OFFSET_T> RegGatherIndex;
        MicroAPI::RegTensor<OFFSET_T> RegXDim1;
        MicroAPI::RegTensor<OFFSET_T> RegXDim2;
        MicroAPI::RegTensor<OFFSET_T> RegIdxStride;
        MicroAPI::RegTensor<OFFSET_T> Regidx;
        MicroAPI::RegTensor<OFFSET_T> Regtmp;
        MicroAPI::RegTensor<OFFSET_T> RegDim0Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim1Stride;
        MicroAPI::RegTensor<X_T> RegY;
        MicroAPI::RegTensor<OFFSET_T> RegArange;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg maskRegInput;
        for (uint16_t i = 0; i < repeatNum; i++) {
            maskReg = MicroAPI::CreateMask<OFFSET_T, MicroAPI::MaskPattern::ALL>();
            if constexpr ((std::is_same<X_T, uint64_t>::value || std::is_same<X_T, int64_t>::value) && (std::is_same<INDEX_T, int32_t>::value)) {
                maskRegInput = MicroAPI::UpdateMask<X_T>(inputMask);
            } else {
                maskRegInput = MicroAPI::UpdateMask<INDEX_T>(inputMask);
            }
            MicroAPI::DataCopy(RegIndextmp, idxPtr + i * oneRepeatSize * dtypeFactor);
            if constexpr (std::is_same<INDEX_T, int64_t>::value) {
                MicroAPI::RegTensor<uint32_t>RegUint32Index;
                MicroAPI::DeInterleave(
                        RegIndextmp, RegUint32Index, RegIndextmp, RegIndextmp);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::RegTensor<uint16_t>RegUint16Index;
                MicroAPI::DeInterleave(
                        RegIndex, RegUint16Index, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp);
            } else {
                MicroAPI::Copy(RegIndex, RegIndextmp, maskReg);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::Arange((MicroAPI::RegTensor<int16_t> &)RegArange, (int16_t)idxOffset + i * oneRepeatSize);
            } else {
                MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)RegArange, (int32_t)idxOffset + i * oneRepeatSize);
            }
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr0);
            MicroAPI::Div(Regtmp, RegArange, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegIdxStride, Regtmp, maskReg);
            MicroAPI::Sub(Regidx, RegArange, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr1);
            MicroAPI::Div(RegXDim1, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim1, RegIdxStride, maskReg);
            MicroAPI::Sub(RegXDim2, Regidx, Regtmp, maskReg);
            MicroAPI::Muls(RegDim0Stride, RegIndex, (OFFSET_T)xStrideArr0, maskReg);
            MicroAPI::Muls(RegDim1Stride, RegXDim1, (OFFSET_T)xStrideArr1, maskReg);
            MicroAPI::Add(RegGatherIndex, RegDim0Stride, RegDim1Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegXDim2, maskReg);
            GatherValue<X_T, INDEX_T, OFFSET_T>(maskRegInput, i, oneRepeatSize, xPtr, yPtr, RegGatherIndex);
        }
    }
}

template <typename X_T, typename INDEX_T, typename OFFSET_T>
__aicore__ inline void ComputeDim4Vf(LocalTensor<X_T> &xLocal, LocalTensor<INDEX_T> &idxLocal, LocalTensor<X_T> &yLocal, int64_t xRegOffset, int64_t idxRegOffset, int64_t yRegOffset, uint16_t repeatNum,
                                            uint16_t oneRepeatSize, int64_t idxOffset, uint32_t indexUbFactor, int64_t indexStrideArr0, int64_t xStrideArr0,
                                            int64_t indexStrideArr1, int64_t xStrideArr1, int64_t indexStrideArr2, int64_t xStrideArr2)
{
    __local_mem__ X_T* xPtr = (__local_mem__ X_T*)(xLocal.GetPhyAddr() + xRegOffset);
    __local_mem__ uint32_t* idxPtr = (__local_mem__ uint32_t*)(idxLocal.GetPhyAddr() + idxRegOffset);
    __local_mem__ X_T* yPtr = (__local_mem__ X_T*)(yLocal.GetPhyAddr() + yRegOffset);
    uint16_t dtypeFactor = sizeof(INDEX_T) / sizeof(uint32_t);
    uint32_t inputMask = indexUbFactor;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<OFFSET_T> RegIndex;
        MicroAPI::RegTensor<uint32_t> RegIndextmp;
        MicroAPI::RegTensor<OFFSET_T> RegGatherIndex;
        MicroAPI::RegTensor<OFFSET_T> RegXDim1;
        MicroAPI::RegTensor<OFFSET_T> RegXDim2;
        MicroAPI::RegTensor<OFFSET_T> RegXDim3;
        MicroAPI::RegTensor<OFFSET_T> RegIdxStride;
        MicroAPI::RegTensor<OFFSET_T> Regidx;
        MicroAPI::RegTensor<OFFSET_T> Regtmp;
        MicroAPI::RegTensor<OFFSET_T> RegDim0Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim1Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim2Stride;
        MicroAPI::RegTensor<X_T> RegY;
        MicroAPI::RegTensor<OFFSET_T> RegArange;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg maskRegInput;
        for (uint16_t i = 0; i < repeatNum; i++) {
            maskReg = MicroAPI::CreateMask<OFFSET_T, MicroAPI::MaskPattern::ALL>();
            if constexpr ((std::is_same<X_T, uint64_t>::value || std::is_same<X_T, int64_t>::value) && (std::is_same<INDEX_T, int32_t>::value)) {
                maskRegInput = MicroAPI::UpdateMask<X_T>(inputMask);
            } else {
                maskRegInput = MicroAPI::UpdateMask<INDEX_T>(inputMask);
            }
            MicroAPI::DataCopy(RegIndextmp, idxPtr + i * oneRepeatSize * dtypeFactor);
            if constexpr (std::is_same<INDEX_T, int64_t>::value) {
                MicroAPI::RegTensor<uint32_t>RegUint32Index;
                MicroAPI::DeInterleave(
                        RegIndextmp, RegUint32Index, RegIndextmp, RegIndextmp);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::RegTensor<uint16_t>RegUint16Index;
                MicroAPI::DeInterleave(
                        RegIndex, RegUint16Index, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp);
            } else {
                MicroAPI::Copy(RegIndex, RegIndextmp, maskReg);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::Arange((MicroAPI::RegTensor<int16_t> &)RegArange, (int16_t)idxOffset + i * oneRepeatSize);
            } else {
                MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)RegArange, (int32_t)idxOffset + i * oneRepeatSize);
            }
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr0);
            MicroAPI::Div(Regtmp, RegArange, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegIdxStride, Regtmp, maskReg);
            MicroAPI::Sub(Regidx, RegArange, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr1);
            MicroAPI::Div(RegXDim1, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim1, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr2);
            MicroAPI::Div(RegXDim2, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim2, RegIdxStride, maskReg);
            MicroAPI::Sub(RegXDim3, Regidx, Regtmp, maskReg);
            MicroAPI::Muls(RegDim0Stride, RegIndex, (OFFSET_T)xStrideArr0, maskReg);
            MicroAPI::Muls(RegDim1Stride, RegXDim1, (OFFSET_T)xStrideArr1, maskReg);
            MicroAPI::Muls(RegDim2Stride, RegXDim2, (OFFSET_T)xStrideArr2, maskReg);
            MicroAPI::Add(RegGatherIndex, RegDim0Stride, RegDim1Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim2Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegXDim3, maskReg);
            GatherValue<X_T, INDEX_T, OFFSET_T>(maskRegInput, i, oneRepeatSize, xPtr, yPtr, RegGatherIndex);
        }
    }
}

template <typename X_T, typename INDEX_T, typename OFFSET_T>
__aicore__ inline void ComputeDim5Vf(LocalTensor<X_T> &xLocal, LocalTensor<INDEX_T> &idxLocal, LocalTensor<X_T> &yLocal, int64_t xRegOffset, int64_t idxRegOffset, int64_t yRegOffset, uint16_t repeatNum,
                                            uint16_t oneRepeatSize, int64_t idxOffset, uint32_t indexUbFactor, int64_t indexStrideArr0, int64_t xStrideArr0,
                                            int64_t indexStrideArr1, int64_t xStrideArr1, int64_t indexStrideArr2, int64_t xStrideArr2, int64_t indexStrideArr3,
                                            int64_t xStrideArr3)
{
    __local_mem__ X_T* xPtr = (__local_mem__ X_T*)(xLocal.GetPhyAddr() + xRegOffset);
    __local_mem__ uint32_t* idxPtr = (__local_mem__ uint32_t*)(idxLocal.GetPhyAddr() + idxRegOffset);
    __local_mem__ X_T* yPtr = (__local_mem__ X_T*)(yLocal.GetPhyAddr() + yRegOffset);
    uint16_t dtypeFactor = sizeof(INDEX_T) / sizeof(uint32_t);
    uint32_t inputMask = indexUbFactor;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<OFFSET_T> RegIndex;
        MicroAPI::RegTensor<uint32_t> RegIndextmp;
        MicroAPI::RegTensor<OFFSET_T> RegGatherIndex;
        MicroAPI::RegTensor<OFFSET_T> RegXDim1;
        MicroAPI::RegTensor<OFFSET_T> RegXDim2;
        MicroAPI::RegTensor<OFFSET_T> RegXDim3;
        MicroAPI::RegTensor<OFFSET_T> RegXDim4;
        MicroAPI::RegTensor<OFFSET_T> RegIdxStride;
        MicroAPI::RegTensor<OFFSET_T> Regidx;
        MicroAPI::RegTensor<OFFSET_T> Regtmp;
        MicroAPI::RegTensor<OFFSET_T> RegDim0Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim1Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim2Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim3Stride;
        MicroAPI::RegTensor<X_T> RegY;
        MicroAPI::RegTensor<OFFSET_T> RegArange;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg maskRegInput;
        for (uint16_t i = 0; i < repeatNum; i++) {
            maskReg = MicroAPI::CreateMask<OFFSET_T, MicroAPI::MaskPattern::ALL>();
            if constexpr ((std::is_same<X_T, uint64_t>::value || std::is_same<X_T, int64_t>::value) && (std::is_same<INDEX_T, int32_t>::value)) {
                maskRegInput = MicroAPI::UpdateMask<X_T>(inputMask);
            } else {
                maskRegInput = MicroAPI::UpdateMask<INDEX_T>(inputMask);
            }
            MicroAPI::DataCopy(RegIndextmp, idxPtr + i * oneRepeatSize * dtypeFactor);
            if constexpr (std::is_same<INDEX_T, int64_t>::value) {
                MicroAPI::RegTensor<uint32_t>RegUint32Index;
                MicroAPI::DeInterleave(
                        RegIndextmp, RegUint32Index, RegIndextmp, RegIndextmp);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::RegTensor<uint16_t>RegUint16Index;
                MicroAPI::DeInterleave(
                        RegIndex, RegUint16Index, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp);
            } else {
                MicroAPI::Copy(RegIndex, RegIndextmp, maskReg);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::Arange((MicroAPI::RegTensor<int16_t> &)RegArange, (int16_t)idxOffset + i * oneRepeatSize);
            } else {
                MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)RegArange, (int32_t)idxOffset + i * oneRepeatSize);
            }
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr0);
            MicroAPI::Div(Regtmp, RegArange, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegIdxStride, Regtmp, maskReg);
            MicroAPI::Sub(Regidx, RegArange, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr1);
            MicroAPI::Div(RegXDim1, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim1, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr2);
            MicroAPI::Div(RegXDim2, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim2, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr3);
            MicroAPI::Div(RegXDim3, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim3, RegIdxStride, maskReg);
            MicroAPI::Sub(RegXDim4, Regidx, Regtmp, maskReg);
            MicroAPI::Muls(RegDim0Stride, RegIndex, (OFFSET_T)xStrideArr0, maskReg);
            MicroAPI::Muls(RegDim1Stride, RegXDim1, (OFFSET_T)xStrideArr1, maskReg);
            MicroAPI::Muls(RegDim2Stride, RegXDim2, (OFFSET_T)xStrideArr2, maskReg);
            MicroAPI::Muls(RegDim3Stride, RegXDim3, (OFFSET_T)xStrideArr3, maskReg);
            MicroAPI::Add(RegGatherIndex, RegDim0Stride, RegDim1Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim2Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim3Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegXDim4, maskReg);
            GatherValue<X_T, INDEX_T, OFFSET_T>(maskRegInput, i, oneRepeatSize, xPtr, yPtr, RegGatherIndex);
        }
    }
}

template <typename X_T, typename INDEX_T, typename OFFSET_T>
__aicore__ inline void ComputeDim6Vf(LocalTensor<X_T> &xLocal, LocalTensor<INDEX_T> &idxLocal, LocalTensor<X_T> &yLocal, int64_t xRegOffset, int64_t idxRegOffset, int64_t yRegOffset, uint16_t repeatNum,
                                            uint16_t oneRepeatSize, int64_t idxOffset, uint32_t indexUbFactor, int64_t indexStrideArr0, int64_t xStrideArr0,
                                            int64_t indexStrideArr1, int64_t xStrideArr1, int64_t indexStrideArr2, int64_t xStrideArr2, int64_t indexStrideArr3,
                                            int64_t xStrideArr3, int64_t indexStrideArr4, int64_t xStrideArr4)
{
    __local_mem__ X_T* xPtr = (__local_mem__ X_T*)(xLocal.GetPhyAddr() + xRegOffset);
    __local_mem__ uint32_t* idxPtr = (__local_mem__ uint32_t*)(idxLocal.GetPhyAddr() + idxRegOffset);
    __local_mem__ X_T* yPtr = (__local_mem__ X_T*)(yLocal.GetPhyAddr() + yRegOffset);
    uint16_t dtypeFactor = sizeof(INDEX_T) / sizeof(uint32_t);
    uint32_t inputMask = indexUbFactor;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<OFFSET_T> RegIndex;
        MicroAPI::RegTensor<uint32_t> RegIndextmp;
        MicroAPI::RegTensor<OFFSET_T> RegGatherIndex;
        MicroAPI::RegTensor<OFFSET_T> RegXDim1;
        MicroAPI::RegTensor<OFFSET_T> RegXDim2;
        MicroAPI::RegTensor<OFFSET_T> RegXDim3;
        MicroAPI::RegTensor<OFFSET_T> RegXDim4;
        MicroAPI::RegTensor<OFFSET_T> RegXDim5;
        MicroAPI::RegTensor<OFFSET_T> RegIdxStride;
        MicroAPI::RegTensor<OFFSET_T> Regidx;
        MicroAPI::RegTensor<OFFSET_T> Regtmp;
        MicroAPI::RegTensor<OFFSET_T> RegDim0Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim1Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim2Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim3Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim4Stride;
        MicroAPI::RegTensor<X_T> RegY;
        MicroAPI::RegTensor<OFFSET_T> RegArange;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg maskRegInput;
        for (uint16_t i = 0; i < repeatNum; i++) {
            maskReg = MicroAPI::CreateMask<OFFSET_T, MicroAPI::MaskPattern::ALL>();
            if constexpr ((std::is_same<X_T, uint64_t>::value || std::is_same<X_T, int64_t>::value) && (std::is_same<INDEX_T, int32_t>::value)) {
                maskRegInput = MicroAPI::UpdateMask<X_T>(inputMask);
            } else {
                maskRegInput = MicroAPI::UpdateMask<INDEX_T>(inputMask);
            }
            MicroAPI::DataCopy(RegIndextmp, idxPtr + i * oneRepeatSize * dtypeFactor);
            if constexpr (std::is_same<INDEX_T, int64_t>::value) {
                MicroAPI::RegTensor<uint32_t>RegUint32Index;
                MicroAPI::DeInterleave(
                        RegIndextmp, RegUint32Index, RegIndextmp, RegIndextmp);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::RegTensor<uint16_t>RegUint16Index;
                MicroAPI::DeInterleave(
                        RegIndex, RegUint16Index, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp);
            } else {
                MicroAPI::Copy(RegIndex, RegIndextmp, maskReg);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::Arange((MicroAPI::RegTensor<int16_t> &)RegArange, (int16_t)idxOffset + i * oneRepeatSize);
            } else {
                MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)RegArange, (int32_t)idxOffset + i * oneRepeatSize);
            }
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr0);
            MicroAPI::Div(Regtmp, RegArange, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegIdxStride, Regtmp, maskReg);
            MicroAPI::Sub(Regidx, RegArange, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr1);
            MicroAPI::Div(RegXDim1, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim1, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr2);
            MicroAPI::Div(RegXDim2, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim2, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr3);
            MicroAPI::Div(RegXDim3, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim3, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr4);
            MicroAPI::Div(RegXDim4, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim4, RegIdxStride, maskReg);
            MicroAPI::Sub(RegXDim5, Regidx, Regtmp, maskReg);
            MicroAPI::Muls(RegDim0Stride, RegIndex, (OFFSET_T)xStrideArr0, maskReg);
            MicroAPI::Muls(RegDim1Stride, RegXDim1, (OFFSET_T)xStrideArr1, maskReg);
            MicroAPI::Muls(RegDim2Stride, RegXDim2, (OFFSET_T)xStrideArr2, maskReg);
            MicroAPI::Muls(RegDim3Stride, RegXDim3, (OFFSET_T)xStrideArr3, maskReg);
            MicroAPI::Muls(RegDim4Stride, RegXDim4, (OFFSET_T)xStrideArr4, maskReg);
            MicroAPI::Add(RegGatherIndex, RegDim0Stride, RegDim1Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim2Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim3Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim4Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegXDim5, maskReg);
            GatherValue<X_T, INDEX_T, OFFSET_T>(maskRegInput, i, oneRepeatSize, xPtr, yPtr, RegGatherIndex);
        }
    }
}

template <typename X_T, typename INDEX_T, typename OFFSET_T>
__aicore__ inline void ComputeDim7Vf(LocalTensor<X_T> &xLocal, LocalTensor<INDEX_T> &idxLocal, LocalTensor<X_T> &yLocal, int64_t xRegOffset, int64_t idxRegOffset, int64_t yRegOffset, uint16_t repeatNum,
                                            uint16_t oneRepeatSize, int64_t idxOffset, uint32_t indexUbFactor, int64_t indexStrideArr0, int64_t xStrideArr0,
                                            int64_t indexStrideArr1, int64_t xStrideArr1, int64_t indexStrideArr2, int64_t xStrideArr2, int64_t indexStrideArr3,
                                            int64_t xStrideArr3, int64_t indexStrideArr4, int64_t xStrideArr4, int64_t indexStrideArr5, int64_t xStrideArr5)
{
    __local_mem__ X_T* xPtr = (__local_mem__ X_T*)(xLocal.GetPhyAddr() + xRegOffset);
    __local_mem__ uint32_t* idxPtr = (__local_mem__ uint32_t*)(idxLocal.GetPhyAddr() + idxRegOffset);
    __local_mem__ X_T* yPtr = (__local_mem__ X_T*)(yLocal.GetPhyAddr() + yRegOffset);
    uint16_t dtypeFactor = sizeof(INDEX_T) / sizeof(uint32_t);
    uint32_t inputMask = indexUbFactor;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<OFFSET_T> RegIndex;
        MicroAPI::RegTensor<uint32_t> RegIndextmp;
        MicroAPI::RegTensor<OFFSET_T> RegGatherIndex;
        MicroAPI::RegTensor<OFFSET_T> RegXDim1;
        MicroAPI::RegTensor<OFFSET_T> RegXDim2;
        MicroAPI::RegTensor<OFFSET_T> RegXDim3;
        MicroAPI::RegTensor<OFFSET_T> RegXDim4;
        MicroAPI::RegTensor<OFFSET_T> RegXDim5;
        MicroAPI::RegTensor<OFFSET_T> RegXDim6;
        MicroAPI::RegTensor<OFFSET_T> RegIdxStride;
        MicroAPI::RegTensor<OFFSET_T> Regidx;
        MicroAPI::RegTensor<OFFSET_T> Regtmp;
        MicroAPI::RegTensor<OFFSET_T> RegDim0Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim1Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim2Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim3Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim4Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim5Stride;
        MicroAPI::RegTensor<X_T> RegY;
        MicroAPI::RegTensor<OFFSET_T> RegArange;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg maskRegInput;
        for (uint16_t i = 0; i < repeatNum; i++) {
            maskReg = MicroAPI::CreateMask<OFFSET_T, MicroAPI::MaskPattern::ALL>();
            if constexpr ((std::is_same<X_T, uint64_t>::value || std::is_same<X_T, int64_t>::value) && (std::is_same<INDEX_T, int32_t>::value)) {
                maskRegInput = MicroAPI::UpdateMask<X_T>(inputMask);
            } else {
                maskRegInput = MicroAPI::UpdateMask<INDEX_T>(inputMask);
            }
            MicroAPI::DataCopy(RegIndextmp, idxPtr + i * oneRepeatSize * dtypeFactor);
            if constexpr (std::is_same<INDEX_T, int64_t>::value) {
                MicroAPI::RegTensor<uint32_t>RegUint32Index;
                MicroAPI::DeInterleave(
                        RegIndextmp, RegUint32Index, RegIndextmp, RegIndextmp);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::RegTensor<uint16_t>RegUint16Index;
                MicroAPI::DeInterleave(
                        RegIndex, RegUint16Index, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp);
            } else {
                MicroAPI::Copy(RegIndex, RegIndextmp, maskReg);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::Arange((MicroAPI::RegTensor<int16_t> &)RegArange, (int16_t)idxOffset + i * oneRepeatSize);
            } else {
                MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)RegArange, (int32_t)idxOffset + i * oneRepeatSize);
            }
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr0);
            MicroAPI::Div(Regtmp, RegArange, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegIdxStride, Regtmp, maskReg);
            MicroAPI::Sub(Regidx, RegArange, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr1);
            MicroAPI::Div(RegXDim1, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim1, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr2);
            MicroAPI::Div(RegXDim2, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim2, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr3);
            MicroAPI::Div(RegXDim3, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim3, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr4);
            MicroAPI::Div(RegXDim4, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim4, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr5);
            MicroAPI::Div(RegXDim5, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim5, RegIdxStride, maskReg);
            MicroAPI::Sub(RegXDim6, Regidx, Regtmp, maskReg);
            MicroAPI::Muls(RegDim0Stride, RegIndex, (OFFSET_T)xStrideArr0, maskReg);
            MicroAPI::Muls(RegDim1Stride, RegXDim1, (OFFSET_T)xStrideArr1, maskReg);
            MicroAPI::Muls(RegDim2Stride, RegXDim2, (OFFSET_T)xStrideArr2, maskReg);
            MicroAPI::Muls(RegDim3Stride, RegXDim3, (OFFSET_T)xStrideArr3, maskReg);
            MicroAPI::Muls(RegDim4Stride, RegXDim4, (OFFSET_T)xStrideArr4, maskReg);
            MicroAPI::Muls(RegDim5Stride, RegXDim5, (OFFSET_T)xStrideArr5, maskReg);
            MicroAPI::Add(RegGatherIndex, RegDim0Stride, RegDim1Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim2Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim3Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim4Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim5Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegXDim6, maskReg);
            GatherValue<X_T, INDEX_T, OFFSET_T>(maskRegInput, i, oneRepeatSize, xPtr, yPtr, RegGatherIndex);
        }
    }
}

template <typename X_T, typename INDEX_T, typename OFFSET_T>
__aicore__ inline void ComputeDim8Vf(LocalTensor<X_T> &xLocal, LocalTensor<INDEX_T> &idxLocal, LocalTensor<X_T> &yLocal, int64_t xRegOffset, int64_t idxRegOffset, int64_t yRegOffset, uint16_t repeatNum,
                                            uint16_t oneRepeatSize, int64_t idxOffset, uint32_t indexUbFactor, int64_t indexStrideArr0, int64_t xStrideArr0,
                                            int64_t indexStrideArr1, int64_t xStrideArr1, int64_t indexStrideArr2, int64_t xStrideArr2, int64_t indexStrideArr3,
                                            int64_t xStrideArr3, int64_t indexStrideArr4, int64_t xStrideArr4, int64_t indexStrideArr5, int64_t xStrideArr5,
                                            int64_t indexStrideArr6, int64_t xStrideArr6)
{
    __local_mem__ X_T* xPtr = (__local_mem__ X_T*)(xLocal.GetPhyAddr() + xRegOffset);
    __local_mem__ uint32_t* idxPtr = (__local_mem__ uint32_t*)(idxLocal.GetPhyAddr() + idxRegOffset);
    __local_mem__ X_T* yPtr = (__local_mem__ X_T*)(yLocal.GetPhyAddr() + yRegOffset);
    uint16_t dtypeFactor = sizeof(INDEX_T) / sizeof(uint32_t);
    uint32_t inputMask = indexUbFactor;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<OFFSET_T> RegIndex;
        MicroAPI::RegTensor<uint32_t> RegIndextmp;
        MicroAPI::RegTensor<OFFSET_T> RegGatherIndex;
        MicroAPI::RegTensor<OFFSET_T> RegXDim1;
        MicroAPI::RegTensor<OFFSET_T> RegXDim2;
        MicroAPI::RegTensor<OFFSET_T> RegXDim3;
        MicroAPI::RegTensor<OFFSET_T> RegXDim4;
        MicroAPI::RegTensor<OFFSET_T> RegXDim5;
        MicroAPI::RegTensor<OFFSET_T> RegXDim6;
        MicroAPI::RegTensor<OFFSET_T> RegXDim7;
        MicroAPI::RegTensor<OFFSET_T> RegIdxStride;
        MicroAPI::RegTensor<OFFSET_T> Regidx;
        MicroAPI::RegTensor<OFFSET_T> Regtmp;
        MicroAPI::RegTensor<OFFSET_T> RegDim0Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim1Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim2Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim3Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim4Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim5Stride;
        MicroAPI::RegTensor<OFFSET_T> RegDim6Stride;
        MicroAPI::RegTensor<X_T> RegY;
        MicroAPI::RegTensor<OFFSET_T> RegArange;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg maskRegInput;
        for (uint16_t i = 0; i < repeatNum; i++) {
            maskReg = MicroAPI::CreateMask<OFFSET_T, MicroAPI::MaskPattern::ALL>();
            if constexpr ((std::is_same<X_T, uint64_t>::value || std::is_same<X_T, int64_t>::value) && (std::is_same<INDEX_T, int32_t>::value)) {
                maskRegInput = MicroAPI::UpdateMask<X_T>(inputMask);
            } else {
                maskRegInput = MicroAPI::UpdateMask<INDEX_T>(inputMask);
            }
            MicroAPI::DataCopy(RegIndextmp, idxPtr + i * oneRepeatSize * dtypeFactor);
            if constexpr (std::is_same<INDEX_T, int64_t>::value) {
                MicroAPI::RegTensor<uint32_t>RegUint32Index;
                MicroAPI::DeInterleave(
                        RegIndextmp, RegUint32Index, RegIndextmp, RegIndextmp);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::RegTensor<uint16_t>RegUint16Index;
                MicroAPI::DeInterleave(
                        RegIndex, RegUint16Index, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp, (MicroAPI::RegTensor<uint16_t> &)RegIndextmp);
            } else {
                MicroAPI::Copy(RegIndex, RegIndextmp, maskReg);
            }
            if constexpr (std::is_same<OFFSET_T, uint16_t>::value) {
                MicroAPI::Arange((MicroAPI::RegTensor<int16_t> &)RegArange, (int16_t)idxOffset + i * oneRepeatSize);
            } else {
                MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)RegArange, (int32_t)idxOffset + i * oneRepeatSize);
            }
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr0);
            MicroAPI::Div(Regtmp, RegArange, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegIdxStride, Regtmp, maskReg);
            MicroAPI::Sub(Regidx, RegArange, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr1);
            MicroAPI::Div(RegXDim1, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim1, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr2);
            MicroAPI::Div(RegXDim2, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim2, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr3);
            MicroAPI::Div(RegXDim3, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim3, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr4);
            MicroAPI::Div(RegXDim4, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim4, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr5);
            MicroAPI::Div(RegXDim5, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim5, RegIdxStride, maskReg);
            MicroAPI::Sub(Regidx, Regidx, Regtmp, maskReg);
            MicroAPI::Duplicate(RegIdxStride, (OFFSET_T)indexStrideArr6);
            MicroAPI::Div(RegXDim6, Regidx, RegIdxStride, maskReg);
            MicroAPI::Mul(Regtmp, RegXDim6, RegIdxStride, maskReg);
            MicroAPI::Sub(RegXDim7, Regidx, Regtmp, maskReg);
            MicroAPI::Muls(RegDim0Stride, RegIndex, (OFFSET_T)xStrideArr0, maskReg);
            MicroAPI::Muls(RegDim1Stride, RegXDim1, (OFFSET_T)xStrideArr1, maskReg);
            MicroAPI::Muls(RegDim2Stride, RegXDim2, (OFFSET_T)xStrideArr2, maskReg);
            MicroAPI::Muls(RegDim3Stride, RegXDim3, (OFFSET_T)xStrideArr3, maskReg);
            MicroAPI::Muls(RegDim4Stride, RegXDim4, (OFFSET_T)xStrideArr4, maskReg);
            MicroAPI::Muls(RegDim5Stride, RegXDim5, (OFFSET_T)xStrideArr5, maskReg);
            MicroAPI::Muls(RegDim6Stride, RegXDim6, (OFFSET_T)xStrideArr6, maskReg);
            MicroAPI::Add(RegGatherIndex, RegDim0Stride, RegDim1Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim2Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim3Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim4Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim5Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegDim6Stride, maskReg);
            MicroAPI::Add(RegGatherIndex, RegGatherIndex, RegXDim7, maskReg);
            GatherValue<X_T, INDEX_T, OFFSET_T>(maskRegInput, i, oneRepeatSize, xPtr, yPtr, RegGatherIndex);
        }
    }
}

template <typename X_T, typename INDEX_T, int32_t DIM_NUM, int32_t AXIS>
__aicore__ inline void GatherElementsFullLoadKernel<X_T, INDEX_T, DIM_NUM, AXIS>::CopyOut(int64_t offset, uint16_t rows, int64_t dataLen)
{
    DataCopyExtParams outParams = { rows, static_cast<uint32_t>(dataLen * sizeof(X_T)), 0, 0, 0 };
    LocalTensor<X_T> yLocal = yQueue_.DeQue<X_T>();
    DataCopyPad(yGm_[offset], yLocal, outParams);
    yQueue_.FreeTensor(yLocal);
}

template <typename X_T, typename INDEX_T, int32_t DIM_NUM, int32_t AXIS>
__aicore__ inline void GatherElementsFullLoadKernel<X_T, INDEX_T, DIM_NUM, AXIS>::Compute(LocalTensor<X_T> &xLocal, LocalTensor<INDEX_T> &idxLocal, LocalTensor<X_T> &yLocal,
                                                                                        int64_t indexUbFactor, int64_t idxOffset, int64_t xRegOffset, int64_t idxRegOffset, int64_t yRegOffset)
{
    using OFFSET_T = typename std::conditional<sizeof(X_T) <= INT16DTYPESIZE, uint16_t, uint32_t>::type;
    uint16_t oneRepeatSize = GetVecLen() / sizeof(INDEX_T);
    if (sizeof(X_T) == INT64DTYPESIZE) {
        oneRepeatSize = GetVecLen() / INT64DTYPESIZE;
    }
    uint16_t repeatNum = CeilDivision(indexUbFactor, oneRepeatSize);
    if constexpr (DIM_NUM == DIM1) {
        ComputeDim1Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, (uint32_t)indexUbFactor);
    } else if constexpr (DIM_NUM == DIM2) {
        if constexpr (AXIS == 0) {
            ComputeDim2Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 1) {
            ComputeDim1Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, (uint32_t)indexUbFactor);
        }
    } else if constexpr (DIM_NUM == DIM3) {
        if constexpr (AXIS == 0) {
            ComputeDim3Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5],
                                        tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 1) {
            ComputeDim2Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 2) {
            ComputeDim1Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, (uint32_t)indexUbFactor);
        }
    } else if constexpr (DIM_NUM == DIM4) {
        if constexpr (AXIS == 0) {
            ComputeDim4Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4],
                                        tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 1) {
            ComputeDim3Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5],
                                        tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 2) {
            ComputeDim2Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 3) {
            ComputeDim1Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, (uint32_t)indexUbFactor);
        }
    } else if constexpr (DIM_NUM == DIM5) {
        if constexpr (AXIS == 0) {
            ComputeDim5Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[3], tilingData_->xStrideArr[3],
                                        tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4], tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6],
                                        tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 1) {
            ComputeDim4Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4],
                                        tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 2) {
            ComputeDim3Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5],
                                        tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 3) {
            ComputeDim2Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 4) {
            ComputeDim1Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, (uint32_t)indexUbFactor);
        }
    } else if constexpr (DIM_NUM == DIM6) {
        if constexpr (AXIS == 0) {
            ComputeDim6Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[2], tilingData_->xStrideArr[2],
                                        tilingData_->indexStrideArr[3], tilingData_->xStrideArr[3], tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4], tilingData_->indexStrideArr[5],
                                        tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 1) {
            ComputeDim5Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[3], tilingData_->xStrideArr[3],
                                        tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4], tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6],
                                        tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 2) {
            ComputeDim4Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4],
                                        tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 3) {
            ComputeDim3Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5],
                                        tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 4) {
            ComputeDim2Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 5) {
            ComputeDim1Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, (uint32_t)indexUbFactor);
        }
    } else if constexpr (DIM_NUM == DIM7) {
        if constexpr (AXIS == 0) {
            ComputeDim7Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[1], tilingData_->xStrideArr[1],
                                        tilingData_->indexStrideArr[2], tilingData_->xStrideArr[2], tilingData_->indexStrideArr[3], tilingData_->xStrideArr[3], tilingData_->indexStrideArr[4],
                                        tilingData_->xStrideArr[4], tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 1) {
            ComputeDim6Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[2], tilingData_->xStrideArr[2],
                                        tilingData_->indexStrideArr[3], tilingData_->xStrideArr[3], tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4], tilingData_->indexStrideArr[5],
                                        tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 2) {
            ComputeDim5Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[3], tilingData_->xStrideArr[3],
                                        tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4], tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6],
                                        tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 3) {
            ComputeDim4Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4],
                                        tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 4) {
            ComputeDim3Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5],
                                        tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 5) {
            ComputeDim2Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 6) {
            ComputeDim1Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, (uint32_t)indexUbFactor);
        }
    } else if constexpr (DIM_NUM == DIM8) {
        if constexpr (AXIS == 0) {
            ComputeDim8Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[0], tilingData_->xStrideArr[0],
                                        tilingData_->indexStrideArr[1], tilingData_->xStrideArr[1], tilingData_->indexStrideArr[2], tilingData_->xStrideArr[2], tilingData_->indexStrideArr[3],
                                        tilingData_->xStrideArr[3], tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4], tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5],
                                        tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 1) {
            ComputeDim7Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[1], tilingData_->xStrideArr[1],
                                        tilingData_->indexStrideArr[2], tilingData_->xStrideArr[2], tilingData_->indexStrideArr[3], tilingData_->xStrideArr[3], tilingData_->indexStrideArr[4],
                                        tilingData_->xStrideArr[4], tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 2) {
            ComputeDim6Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[2], tilingData_->xStrideArr[2],
                                        tilingData_->indexStrideArr[3], tilingData_->xStrideArr[3], tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4], tilingData_->indexStrideArr[5],
                                        tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 3) {
            ComputeDim5Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[3], tilingData_->xStrideArr[3],
                                        tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4], tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6],
                                        tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 4) {
            ComputeDim4Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[4], tilingData_->xStrideArr[4],
                                        tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5], tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 5) {
            ComputeDim3Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[5], tilingData_->xStrideArr[5],
                                        tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 6) {
            ComputeDim2Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, idxOffset, (uint32_t)indexUbFactor, tilingData_->indexStrideArr[6], tilingData_->xStrideArr[6]);
        } else if constexpr (AXIS == 7) {
            ComputeDim1Vf<X_T, INDEX_T, OFFSET_T>(xLocal, idxLocal, yLocal, xRegOffset, idxRegOffset, yRegOffset, repeatNum, oneRepeatSize, (uint32_t)indexUbFactor);
        }
    }
    yQueue_.EnQue(yLocal);
    indexQueue_.FreeTensor(idxLocal);
}

template <typename X_T, typename INDEX_T, int32_t DIM_NUM, int32_t AXIS>
__aicore__ inline void GatherElementsFullLoadKernel<X_T, INDEX_T, DIM_NUM, AXIS>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCore) {
        return;
    }
    uint32_t blockSize = platform::GetUbBlockSize();
    int64_t idxPerAxisOffset = 0;
    int64_t xPerAxisOffset = 0;
    int64_t xRegOffset = 0;
    int64_t idxRegOffset = 0;
    int64_t yRegOffset = 0;
    int64_t indexLoadInUbNum = tilingData_->indexLoadInUbNum;
    int64_t xOffset = 0;
    int64_t yOffset = GetBlockIdx() * idxAfterAxis_ * tilingData_->perCoreNum;
    int64_t idxOffset = GetBlockIdx() * idxAfterAxis_ * tilingData_->perCoreNum;
    int64_t LoopSize = ops::CeilDiv(tilingData_->perCoreNum, indexLoadInUbNum);
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    int64_t dim3 = 0;
    int64_t dim4 = 0;
    int64_t dim5 = 0;
    int64_t dim6 = 0;
    int64_t dim7 = 0;
    int64_t idx = 0;
    int64_t coreNum = tilingData_->perCoreNum;
    if (GetBlockIdx() == tilingData_->usedCore - 1) {
        LoopSize = ops::CeilDiv(tilingData_->tailCoreNum, indexLoadInUbNum);
        coreNum = tilingData_->tailCoreNum;
    }
    int64_t tailComputeNum = coreNum % indexLoadInUbNum == 0 ? indexLoadInUbNum : coreNum % indexLoadInUbNum;
    if constexpr (DIM_NUM == DIM1) {
        CopyInX(xOffset, 1, xAfterAxis_);
        CopyInIdx(idxOffset, 1, idxAfterAxis_);
        LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
        LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
        LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
        Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, 0, 0, 0);
        xQueue_.FreeTensor(xLocal);
        indexQueue_.FreeTensor(idxLocal);
        yQueue_.EnQue(yLocal);
        CopyOut(yOffset, 1, idxAfterAxis_);
    } else if constexpr (DIM_NUM == DIM2) {
        if constexpr (AXIS == 0) {
            CopyInX(xOffset, 1, xAfterAxis_);
            CopyInIdx(idxOffset, 1, idxAfterAxis_);
            LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
            LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
            LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, 0, 0, 0);
            xQueue_.FreeTensor(xLocal);
            indexQueue_.FreeTensor(idxLocal);
            yQueue_.EnQue(yLocal);
            CopyOut(yOffset, 1, idxAfterAxis_);
        } else if constexpr (AXIS == 1) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset;
                xOffset = dim1 * tilingData_->xStrideArr[6];
                xPerAxisOffset = dim1;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset;
                        if (dim1 < tilingData_->indexShape[0]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        }
    } else if constexpr (DIM_NUM == DIM3) {
        if constexpr (AXIS == 0) {
            CopyInX(xOffset, 1, xAfterAxis_);
            CopyInIdx(idxOffset, 1, idxAfterAxis_);
            LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
            LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
            LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, 0, 0, 0);
            xQueue_.FreeTensor(xLocal);
            indexQueue_.FreeTensor(idxLocal);
            yQueue_.EnQue(yLocal);
            CopyOut(yOffset, 1, idxAfterAxis_);
        } else if constexpr (AXIS == 1) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset;
                xOffset = dim1 * tilingData_->xStrideArr[5];
                xPerAxisOffset = dim1;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset;
                        if (dim1 < tilingData_->indexShape[0]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 2) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / tilingData_->indexShape[1];
                dim2 = idxPerAxisOffset - dim1 * tilingData_->indexShape[1];
                xOffset = dim1 * tilingData_->xStrideArr[5] + dim2 * tilingData_->xStrideArr[6];
                xPerAxisOffset = dim1 * tilingData_->xShape[1] + dim2;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / tilingData_->xShape[1];
                        dim2 = xPerAxisOffset - dim1 * tilingData_->xShape[1];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        }
    } else if constexpr (DIM_NUM == DIM4) {
        if constexpr (AXIS == 0) {
            CopyInX(xOffset, 1, xAfterAxis_);
            CopyInIdx(idxOffset, 1, idxAfterAxis_);
            LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
            LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
            LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, 0, 0, 0);
            xQueue_.FreeTensor(xLocal);
            indexQueue_.FreeTensor(idxLocal);
            yQueue_.EnQue(yLocal);
            CopyOut(yOffset, 1, idxAfterAxis_);
        } else if constexpr (AXIS == 1) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset;
                xOffset = dim1 * tilingData_->xStrideArr[4];
                xPerAxisOffset = dim1;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset;
                        if (dim1 < tilingData_->indexShape[0]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 2) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / tilingData_->indexShape[1];
                dim2 = idxPerAxisOffset - dim1 * tilingData_->indexShape[1];
                xOffset = dim1 * tilingData_->xStrideArr[4] + dim2 * tilingData_->xStrideArr[5];
                xPerAxisOffset = dim1 * tilingData_->xShape[1] + dim2;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / tilingData_->xShape[1];
                        dim2 = xPerAxisOffset - dim1 * tilingData_->xShape[1];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 3) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / tilingData_->indexShape[2];
                dim3 = idx - dim2 * tilingData_->indexShape[2];
                xOffset = dim1 * tilingData_->xStrideArr[4] + dim2 * tilingData_->xStrideArr[5] + dim3 * tilingData_->xStrideArr[6];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * tilingData_->xShape[2] + dim3;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / tilingData_->xShape[2];
                        dim3 = idx - dim2 * tilingData_->xShape[2];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        }
    } else if constexpr (DIM_NUM == DIM5) {
        if constexpr (AXIS == 0) {
            CopyInX(xOffset, 1, xAfterAxis_);
            CopyInIdx(idxOffset, 1, idxAfterAxis_);
            LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
            LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
            LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, 0, 0, 0);
            xQueue_.FreeTensor(xLocal);
            indexQueue_.FreeTensor(idxLocal);
            yQueue_.EnQue(yLocal);
            CopyOut(yOffset, 1, idxAfterAxis_);
        } else if constexpr (AXIS == 1) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset;
                xOffset = dim1 * tilingData_->xStrideArr[3];
                xPerAxisOffset = dim1;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset;
                        if (dim1 < tilingData_->indexShape[0]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 2) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / tilingData_->indexShape[1];
                dim2 = idxPerAxisOffset - dim1 * tilingData_->indexShape[1];
                xOffset = dim1 * tilingData_->xStrideArr[3] + dim2 * tilingData_->xStrideArr[4];
                xPerAxisOffset = dim1 * tilingData_->xShape[1] + dim2;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / tilingData_->xShape[1];
                        dim2 = xPerAxisOffset - dim1 * tilingData_->xShape[1];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 3) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / tilingData_->indexShape[2];
                dim3 = idx - dim2 * tilingData_->indexShape[2];
                xOffset = dim1 * tilingData_->xStrideArr[3] + dim2 * tilingData_->xStrideArr[4] + dim3 * tilingData_->xStrideArr[5];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * tilingData_->xShape[2] + dim3;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / tilingData_->xShape[2];
                        dim3 = idx - dim2 * tilingData_->xShape[2];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 4) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2] * tilingData_->indexShape[3];
            int64_t idxPerStride2 = tilingData_->indexShape[2] * tilingData_->indexShape[3];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2] * tilingData_->xShape[3];
            int64_t xPerStride2 = tilingData_->xShape[2] * tilingData_->xShape[3];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / idxPerStride2;
                idx = idx - dim2 * idxPerStride2;
                dim3 = idx / tilingData_->indexShape[3];
                dim4 = idx - dim3 * tilingData_->indexShape[3];
                xOffset = dim1 * tilingData_->xStrideArr[3] + dim2 * tilingData_->xStrideArr[4] + dim3 * tilingData_->xStrideArr[5] +
                          dim4 * tilingData_->xStrideArr[6];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * xPerStride2 + dim3 * tilingData_->xShape[3] + dim4;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / xPerStride2;
                        idx = idx - dim2 * xPerStride2;
                        dim3 = idx / tilingData_->xShape[3];
                        dim4 = idx - dim3 * tilingData_->xShape[3];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2] && dim4 < tilingData_->indexShape[3]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        }
    } else if constexpr (DIM_NUM == DIM6) {
        if constexpr (AXIS == 0) {
            CopyInX(xOffset, 1, xAfterAxis_);
            CopyInIdx(idxOffset, 1, idxAfterAxis_);
            LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
            LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
            LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, 0, 0, 0);
            xQueue_.FreeTensor(xLocal);
            indexQueue_.FreeTensor(idxLocal);
            yQueue_.EnQue(yLocal);
            CopyOut(yOffset, 1, idxAfterAxis_);
        } else if constexpr (AXIS == 1) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset;
                xOffset = dim1 * tilingData_->xStrideArr[2];
                xPerAxisOffset = dim1;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset;
                        if (dim1 < tilingData_->indexShape[0]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 2) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / tilingData_->indexShape[1];
                dim2 = idxPerAxisOffset - dim1 * tilingData_->indexShape[1];
                xOffset = dim1 * tilingData_->xStrideArr[2] + dim2 * tilingData_->xStrideArr[3];
                xPerAxisOffset = dim1 * tilingData_->xShape[1] + dim2;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / tilingData_->xShape[1];
                        dim2 = xPerAxisOffset - dim1 * tilingData_->xShape[1];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 3) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / tilingData_->indexShape[2];
                dim3 = idx - dim2 * tilingData_->indexShape[2];
                xOffset = dim1 * tilingData_->xStrideArr[2] + dim2 * tilingData_->xStrideArr[3] + dim3 * tilingData_->xStrideArr[4];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * tilingData_->xShape[2] + dim3;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / tilingData_->xShape[2];
                        dim3 = idx - dim2 * tilingData_->xShape[2];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 4) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2] * tilingData_->indexShape[3];
            int64_t idxPerStride2 = tilingData_->indexShape[2] * tilingData_->indexShape[3];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2] * tilingData_->xShape[3];
            int64_t xPerStride2 = tilingData_->xShape[2] * tilingData_->xShape[3];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / idxPerStride2;
                idx = idx - dim2 * idxPerStride2;
                dim3 = idx / tilingData_->indexShape[3];
                dim4 = idx - dim3 * tilingData_->indexShape[3];
                xOffset = dim1 * tilingData_->xStrideArr[2] + dim2 * tilingData_->xStrideArr[3] + dim3 * tilingData_->xStrideArr[4] +
                          dim4 * tilingData_->xStrideArr[5];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * xPerStride2 + dim3 * tilingData_->xShape[3] + dim4;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / xPerStride2;
                        idx = idx - dim2 * xPerStride2;
                        dim3 = idx / tilingData_->xShape[3];
                        dim4 = idx - dim3 * tilingData_->xShape[3];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2] && dim4 < tilingData_->indexShape[3]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 5) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2] * tilingData_->indexShape[3] *
                                    tilingData_->indexShape[4];
            int64_t idxPerStride2 = tilingData_->indexShape[2] * tilingData_->indexShape[3] * tilingData_->indexShape[4];
            int64_t idxPerStride3 = tilingData_->indexShape[3] * tilingData_->indexShape[4];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4];
            int64_t xPerStride2 = tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4];
            int64_t xPerStride3 = tilingData_->xShape[3] * tilingData_->xShape[4];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / idxPerStride2;
                idx = idx - dim2 * idxPerStride2;
                dim3 = idx / idxPerStride3;
                idx = idx - dim3 * idxPerStride3;
                dim4 = idx / tilingData_->indexShape[4];
                dim5 = idx - dim4 * tilingData_->indexShape[4];
                xOffset = dim1 * tilingData_->xStrideArr[2] + dim2 * tilingData_->xStrideArr[3] + dim3 * tilingData_->xStrideArr[4] +
                            dim4 * tilingData_->xStrideArr[5] + dim5 * tilingData_->xStrideArr[6];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * xPerStride2 + dim3 * xPerStride3 + dim4 * tilingData_->xShape[4] + dim5;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / xPerStride2;
                        idx = idx - dim2 * xPerStride2;
                        dim3 = idx / xPerStride3;
                        idx = idx - dim3 * xPerStride3;
                        dim4 = idx / tilingData_->xShape[4];
                        dim5 = idx - dim4 * tilingData_->xShape[4];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2] && dim4 < tilingData_->indexShape[3] &&
                            dim5 < tilingData_->indexShape[4]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        }
    } else if constexpr (DIM_NUM == DIM7) {
        if constexpr (AXIS == 0) {
            CopyInX(xOffset, 1, xAfterAxis_);
            CopyInIdx(idxOffset, 1, idxAfterAxis_);
            LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
            LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
            LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, 0, 0, 0);
            xQueue_.FreeTensor(xLocal);
            indexQueue_.FreeTensor(idxLocal);
            yQueue_.EnQue(yLocal);
            CopyOut(yOffset, 1, idxAfterAxis_);
        } else if constexpr (AXIS == 1) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset;
                xOffset = dim1 * tilingData_->xStrideArr[1];
                xPerAxisOffset = dim1;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset;
                        if (dim1 < tilingData_->indexShape[0]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 2) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / tilingData_->indexShape[1];
                dim2 = idxPerAxisOffset - dim1 * tilingData_->indexShape[1];
                xOffset = dim1 * tilingData_->xStrideArr[1] + dim2 * tilingData_->xStrideArr[2];
                xPerAxisOffset = dim1 * tilingData_->xShape[1] + dim2;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / tilingData_->xShape[1];
                        dim2 = xPerAxisOffset - dim1 * tilingData_->xShape[1];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 3) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / tilingData_->indexShape[2];
                dim3 = idx - dim2 * tilingData_->indexShape[2];
                xOffset = dim1 * tilingData_->xStrideArr[1] + dim2 * tilingData_->xStrideArr[2] + dim3 * tilingData_->xStrideArr[3];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * tilingData_->xShape[2] + dim3;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / tilingData_->xShape[2];
                        dim3 = idx - dim2 * tilingData_->xShape[2];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 4) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2] * tilingData_->indexShape[3];
            int64_t idxPerStride2 = tilingData_->indexShape[2] * tilingData_->indexShape[3];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2] * tilingData_->xShape[3];
            int64_t xPerStride2 = tilingData_->xShape[2] * tilingData_->xShape[3];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / idxPerStride2;
                idx = idx - dim2 * idxPerStride2;
                dim3 = idx / tilingData_->indexShape[3];
                dim4 = idx - dim3 * tilingData_->indexShape[3];
                xOffset = dim1 * tilingData_->xStrideArr[1] + dim2 * tilingData_->xStrideArr[2] + dim3 * tilingData_->xStrideArr[3] +
                          dim4 * tilingData_->xStrideArr[4];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * xPerStride2 + dim3 * tilingData_->xShape[3] + dim4;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / xPerStride2;
                        idx = idx - dim2 * xPerStride2;
                        dim3 = idx / tilingData_->xShape[3];
                        dim4 = idx - dim3 * tilingData_->xShape[3];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2] && dim4 < tilingData_->indexShape[3]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 5) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2] * tilingData_->indexShape[3] *
                                    tilingData_->indexShape[4];
            int64_t idxPerStride2 = tilingData_->indexShape[2] * tilingData_->indexShape[3] * tilingData_->indexShape[4];
            int64_t idxPerStride3 = tilingData_->indexShape[3] * tilingData_->indexShape[4];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4];
            int64_t xPerStride2 = tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4];
            int64_t xPerStride3 = tilingData_->xShape[3] * tilingData_->xShape[4];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / idxPerStride2;
                idx = idx - dim2 * idxPerStride2;
                dim3 = idx / idxPerStride3;
                idx = idx - dim3 * idxPerStride3;
                dim4 = idx / tilingData_->indexShape[4];
                dim5 = idx - dim4 * tilingData_->indexShape[4];
                xOffset = dim1 * tilingData_->xStrideArr[1] + dim2 * tilingData_->xStrideArr[2] + dim3 * tilingData_->xStrideArr[3] +
                          dim4 * tilingData_->xStrideArr[4] + dim5 * tilingData_->xStrideArr[5];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * xPerStride2 + dim3 * xPerStride3 + dim4 * tilingData_->xShape[4] + dim5;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / xPerStride2;
                        idx = idx - dim2 * xPerStride2;
                        dim3 = idx / xPerStride3;
                        idx = idx - dim3 * xPerStride3;
                        dim4 = idx / tilingData_->xShape[4];
                        dim5 = idx - dim4 * tilingData_->xShape[4];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2] && dim4 < tilingData_->indexShape[3] &&
                            dim5 < tilingData_->indexShape[4]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 6) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2] * tilingData_->indexShape[3] *
                                    tilingData_->indexShape[4] * tilingData_->indexShape[5];
            int64_t idxPerStride2 = tilingData_->indexShape[2] * tilingData_->indexShape[3] * tilingData_->indexShape[4] *
                                    tilingData_->indexShape[5];
            int64_t idxPerStride3 = tilingData_->indexShape[3] * tilingData_->indexShape[4] * tilingData_->indexShape[5];
            int64_t idxPerStride4 = tilingData_->indexShape[4] * tilingData_->indexShape[5];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4] *
                                  tilingData_->xShape[5];
            int64_t xPerStride2 = tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4] * tilingData_->xShape[5];
            int64_t xPerStride3 = tilingData_->xShape[3] * tilingData_->xShape[4] * tilingData_->xShape[5];
            int64_t xPerStride4 = tilingData_->xShape[4] * tilingData_->xShape[5];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / idxPerStride2;
                idx = idx - dim2 * idxPerStride2;
                dim3 = idx / idxPerStride3;
                idx = idx - dim3 * idxPerStride3;
                dim4 = idx / idxPerStride4;
                idx = idx - dim4 * idxPerStride4;
                dim5 = idx / tilingData_->indexShape[5];
                dim6 = idx - dim5 * tilingData_->indexShape[5];
                xOffset = dim1 * tilingData_->xStrideArr[1] + dim2 * tilingData_->xStrideArr[2] + dim3 * tilingData_->xStrideArr[3] +
                          dim4 * tilingData_->xStrideArr[4] + dim5 * tilingData_->xStrideArr[5] + dim6 * tilingData_->xStrideArr[6];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * xPerStride2 + dim3 * xPerStride3 + dim4 * xPerStride4 + dim5 * tilingData_->xShape[5] +
                                dim6;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / xPerStride2;
                        idx = idx - dim2 * xPerStride2;
                        dim3 = idx / xPerStride3;
                        idx = idx - dim3 * xPerStride3;
                        dim4 = idx / xPerStride4;
                        idx = idx - dim4 * xPerStride4;
                        dim5 = idx / tilingData_->xShape[5];
                        dim6 = idx - dim5 * tilingData_->xShape[5];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2] && dim4 < tilingData_->indexShape[3] &&
                            dim5 < tilingData_->indexShape[4] && dim6 < tilingData_->indexShape[5]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        }
    } else if constexpr (DIM_NUM == DIM8) {
        if constexpr (AXIS == 0) {
            CopyInX(xOffset, 1, xAfterAxis_);
            CopyInIdx(idxOffset, 1, idxAfterAxis_);
            LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
            LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
            LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, 0, 0, 0);
            xQueue_.FreeTensor(xLocal);
            indexQueue_.FreeTensor(idxLocal);
            yQueue_.EnQue(yLocal);
            CopyOut(yOffset, 1, idxAfterAxis_);
        } else if constexpr (AXIS == 1) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset;
                xOffset = dim1 * tilingData_->xStrideArr[0];
                xPerAxisOffset = dim1;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset;
                        if (dim1 < tilingData_->indexShape[0]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 2) {
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / tilingData_->indexShape[1];
                dim2 = idxPerAxisOffset - dim1 * tilingData_->indexShape[1];
                xOffset = dim1 * tilingData_->xStrideArr[0] + dim2 * tilingData_->xStrideArr[1];
                xPerAxisOffset = dim1 * tilingData_->xShape[1] + dim2;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / tilingData_->xShape[1];
                        dim2 = xPerAxisOffset - dim1 * tilingData_->xShape[1];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 3) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / tilingData_->indexShape[2];
                dim3 = idx - dim2 * tilingData_->indexShape[2];
                xOffset = dim1 * tilingData_->xStrideArr[0] + dim2 * tilingData_->xStrideArr[1] + dim3 * tilingData_->xStrideArr[2];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * tilingData_->xShape[2] + dim3;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / tilingData_->xShape[2];
                        dim3 = idx - dim2 * tilingData_->xShape[2];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 4) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2] * tilingData_->indexShape[3];
            int64_t idxPerStride2 = tilingData_->indexShape[2] * tilingData_->indexShape[3];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2] * tilingData_->xShape[3];
            int64_t xPerStride2 = tilingData_->xShape[2] * tilingData_->xShape[3];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / idxPerStride2;
                idx = idx - dim2 * idxPerStride2;
                dim3 = idx / tilingData_->indexShape[3];
                dim4 = idx - dim3 * tilingData_->indexShape[3];
                xOffset = dim1 * tilingData_->xStrideArr[0] + dim2 * tilingData_->xStrideArr[1] + dim3 * tilingData_->xStrideArr[2] +
                          dim4 * tilingData_->xStrideArr[3];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * xPerStride2 + dim3 * tilingData_->xShape[3] + dim4;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / xPerStride2;
                        idx = idx - dim2 * xPerStride2;
                        dim3 = idx / tilingData_->xShape[3];
                        dim4 = idx - dim3 * tilingData_->xShape[3];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2] && dim4 < tilingData_->indexShape[3]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 5) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2] * tilingData_->indexShape[3] *
                                    tilingData_->indexShape[4];
            int64_t idxPerStride2 = tilingData_->indexShape[2] * tilingData_->indexShape[3] * tilingData_->indexShape[4];
            int64_t idxPerStride3 = tilingData_->indexShape[3] * tilingData_->indexShape[4];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4];
            int64_t xPerStride2 = tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4];
            int64_t xPerStride3 = tilingData_->xShape[3] * tilingData_->xShape[4];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / idxPerStride2;
                idx = idx - dim2 * idxPerStride2;
                dim3 = idx / idxPerStride3;
                idx = idx - dim3 * idxPerStride3;
                dim4 = idx / tilingData_->indexShape[4];
                dim5 = idx - dim4 * tilingData_->indexShape[4];
                xOffset = dim1 * tilingData_->xStrideArr[0] + dim2 * tilingData_->xStrideArr[1] + dim3 * tilingData_->xStrideArr[2] +
                          dim4 * tilingData_->xStrideArr[3] + dim5 * tilingData_->xStrideArr[4];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * xPerStride2 + dim3 * xPerStride3 + dim4 * tilingData_->xShape[4] + dim5;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / xPerStride2;
                        idx = idx - dim2 * xPerStride2;
                        dim3 = idx / xPerStride3;
                        idx = idx - dim3 * xPerStride3;
                        dim4 = idx / tilingData_->xShape[4];
                        dim5 = idx - dim4 * tilingData_->xShape[4];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2] && dim4 < tilingData_->indexShape[3] &&
                            dim5 < tilingData_->indexShape[4]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 6) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2] * tilingData_->indexShape[3] *
                                    tilingData_->indexShape[4] * tilingData_->indexShape[5];
            int64_t idxPerStride2 = tilingData_->indexShape[2] * tilingData_->indexShape[3] * tilingData_->indexShape[4] *
                                    tilingData_->indexShape[5];
            int64_t idxPerStride3 = tilingData_->indexShape[3] * tilingData_->indexShape[4] * tilingData_->indexShape[5];
            int64_t idxPerStride4 = tilingData_->indexShape[4] * tilingData_->indexShape[5];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4] *
                                    tilingData_->xShape[5];
            int64_t xPerStride2 = tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4] * tilingData_->xShape[5];
            int64_t xPerStride3 = tilingData_->xShape[3] * tilingData_->xShape[4] * tilingData_->xShape[5];
            int64_t xPerStride4 = tilingData_->xShape[4] * tilingData_->xShape[5];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / idxPerStride2;
                idx = idx - dim2 * idxPerStride2;
                dim3 = idx / idxPerStride3;
                idx = idx - dim3 * idxPerStride3;
                dim4 = idx / idxPerStride4;
                idx = idx - dim4 * idxPerStride4;
                dim5 = idx / tilingData_->indexShape[5];
                dim6 = idx - dim5 * tilingData_->indexShape[5];
                xOffset = dim1 * tilingData_->xStrideArr[0] + dim2 * tilingData_->xStrideArr[1] + dim3 * tilingData_->xStrideArr[2] +
                          dim4 * tilingData_->xStrideArr[3] + dim5 * tilingData_->xStrideArr[4] + dim6 * tilingData_->xStrideArr[5];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * xPerStride2 + dim3 * xPerStride3 + dim4 * xPerStride4 + dim5 * tilingData_->xShape[5] +
                                dim6;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / xPerStride2;
                        idx = idx - dim2 * xPerStride2;
                        dim3 = idx / xPerStride3;
                        idx = idx - dim3 * xPerStride3;
                        dim4 = idx / xPerStride4;
                        idx = idx - dim4 * xPerStride4;
                        dim5 = idx / tilingData_->xShape[5];
                        dim6 = idx - dim5 * tilingData_->xShape[5];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2] && dim4 < tilingData_->indexShape[3] &&
                            dim5 < tilingData_->indexShape[4] && dim6 < tilingData_->indexShape[5]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        } else if constexpr (AXIS == 7) {
            int64_t idxPerStride1 = tilingData_->indexShape[1] * tilingData_->indexShape[2] * tilingData_->indexShape[3] *
                                    tilingData_->indexShape[4] * tilingData_->indexShape[5] * tilingData_->indexShape[6];
            int64_t idxPerStride2 = tilingData_->indexShape[2] * tilingData_->indexShape[3] * tilingData_->indexShape[4] *
                                    tilingData_->indexShape[5] * tilingData_->indexShape[6];
            int64_t idxPerStride3 = tilingData_->indexShape[3] * tilingData_->indexShape[4] * tilingData_->indexShape[5] *
                                    tilingData_->indexShape[6];
            int64_t idxPerStride4 = tilingData_->indexShape[4] * tilingData_->indexShape[5] * tilingData_->indexShape[6];
            int64_t idxPerStride5 = tilingData_->indexShape[5] * tilingData_->indexShape[6];
            int64_t xPerStride1 = tilingData_->xShape[1] * tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4] *
                                  tilingData_->xShape[5] * tilingData_->xShape[6];
            int64_t xPerStride2 = tilingData_->xShape[2] * tilingData_->xShape[3] * tilingData_->xShape[4] * tilingData_->xShape[5] *
                                  tilingData_->xShape[6];
            int64_t xPerStride3 = tilingData_->xShape[3] * tilingData_->xShape[4] * tilingData_->xShape[5] * tilingData_->xShape[6];
            int64_t xPerStride4 = tilingData_->xShape[4] * tilingData_->xShape[5] * tilingData_->xShape[6];
            int64_t xPerStride5 = tilingData_->xShape[5] * tilingData_->xShape[6];
            int64_t idxRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(INDEX_T), static_cast<uint64_t>(blockSize));
            int64_t yRegOffsetStride = ops::CeilAlign((uint64_t)idxAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            int64_t xRegOffsetStride = ops::CeilAlign((uint64_t)xAfterAxis_ * sizeof(X_T), static_cast<uint64_t>(blockSize));
            for (int64_t i = 0; i < LoopSize; i++) {
                int64_t needComputeNum = i == LoopSize - 1 ? tailComputeNum : indexLoadInUbNum;
                int64_t copyInXNum = i == LoopSize - 1 ? needComputeNum : tilingData_->xLoadInUbNum;
                idxPerAxisOffset = GetBlockIdx() * tilingData_->perCoreNum + i * indexLoadInUbNum;
                idxRegOffset = 0;
                yRegOffset = 0;
                CopyInIdx(idxOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                LocalTensor<INDEX_T> idxLocal = indexQueue_.DeQue<INDEX_T>();
                LocalTensor<X_T> yLocal = yQueue_.AllocTensor<X_T>();
                dim1 = idxPerAxisOffset / idxPerStride1;
                idx = idxPerAxisOffset - dim1 * idxPerStride1;
                dim2 = idx / idxPerStride2;
                idx = idx - dim2 * idxPerStride2;
                dim3 = idx / idxPerStride3;
                idx = idx - dim3 * idxPerStride3;
                dim4 = idx / idxPerStride4;
                idx = idx - dim4 * idxPerStride4;
                dim5 = idx / idxPerStride5;
                idx = idx - dim5 * idxPerStride5;
                dim6 = idx / tilingData_->indexShape[6];
                dim7 = idx - dim6 * tilingData_->indexShape[6];
                xOffset = dim1 * tilingData_->xStrideArr[0] + dim2 * tilingData_->xStrideArr[1] + dim3 * tilingData_->xStrideArr[2] +
                          dim4 * tilingData_->xStrideArr[3] + dim5 * tilingData_->xStrideArr[4] + dim6 * tilingData_->xStrideArr[5] +
                          dim7 * tilingData_->xStrideArr[6];
                xPerAxisOffset = dim1 * xPerStride1 + dim2 * xPerStride2 + dim3 * xPerStride3 + dim4 * xPerStride4 + dim5 * xPerStride5 +
                                dim6 * tilingData_->xShape[6] + dim7;
                for (int64_t computeNum = 0; computeNum < needComputeNum;) {
                    xRegOffset = 0;
                    CopyInX(xOffset, (uint16_t)copyInXNum, xAfterAxis_);
                    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
                    for (int64_t i = 0; i < copyInXNum && computeNum != needComputeNum; i++) {
                        dim1 = xPerAxisOffset / xPerStride1;
                        idx = xPerAxisOffset - dim1 * xPerStride1;
                        dim2 = idx / xPerStride2;
                        idx = idx - dim2 * xPerStride2;
                        dim3 = idx / xPerStride3;
                        idx = idx - dim3 * xPerStride3;
                        dim4 = idx / xPerStride4;
                        idx = idx - dim4 * xPerStride4;
                        dim5 = idx / xPerStride5;
                        idx = idx - dim5 * xPerStride5;
                        dim6 = idx / tilingData_->xShape[6];
                        dim7 = idx - dim6 * tilingData_->xShape[6];
                        if (dim1 < tilingData_->indexShape[0] && dim2 < tilingData_->indexShape[1] &&
                            dim3 < tilingData_->indexShape[2] && dim4 < tilingData_->indexShape[3] &&
                            dim5 < tilingData_->indexShape[4] && dim6 < tilingData_->indexShape[5] &&
                            dim7 < tilingData_->indexShape[6]) {
                            Compute(xLocal, idxLocal, yLocal, idxAfterAxis_, 0, xRegOffset, idxRegOffset, yRegOffset);
                            computeNum++;
                            idxRegOffset += idxRegOffsetStride;
                            yRegOffset += yRegOffsetStride;
                        }
                        xPerAxisOffset++;
                        xRegOffset += xRegOffsetStride;
                    }
                    xQueue_.FreeTensor(xLocal);
                    xOffset += xAfterAxis_ * copyInXNum;
                }
                indexQueue_.FreeTensor(idxLocal);
                yQueue_.EnQue(yLocal);
                CopyOut(yOffset, (uint16_t)needComputeNum, idxAfterAxis_);
                idxOffset += idxAfterAxis_ * indexLoadInUbNum;
                yOffset = idxOffset;
            }
        }
    }
}
} // namespace GatherElements
#endif // GATHER_ELEMENTS_FULL_LOAD_H