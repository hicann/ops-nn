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
 * \file repeat_interleave.h
 * \brief
 */

#ifndef REPEAT_INTERLEAVE_H
#define REPEAT_INTERLEAVE_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"

namespace RepeatInterleave {
using namespace AscendC;

constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t MIN_CP_THRESHOLD = 128;

template <typename T, typename U>
class RepeatInterleaveImpl {
public:
    __aicore__ inline RepeatInterleaveImpl(const RepeatInterleaveTilingKernelDataNorm& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyInRepeats(int64_t repeatDimIdx);
    __aicore__ inline void ComputeRepeatsOnCurDim(int64_t repeatDimIdx);
    __aicore__ inline void ComputeStartOffset();
    __aicore__ inline void CopyInX(int64_t repeatDimIdx, int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void CopyXToOut(int64_t dataCount);
    __aicore__ inline void CopyOneCpToRepeatOut(const LocalTensor<T> xInLocal, uint16_t repeatTimes);
    __aicore__ inline void CopyXToMatchOut(int64_t startCpIdx, int64_t cpCount);
    __aicore__ inline void CopyOutY(int64_t repeatDimIdx, int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void CopyMatchOutToY();
    __aicore__ inline void ProcessCpMatchToUb(int64_t startCpIdx, int64_t handleCpCount);
    __aicore__ inline void ProcessWholeCp();
    __aicore__ inline void ProcessSplitCp();
    __aicore__ inline void Process();

private:
    TPipe& pipe_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<U> repeatsGm_;
    AscendC::GlobalTensor<T> yGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xInQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> xOutQueue_;
    TQue<QuePosition::VECIN, 1> repeatsQueue_;
    const RepeatInterleaveTilingKernelDataNorm& tilingData_;

    int64_t cpTileOffset_{0};                // 源数据在cp轴上cpTile的偏移
    int64_t repeatsScalarValue_{0};          // repeats为标量时，其值
    int64_t curCoreFinishCount_{0};          // 当前核已经处理的repeats个数
    int64_t curCoreRepeatsCountOnCurDim_{0}; // 当前核在当前repeat轴上要处理的repeats个数

    int64_t eachLoopHandleNum_{0};
    int64_t tailLoopHandleNum_{0};

    bool isRepeatsScalar_{false};
    int64_t cpCountInUbFactor_{0}; // ubFactor大小的空间中能放下的cp轴数量
    int64_t copyFromXNum_{0};
    int64_t copyToMatchOutNum_{0};
    int64_t copyToGmNum_{0};
    int64_t outStartOffset_{0};
};

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::Init(GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace)
{
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    repeatsGm_.SetGlobalBuffer((__gm__ U*)repeats);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_.InitBuffer(repeatsQueue_, 1, tilingData_.ubFactor * sizeof(U));
    pipe_.InitBuffer(xInQueue_, DOUBLE_BUFFER, tilingData_.ubFactor * sizeof(T));
    pipe_.InitBuffer(xOutQueue_, DOUBLE_BUFFER, tilingData_.ubFactor * sizeof(T));

    cpCountInUbFactor_ = tilingData_.ubFactor / tilingData_.mergedDims[2];
    /* repeats输入时scalar的情况，直接取tiling传入的scalar值 */
    if (tilingData_.repeatsCount > -1) {
        isRepeatsScalar_ = true;
        repeatsScalarValue_ = tilingData_.repeatsCount;
    }
    ComputeStartOffset();
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::CopyInRepeats(int64_t repeatDimIdx)
{
    if (isRepeatsScalar_) {
        return;
    }

    /* repeats搬入地址起始位置 */
    int64_t offset = repeatDimIdx % tilingData_.mergedDims[1];
    int64_t dataLen =
        ((offset + tilingData_.ubFactor) > tilingData_.mergedDims[1] ? (tilingData_.mergedDims[1] - offset) :
                                                                       tilingData_.ubFactor) *
        sizeof(U);

    DataCopyExtParams inParams = {1, static_cast<uint32_t>(dataLen), 0, 0, 0};
    DataCopyPadExtParams<U> padParams = {false, 0, 0, 0};
    LocalTensor<U> repeatsLocal = repeatsQueue_.AllocTensor<U>();
    DataCopyPad(repeatsLocal, repeatsGm_[offset], inParams, padParams);
    repeatsQueue_.EnQue(repeatsLocal);
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::ComputeRepeatsOnCurDim(int64_t repeatDimIdx)
{
    /* repeats输入是scalar时直接取值 */
    if (isRepeatsScalar_) {
        curCoreRepeatsCountOnCurDim_ = repeatsScalarValue_;
        return;
    }
    LocalTensor<U> repeatsLocal = repeatsQueue_.DeQue<U>();
    uint32_t repeatOffset = repeatDimIdx % tilingData_.mergedDims[1] % tilingData_.ubFactor;
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    curCoreRepeatsCountOnCurDim_ = repeatsLocal.GetValue(repeatOffset);
    repeatsQueue_.FreeTensor(repeatsLocal);
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::ComputeStartOffset()
{
    int64_t batchIdx = 0;
    int64_t repeatOffset = 0;
    if (tilingData_.isSplitCP == 1) {
        batchIdx = GetBlockIdx() / tilingData_.cpSlice;
    } else {
        batchIdx = GetBlockIdx() * tilingData_.eachCoreBatchCount;
    }
    if (isRepeatsScalar_) {
        outStartOffset_ = (batchIdx * repeatsScalarValue_) * tilingData_.mergedDims[2];
    } else {
        outStartOffset_ = (batchIdx * tilingData_.totalRepeatSum) * tilingData_.mergedDims[2];
    }
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::CopyInX(int64_t repeatDimIdx, int64_t loopIdx, int64_t dataCount)
{
    /* 源数据在repeat轴上的偏移 */
    int64_t repeatOffset = repeatDimIdx * tilingData_.mergedDims[2];
    /* 源地址在当前轴上loop处理的偏移 */
    int64_t loopOffset = loopIdx * eachLoopHandleNum_;
    int64_t offset = repeatOffset + cpTileOffset_ + loopOffset;
    uint32_t dataLen = dataCount * sizeof(T);

    DataCopyExtParams inParams = {1, dataLen, 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    LocalTensor<T> xInLocal = xInQueue_.AllocTensor<T>();
    DataCopyPad(xInLocal, xGm_[offset], inParams, padParams);
    xInQueue_.EnQue(xInLocal);
    copyFromXNum_ = 0;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::CopyOneCpToRepeatOut(
    const LocalTensor<T> xInLocal, uint16_t repeatTimes)
{
    LocalTensor<T> xOutLocal = xOutQueue_.DeQue<T>();
    __local_mem__ T* xInLocalPtr = ((__local_mem__ T*)xInLocal.GetPhyAddr()) + copyFromXNum_;
    __local_mem__ T* xOutLocalPtr = ((__local_mem__ T*)xOutLocal.GetPhyAddr()) + copyToMatchOutNum_;

    int64_t dataCount = tilingData_.mergedDims[2];
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::UnalignReg uIn;
        AscendC::MicroAPI::UnalignReg uOut;
        AscendC::MicroAPI::RegTensor<T> inputRegTensor;
        AscendC::MicroAPI::DataCopyUnAlignPre(uIn, xInLocalPtr);
        AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            inputRegTensor, uIn, xInLocalPtr, dataCount);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                xOutLocalPtr, inputRegTensor, uOut, dataCount);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(xOutLocalPtr, uOut, 0);
    }

    copyToMatchOutNum_ += repeatTimes * dataCount;
    xOutQueue_.EnQue(xOutLocal);
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::CopyXToMatchOut(int64_t startCpIdx, int64_t cpCount)
{
    LocalTensor<T> xInLocal = xInQueue_.DeQue<T>();
    LocalTensor<T> xOutLocal = xOutQueue_.AllocTensor<T>();
    xOutQueue_.EnQue(xOutLocal);

    for (int64_t repeatDimIdx = 0; repeatDimIdx < cpCount; repeatDimIdx++) {
        if (!isRepeatsScalar_ && (repeatDimIdx % tilingData_.mergedDims[2]) >= tilingData_.ubFactor) {
            CopyInRepeats(repeatDimIdx);
        }
        ComputeRepeatsOnCurDim(startCpIdx + repeatDimIdx);
        int64_t loopSize = (curCoreRepeatsCountOnCurDim_ + cpCountInUbFactor_ - 1) / cpCountInUbFactor_;
        int64_t mainRepeatTimes = cpCountInUbFactor_;
        int64_t tailRepeatTimes = curCoreRepeatsCountOnCurDim_ - cpCountInUbFactor_ * (loopSize - 1);
        if (curCoreRepeatsCountOnCurDim_ == 0) {
            copyFromXNum_ += tilingData_.mergedDims[2];
            continue;
        }
        for (int64_t loopIdx = 0; loopIdx < (loopSize - 1); loopIdx++) {
            CopyOneCpToRepeatOut(xInLocal, mainRepeatTimes);
            CopyMatchOutToY();
            xOutLocal = xOutQueue_.AllocTensor<T>();
            xOutQueue_.EnQue(xOutLocal);
        }
        if (copyToMatchOutNum_ > tilingData_.ubFactor - tailRepeatTimes * tilingData_.mergedDims[2]) {
            CopyMatchOutToY();
            xOutLocal = xOutQueue_.AllocTensor<T>();
            xOutQueue_.EnQue(xOutLocal);
        }
        CopyOneCpToRepeatOut(xInLocal, tailRepeatTimes);
        copyFromXNum_ += tilingData_.mergedDims[2];
    }
    xInQueue_.FreeTensor(xInLocal);
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::CopyXToOut(int64_t dataCount)
{
    LocalTensor<T> xInLocal = xInQueue_.DeQue<T>();
    LocalTensor<T> xOutLocal = xOutQueue_.AllocTensor<T>();

    __local_mem__ int8_t* xInLocalPtr = (__local_mem__ int8_t*)xInLocal.GetPhyAddr();
    __local_mem__ int8_t* xOutLocalPtr = (__local_mem__ int8_t*)xOutLocal.GetPhyAddr();

    uint32_t totalBytes = dataCount * sizeof(T);
    uint16_t stride = Ops::Base::GetVRegSize();
    uint16_t size = (totalBytes + stride - 1) / stride;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int8_t> inputRegTensor;
        uint32_t sreg = totalBytes;
        AscendC::MicroAPI::MaskReg preg;

        for (uint16_t i = 0; i < size; i++) {
            preg = AscendC::MicroAPI::UpdateMask<int8_t>(sreg);
            AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<int8_t>(i, stride);
            AscendC::MicroAPI::DataCopy(inputRegTensor, xInLocalPtr, offset);
            AscendC::MicroAPI::DataCopy(xOutLocalPtr, inputRegTensor, offset, preg);
        }
    }
    xOutQueue_.EnQue(xOutLocal);
    xInQueue_.FreeTensor(xInLocal);
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::CopyMatchOutToY()
{
    LocalTensor<T> xOutLocal = xOutQueue_.DeQue<T>();
    DataCopyExtParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = copyToMatchOutNum_ * sizeof(T);
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(yGm_[outStartOffset_ + copyToGmNum_], xOutLocal, outParams);
    copyToGmNum_ += copyToMatchOutNum_;
    copyToMatchOutNum_ = 0;
    xOutQueue_.FreeTensor(xOutLocal);
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::CopyOutY(int64_t repeatDimIdx, int64_t loopIdx, int64_t dataCount)
{
    int64_t loopOffset = loopIdx * eachLoopHandleNum_;
    int64_t offset = outStartOffset_ + curCoreFinishCount_ * tilingData_.mergedDims[2] + cpTileOffset_ + loopOffset;

    CopyXToOut(dataCount);
    LocalTensor<T> xOutLocal = xOutQueue_.DeQue<T>();
    LoopModeParams loopParams;
    loopParams.loop1Size = curCoreRepeatsCountOnCurDim_;
    loopParams.loop2Size = 1;
    loopParams.loop1SrcStride = 0;
    loopParams.loop2SrcStride = 0;
    loopParams.loop1DstStride = tilingData_.mergedDims[2] * sizeof(T);
    loopParams.loop2DstStride = 0;
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = dataCount * sizeof(T);
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad<T, PaddingMode::Compact>(yGm_[offset], xOutLocal, outParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    xOutQueue_.FreeTensor(xOutLocal);
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::ProcessCpMatchToUb(int64_t startCpIdx, int64_t handleCpCount)
{
    int64_t loopSize = (handleCpCount + cpCountInUbFactor_ - 1) / cpCountInUbFactor_;
    int64_t mainCpNum = cpCountInUbFactor_;
    int64_t tailCpNum = handleCpCount - cpCountInUbFactor_ * (loopSize - 1);

    CopyInRepeats(0);
    int64_t handleStartCpIdx = 0;
    for (int64_t loopIdx = 0; loopIdx < (loopSize - 1); loopIdx++) {
        handleStartCpIdx = startCpIdx + loopIdx * mainCpNum;
        CopyInX(handleStartCpIdx, 0, mainCpNum * tilingData_.mergedDims[2]);
        CopyXToMatchOut(handleStartCpIdx, mainCpNum);
        CopyMatchOutToY();
    }
    handleStartCpIdx = startCpIdx + (loopSize - 1) * mainCpNum;
    CopyInX(handleStartCpIdx, 0, tailCpNum * tilingData_.mergedDims[2]);
    CopyXToMatchOut(handleStartCpIdx, tailCpNum);
    CopyMatchOutToY();
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::ProcessWholeCp()
{
    int64_t curCoreCpCount = tilingData_.eachCoreBatchCount;
    if (GetBlockIdx() == GetBlockNum() - 1) {
        curCoreCpCount = tilingData_.tailCoreBatchCount;
    }
    curCoreCpCount *= tilingData_.mergedDims[1];
    int64_t startCpIdx = GetBlockIdx() * tilingData_.eachCoreBatchCount * tilingData_.mergedDims[1];
    if (tilingData_.mergedDims[2] * sizeof(T) < MIN_CP_THRESHOLD) {
        /* 从每个核要复制的起始轴开始依次处理,不需要对cp轴分loop */
        ProcessCpMatchToUb(startCpIdx, curCoreCpCount);
        return;
    }

    /* cp轴大于128Byte时考虑对每次处理的copy轴数据分loop */
    int64_t loopSize = (tilingData_.mergedDims[2] + tilingData_.ubFactor - 1) / tilingData_.ubFactor;
    eachLoopHandleNum_ = tilingData_.ubFactor;
    tailLoopHandleNum_ = tilingData_.mergedDims[2] - tilingData_.ubFactor * (loopSize - 1);

    /* 先完成一次repeats搬运，当处理的repeatDim超过搬运的repeats大小时，再次搬运 */
    CopyInRepeats(0);
    for (int64_t repeatDimIdx = startCpIdx; repeatDimIdx < startCpIdx + curCoreCpCount; repeatDimIdx++) {
        if (!isRepeatsScalar_ && (repeatDimIdx % tilingData_.mergedDims[2]) >= tilingData_.ubFactor) {
            CopyInRepeats(repeatDimIdx);
        }
        ComputeRepeatsOnCurDim(repeatDimIdx);
        for (int64_t loopIdx = 0; loopIdx < loopSize - 1; loopIdx++) {
            CopyInX(repeatDimIdx, loopIdx, eachLoopHandleNum_);
            CopyOutY(repeatDimIdx, loopIdx, eachLoopHandleNum_);
        }
        CopyInX(repeatDimIdx, (loopSize - 1), tailLoopHandleNum_);
        CopyOutY(repeatDimIdx, (loopSize - 1), tailLoopHandleNum_);
        curCoreFinishCount_ += curCoreRepeatsCountOnCurDim_;
    }
}
template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::ProcessSplitCp()
{
    int64_t curCoreHandleCpTile = tilingData_.normalCP;
    if (GetBlockIdx() % tilingData_.cpSlice == (tilingData_.cpSlice - 1)) {
        curCoreHandleCpTile = tilingData_.tailCP;
    }

    /* 对每次处理的copy轴数据分loop */
    int64_t loopSize = (curCoreHandleCpTile + tilingData_.ubFactor - 1) / tilingData_.ubFactor;
    eachLoopHandleNum_ = tilingData_.ubFactor;
    tailLoopHandleNum_ = curCoreHandleCpTile - tilingData_.ubFactor * (loopSize - 1);
    cpTileOffset_ = GetBlockIdx() % tilingData_.cpSlice * tilingData_.normalCP;

    int64_t startCpIdx = (GetBlockIdx() / tilingData_.cpSlice) * tilingData_.mergedDims[1];
    /* 先完成一次repeats搬运，当处理的repeatDim超过搬运的repeats大小时，再次搬运 */
    CopyInRepeats(0);
    for (int64_t repeatDimIdx = startCpIdx; repeatDimIdx < startCpIdx + tilingData_.mergedDims[1]; repeatDimIdx++) {
        if (!isRepeatsScalar_ && (repeatDimIdx % tilingData_.mergedDims[2]) >= tilingData_.ubFactor) {
            CopyInRepeats(repeatDimIdx);
        }
        ComputeRepeatsOnCurDim(repeatDimIdx);
        for (int64_t loop = 0; loop < loopSize - 1; loop++) {
            CopyInX(repeatDimIdx, loop, eachLoopHandleNum_);
            CopyOutY(repeatDimIdx, loop, eachLoopHandleNum_);
        }
        CopyInX(repeatDimIdx, (loopSize - 1), tailLoopHandleNum_);
        CopyOutY(repeatDimIdx, (loopSize - 1), tailLoopHandleNum_);
        curCoreFinishCount_ += curCoreRepeatsCountOnCurDim_;
    }
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveImpl<T, U>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    if (tilingData_.isSplitCP == 1) {
        ProcessSplitCp();
    } else {
        ProcessWholeCp();
    }
}
} // namespace RepeatInterleave

#endif // REPEAT_INTERLEAVE_H
