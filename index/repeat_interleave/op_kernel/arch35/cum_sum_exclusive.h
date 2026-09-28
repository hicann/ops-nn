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
 * \file cum_sum_exclusive.h
 * \brief
 */

#ifndef CUM_SUM_EXCLUSIVE_H
#define CUM_SUM_EXCLUSIVE_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "../inc/platform.h"

namespace RepeatInterleave {
using namespace AscendC;

constexpr uint64_t CUMSUM_BUFFER = 2;
constexpr uint32_t NUM_TWO = 2;

template <typename V>
using IndexRegType = typename std::conditional<IsSameType<V, int64_t>::value,
                                               typename AscendC::Reg::RegTensor<uint64_t, AscendC::Reg::RegTraitNumTwo>,
                                               typename AscendC::Reg::RegTensor<uint32_t>>::type;

template <typename V>
using InnerRegType = typename std::conditional<IsSameType<V, int64_t>::value,
                                               typename AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo>,
                                               typename AscendC::Reg::RegTensor<int32_t>>::type;

template <typename V>
__simd_callee__ inline AscendC::Reg::MaskReg CreateCountMask(uint32_t& count)
{
    if constexpr (IsSameType<V, int64_t>::value) {
        return AscendC::Reg::UpdateMask<int64_t, AscendC::Reg::RegTraitNumTwo>(count);
    } else {
        return AscendC::Reg::UpdateMask<V>(count);
    }
}

template <typename V>
__simd_callee__ inline AscendC::Reg::MaskReg CreateFullMask()
{
    if constexpr (IsSameType<V, int64_t>::value) {
        return AscendC::Reg::CreateMask<int64_t, AscendC::Reg::MaskPattern::ALL, AscendC::Reg::RegTraitNumTwo>();
    } else {
        return AscendC::Reg::CreateMask<V, AscendC::Reg::MaskPattern::ALL>();
    }
}

template <typename T, typename U, typename V, typename TilingDataT>
class CumSumExclusive {
public:
    __aicore__ inline CumSumExclusive(const TilingDataT& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyInRepeat(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyInRepeatSum(int64_t offset, int64_t dataLen);
    __aicore__ inline void ComputePrefixSum(int64_t dataLen);
    __aicore__ inline void ComputeSingleLoopPrefixSum(LocalTensor<U> repeatsLocal, uint32_t dataLen,
                                                      __ubuf__ V* prefixSumAddr, __ubuf__ V* tmpAddr);
    template <bool isCopyOutCoreSum = false, bool notNeedCopyPrefixSum = false>
    __aicore__ inline void CopyOutRepeatSumExclusive(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOutRepeatSum(int64_t offset, int64_t dataLen);
    __aicore__ inline void Process();
    __aicore__ inline void ComputePreCoreMaskSum();
    __aicore__ inline void CustomReduceSum(const LocalTensor<V>& dst, const LocalTensor<V>& src, uint16_t dataLen);

private:
    AscendC::GlobalTensor<U> repeatsGm_;
    AscendC::GlobalTensor<V> prefixSumGm_;
    AscendC::GlobalTensor<V> prefixSumGmTmp_;
    AscendC::GlobalTensor<V> coreSumGm_;

    TQue<QuePosition::VECIN, CUMSUM_BUFFER> repeatsQueue_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TQue<QuePosition::VECOUT, 1> prefixSumQueue_;

    TPipe& pipe_;
    const TilingDataT& tilingData_;

    V curRepeatSum_{0};
    V preCoreMaskSum_{0};
};

template <typename T, typename U, typename V, typename TilingDataT>
__aicore__ inline void CumSumExclusive<T, U, V, TilingDataT>::Init(GM_ADDR x, GM_ADDR repeats, GM_ADDR y,
                                                                   GM_ADDR workspace)
{
    repeatsGm_.SetGlobalBuffer((__gm__ U*)repeats + AscendC::GetBlockIdx() * tilingData_.cumSumNormalCoreRepeatsCount);
    coreSumGm_.SetGlobalBuffer((__gm__ V*)(workspace) + tilingData_.mergedDims[1]);
    prefixSumGm_.SetGlobalBuffer((__gm__ V*)(workspace) +
                                 AscendC::GetBlockIdx() * tilingData_.cumSumNormalCoreRepeatsCount);
    prefixSumGmTmp_ = prefixSumGm_[1];

    pipe_.InitBuffer(repeatsQueue_, CUMSUM_BUFFER, tilingData_.cumSumNormalUbFactors * sizeof(V));

    if constexpr (!std::is_same_v<U, V>) {
        pipe_.InitBuffer(tmpBuf_, tilingData_.cumSumNormalUbFactors * sizeof(V));
    } else {
        pipe_.InitBuffer(tmpBuf_, NUM_TWO * platform::GetUbBlockSize());
    }

    pipe_.InitBuffer(prefixSumQueue_, 1, tilingData_.cumSumNormalUbFactors * sizeof(V));
}

template <typename T, typename U, typename V, typename TilingDataT>
__aicore__ inline void CumSumExclusive<T, U, V, TilingDataT>::CopyInRepeat(int64_t offset, int64_t dataLen)
{
    DataCopyPadExtParams<U> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyExtParams inParams;
    inParams.blockCount = 1;
    inParams.blockLen = dataLen * sizeof(U);
    inParams.srcStride = 0;
    inParams.dstStride = 0;

    LocalTensor<U> repeatsLocal = repeatsQueue_.AllocTensor<U>();
    DataCopyPad(repeatsLocal, repeatsGm_[offset], inParams, padParams);
    repeatsQueue_.EnQue(repeatsLocal);
}

template <typename T, typename U, typename V, typename TilingDataT>
__aicore__ inline void CumSumExclusive<T, U, V, TilingDataT>::CopyInRepeatSum(int64_t offset, int64_t dataLen)
{
    DataCopyPadExtParams<V> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyExtParams inParams;
    inParams.blockCount = 1;
    inParams.blockLen = dataLen * sizeof(V);
    inParams.srcStride = 0;
    inParams.dstStride = 0;

    LocalTensor<V> prefixSumLocal = prefixSumQueue_.AllocTensor<V>();
    DataCopyPad(prefixSumLocal, prefixSumGm_[offset], inParams, padParams);

    auto vWiatMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(vWiatMTEEventID);
    WaitFlag<HardEvent::MTE2_V>(vWiatMTEEventID);

    Adds(prefixSumLocal, prefixSumLocal, preCoreMaskSum_, dataLen);

    prefixSumQueue_.EnQue(prefixSumLocal);
}

template <typename U, typename V>
__simd_callee__ inline void CopyRepeatsToPrefixSumVf(__ubuf__ U* repeatsAddr, __ubuf__ V* prefixSumAddr,
                                                     uint32_t vfLenB64, uint16_t loopB64)
{
    uint32_t main = vfLenB64;
    uint32_t stride = vfLenB64;

    AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<V>(main);

    AscendC::Reg::RegTensor<U> srcRegB32;
    AscendC::Reg::RegTensor<V> dstReg;
    auto prefixSumTmpAddr = prefixSumAddr;
    for (uint16_t i = 0; i < loopB64; ++i) {
        AscendC::Reg::LoadAlign<U, AscendC::Reg::LoadDist::DIST_UNPACK_B32>(srcRegB32, repeatsAddr + i * stride);
        AscendC::Reg::Cast<V, U, castTraitB322B64>(dstReg, srcRegB32, p0);
        AscendC::Reg::StoreAlign(prefixSumTmpAddr + i * stride, dstReg, p0);
    }
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
}

template <typename V>
__simd_callee__ inline void ComputeColPrefixSumVf(__ubuf__ V* prefixSumAddr, __ubuf__ V* tmpAddr, uint32_t rows,
                                                  uint32_t cols, uint16_t size0)
{
    uint32_t stride = rows;
    InnerRegType<V> tmp;
    IndexRegType<V> sequence;
    IndexRegType<V> index;
    AscendC::Reg::Arange(tmp, 0);
    sequence = (IndexRegType<V>&)tmp;
    InnerRegType<V> v0;
    InnerRegType<V> v1;
    InnerRegType<V> v2;
    uint32_t main = rows;
    AscendC::Reg::MaskReg p0 = CreateCountMask<V>(main);
    AscendC::Reg::Duplicate(v1, 0, p0);
    AscendC::Reg::Muls(sequence, sequence, cols, p0);
    auto prefixSumTmpAddr = prefixSumAddr;
    auto tempAddr = tmpAddr;
    for (uint16_t i = 0; i < size0; ++i) {
        AscendC::Reg::Adds(index, sequence, (uint32_t)i, p0);
        AscendC::Reg::Gather(v0, prefixSumTmpAddr, index, p0);
        AscendC::Reg::Add(v2, v0, v1, p0);
        AscendC::Reg::Move(v1, v2, p0);
        AscendC::Reg::StoreAlign(tempAddr + i * stride, v2, p0);
    }
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
}

template <typename V>
__simd_callee__ inline void ComputeRowPrefixSumVf(__ubuf__ V* prefixSumAddr, __ubuf__ V* tmpAddr, uint32_t rows,
                                                  uint32_t cols, uint16_t size1, uint16_t tailSize1, uint32_t vfLen)
{
    InnerRegType<V> tmp;
    IndexRegType<V> sequence;
    IndexRegType<V> index;
    AscendC::Reg::Arange(tmp, 0);
    sequence = (IndexRegType<V>&)tmp;
    AscendC::Reg::MaskReg pregFull = CreateFullMask<V>();
    AscendC::Reg::Muls(sequence, sequence, rows, pregFull);
    uint32_t spReg1 = tailSize1 - 1;
    AscendC::Reg::MaskReg sp1 = CreateCountMask<V>(spReg1);
    InnerRegType<V> v0;
    InnerRegType<V> v1;
    InnerRegType<V> v2;
    InnerRegType<V> v3;
    InnerRegType<V> v4;
    IndexRegType<V> vdSque;
    AscendC::Reg::UnalignRegForStore u1;
    AscendC::Reg::Duplicate(v1, 0, pregFull);
    for (uint16_t i = 0; i < static_cast<uint16_t>(rows); i++) {
        uint32_t mainCols = cols;
        AscendC::Reg::Adds(vdSque, sequence, (uint32_t)i, pregFull);
        for (uint16_t j = 0; j < size1; j++) {
            AscendC::Reg::MaskReg p1 = CreateCountMask<V>(mainCols);
            AscendC::Reg::Adds(index, vdSque, (uint32_t)(j * vfLen * vfLen), p1);
            AscendC::Reg::Gather(v0, tmpAddr, index, p1);
            AscendC::Reg::Add(v2, v0, v1, p1);
            AscendC::Reg::StoreUnAlign(prefixSumAddr, v2, u1, rows);
        }
        AscendC::Reg::StoreUnAlignPost(prefixSumAddr, u1, 0);
        AscendC::Reg::MaskReg p1 = CreateCountMask<V>(mainCols);
        AscendC::Reg::Adds(index, vdSque, (uint32_t)(size1 * vfLen * vfLen), p1);
        AscendC::Reg::Gather(v0, tmpAddr, index, p1);
        AscendC::Reg::Add(v2, v0, v1, p1);
        AscendC::Reg::StoreUnAlign(prefixSumAddr, v2, u1, tailSize1);
        AscendC::Reg::Duplicate(v4, 0, sp1);
        AscendC::Reg::Move<V, AscendC::Reg::MaskMergeMode::MERGING>(v2, v4, sp1);
        AscendC::Reg::Reduce<Reg::ReduceType::SUM>(v3, v2, p1);
        AscendC::Reg::Duplicate(v1, v3, pregFull);
        AscendC::Reg::StoreUnAlignPost(prefixSumAddr, u1, 0);
    }
}

template <typename U, typename V>
__simd_vf__ inline void ComputeSingleLoopPrefixSumVf(__ubuf__ U* repeatsAddr, __ubuf__ V* prefixSumAddr,
                                                     __ubuf__ V* tmpAddr, uint32_t vfLen, uint32_t rows, uint32_t cols,
                                                     uint16_t size0, uint16_t size1, uint16_t tailSize1,
                                                     uint32_t vfLenB64, uint16_t loopB64)
{
    if constexpr (!std::is_same_v<U, V>) {
        CopyRepeatsToPrefixSumVf<U, V>(repeatsAddr, prefixSumAddr, vfLenB64, loopB64);
        ComputeColPrefixSumVf<V>(prefixSumAddr, tmpAddr, rows, cols, size0);
        ComputeRowPrefixSumVf<V>(prefixSumAddr, tmpAddr, rows, cols, size1, tailSize1, vfLen);
    } else {
        ComputeColPrefixSumVf<V>(repeatsAddr, tmpAddr, rows, cols, size0);
        ComputeRowPrefixSumVf<V>(repeatsAddr, tmpAddr, rows, cols, size1, tailSize1, vfLen);
    }
}

template <typename T, typename U, typename V, typename TilingDataT>
__aicore__ inline void CumSumExclusive<T, U, V, TilingDataT>::ComputeSingleLoopPrefixSum(LocalTensor<U> repeatsLocal,
                                                                                         uint32_t dataLen,
                                                                                         __ubuf__ V* prefixSumAddr,
                                                                                         __ubuf__ V* tmpAddr)
{
    auto repeatsAddr = (__ubuf__ U*)repeatsLocal.GetPhyAddr();
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint32_t vfLenB64 = Ops::Base::GetVRegSize() / sizeof(int64_t);
    uint32_t rows = vfLen;
    uint32_t cols = (dataLen + vfLen - 1) / vfLen;
    uint32_t loopB64 = (dataLen + vfLenB64 - 1) / vfLenB64;

    uint16_t size0 = cols;
    uint16_t size1 = cols / vfLen;
    uint16_t tailSize1 = cols - size1 * vfLen;
    if (tailSize1 == 0) {
        size1--;
        tailSize1 = vfLen;
    }
    ComputeSingleLoopPrefixSumVf<U, V>(repeatsAddr, prefixSumAddr, tmpAddr, vfLen, rows, cols, size0, size1, tailSize1,
                                       vfLenB64, loopB64);
}

template <typename T, typename U, typename V, typename TilingDataT>
__aicore__ inline void CumSumExclusive<T, U, V, TilingDataT>::ComputePrefixSum(int64_t dataLen)
{
    LocalTensor<V> prefixSumLocal = prefixSumQueue_.AllocTensor<V>();
    auto prefixSumAddr = (__ubuf__ V*)prefixSumLocal.GetPhyAddr();

    LocalTensor<V> tmpLocal;
    __ubuf__ V* tmpAddr = nullptr;
    if constexpr (!std::is_same_v<U, V>) {
        tmpLocal = tmpBuf_.Get<V>();
        tmpAddr = (__ubuf__ V*)tmpLocal.GetPhyAddr();
    }

    LocalTensor<U> repeatsLocal = repeatsQueue_.DeQue<U>();
    ComputeSingleLoopPrefixSum(repeatsLocal, dataLen, tmpAddr, prefixSumAddr);
    if constexpr (!std::is_same_v<U, V>) {
        Adds(prefixSumLocal, tmpLocal, curRepeatSum_, dataLen);
    } else {
        Adds(prefixSumLocal, repeatsLocal, curRepeatSum_, dataLen);
    }
    repeatsQueue_.FreeTensor(repeatsLocal);

    auto sWiatVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(sWiatVEventID);
    WaitFlag<HardEvent::V_S>(sWiatVEventID);
    curRepeatSum_ = prefixSumLocal.GetValue(dataLen - 1);
    prefixSumQueue_.EnQue(prefixSumLocal);
}

template <typename T, typename U, typename V, typename TilingDataT>
template <bool isCopyOutCoreSum, bool notNeedCopyPrefixSum>
__aicore__ inline void CumSumExclusive<T, U, V, TilingDataT>::CopyOutRepeatSumExclusive(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = dataLen * sizeof(V);
    outParams.srcStride = 0;
    outParams.dstStride = 0;

    LocalTensor<V> prefixSumLocal = prefixSumQueue_.DeQue<V>();

    if constexpr (!notNeedCopyPrefixSum) {
        DataCopyPad(prefixSumGmTmp_[offset], prefixSumLocal, outParams);
    }

    if constexpr (isCopyOutCoreSum) {
        outParams.blockCount = 1;
        outParams.blockLen = 1 * sizeof(V);
        outParams.srcStride = 0;
        outParams.dstStride = 0;

        int64_t customOffset = platform::GetUbBlockSize() / sizeof(V);
        LocalTensor<V> coreSumLocal = tmpBuf_.Get<V>();
        coreSumLocal.SetValue(0, curRepeatSum_);
        coreSumLocal.SetValue(customOffset, 0);
        auto mte3WiatSEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(mte3WiatSEventID);
        WaitFlag<HardEvent::S_MTE3>(mte3WiatSEventID);
        DataCopyPad(coreSumGm_[AscendC::GetBlockIdx()], coreSumLocal, outParams);
        DataCopyPad(prefixSumGm_, coreSumLocal[customOffset], outParams);
    }

    prefixSumQueue_.FreeTensor(prefixSumLocal);
}

template <typename T, typename U, typename V, typename TilingDataT>
__aicore__ inline void CumSumExclusive<T, U, V, TilingDataT>::CopyOutRepeatSum(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = dataLen * sizeof(V);
    outParams.srcStride = 0;
    outParams.dstStride = 0;

    LocalTensor<V> prefixSumLocal = prefixSumQueue_.DeQue<V>();

    DataCopyPad(prefixSumGm_[offset], prefixSumLocal, outParams);

    prefixSumQueue_.FreeTensor(prefixSumLocal);
}

template <typename V>
__simd_vf__ inline void CustomReduceSumVf(__ubuf__ V* srcAddr, __ubuf__ V* dstAddr, uint16_t dataLen, uint16_t vfLen,
                                          uint16_t loopSize)
{
    AscendC::Reg::RegTensor<V> src;
    AscendC::Reg::RegTensor<V> dst;
    AscendC::Reg::RegTensor<V> tmpSum;
    uint32_t pnum = static_cast<uint32_t>(dataLen);
    uint32_t sumMask = 1;
    AscendC::Reg::MaskReg oneMask = AscendC::Reg::UpdateMask<V>(sumMask);
    AscendC::Reg::Duplicate(dst, 0, oneMask);
    for (uint16_t i = 0; i < loopSize; i++) {
        AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<V>(pnum);
        AscendC::Reg::LoadAlign<V, Reg::PostLiteral::POST_MODE_UPDATE>(src, srcAddr, vfLen);
        AscendC::Reg::Reduce<Reg::ReduceType::SUM>(tmpSum, src, pMask);
        AscendC::Reg::Add(dst, dst, tmpSum, oneMask);
    }
    AscendC::Reg::StoreAlign<V, Reg::PostLiteral::POST_MODE_UPDATE>(dstAddr, dst, 0, oneMask);
}

template <typename T, typename U, typename V, typename TilingDataT>
__aicore__ inline void CumSumExclusive<T, U, V, TilingDataT>::CustomReduceSum(const LocalTensor<V>& dst,
                                                                              const LocalTensor<V>& src,
                                                                              uint16_t dataLen)
{
    uint16_t vfLen = Ops::Base::GetVRegSize() / sizeof(V);
    uint16_t loopSize = (dataLen + vfLen - 1) / vfLen;
    auto srcAddr = (__ubuf__ V*)src.GetPhyAddr();
    auto dstAddr = (__ubuf__ V*)dst.GetPhyAddr();
    CustomReduceSumVf<V>(srcAddr, dstAddr, dataLen, vfLen, loopSize);
}

template <typename T, typename U, typename V, typename TilingDataT>
__aicore__ inline void CumSumExclusive<T, U, V, TilingDataT>::ComputePreCoreMaskSum()
{
    LocalTensor<V> tmpLocal = tmpBuf_.Get<V>();

    DataCopyExtParams inParams = {1, static_cast<uint32_t>(tilingData_.cumSumCoreNum * sizeof(V)), 0, 0, 0};
    DataCopyPadExtParams<V> padParams = {false, 0, 0, 0};
    DataCopyPad(tmpLocal, coreSumGm_, inParams, padParams);

    if (AscendC::GetBlockIdx() == 1) {
        auto sWiatVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(sWiatVEventID);
        WaitFlag<HardEvent::MTE2_S>(sWiatVEventID);
        preCoreMaskSum_ = tmpLocal.GetValue(0);
    } else { // AscendC::GetBlockIdx() != 0
        auto vWiatMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(vWiatMTEEventID);
        WaitFlag<HardEvent::MTE2_V>(vWiatMTEEventID);
        CustomReduceSum(tmpLocal, tmpLocal, AscendC::GetBlockIdx());
        auto sWiatVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(sWiatVEventID);
        WaitFlag<HardEvent::V_S>(sWiatVEventID);
        preCoreMaskSum_ = tmpLocal.GetValue(0);
    }
}

template <typename T, typename U, typename V, typename TilingDataT>
__aicore__ inline void CumSumExclusive<T, U, V, TilingDataT>::Process()
{
    int64_t loopSize = tilingData_.cumSumNormalCoreLoops;
    int64_t tailUbFactor = tilingData_.cumSumNormalCoreTailUbFactors;
    int64_t offset = 0;
    int64_t dataLen = tilingData_.cumSumNormalUbFactors;

    if (AscendC::GetBlockIdx() < tilingData_.cumSumCoreNum) {
        if (AscendC::GetBlockIdx() == tilingData_.cumSumCoreNum - 1) {
            loopSize = tilingData_.cumSumTailCoreLoops;
            tailUbFactor = tilingData_.cumSumTailCoreTailUbFactors;
        }

        for (int64_t i = 0; i < loopSize - 1; ++i) {
            offset = i * tilingData_.cumSumNormalUbFactors;
            CopyInRepeat(offset, dataLen);
            ComputePrefixSum(dataLen);
            CopyOutRepeatSumExclusive(offset, dataLen);
        }
        offset = (loopSize - 1) * tilingData_.cumSumNormalUbFactors;
        dataLen = tailUbFactor;
        CopyInRepeat(offset, dataLen);
        ComputePrefixSum(dataLen);
        if (dataLen == 1) {
            CopyOutRepeatSumExclusive<true, true>(offset, dataLen - 1);
        } else {
            CopyOutRepeatSumExclusive<true, false>(offset, dataLen - 1);
        }
    }

    // 通过核内前缀和计算全局前缀和
    if (tilingData_.cumSumCoreNum > 1) {
        SyncAll();
        pipe_.Reset();
        pipe_.InitBuffer(tmpBuf_, ops::CeilAlign(tilingData_.cumSumCoreNum * sizeof(V),
                                                 static_cast<uint64_t>(platform::GetUbBlockSize())));
        pipe_.InitBuffer(prefixSumQueue_, 1, tilingData_.cumSumNormalUbFactors * sizeof(V));

        if (AscendC::GetBlockIdx() == 0 || AscendC::GetBlockIdx() >= tilingData_.cumSumCoreNum) {
            return;
        }

        ComputePreCoreMaskSum();

        offset = 0;
        dataLen = tilingData_.cumSumNormalUbFactors;
        for (int64_t i = 0; i < loopSize - 1; ++i) {
            offset = i * tilingData_.cumSumNormalUbFactors;
            CopyInRepeatSum(offset, dataLen);
            CopyOutRepeatSum(offset, dataLen);

            auto vWiatMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(vWiatMTEEventID);
            WaitFlag<HardEvent::MTE3_MTE2>(vWiatMTEEventID);
        }
        offset = (loopSize - 1) * tilingData_.cumSumNormalUbFactors;
        dataLen = tailUbFactor;
        CopyInRepeatSum(offset, dataLen);
        CopyOutRepeatSum(offset, dataLen);
    }
}

} // namespace RepeatInterleave

#endif
