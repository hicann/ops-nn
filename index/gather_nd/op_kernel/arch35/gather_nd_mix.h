/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_nd_mix.h
 * \brief
 */
#ifndef ASCENDC_GATHER_ND_GATHER_ND_MIX_H_
#define ASCENDC_GATHER_ND_GATHER_ND_MIX_H_

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace GatherNd
{
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_SUPPORT_DIM = 8;
constexpr uint32_t T_BUFFER_SIZE = 1024;
constexpr uint32_t T_BUFFER_OFFSET_512 = 512;
constexpr uint32_t T_BUFFER_OFFSET_640 = 640;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUMS = 128;
#else
constexpr uint32_t THREAD_NUMS = 2048;
#endif


template <typename T2, typename T3>
__aicore__ inline MicroAPI::MaskReg GenT2Mask(uint32_t& maskCount)
{
    MicroAPI::MaskReg reg;
    if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int64_t>::value) {
        reg = AscendC::MicroAPI::UpdateMask<T2, AscendC::MicroAPI::RegTraitNumTwo>(maskCount);
    } else {
        reg = AscendC::MicroAPI::UpdateMask<T2>(maskCount);
    }
    return reg;
}

// T2 为 conditionalType
template <typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUMS) inline void MixSimt( __ubuf__ T1 *yAddr, __ubuf__ T3 *indicesAddr,  __gm__ T1 *xAddr, 
    const T2 yCount,  const T3 aMergeAxisSize, const T2 shift, const T2 m)
{
    for (T2 idx = Simt::GetThreadIdx(); idx < yCount; idx += Simt::GetThreadNum()) {
        T3 indicesCurrentIdx = Simt::UintDiv(idx, m, shift);
        T3 colOffset = idx - indicesCurrentIdx * aMergeAxisSize;

        T3 rowStart = indicesAddr[indicesCurrentIdx];
        bool idxOutOfBound = rowStart < 0;
        yAddr[idx] = idxOutOfBound ? 0 : xAddr[rowStart + colOffset];
    }
}

// T2 为 conditionalType
template <typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUMS) inline void MixSimtWithLastAxisOne( __ubuf__ T1 *yAddr, __ubuf__ T3 *indicesAddr,  __gm__ T1 *xAddr, 
    const T2 yCount)
{
    for (T2 idx = Simt::GetThreadIdx(); idx < yCount; idx += Simt::GetThreadNum()) {
        T3 rowStart = indicesAddr[idx];
        bool idxOutOfBound = rowStart < 0;
        yAddr[idx] = idxOutOfBound ? 0 : xAddr[rowStart];
    }
}

template <typename T1, typename T2, typename T3>
class GatherNdMixKernel
{
public:
    __aicore__ inline GatherNdMixKernel(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace,
                                const GatherNdTilingData* __restrict ordTilingData, TPipe* pipeIn);
    __aicore__ inline void CopyInIndices(LocalTensor<T2>& dstTensor, int32_t indicesCount, int64_t indicesOffset, uint16_t rank);
    __aicore__ inline void CopyOutY(LocalTensor<T1>& yTensor, uint32_t yCount, int64_t yOffset);
    __aicore__ inline void Process();
    __aicore__ inline void GenGatherIndex();
    template <const bool NIS>
    __aicore__ inline void FixIndicesVf(LocalTensor<T2>& indicesTensor, LocalTensor<T3>& maxGatherShapeTensor, LocalTensor<T3>& strideShapeTensor, int32_t indicesNumPro, uint16_t rank, T3 aMergeAxisSize);

protected:
    TQue<QuePosition::VECIN, BUFFER_NUM> indicesQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue_;
    TBuf<TPosition::VECCALC> helpTBuf_;

    const GatherNdTilingData* __restrict tilingData_;

    // input
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> indicesGm_;

    // output
    GlobalTensor<T1> yGm_;
    int32_t blockIdx_;
};

template <typename T1, typename T2, typename T3>
__aicore__ inline void GatherNdMixKernel<T1, T2, T3>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace,
                                                           const GatherNdTilingData* __restrict ordTilingData,
                                                           TPipe* pipeIn)
{
    tilingData_ = ordTilingData;
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_->blockNum) {
        return;
    }

    // init gm
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    indicesGm_.SetGlobalBuffer((__gm__ T2*)indices + blockIdx_ * tilingData_->normalCoreIndicesNum * tilingData_->rank);
    yGm_.SetGlobalBuffer((__gm__ T1*)y + blockIdx_ * tilingData_->normalCoreIndicesNum * tilingData_->gatherSize);

    pipeIn->InitBuffer(indicesQue_, BUFFER_NUM, tilingData_->indicesUbSize);
    pipeIn->InitBuffer(yQue_, BUFFER_NUM, tilingData_->xUbSize);
    pipeIn->InitBuffer(helpTBuf_, T_BUFFER_SIZE);
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void GatherNdMixKernel<T1, T2, T3>::GenGatherIndex()
{
    LocalTensor<int32_t> helpTensor = helpTBuf_.Get<int32_t>();    // 只用前256B
    __local_mem__ int32_t* helpAddr = (__local_mem__ int32_t*)helpTensor.GetPhyAddr();
    int32_t eleStride = tilingData_->rank;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> reg0;
        AscendC::MicroAPI::RegTensor<int32_t> reg1;

        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Arange(reg0, 0);
        AscendC::MicroAPI::Muls(reg1, reg0, eleStride, preg);

        AscendC::MicroAPI::DataCopy(helpAddr, reg1, preg);
    }
}

template <typename T1, typename T2, typename T3>
template <const bool NIS>
__aicore__ inline void GatherNdMixKernel<T1, T2, T3>::FixIndicesVf(
    LocalTensor<T2>& indicesTensor, LocalTensor<T3>& maxGatherShapeTensor, LocalTensor<T3>& strideShapeTensor, int32_t indicesNumPro, uint16_t rank, T3 aMergeAxisSize)
{
    __local_mem__ T2* indicesAddr = (__local_mem__ T2*)indicesTensor.GetPhyAddr();
    __local_mem__ T3* maxGatherShapeAddr = (__local_mem__ T3*)maxGatherShapeTensor.GetPhyAddr();
    __local_mem__ T3* strideShapeAddr = (__local_mem__ T3*)strideShapeTensor.GetPhyAddr();

    int32_t indicesStride = rank;
    uint16_t computeSizeT3 = platform::GetVRegSize() / sizeof(T3);
    uint16_t repeatimes = (indicesNumPro + computeSizeT3 - 1) / computeSizeT3;

    uint32_t repeatStride = computeSizeT3 * indicesStride;

    LocalTensor<uint32_t> helpTensor = helpTBuf_.Get<uint32_t>();   // uint32   0  8  16 ...    uint64  0  4  8 ...
    __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> indexStart;
        AscendC::MicroAPI::RegTensor<uint32_t> curIndex;

        AscendC::MicroAPI::RegTensor<T3> zeroConstReg;
        AscendC::MicroAPI::RegTensor<T3> upLimitConstReg;
        AscendC::MicroAPI::Duplicate(zeroConstReg, T3(0));

        AscendC::MicroAPI::RegTensor<T3> indicesResReg;
        AscendC::MicroAPI::RegTensor<T3> tmpReg;
        AscendC::MicroAPI::RegTensor<T3> curIndicesReg;
        AscendC::MicroAPI::MaskReg tmpMask;
        AscendC::MicroAPI::MaskReg pregAll = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t indicesMask = indicesNumPro;
        uint32_t indicesMaskU32 = indicesNumPro;
        
        AscendC::MicroAPI::DataCopy<uint32_t>(indexStart, helpAddr);

        for (uint16_t i = 0; i < repeatimes; i++) {
            uint32_t indicesOffset = i * repeatStride;
            AscendC::MicroAPI::Duplicate(indicesResReg, T3(0));

            AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<T3>(indicesMask);
            AscendC::MicroAPI::MaskReg pregT2 = GenT2Mask<T2, T3>(indicesMaskU32);
            AscendC::MicroAPI::MaskReg overstepMask = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALLF>();
            for (uint16_t k = 0; k < rank; k++) {
                AscendC::MicroAPI::Adds(curIndex, indexStart, (indicesOffset + k), pregAll);

                // T2 --> T3
                if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int32_t>::value) {
                    AscendC::MicroAPI::DataCopyGather(curIndicesReg, indicesAddr, curIndex, pregT2);
                } else if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int64_t>::value) {
                    AscendC::MicroAPI::RegTensor<T2, AscendC::MicroAPI::RegTraitNumTwo> curIndicesRegTwo;
                    AscendC::MicroAPI::DataCopyGather(curIndicesRegTwo, indicesAddr, curIndex, pregT2);
                    curIndicesReg = (AscendC::MicroAPI::RegTensor<T3>&)curIndicesRegTwo.reg[0];
                } else if constexpr (std::is_same<T3, int64_t>::value && std::is_same<T2, int64_t>::value) {
                    AscendC::MicroAPI::DataCopyGather(curIndicesReg, indicesAddr, curIndex, pregT2);
                }

                T3 maxGatherDimKSize = maxGatherShapeAddr[k];
                if constexpr (NIS) {
                    AscendC::MicroAPI::Compare<T3, CMPMODE::LT>(tmpMask, curIndicesReg, zeroConstReg, preg);
                    AscendC::MicroAPI::Adds(tmpReg, curIndicesReg, maxGatherDimKSize, tmpMask); // 补偿负索引场景
                    Copy<T3, AscendC::MicroAPI::MaskMergeMode::MERGING>(curIndicesReg, tmpReg, tmpMask);
                }

                AscendC::MicroAPI::Compare<T3, CMPMODE::LT>(tmpMask, curIndicesReg, zeroConstReg, preg); // 负越界
                AscendC::MicroAPI::MaskOr(overstepMask, overstepMask, tmpMask, preg);

                AscendC::MicroAPI::Duplicate(upLimitConstReg, T3(maxGatherDimKSize));
                AscendC::MicroAPI::Compare<T3, CMPMODE::GE>(tmpMask, curIndicesReg, upLimitConstReg, preg); // 正越界
                AscendC::MicroAPI::MaskOr(overstepMask, overstepMask, tmpMask, preg);

                T3 strideDimKSize = strideShapeAddr[k];
                AscendC::MicroAPI::Muls(curIndicesReg, curIndicesReg, strideDimKSize, preg);
                AscendC::MicroAPI::Add(indicesResReg, indicesResReg, curIndicesReg, preg);
            }

            AscendC::MicroAPI::Muls(indicesResReg, indicesResReg, aMergeAxisSize, preg);
            AscendC::MicroAPI::Duplicate(tmpReg, T3(-1), overstepMask); // 补偿越界
            Copy<T3, AscendC::MicroAPI::MaskMergeMode::MERGING>(indicesResReg, tmpReg, overstepMask);

            AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<T3>(i, computeSizeT3);
            AscendC::MicroAPI::DataCopy((__local_mem__ T3*)indicesAddr, indicesResReg, offset, preg);
        }
    }
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void GatherNdMixKernel<T1, T2, T3>::CopyInIndices(LocalTensor<T2>& dstTensor, int32_t indicesCount, int64_t indicesOffset, uint16_t rank)
{
    DataCopyPad(dstTensor, indicesGm_[indicesOffset * rank],
                {static_cast<uint16_t>(1), static_cast<uint32_t>(indicesCount * rank * sizeof(T2)), 0, 0, 0},
                {false, 0, 0, 0});
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void GatherNdMixKernel<T1, T2, T3>::CopyOutY(LocalTensor<T1>& yTensor, uint32_t yCount, int64_t yOffset)
{
    DataCopyPad(yGm_[yOffset], yTensor, {static_cast<uint16_t>(1), static_cast<uint32_t>(yCount * sizeof(T1)), 0, 0, 0});
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void GatherNdMixKernel<T1, T2, T3>::Process()
{
    if (blockIdx_ >= tilingData_->blockNum) {
        return;
    }

    GenGatherIndex();
    AscendC::LocalTensor<T3> helpTensor = helpTBuf_.Get<T3>();
    auto maxGatherShapeTensor = helpTensor[T_BUFFER_OFFSET_512 / sizeof(T3)];
    for (uint32_t i = 0; i < MAX_SUPPORT_DIM; i++) {
        maxGatherShapeTensor.SetValue(i, (T3)(tilingData_->xShape[i]));
    }

    auto strideShapeTensor = helpTensor[T_BUFFER_OFFSET_640 / sizeof(T3)];
    for (uint32_t i = 0; i < MAX_SUPPORT_DIM; i++) {
        strideShapeTensor.SetValue(i, (T3)(tilingData_->strideShape[i]));
    }
    DataSyncBarrier<MemDsbT::UB>();

    using conditionalType = typename std::conditional<std::is_same<T2, int64_t>::value && std::is_same<T3, int64_t>::value, uint64_t, uint32_t>::type;

    conditionalType shift = 0;
    conditionalType m = 0;
    conditionalType aMergeAxisSize = tilingData_->gatherSize;
    bool isLastAxisOne = true;
    if (aMergeAxisSize != 1) {
        isLastAxisOne = false;
        GetUintDivMagicAndShift(m, shift, aMergeAxisSize);
    }

    bool negativeIndexSupport = tilingData_->negativeIndexSupport == 1;
    uint16_t rank = tilingData_->rank;
    int64_t indicesBufEleNum = tilingData_->singleUbProNum;
    int64_t curCoreIndicesNum_ = (blockIdx_ + 1 == tilingData_->blockNum) ? tilingData_->tailCoreIndicesNum :
                                                                           tilingData_->normalCoreIndicesNum;
    int64_t ubRepeatimes = (curCoreIndicesNum_  + indicesBufEleNum - 1) / indicesBufEleNum;
    for (int64_t i = 0; i < ubRepeatimes; i++) {
        int32_t indicesNumPro = i == (ubRepeatimes - 1) ? curCoreIndicesNum_ - (ubRepeatimes - 1) * indicesBufEleNum : indicesBufEleNum;
        int64_t indicesNumOffset = i * indicesBufEleNum;
        LocalTensor<T2> indicesTensor = indicesQue_.AllocTensor<T2>();
        CopyInIndices(indicesTensor, indicesNumPro, indicesNumOffset, rank);
        indicesQue_.EnQue(indicesTensor);
        indicesQue_.DeQue<T2>();
        if (negativeIndexSupport) {
            FixIndicesVf<true>(indicesTensor, maxGatherShapeTensor, strideShapeTensor, indicesNumPro, rank, aMergeAxisSize);
        } else {
            FixIndicesVf<false>(indicesTensor, maxGatherShapeTensor, strideShapeTensor, indicesNumPro, rank, aMergeAxisSize);
        }

        LocalTensor<T1> yTensor = yQue_.AllocTensor<T1>();

        uint32_t yCount = indicesNumPro * aMergeAxisSize;
        if (isLastAxisOne) {
            Simt::VF_CALL<MixSimtWithLastAxisOne<T1, conditionalType, T3>>(Simt::Dim3(static_cast<uint32_t>(THREAD_NUMS)),
                                                    (__ubuf__ T1*)yTensor.GetPhyAddr(),
                                                    (__ubuf__ T3*)indicesTensor.GetPhyAddr(),
                                                    (__gm__ T1*)xGm_.GetPhyAddr(),
                                                    yCount);
        } else {
            Simt::VF_CALL<MixSimt<T1, conditionalType, T3>>(Simt::Dim3(static_cast<uint32_t>(THREAD_NUMS)),
                                                    (__ubuf__ T1*)yTensor.GetPhyAddr(),
                                                    (__ubuf__ T3*)indicesTensor.GetPhyAddr(),
                                                    (__gm__ T1*)xGm_.GetPhyAddr(),
                                                    yCount, aMergeAxisSize, shift, m);
        }

        indicesQue_.FreeTensor(indicesTensor);
        yQue_.EnQue(yTensor);
        yQue_.DeQue<T1>();

        int64_t yOffset = indicesNumOffset * aMergeAxisSize;
        CopyOutY(yTensor, yCount, yOffset);
        yQue_.FreeTensor(yTensor);
    }
}
}  // namespace GatherNd

#endif  // ASCENDC_GATHER_ND_GATHER_ND_MIX_H_