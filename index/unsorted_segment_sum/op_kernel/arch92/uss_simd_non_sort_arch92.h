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
 * \file uss_simd_non_sort.h
 * \brief uss_simd_non_sort
 */
#ifndef ASCENDC_USS_SIMD_NON_SORT_ARCH92_H_
#define ASCENDC_USS_SIMD_NON_SORT_ARCH92_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "unsorted_segment_base_arch92.h"

namespace UnsortedSegmentSum {
using namespace AscendC;
constexpr uint32_t NON_SORT_DB_BUF = 2;
constexpr uint64_t NON_SORT_OFFSET_TYPESIZE = 8; // int64_t offset for ScatterDataCopy

template <typename IDS_T>
__simd_callee__ inline void ComputeScatterOffsetsNonSort(__local_mem__ int64_t* offsetAddr,
                                                         __local_mem__ IDS_T* idsAddr, uint32_t rows,
                                                         uint64_t outputOuterDim, uint64_t innerDim, uint64_t aIdx,
                                                         uint64_t normBlockA, uint64_t aLoop, uint64_t baseA,
                                                         uint64_t valueTypeBytes)
{
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int64_t);
    uint32_t remaining = rows;
    uint64_t colOffsetEle = aIdx * normBlockA + aLoop * baseA;
    int64_t colOffsetBytes = static_cast<int64_t>(colOffsetEle * valueTypeBytes);
    int64_t rowStrideBytes = static_cast<int64_t>(innerDim * valueTypeBytes);
    int64_t maxIdx = static_cast<int64_t>(outputOuterDim);
    uint16_t loopCnt = static_cast<uint16_t>((rows + vfLen - 1) / vfLen);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<IDS_T> invalidReg;
        MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> offsetReg;
        MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> tmpReg;
        MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> idxI64Reg;
        MicroAPI::RegTensor<int32_t> castReg32;
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg cmpMask;
        MicroAPI::MaskReg pregLoopI64;
        static constexpr MicroAPI::CastTrait castTraitB322B64 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        MicroAPI::Duplicate(invalidReg, static_cast<IDS_T>(-1));

        for (uint16_t i = 0; i < loopCnt; i++) {
            uint32_t remainingForIds = remaining;
            uint32_t remainingForI64 = remaining;
            pregLoop = MicroAPI::UpdateMask<IDS_T>(remainingForIds);
            pregLoopI64 = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(remainingForI64);

            if constexpr (IsSameType<IDS_T, int32_t>::value) {
                MicroAPI::RegTensor<int32_t> idsReg;
                MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(idsReg, idsAddr + i * vfLen);
                MicroAPI::Compares<int32_t, CMPMODE::GE>(cmpMask, idsReg, maxIdx, pregLoop);
                MicroAPI::Select<int32_t>(idsReg, invalidReg, idsReg, cmpMask);
                MicroAPI::Cast<int64_t, int32_t, castTraitB322B64>(idxI64Reg, idsReg, pregLoopI64);
            } else {
                MicroAPI::DataCopy<int64_t>(idxI64Reg, reinterpret_cast<__ubuf__ int64_t*>(idsAddr + i * vfLen));
                MicroAPI::Compares<int64_t, CMPMODE::GE>(cmpMask, idxI64Reg, maxIdx, pregLoopI64);
                MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> invalidI64Reg;
                MicroAPI::Duplicate<int64_t>(invalidI64Reg, static_cast<int64_t>(-1));
                MicroAPI::Select<int64_t>(idxI64Reg, invalidI64Reg, idxI64Reg, cmpMask);
            }
            MicroAPI::Muls(tmpReg, idxI64Reg, rowStrideBytes, pregLoopI64);
            MicroAPI::Adds(offsetReg, tmpReg, colOffsetBytes, pregLoopI64);
            MicroAPI::DataCopy(offsetAddr + i * vfLen, offsetReg, pregLoopI64);
            remaining = remaining > vfLen ? remaining - vfLen : 0;
        }
    }
}

template <typename X_T, typename IDS_T>
class USSKernelSimdNonSort {
public:
    __aicore__ inline USSKernelSimdNonSort(const UnsortedSegmentSumSimdNonSortTilingData* tiling, TPipe* pipe)
        : td_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output);
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<X_T> yGm_;
    AscendC::GlobalTensor<IDS_T> idsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, NON_SORT_DB_BUF> xQue_;
    TQue<QuePosition::VECIN, NON_SORT_DB_BUF> idsQue_;
    TBuf<QuePosition::VECCALC> offsetBuf_; // int64_t offsets for ScatterDataCopy
    TPipe* pipe_ = nullptr;
    const UnsortedSegmentSumSimdNonSortTilingData* td_;
};

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelSimdNonSort<X_T, IDS_T>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
{
    InitGm<X_T>(output, td_->outputOuterDim * td_->innerDim);

    xGm_.SetGlobalBuffer((__gm__ X_T*)(x));
    idsGm_.SetGlobalBuffer((__gm__ IDS_T*)(segmentIds));
    yGm_.SetGlobalBuffer((__gm__ X_T*)(output));

    pipe_->InitBuffer(xQue_, NON_SORT_DB_BUF, td_->baseS * td_->baseA * sizeof(X_T));
    pipe_->InitBuffer(idsQue_, NON_SORT_DB_BUF,
                      Ops::Base::CeilAlign(static_cast<uint64_t>(td_->baseS * sizeof(IDS_T)), ONE_BLOCK_SIZE));
    if (td_->useScatterDMA != 0) {
        uint64_t offsetBufBytes = Ops::Base::CeilAlign(td_->baseS * NON_SORT_OFFSET_TYPESIZE, ONE_BLOCK_SIZE);
        pipe_->InitBuffer(offsetBuf_, offsetBufBytes);
    }
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelSimdNonSort<X_T, IDS_T>::Process()
{
    if (GetBlockIdx() >= td_->usedCoreNum) {
        return;
    }

    uint64_t sIdx = GetBlockIdx() / td_->aTileNum;
    uint64_t aIdx = GetBlockIdx() % td_->aTileNum;
    uint64_t curCoreRows = sIdx != (td_->sTileNum - 1) ? td_->normBlockS : td_->tailBlockS;
    uint64_t curCoreCols = aIdx != (td_->aTileNum - 1) ? td_->normBlockA : td_->tailBlockA;

    uint64_t blockOffsetIdx = sIdx * td_->normBlockS;
    uint64_t blockOffsetX = sIdx * td_->normBlockS * td_->innerDim + aIdx * td_->normBlockA;

    uint64_t aLoopNum = Ops::Base::CeilDiv(curCoreCols, td_->baseA);
    uint64_t sLoopNum = Ops::Base::CeilDiv(curCoreRows, td_->baseS);

    bool useScatterDMA = (td_->useScatterDMA != 0);

    for (uint64_t sLoop = 0; sLoop < sLoopNum; sLoop++) {
        uint64_t rows = (sLoop == sLoopNum - 1) ? (curCoreRows - sLoop * td_->baseS) : td_->baseS;

        LocalTensor<IDS_T> idsLocal = idsQue_.AllocTensor<IDS_T>();
        CopyIn(idsLocal, idsGm_, blockOffsetIdx + sLoop * td_->baseS, 1, rows);
        idsQue_.EnQue<IDS_T>(idsLocal);

        idsLocal = idsQue_.DeQue<IDS_T>();

        for (uint64_t aLoop = 0; aLoop < aLoopNum; aLoop++) {
            uint64_t cols = (aLoop == aLoopNum - 1) ? (curCoreCols - aLoop * td_->baseA) : td_->baseA;
            uint64_t colsAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(cols * sizeof(X_T)), ONE_BLOCK_SIZE) /
                                 sizeof(X_T);

            LocalTensor<X_T> xLocal = xQue_.AllocTensor<X_T>();
            uint64_t offset = blockOffsetX + sLoop * td_->baseS * td_->innerDim + aLoop * td_->baseA;
            CopyIn(xLocal, xGm_, offset, rows, cols, td_->innerDim - cols);
            xQue_.EnQue<X_T>(xLocal);

            xLocal = xQue_.DeQue<X_T>();
            if (useScatterDMA) {
                LocalTensor<int64_t> offsetLocal = offsetBuf_.Get<int64_t>();
                __local_mem__ IDS_T* idsAddr = (__local_mem__ IDS_T*)idsLocal.GetPhyAddr();
                __local_mem__ int64_t* offsetAddr = (__local_mem__ int64_t*)offsetLocal.GetPhyAddr();
                ComputeScatterOffsetsNonSort<IDS_T>(offsetAddr, idsAddr, static_cast<uint32_t>(rows),
                                                    td_->outputOuterDim, td_->innerDim, aIdx, td_->normBlockA, aLoop,
                                                    td_->baseA, sizeof(X_T));

                event_t eventV2MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventV2MTE3);
                WaitFlag<HardEvent::V_MTE3>(eventV2MTE3);

                SetAtomicAdd<X_T>();
                ScatterParams scatterParams;
                scatterParams.offsetNum = static_cast<uint16_t>(rows);
                scatterParams.burstLen = cols * sizeof(X_T);
                scatterParams.srcStride = colsAlign * sizeof(X_T);
                ScatterDataCopy(yGm_, xLocal, offsetLocal, scatterParams);
                SetAtomicNone();
            } else {
                event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                SetAtomicAdd<X_T>();
                for (uint64_t i = 0; i < rows; i++) {
                    uint64_t dstIdx = idsLocal.GetValue(i);
                    if (dstIdx < 0 || dstIdx >= td_->outputOuterDim) {
                        continue;
                    }
                    uint64_t dstOffset = dstIdx * td_->innerDim + aIdx * td_->normBlockA + aLoop * td_->baseA;
                    CopyOut(yGm_, xLocal[i * colsAlign], dstOffset, 1, cols);
                }
                SetAtomicNone();
            }
            xQue_.FreeTensor(xLocal);
        }
        idsQue_.FreeTensor(idsLocal);
    }
}
} // namespace UnsortedSegmentSum

#endif
