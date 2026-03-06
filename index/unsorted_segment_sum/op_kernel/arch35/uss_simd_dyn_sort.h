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
 * \file uss_simd_dyn_sort.h
 * \brief uss_simd_dyn_sort
 */
#ifndef ASCENDC_USS_SIMD_DYN_SORT_H_
#define ASCENDC_USS_SIMD_DYN_SORT_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "unsorted_segment_base.h"
#include "kernel_tiling/kernel_tiling.h"

namespace UnsortedSegmentSum {
using namespace AscendC;
constexpr uint32_t DYN_SORT_DB_BUF = 1;
constexpr uint32_t SORT_PADDING = 64;
constexpr uint32_t HELP_FRE = 2;

template <typename X_T, typename IDS_T>
class USSKernelSimdDynSort
{
public:
    __aicore__ inline USSKernelSimdDynSort(const UnsortedSegmentSumSimdDynSortTilingData* tiling, TPipe* pipe)
        : td_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output);
    __aicore__ inline void ProcessIndices(uint64_t blockOffsetIdx, uint64_t sLoop, uint32_t rows, int64_t& arNum);
    template <typename VGatherIndexDType>
    __aicore__ inline void ComputeXSum(uint32_t cols, uint32_t colsAlign, int64_t arNum);
    __aicore__ inline void CopyResToGm(uint32_t cols, uint32_t colsAlign, uint64_t ubOffset, int64_t& arNum);
    template <typename VGatherIndexDType>
    __aicore__ inline void Compute();
    __aicore__ inline void Process();
    template <typename VGatherIndexDType>
    __aicore__ inline void ProcessPerXGroup(
        __local_mem__ X_T* xLocalAddr, __local_mem__ X_T* resBufAddr, MicroAPI::MaskReg& maskRegUpdate,
        MicroAPI::RegTensor<int32_t>& serReg, MicroAPI::AddrReg& addrReg, xAddParams& params);

private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<X_T> yGm_;
    AscendC::GlobalTensor<IDS_T> idsGm_;
    TQue<QuePosition::VECIN, DYN_SORT_DB_BUF> xQue_;
    TQue<QuePosition::VECIN, DYN_SORT_DB_BUF> idsQue_;
    TQue<QuePosition::VECOUT, 1> outQueueRes_;
    TBuf<QuePosition::VECCALC> noDupBuf_;
    TBuf<QuePosition::VECCALC> sortedIdxBuf_;
    TBuf<QuePosition::VECCALC> sortedKeyBuf_;
    TBuf<QuePosition::VECCALC> sharedTmpBuf_;
    TPipe* pipe_ = nullptr;
    const UnsortedSegmentSumSimdDynSortTilingData* td_;
    static constexpr uint32_t vfLengthX_ = VF_SIZE / sizeof(X_T);
    static constexpr uint32_t shiftOffset_ = ONE_BLOCK_SIZE / sizeof(IDS_T);
    static constexpr uint32_t vfIndicesNum_ = VF_SIZE / sizeof(IDS_T);
};

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
{
    InitGm<X_T>(output, td_->outputOuterDim * td_->innerDim);

    xGm_.SetGlobalBuffer((__gm__ X_T*)(x));
    idsGm_.SetGlobalBuffer((__gm__ IDS_T*)(segmentIds));
    yGm_.SetGlobalBuffer((__gm__ X_T*)(output));

    pipe_->InitBuffer(xQue_, DYN_SORT_DB_BUF, td_->sortBaseS * td_->sortBaseA * sizeof(X_T));
    pipe_->InitBuffer(outQueueRes_, 1, td_->sortBaseS * td_->sortBaseA * sizeof(X_T));

    uint64_t idsAlignB32 = Ops::Base::CeilAlign(static_cast<uint64_t>(td_->sortBaseS * sizeof(uint32_t)), ONE_BLOCK_SIZE);
    uint64_t idsAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(td_->sortBaseS * sizeof(IDS_T)), ONE_BLOCK_SIZE);

    pipe_->InitBuffer(idsQue_, DYN_SORT_DB_BUF, idsAlign);
    pipe_->InitBuffer(noDupBuf_, idsAlignB32 + SORT_PADDING);
    pipe_->InitBuffer(sortedKeyBuf_, idsAlign + SORT_PADDING);
    pipe_->InitBuffer(sortedIdxBuf_, idsAlignB32);
    pipe_->InitBuffer(sharedTmpBuf_, Ops::Base::CeilAlign(static_cast<uint64_t>(td_->sortSharedBufSize), ONE_BLOCK_SIZE));
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T>::ProcessIndices(
    uint64_t blockOffsetIdx, uint64_t sLoop, uint32_t rows, int64_t& arNum)
{
    LocalTensor<IDS_T> idsLocal = idsQue_.AllocTensor<IDS_T>();
    CopyIn(idsLocal, idsGm_, blockOffsetIdx + sLoop * td_->sortBaseS, 1, rows, 0);
    idsQue_.EnQue<IDS_T>(idsLocal);

    idsLocal = idsQue_.DeQue<IDS_T>();
    LocalTensor<uint32_t> sortedIdxLocal = sortedIdxBuf_.Get<uint32_t>();
    LocalTensor<IDS_T> sortedKeyLocal = sortedKeyBuf_.Get<IDS_T>();
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTmpBuf_.Get<uint8_t>();

    Duplicate(sortedKeyLocal, static_cast<IDS_T>(-1), static_cast<uint32_t>(shiftOffset_ * HELP_FRE + rows));
    LocalTensor<IDS_T> sortedDstLocal = sortedKeyLocal[shiftOffset_];
    static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};
    AscendC::Sort<IDS_T, false, sortConfig>(sortedDstLocal, sortedIdxLocal, idsLocal, sharedTmpBuffer, rows);

    LocalTensor<int32_t> noDupRes = noDupBuf_.Get<int32_t>();
    Duplicate(noDupRes, 0, rows);
    UniqueGetElm(sortedKeyLocal, noDupRes, rows, shiftOffset_, vfIndicesNum_, arNum);
    arNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
    UniqueStat(noDupRes, arNum);
    idsQue_.FreeTensor(idsLocal);
}

template <typename X_T, typename IDS_T>
template <typename VGatherIndexDType>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T>::ProcessPerXGroup(
    __local_mem__ X_T* xLocalAddr, __local_mem__ X_T* resBufAddr, MicroAPI::MaskReg& maskRegUpdate,
    MicroAPI::RegTensor<int32_t>& serReg, MicroAPI::AddrReg& addrReg, xAddParams& params)
{
    MicroAPI::RegTensor<VGatherIndexDType> initIdsReg, idsReg;
    MicroAPI::RegTensor<X_T> gatherOut;
    MicroAPI::RegTensor<X_T> outReg;
    MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<X_T, MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate(outReg, (X_T)0, maskReg);
    for (uint16_t pIdx = 0; pIdx < params.segCount; ++pIdx) {
        MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(
            (MicroAPI::RegTensor<uint32_t>&)initIdsReg, params.sortedIdxAddr + (params.outGmIndex + pIdx));
        MicroAPI::Muls(idsReg, initIdsReg, params.xPerRowNum, maskRegUpdate);
        MicroAPI::Add(idsReg, (MicroAPI::RegTensor<VGatherIndexDType>&)serReg, idsReg, maskRegUpdate);
        MicroAPI::DataCopyGather(gatherOut, xLocalAddr, idsReg, maskRegUpdate);
        MicroAPI::Add(outReg, outReg, gatherOut, maskRegUpdate);
    }

    MicroAPI::DataCopy(resBufAddr, outReg, addrReg, maskRegUpdate);
}

template <typename X_T, typename IDS_T>
template <typename VGatherIndexDType>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T>::ComputeXSum(uint32_t cols, uint32_t colsAlign, int64_t arNum)
{
    LocalTensor<uint32_t> sortedIdxLocal = sortedIdxBuf_.template Get<uint32_t>();
    LocalTensor<int32_t> noDupRes = noDupBuf_.template Get<int32_t>();
    LocalTensor<X_T> resLocal = outQueueRes_.AllocTensor<X_T>();
    LocalTensor<X_T> xLocal = xQue_.DeQue<X_T>();
    __local_mem__ X_T* xLocalAddr = (__ubuf__ X_T*)xLocal.GetPhyAddr();
    __local_mem__ X_T* resBufAddr = (__ubuf__ X_T*)resLocal.GetPhyAddr();
    __local_mem__ X_T* resBufBaseAddr = resBufAddr;

    int32_t sclar0 = 0;
    uint32_t loopPerRow = (cols + vfLengthX_ - 1) / vfLengthX_;

    xAddParams params;
    params.xPerRowNum = colsAlign;
    params.outGmIndex = 0;
    params.sortedIdxAddr = (__ubuf__ uint32_t*)sortedIdxLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        for (uint16_t i = 0; i < (uint16_t)arNum; ++i) {
            MicroAPI::RegTensor<int32_t> serReg;
            MicroAPI::RegTensor<int32_t> serRegBase;

            MicroAPI::Arange(serRegBase, sclar0);

            params.segCount = static_cast<uint16_t>(noDupRes(i));

            uint32_t colCount = cols;
            resBufAddr = resBufBaseAddr + i * colsAlign;
            for (uint16_t j = 0; j < (uint16_t)loopPerRow; ++j) {
                MicroAPI::MaskReg maskRegUpdate = MicroAPI::UpdateMask<uint32_t>(colCount);
                auto addrReg = MicroAPI::CreateAddrReg<X_T>(j, static_cast<uint16_t>(vfLengthX_));
                MicroAPI::Adds(serReg, serRegBase, VF_B32 * j, maskRegUpdate);
                ProcessPerXGroup<VGatherIndexDType>(xLocalAddr, resBufAddr, maskRegUpdate, serReg, addrReg, params);
            }
            params.outGmIndex = params.outGmIndex + params.segCount;
        }
    }
    xQue_.FreeTensor<X_T>(xLocal);
    outQueueRes_.EnQue(resLocal);
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T>::CopyResToGm(
    uint32_t cols, uint32_t colsAlign, uint64_t ubOffset, int64_t& arNum)
{
    LocalTensor<X_T> resLocal = outQueueRes_.DeQue<X_T>();
    LocalTensor<IDS_T> sortedKeyLocal = sortedKeyBuf_.Get<IDS_T>();
    LocalTensor<int32_t> noDupRes = noDupBuf_.Get<int32_t>();

    int32_t tmpIndex = shiftOffset_;

    SetAtomicAdd<X_T>();
    for (uint32_t i = 0; i < static_cast<uint32_t>(arNum); i++) {
        uint64_t dstIdx = sortedKeyLocal(tmpIndex);
        uint64_t offset = dstIdx * td_->innerDim + ubOffset;
        tmpIndex = tmpIndex + noDupRes(i);

        if (dstIdx < 0 || dstIdx >= td_->outputOuterDim) {
            continue;
        }

        CopyOut(yGm_, resLocal[i * colsAlign], offset, 1, cols);
    }
    SetAtomicNone();
    outQueueRes_.FreeTensor(resLocal);
}

template <typename X_T, typename IDS_T>
template <typename VGatherIndexDType>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T>::Compute()
{
    uint64_t sIdx = GetBlockIdx() / td_->aTileNum;
    uint64_t aIdx = GetBlockIdx() % td_->aTileNum;
    uint64_t curCoreRows = sIdx != (td_->sTileNum - 1) ? td_->normBlockS : td_->tailBlockS;
    uint64_t curCoreCols = aIdx != (td_->aTileNum - 1) ? td_->normBlockA : td_->tailBlockA;

    uint64_t blockOffsetIdx = sIdx * td_->normBlockS;
    uint64_t blockOffsetX = sIdx * td_->normBlockS * td_->innerDim + aIdx * td_->normBlockA;

    uint64_t aLoopNum = Ops::Base::CeilDiv(curCoreCols, td_->sortBaseA);
    uint64_t sLoopNum = Ops::Base::CeilDiv(curCoreRows, td_->sortBaseS);

    for (uint64_t sLoop = 0; sLoop < sLoopNum; sLoop++) {
        uint32_t rows = (sLoop == sLoopNum - 1) ? (curCoreRows - sLoop * td_->sortBaseS) : td_->sortBaseS;
        int64_t arNum = 0;
        ProcessIndices(blockOffsetIdx, sLoop, rows, arNum);

        for (uint64_t aLoop = 0; aLoop < aLoopNum; aLoop++) {
            uint32_t cols = (aLoop == aLoopNum - 1) ? (curCoreCols - aLoop * td_->sortBaseA) : td_->sortBaseA;
            uint32_t colsAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(cols * sizeof(X_T)), ONE_BLOCK_SIZE) / sizeof(X_T);

            LocalTensor<X_T> xLocal = xQue_.AllocTensor<X_T>();
            uint64_t offset = blockOffsetX + sLoop * td_->sortBaseS * td_->innerDim + aLoop * td_->sortBaseA;
            CopyIn(xLocal, xGm_, offset, rows, cols, td_->innerDim - cols);
            xQue_.EnQue<X_T>(xLocal);

            ComputeXSum<VGatherIndexDType>(cols, colsAlign, arNum);
            uint64_t ubOffset = aIdx * td_->normBlockA + aLoop * td_->sortBaseA;
            CopyResToGm(cols, colsAlign, ubOffset, arNum);
        }
    }
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }

    if constexpr (IsSameType<X_T, half>::value || IsSameType<X_T, bfloat16_t>::value) {
        Compute<uint16_t>();
    } else if constexpr (
        IsSameType<X_T, float>::value || IsSameType<X_T, uint32_t>::value || IsSameType<X_T, int32_t>::value) {
        Compute<uint32_t>();
    } else {
        Compute<uint64_t>();
    }
}
} // namespace UnsortedSegmentSum

#endif