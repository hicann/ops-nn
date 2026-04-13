/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

constexpr AscendC::MicroAPI::CastTrait castTraitU32U16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING
};

template <typename X_T, typename IDS_T, typename CAST_T, uint32_t castType>
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
    template <typename VGatherIndexDType, typename VGatherIndexDTypeInt>
    __aicore__ inline void ProcessPerXGroup(
        __local_mem__ X_T* xLocalAddr, __local_mem__ X_T* resBufAddr, MicroAPI::MaskReg& maskRegUpdate,
        MicroAPI::RegTensor<VGatherIndexDTypeInt>& serReg, MicroAPI::AddrReg& addrReg, xAddParams& params);

private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<X_T> yGm_;
    AscendC::GlobalTensor<IDS_T> idsGm_;
    TQue<QuePosition::VECIN, DYN_SORT_DB_BUF> xQue_;
    TQue<QuePosition::VECIN, DYN_SORT_DB_BUF> idsQue_;
    TQue<QuePosition::VECOUT, 1> outQueueRes_;
    TBuf<QuePosition::VECCALC> noDupBuf_;
    TBuf<QuePosition::VECCALC> sortedIdxBuf_;
    TBuf<QuePosition::VECCALC> castKeyIdxBuf_;
    TBuf<QuePosition::VECCALC> sortedKeyBuf_;
    TBuf<QuePosition::VECCALC> sharedTmpBuf_;
    TPipe* pipe_ = nullptr;
    const UnsortedSegmentSumSimdDynSortTilingData* td_;
    static constexpr uint32_t vfLengthX_ = VF_SIZE / sizeof(X_T);
    static constexpr uint32_t shiftOffset_ = ONE_BLOCK_SIZE / sizeof(CAST_T);
    static constexpr uint32_t vfIndicesNum_ = VF_SIZE / sizeof(CAST_T);
};

template <typename X_T, typename IDS_T, typename CAST_T, uint32_t castType>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T, CAST_T, castType>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
{
    InitGm<X_T>(output, td_->outputOuterDim * td_->innerDim);

    xGm_.SetGlobalBuffer((__gm__ X_T*)(x));
    idsGm_.SetGlobalBuffer((__gm__ IDS_T*)(segmentIds));
    yGm_.SetGlobalBuffer((__gm__ X_T*)(output));

    pipe_->InitBuffer(xQue_, DYN_SORT_DB_BUF, td_->sortBaseS * td_->sortBaseA * sizeof(X_T));
    pipe_->InitBuffer(outQueueRes_, 1, td_->sortBaseS * td_->sortBaseA * sizeof(X_T));

    uint64_t idsAlignB32 = Ops::Base::CeilAlign(static_cast<uint64_t>(td_->sortBaseS * sizeof(uint32_t)), ONE_BLOCK_SIZE);
    uint64_t idsAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(td_->sortBaseS * sizeof(IDS_T)), ONE_BLOCK_SIZE);
    uint64_t idsAlignCast = Ops::Base::CeilAlign(static_cast<uint64_t>(td_->sortBaseS * sizeof(CAST_T)), ONE_BLOCK_SIZE);

    pipe_->InitBuffer(idsQue_, DYN_SORT_DB_BUF, idsAlign);
    pipe_->InitBuffer(noDupBuf_, idsAlignB32 + SORT_PADDING);
    pipe_->InitBuffer(sortedIdxBuf_, idsAlignB32);
    pipe_->InitBuffer(sharedTmpBuf_, Ops::Base::CeilAlign(static_cast<uint64_t>(td_->sortSharedBufSize), ONE_BLOCK_SIZE));
    if constexpr (castType == CAST_NO){
        pipe_->InitBuffer(sortedKeyBuf_, idsAlign + SORT_PADDING);
    } else {
        pipe_->InitBuffer(sortedKeyBuf_, idsAlignCast + SORT_PADDING);
        pipe_->InitBuffer(castKeyIdxBuf_, idsAlignCast);
    }
}

template <typename X_T, typename IDS_T, typename CAST_T, uint32_t castType>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T, CAST_T, castType>::ProcessIndices(
    uint64_t blockOffsetIdx, uint64_t sLoop, uint32_t rows, int64_t& arNum)
{
    LocalTensor<IDS_T> idsLocal = idsQue_.AllocTensor<IDS_T>();
    CopyIn(idsLocal, idsGm_, blockOffsetIdx + sLoop * td_->sortBaseS, 1, rows, 0);
    idsQue_.EnQue<IDS_T>(idsLocal);

    idsLocal = idsQue_.DeQue<IDS_T>();
    LocalTensor<uint32_t> sortedIdxLocal = sortedIdxBuf_.Get<uint32_t>();
    LocalTensor<int32_t> noDupRes = noDupBuf_.Get<int32_t>();
    LocalTensor<CAST_T> sortedKeyLocal = sortedKeyBuf_.Get<CAST_T>();
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTmpBuf_.Get<uint8_t>();

    if constexpr (castType == CAST_NO) {
        Duplicate(sortedKeyLocal, static_cast<CAST_T>(-1), static_cast<uint32_t>(shiftOffset_ * HELP_FRE + rows));
        LocalTensor<CAST_T> sortedDstLocal = sortedKeyLocal[shiftOffset_];
        static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};
        AscendC::Sort<CAST_T, false, sortConfig>(sortedDstLocal, sortedIdxLocal, idsLocal, sharedTmpBuffer, rows);
        arNum = ComputeUniqueIdNum<CAST_T>(sortedKeyLocal, noDupRes, rows);
    } else {
        LocalTensor<CAST_T> indicesCastLocal = castKeyIdxBuf_.Get<CAST_T>();
        IndicesSortCast<IDS_T, CAST_T, castType>(idsLocal, indicesCastLocal, noDupRes, rows);
        Duplicate(sortedKeyLocal, static_cast<CAST_T>(-1), static_cast<uint32_t>(shiftOffset_ * HELP_FRE + rows));
        LocalTensor<CAST_T> sortedDstLocal = sortedKeyLocal[shiftOffset_];
        static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};
        AscendC::Sort<CAST_T, false, sortConfig>(sortedDstLocal, sortedIdxLocal, indicesCastLocal, sharedTmpBuffer, rows);
        Duplicate(noDupRes, 0, rows);
        arNum = ComputeUniqueIdNum<CAST_T>(sortedKeyLocal, noDupRes, rows);
    }

    UniqueStat(noDupRes, arNum);
    idsQue_.FreeTensor(idsLocal);
}

template <typename X_T, typename IDS_T, typename CAST_T, uint32_t castType>
template <typename VGatherIndexDType, typename VGatherIndexDTypeInt>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T, CAST_T, castType>::ProcessPerXGroup(
    __local_mem__ X_T* xLocalAddr, __local_mem__ X_T* resBufAddr, MicroAPI::MaskReg& maskRegUpdate,
    MicroAPI::RegTensor<VGatherIndexDTypeInt>& serReg, MicroAPI::AddrReg& addrReg, xAddParams& params)
{
    if constexpr (IsSameType<VGatherIndexDType, uint16_t>::value) {
        MicroAPI::RegTensor<VGatherIndexDType> initIdsReg, idsReg;
        MicroAPI::RegTensor<VGatherIndexDType> initIdsReg1;
        MicroAPI::RegTensor<uint32_t> tmReg;
        MicroAPI::RegTensor<X_T> gatherOut;
        MicroAPI::RegTensor<X_T> outReg;
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<X_T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate(outReg, (X_T)0, maskReg);
        for (uint16_t pIdx = 0; pIdx < params.segCount; ++pIdx) {
            MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(
                tmReg, params.sortedIdxAddr + (params.outGmIndex + pIdx));

            MicroAPI::Cast<uint16_t, uint32_t, castTraitU32U16>((MicroAPI::RegTensor<uint16_t>&)tmReg, tmReg, maskRegU32);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::HIGHEST>(
                (MicroAPI::RegTensor<uint16_t>&)initIdsReg,
                (MicroAPI::RegTensor<uint32_t>&)tmReg
            );
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint16_t>&)initIdsReg1,
                (MicroAPI::RegTensor<uint32_t>&)tmReg
            );
            MicroAPI::Add(initIdsReg, initIdsReg, initIdsReg1, maskReg);
            MicroAPI::Muls(idsReg, initIdsReg, params.xPerRowNum, maskRegUpdate);
            MicroAPI::Add(idsReg, (MicroAPI::RegTensor<VGatherIndexDType>&)serReg, idsReg, maskRegUpdate);
            MicroAPI::DataCopyGather(gatherOut, xLocalAddr, idsReg, maskRegUpdate);
            MicroAPI::Add(outReg, outReg, gatherOut, maskRegUpdate);
        }

        MicroAPI::DataCopy(resBufAddr, outReg, addrReg, maskRegUpdate);
    } else {
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
}

template <typename X_T, typename IDS_T, typename CAST_T, uint32_t castType>
template <typename VGatherIndexDType>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T, CAST_T, castType>::ComputeXSum(uint32_t cols, uint32_t colsAlign, int64_t arNum)
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
    using VGatherIndexDTypeInt = std::conditional_t<std::is_same_v<VGatherIndexDType, uint16_t>, int16_t, int32_t>;

    __VEC_SCOPE__
    {
        for (uint16_t i = 0; i < (uint16_t)arNum; ++i) {
            MicroAPI::RegTensor<VGatherIndexDTypeInt> serReg;
            MicroAPI::RegTensor<VGatherIndexDTypeInt> serRegBase;

            MicroAPI::Arange(serRegBase, (VGatherIndexDTypeInt)sclar0);

            params.segCount = static_cast<uint16_t>(noDupRes(i));

            uint32_t colCount = cols;
            resBufAddr = resBufBaseAddr + i * colsAlign;
            for (uint16_t j = 0; j < (uint16_t)loopPerRow; ++j) {
                MicroAPI::MaskReg maskRegUpdate = MicroAPI::UpdateMask<VGatherIndexDType>(colCount);
                auto addrReg = MicroAPI::CreateAddrReg<X_T>(j, static_cast<uint16_t>(vfLengthX_));
                MicroAPI::Adds(serReg, serRegBase, (VGatherIndexDTypeInt)(vfLengthX_ * j), maskRegUpdate);
                ProcessPerXGroup<VGatherIndexDType, VGatherIndexDTypeInt>(xLocalAddr, resBufAddr, maskRegUpdate, serReg, addrReg, params);
            }
            params.outGmIndex = params.outGmIndex + params.segCount;
        }
    }
    xQue_.FreeTensor<X_T>(xLocal);
    outQueueRes_.EnQue(resLocal);
}

template <typename X_T, typename IDS_T, typename CAST_T, uint32_t castType>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T, CAST_T, castType>::CopyResToGm(
    uint32_t cols, uint32_t colsAlign, uint64_t ubOffset, int64_t& arNum)
{
    LocalTensor<X_T> resLocal = outQueueRes_.DeQue<X_T>();
    LocalTensor<CAST_T> sortedKeyLocal = sortedKeyBuf_.Get<CAST_T>();
    LocalTensor<int32_t> noDupRes = noDupBuf_.Get<int32_t>();

    int32_t tmpIndex = shiftOffset_;

    SetAtomicAdd<X_T>();
    for (uint32_t i = 0; i < static_cast<uint32_t>(arNum); i++) {
        uint64_t dstIdx = sortedKeyLocal(tmpIndex);
        if (dstIdx < 0 || dstIdx >= td_->outputOuterDim) {
            continue;
        }

        uint64_t offset = dstIdx * td_->innerDim + ubOffset;
        tmpIndex = tmpIndex + noDupRes(i);

        CopyOut(yGm_, resLocal[i * colsAlign], offset, 1, cols);
    }
    SetAtomicNone();
    outQueueRes_.FreeTensor(resLocal);
}

template <typename X_T, typename IDS_T, typename CAST_T, uint32_t castType>
template <typename VGatherIndexDType>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T, CAST_T, castType>::Compute()
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

template <typename X_T, typename IDS_T, typename CAST_T, uint32_t castType>
__aicore__ inline void USSKernelSimdDynSort<X_T, IDS_T, CAST_T, castType>::Process()
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