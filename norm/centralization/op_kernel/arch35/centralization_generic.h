/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_GENERIC_H_
#define OPS_NORM_CENTRALIZATION_GENERIC_H_
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "centralization_tiling_data.h"

namespace Centralization {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;
using NormCommon::V_LENGTH;
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

template <typename T>
class Generic {
public:
    __aicore__ inline Generic(TPipe* pipe, const CentralizationTilingData* tiling) : pipe_(pipe), tiling_(tiling) {}
    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* workspace)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        const bool smallContiguous = tiling_->smallContiguous != 0;
        const bool largeContiguousSmallInner = tiling_->largeContiguousSmallInner != 0;
        const uint32_t queueDepth = (smallContiguous || largeContiguousSmallInner) ? 1U : 2U;
        pipe_->InitBuffer(inQueue_, queueDepth, tiling_->genericInputBufferBytes);
        pipe_->InitBuffer(outQueue_, queueDepth, tiling_->genericOutputBufferBytes);
        pipe_->InitBuffer(sumBuf_, tiling_->genericSumBufferBytes);
        eventMte2ToS_ = pipe_->FetchEventID(HardEvent::MTE2_S);
        eventVToS_ = pipe_->FetchEventID(HardEvent::V_S);
        eventSToMte3_ = pipe_->FetchEventID(HardEvent::S_MTE3);
        eventMte3ToMte2_ = pipe_->FetchEventID(HardEvent::MTE3_MTE2);
    }

    __aicore__ inline int64_t GroupBase(int64_t group) const
    {
        int64_t base = 0;
        int64_t value = group;
        for (int64_t i = 0; i < tiling_->keepRank; ++i) {
            int64_t coord = value / tiling_->keepIndexStrides[i];
            value %= tiling_->keepIndexStrides[i];
            base += coord * tiling_->keepStrides[i];
        }
        return base;
    }

    __aicore__ inline int64_t ReduceOffset(int64_t reduced) const
    {
        int64_t offset = 0;
        int64_t value = reduced;
        for (int64_t i = 0; i < tiling_->reduceRank; ++i) {
            int64_t coord = value / tiling_->reduceIndexStrides[i];
            value %= tiling_->reduceIndexStrides[i];
            offset += coord * tiling_->reduceStrides[i];
        }
        return offset;
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        if (tiling_->smallContiguous != 0) {
            ProcessSmallContiguous(core, cores);
            return;
        }
        if (tiling_->largeContiguousSmallInner != 0) {
            ProcessLargeContiguousSmallInner(core, cores);
            return;
        }
        if (tiling_->contiguousGeneric != 0) {
            ProcessContiguous(core, cores);
            return;
        }
        if (tiling_->irregularVectorizable != 0) {
            ProcessIrregularVector(core, cores);
            return;
        }
        if (tiling_->largePath != 0) {
            ProcessLarge(core, cores);
            return;
        }
        ProcessOutputBlocks(core, cores, false);
    }

private:
    static constexpr int64_t kLargeChunkElements = 256;
    static constexpr uint16_t kMaxAccSlots = 16;
    static constexpr int64_t kMaxIrregularScalarKeepTile = 16;

    __aicore__ inline int64_t Min(int64_t lhs, int64_t rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline int64_t AlignToBlock(int64_t elements) const
    {
        constexpr int64_t blockElements = 32 / sizeof(T);
        return (elements + blockElements - 1) / blockElements * blockElements;
    }

    __aicore__ inline int64_t AlignToVector(int64_t elements) const
    {
        return (elements + V_LENGTH - 1) / V_LENGTH * V_LENGTH;
    }

    __aicore__ inline int64_t WorkspaceStride() const
    {
        return tiling_->alignedCoresPerRow > 0 ? tiling_->alignedCoresPerRow : 32 / sizeof(float);
    }

    __aicore__ inline int64_t WorkspaceSlot(int64_t group, int64_t slot) const
    {
        return group * WorkspaceStride() + slot;
    }

    __aicore__ inline int64_t ContiguousBase(int64_t outer, int64_t innerOffset) const
    {
        return outer * tiling_->reduceCount * tiling_->contiguousInnerCount + innerOffset;
    }

    __aicore__ inline int64_t IrregularKeepOuterBase(int64_t keepOuter) const
    {
        int64_t base = 0;
        int64_t value = keepOuter;
        for (int64_t i = 0; i < tiling_->irregularKeepOuterRank; ++i) {
            const int64_t coord = value / tiling_->irregularKeepOuterIndexStrides[i];
            value %= tiling_->irregularKeepOuterIndexStrides[i];
            base += coord * tiling_->keepStrides[i];
        }
        return base;
    }

    __aicore__ inline int64_t IrregularReduceOuterOffset(int64_t reduceOuter) const
    {
        int64_t offset = 0;
        int64_t value = reduceOuter;
        for (int64_t i = 0; i < tiling_->irregularReduceOuterRank; ++i) {
            const int64_t coord = value / tiling_->irregularReduceOuterIndexStrides[i];
            value %= tiling_->irregularReduceOuterIndexStrides[i];
            offset += coord * tiling_->reduceStrides[i];
        }
        return offset;
    }

    __aicore__ inline void AdvanceReduceOffset(int64_t& currentOffset, int64_t* coords, int64_t rank) const
    {
        for (int64_t i = rank - 1; i >= 0; --i) {
            ++coords[i];
            currentOffset += tiling_->reduceStrides[i];
            if (coords[i] < tiling_->reduceDims[i]) {
                break;
            }
            currentOffset -= coords[i] * tiling_->reduceStrides[i];
            coords[i] = 0;
        }
    }

    __aicore__ inline void CopyInContiguous(int64_t offset, int64_t validElements)
    {
        LocalTensor<T> input = inQueue_.AllocTensor<T>();
        const int64_t alignedElements = AlignToBlock(validElements);
        DataCopyExtParams copy{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(alignedElements - validElements), static_cast<T>(0)};
        DataCopyPad(input, xGm_[static_cast<uint64_t>(offset)], copy, pad);
        inQueue_.EnQue(input);
    }

    __aicore__ inline void ZeroSumTileAt(LocalTensor<float>& sum, int64_t sumOffset, int64_t validElements)
    {
        __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> zeroReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t loop = 0; loop < loops; ++loop) {
                const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
                uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                Duplicate(zeroReg, static_cast<float>(0.0f), pregLoop);
                StoreRegForDtype<float>(sumAddr, zeroReg, pregLoop, static_cast<uint32_t>(sumOffset) + offset);
            }
        }
    }

    __aicore__ inline void ZeroSumTile(LocalTensor<float>& sum, int64_t validElements)
    {
        ZeroSumTileAt(sum, 0, validElements);
    }

    __aicore__ inline void AccumulateContiguousAt(LocalTensor<T>& input, LocalTensor<float>& sum, int64_t sumOffset,
                                                  int64_t validElements, int64_t inputOffset = 0)
    {
        __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
        __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> sumReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t loop = 0; loop < loops; ++loop) {
                const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
                uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                LoadRegForDtype<T>(inputAddr, xReg, pregLoop, static_cast<uint32_t>(inputOffset) + offset);
                LoadRegForDtype<float>(sumAddr, sumReg, pregLoop, static_cast<uint32_t>(sumOffset) + offset);
                Add(sumReg, sumReg, xReg, pregLoop);
                StoreRegForDtype<float>(sumAddr, sumReg, pregLoop, static_cast<uint32_t>(sumOffset) + offset);
            }
        }
    }

    __aicore__ inline void AccumulateContiguous(LocalTensor<T>& input, LocalTensor<float>& sum, int64_t validElements)
    {
        AccumulateContiguousAt(input, sum, 0, validElements);
    }

    // 把 numSlots 个独立的 fp32 部分和按两两折叠归并到槽 0（通用树形归约）。
    // 与 norm_common 的 DichotomyAdd 同一思路，对所有 dtype 生效。
    __aicore__ inline void FoldAccumulatorSlots(LocalTensor<float>& sum, uint16_t numSlots, int64_t stride,
                                                int64_t validElements)
    {
        if (numSlots <= 1) {
            return;
        }
        __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> lhsReg;
            RegTensor<float> rhsReg;
            RegTensor<float> outReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            uint16_t active = numSlots;
            for (uint16_t pass = 0; pass < 16 && active > 1; ++pass) {
                const uint16_t half = static_cast<uint16_t>(active / 2);
                for (uint16_t g = 0; g < half; ++g) {
                    for (uint16_t loop = 0; loop < loops; ++loop) {
                        const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                        const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
                        uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                        MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                        const uint32_t lhsOffset = static_cast<uint32_t>(g) * static_cast<uint32_t>(stride) + offset;
                        const uint32_t rhsOffset = static_cast<uint32_t>(g + half) * static_cast<uint32_t>(stride) +
                                                   offset;
                        LoadRegForDtype<float>(sumAddr, lhsReg, pregLoop, lhsOffset);
                        LoadRegForDtype<float>(sumAddr, rhsReg, pregLoop, rhsOffset);
                        Add(outReg, lhsReg, rhsReg, pregLoop);
                        StoreRegForDtype<float>(sumAddr, outReg, pregLoop, lhsOffset);
                    }
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                active = half;
            }
        }
    }

    __aicore__ inline void ScaleSumToMeanAt(LocalTensor<float>& sum, int64_t sumOffset, int64_t validElements)
    {
        __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
        const float invReduce = 1.0f / static_cast<float>(tiling_->reduceCount);
        const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> meanReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t loop = 0; loop < loops; ++loop) {
                const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
                uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                LoadRegForDtype<float>(sumAddr, meanReg, pregLoop, static_cast<uint32_t>(sumOffset) + offset);
                Muls(meanReg, meanReg, invReduce, pregLoop);
                StoreRegForDtype<float>(sumAddr, meanReg, pregLoop, static_cast<uint32_t>(sumOffset) + offset);
            }
        }
    }

    __aicore__ inline void ScaleSumToMean(LocalTensor<float>& sum, int64_t validElements)
    {
        ScaleSumToMeanAt(sum, 0, validElements);
    }

    // ---- 连续路径补偿归约（Neumaier + hi/lo 均值）----
    __aicore__ inline void AccumulateContiguousNeumaierAt(LocalTensor<T>& input, LocalTensor<float>& sum,
                                                          int64_t sumOffset, int64_t corrOffset, int64_t validElements,
                                                          int64_t inputOffset = 0)
    {
        __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
        __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> sumReg;
            RegTensor<float> corrReg;
            RegTensor<float> tReg;
            RegTensor<float> c1Reg;
            RegTensor<float> c2Reg;
            RegTensor<float> addReg;
            RegTensor<float> absSum;
            RegTensor<float> absX;
            MaskReg dom;
            MaskReg fin;
            RegTensor<float> zeroReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            Duplicate(zeroReg, 0.0f, pregFull);
            for (uint16_t loop = 0; loop < loops; ++loop) {
                const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
                uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                LoadRegForDtype<T>(inputAddr, xReg, pregLoop, static_cast<uint32_t>(inputOffset) + offset);
                LoadRegForDtype<float>(sumAddr, sumReg, pregLoop, static_cast<uint32_t>(sumOffset) + offset);
                LoadRegForDtype<float>(sumAddr, corrReg, pregLoop, static_cast<uint32_t>(corrOffset) + offset);
                // Neumaier: t = sum + x; corr += (sum - t) + x（真值 = sum + corr）
                Add(tReg, sumReg, xReg, pregLoop);
                Abs(absSum, sumReg, pregLoop);
                Abs(absX, xReg, pregLoop);
                Compare<float, CMPMODE::GE>(dom, absSum, absX, pregLoop);
                Sub(c1Reg, sumReg, tReg, pregLoop);
                Add(c1Reg, c1Reg, xReg, pregLoop);
                Sub(c2Reg, xReg, tReg, pregLoop);
                Add(c2Reg, c2Reg, sumReg, pregLoop);
                Select(addReg, c1Reg, c2Reg, dom);
                // inf/NaN 保护：t 非有限时补偿置 0，避免 inf-inf=NaN 污染
                Abs(absSum, tReg, pregLoop);
                Compares<float, CMPMODE::LE>(fin, absSum, 3.4028235e38f, pregLoop);
                Select(addReg, addReg, zeroReg, fin);
                Add(corrReg, corrReg, addReg, pregLoop);
                StoreRegForDtype<float>(sumAddr, tReg, pregLoop, static_cast<uint32_t>(sumOffset) + offset);
                StoreRegForDtype<float>(sumAddr, corrReg, pregLoop, static_cast<uint32_t>(corrOffset) + offset);
            }
        }
    }

    __aicore__ inline void FoldAccumulatorSlotsNeumaier(LocalTensor<float>& sum, uint16_t numSlots, int64_t stride,
                                                        int64_t corrBase, int64_t validElements)
    {
        if (numSlots <= 1) {
            return;
        }
        __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> lhsS;
            RegTensor<float> lhsC;
            RegTensor<float> rhsS;
            RegTensor<float> rhsC;
            RegTensor<float> tReg;
            RegTensor<float> c1Reg;
            RegTensor<float> c2Reg;
            RegTensor<float> addReg;
            RegTensor<float> absL;
            RegTensor<float> absR;
            MaskReg dom;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            uint16_t active = numSlots;
            for (uint16_t pass = 0; pass < 16 && active > 1; ++pass) {
                const uint16_t half = static_cast<uint16_t>(active / 2);
                for (uint16_t g = 0; g < half; ++g) {
                    for (uint16_t loop = 0; loop < loops; ++loop) {
                        const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                        const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
                        uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                        MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                        const uint32_t lo = static_cast<uint32_t>(g) * static_cast<uint32_t>(stride) + offset;
                        const uint32_t hi = static_cast<uint32_t>(g + half) * static_cast<uint32_t>(stride) + offset;
                        const uint32_t clo = static_cast<uint32_t>(corrBase) + lo;
                        const uint32_t chi = static_cast<uint32_t>(corrBase) + hi;
                        LoadRegForDtype<float>(sumAddr, lhsS, pregLoop, lo);
                        LoadRegForDtype<float>(sumAddr, lhsC, pregLoop, clo);
                        LoadRegForDtype<float>(sumAddr, rhsS, pregLoop, hi);
                        LoadRegForDtype<float>(sumAddr, rhsC, pregLoop, chi);
                        Add(tReg, lhsS, rhsS, pregLoop);
                        Abs(absL, lhsS, pregLoop);
                        Abs(absR, rhsS, pregLoop);
                        Compare<float, CMPMODE::GE>(dom, absL, absR, pregLoop);
                        Sub(c1Reg, lhsS, tReg, pregLoop);
                        Add(c1Reg, c1Reg, rhsS, pregLoop);
                        Sub(c2Reg, rhsS, tReg, pregLoop);
                        Add(c2Reg, c2Reg, lhsS, pregLoop);
                        Select(addReg, c1Reg, c2Reg, dom);
                        Add(lhsC, lhsC, addReg, pregLoop);
                        Add(lhsC, lhsC, rhsC, pregLoop);
                        StoreRegForDtype<float>(sumAddr, tReg, pregLoop, lo);
                        StoreRegForDtype<float>(sumAddr, lhsC, pregLoop, clo);
                    }
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                active = half;
            }
        }
    }

    __aicore__ inline void FinalizeMeanHiLoAt(LocalTensor<float>& sum, int64_t sumOffset, int64_t corrOffset,
                                              int64_t validElements)
    {
        __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
        const float invReduce = 1.0f / static_cast<float>(tiling_->reduceCount);
        const float nValue = static_cast<float>(tiling_->reduceCount);
        const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> sumReg;
            RegTensor<float> corrReg;
            RegTensor<float> meanReg;
            RegTensor<float> meanLowReg;
            RegTensor<float> nReg;
            RegTensor<float> cReg;
            RegTensor<float> mHi;
            RegTensor<float> mLo;
            RegTensor<float> nHi;
            RegTensor<float> nLo;
            RegTensor<float> tmpReg;
            RegTensor<float> pReg;
            RegTensor<float> eReg;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> rReg;
            RegTensor<float> negMeanReg;
            RegTensor<float> absMeanReg;
            RegTensor<float> zeroReg;
            MaskReg fin;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            Duplicate(zeroReg, 0.0f, pregFull);
            for (uint16_t loop = 0; loop < loops; ++loop) {
                const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
                uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                LoadRegForDtype<float>(sumAddr, sumReg, pregLoop, static_cast<uint32_t>(sumOffset) + offset);
                LoadRegForDtype<float>(sumAddr, corrReg, pregLoop, static_cast<uint32_t>(corrOffset) + offset);
                Duplicate(nReg, nValue, pregLoop);
                Div(meanReg, sumReg, nReg, pregLoop);
                // r = sum - mean*N：向量 FMA（MulDstAdd），精确除法残差
                Muls(negMeanReg, meanReg, -1.0f, pregLoop);
                MulDstAdd<float, AscendC::Reg::MaskMergeMode::ZEROING>(negMeanReg, nReg, sumReg, pregLoop);
                Add(tmpReg, negMeanReg, corrReg, pregLoop);
                Div(meanLowReg, tmpReg, nReg, pregLoop);
                // mean 非有限（组内含 inf/NaN）时，除法残差无意义且会引入 NaN；置 0 以保持 y = x - mean
                Abs(absMeanReg, meanReg, pregLoop);
                Compares<float, CMPMODE::LE>(fin, absMeanReg, 3.4028235e38f, pregLoop);
                Select(meanLowReg, meanLowReg, zeroReg, fin);
                StoreRegForDtype<float>(sumAddr, meanReg, pregLoop, static_cast<uint32_t>(sumOffset) + offset);
                StoreRegForDtype<float>(sumAddr, meanLowReg, pregLoop, static_cast<uint32_t>(corrOffset) + offset);
            }
        }
    }

    __aicore__ inline void CentralizeContiguousHiLoAt(LocalTensor<T>& input, LocalTensor<T>& output,
                                                      LocalTensor<float>& mean, int64_t meanOffset,
                                                      int64_t meanLowOffset, int64_t validElements,
                                                      int64_t inputOffset = 0, int64_t outputOffset = 0)
    {
        __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
        __local_mem__ T* outputAddr = (__local_mem__ T*)output.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> hiReg;
            RegTensor<float> loReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t loop = 0; loop < loops; ++loop) {
                const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
                uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                LoadRegForDtype<T>(inputAddr, xReg, pregLoop, static_cast<uint32_t>(inputOffset) + offset);
                LoadRegForDtype<float>(meanAddr, hiReg, pregLoop, static_cast<uint32_t>(meanOffset) + offset);
                LoadRegForDtype<float>(meanAddr, loReg, pregLoop, static_cast<uint32_t>(meanLowOffset) + offset);
                Sub(xReg, xReg, hiReg, pregLoop);
                Sub(xReg, xReg, loReg, pregLoop);
                StoreRegForDtype<T>(outputAddr, xReg, pregLoop, static_cast<uint32_t>(outputOffset) + offset);
            }
        }
    }

    __aicore__ inline void CentralizeContiguousAt(LocalTensor<T>& input, LocalTensor<T>& output,
                                                  LocalTensor<float>& mean, int64_t meanOffset, int64_t validElements,
                                                  int64_t inputOffset = 0, int64_t outputOffset = 0)
    {
        __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
        __local_mem__ T* outputAddr = (__local_mem__ T*)output.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> meanReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t loop = 0; loop < loops; ++loop) {
                const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
                uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                LoadRegForDtype<T>(inputAddr, xReg, pregLoop, static_cast<uint32_t>(inputOffset) + offset);
                LoadRegForDtype<float>(meanAddr, meanReg, pregLoop, static_cast<uint32_t>(meanOffset) + offset);
                Sub(xReg, xReg, meanReg, pregLoop);
                StoreRegForDtype<T>(outputAddr, xReg, pregLoop, static_cast<uint32_t>(outputOffset) + offset);
            }
        }
    }

    __aicore__ inline void CentralizeContiguous(LocalTensor<T>& input, LocalTensor<T>& output, LocalTensor<float>& mean,
                                                int64_t validElements)
    {
        CentralizeContiguousAt(input, output, mean, 0, validElements);
    }

    __aicore__ inline void CopyOutContiguous(int64_t offset, int64_t validElements)
    {
        LocalTensor<T> output = outQueue_.DeQue<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[static_cast<uint64_t>(offset)], output, copy);
        outQueue_.FreeTensor(output);
    }

    __aicore__ inline void CopyInSmallPadded(int64_t offset, int64_t rowElements, int64_t rowStride)
    {
        LocalTensor<T> input = inQueue_.AllocTensor<T>();
        const int64_t copyAlignedElements = AlignToBlock(rowElements);
        const uint32_t dstStride = static_cast<uint32_t>((rowStride - copyAlignedElements) * sizeof(T) / 32);
        DataCopyExtParams copy{static_cast<uint16_t>(tiling_->reduceCount),
                               static_cast<uint32_t>(rowElements * sizeof(T)), 0, dstStride, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(copyAlignedElements - rowElements),
                                    static_cast<T>(0)};
        DataCopyPad(input, xGm_[static_cast<uint64_t>(offset)], copy, pad);
        inQueue_.EnQue(input);
    }

    __aicore__ inline void CopyOutSmallPadded(int64_t offset, int64_t rowElements, int64_t rowStride)
    {
        LocalTensor<T> output = outQueue_.DeQue<T>();
        const int64_t copyAlignedElements = AlignToBlock(rowElements);
        DataCopyExtParams copy{static_cast<uint16_t>(tiling_->reduceCount),
                               static_cast<uint32_t>(rowElements * sizeof(T)),
                               static_cast<uint32_t>((rowStride - copyAlignedElements) * sizeof(T) / 32), 0, 0};
        DataCopyPad(yGm_[static_cast<uint64_t>(offset)], output, copy);
        outQueue_.FreeTensor(output);
    }

    __aicore__ inline void CopyInRowsPadded(int64_t offset, int64_t rowCount, int64_t rowElements, int64_t rowStride)
    {
        LocalTensor<T> input = inQueue_.AllocTensor<T>();
        const int64_t copyAlignedElements = AlignToBlock(rowElements);
        const uint32_t dstStride = static_cast<uint32_t>((rowStride - copyAlignedElements) * sizeof(T) / 32);
        DataCopyExtParams copy{static_cast<uint16_t>(rowCount), static_cast<uint32_t>(rowElements * sizeof(T)), 0,
                               dstStride, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(copyAlignedElements - rowElements),
                                    static_cast<T>(0)};
        DataCopyPad(input, xGm_[static_cast<uint64_t>(offset)], copy, pad);
        inQueue_.EnQue(input);
    }

    __aicore__ inline void CopyOutRowsPadded(int64_t offset, int64_t rowCount, int64_t rowElements, int64_t rowStride)
    {
        LocalTensor<T> output = outQueue_.DeQue<T>();
        const int64_t copyAlignedElements = AlignToBlock(rowElements);
        const uint32_t srcStride = static_cast<uint32_t>((rowStride - copyAlignedElements) * sizeof(T) / 32);
        DataCopyExtParams copy{static_cast<uint16_t>(rowCount), static_cast<uint32_t>(rowElements * sizeof(T)),
                               srcStride, 0, 0};
        DataCopyPad(yGm_[static_cast<uint64_t>(offset)], output, copy);
        outQueue_.FreeTensor(output);
    }

    __aicore__ inline float ReduceSegmentToScalar(LocalTensor<T>& input, int64_t validElements, int64_t inputOffset = 0)
    {
        LocalTensor<float> sumTensor = sumBuf_.Get<float>();
        __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
        __local_mem__ float* sumAddr = (__local_mem__ float*)sumTensor.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> partialReg;
            RegTensor<float> sumReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            Duplicate(sumReg, static_cast<float>(0.0f), pregOne);
            for (uint16_t loop = 0; loop < loops; ++loop) {
                const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
                uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                LoadRegForDtype<T>(inputAddr, xReg, pregLoop, static_cast<uint32_t>(inputOffset) + offset);
                AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(partialReg, xReg, pregLoop);
                Add(sumReg, sumReg, partialReg, pregOne);
            }
            StoreRegForDtype<float>(sumAddr, sumReg, pregOne, 0);
        }
        SetFlag<HardEvent::V_S>(eventVToS_);
        WaitFlag<HardEvent::V_S>(eventVToS_);
        return sumTensor.GetValue(0);
    }

    __aicore__ inline void CentralizeSegmentScalar(LocalTensor<T>& input, LocalTensor<T>& output, int64_t validElements,
                                                   float mean, int64_t inputOffset = 0, int64_t outputOffset = 0)
    {
        __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
        __local_mem__ T* outputAddr = (__local_mem__ T*)output.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> meanReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t loop = 0; loop < loops; ++loop) {
                const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
                uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                LoadRegForDtype<T>(inputAddr, xReg, pregLoop, static_cast<uint32_t>(inputOffset) + offset);
                Duplicate(meanReg, mean, pregLoop);
                Sub(xReg, xReg, meanReg, pregLoop);
                StoreRegForDtype<T>(outputAddr, xReg, pregLoop, static_cast<uint32_t>(outputOffset) + offset);
            }
        }
    }

#include "centralization_generic_contiguous_impl.h"
#include "centralization_generic_irregular_impl.h"
#include "centralization_generic_large_impl.h"

    TPipe* pipe_;
    const CentralizationTilingData* tiling_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> workspaceGm_;
    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    TBuf<TPosition::VECCALC> sumBuf_;
    TEventID eventMte2ToS_{};
    TEventID eventVToS_{};
    TEventID eventSToMte3_{};
    TEventID eventMte3ToMte2_{};
};
} // namespace Centralization
#endif
