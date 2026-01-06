/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nll_loss_mix.h
 * \brief nll_loss_mix
 */

#ifndef NLL_LOSS_SIMD_H
#define NLL_LOSS_SIMD_H

#include <cmath>
#include <cstdint>
#include "kernel_operator.h"
#include "../../inc/platform.h"

using namespace AscendC;

constexpr int32_t VL_SIZE = platform::GetVRegSize();
constexpr uint32_t REDUCE_256 = 256;
constexpr uint32_t MID_RES_128 = 128;
constexpr uint32_t DOUBLE = 2;
constexpr uint32_t ALIGN_32B = 8;
constexpr uint32_t MIN_THREAD = 32;
constexpr uint32_t MAX_UINT8 = 256;
constexpr uint32_t THIRD = 3;
constexpr uint32_t TWO_NUMBER = 2;
constexpr uint32_t FOUR_NUMBER = 4;

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtZeroPadTo256B(
    const uint64_t start, const uint64_t padSize, __ubuf__ float* outBuffer)
{
    for (uint32_t ubIdx = Simt::GetThreadIdx(); ubIdx < padSize; ubIdx += Simt::GetThreadNum()) {
        outBuffer[start + ubIdx] = static_cast<float>(0);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(MIN_THREAD) inline void GetReduceValueGm(
    int64_t blockId_, __ubuf__ float* srcBuffer, __gm__ volatile float* dstBuffer)
{
    if (Simt::GetThreadIdx() == 0) {
        dstBuffer[blockId_] = srcBuffer[0];
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(MIN_THREAD) inline void GetReduceValue(
    uint32_t level1Idx, __ubuf__ float* srcBuffer, __ubuf__ float* dstBuffer)
{
    if (Simt::GetThreadIdx() == 0) {
        dstBuffer[level1Idx] = srcBuffer[0];
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MIN_THREAD) inline void GetRes(
    uint32_t reduction_, __gm__ T* yGm_, __gm__ T* totalWeight_, __ubuf__ float* midOutResUB_,
    __ubuf__ float* midWeightResUB_)
{
    if (Simt::GetThreadIdx() == 0) {
        if (reduction_ == 1) {
            yGm_[0] = static_cast<T>(midOutResUB_[0] / midWeightResUB_[0]);
        } else {
            yGm_[0] = static_cast<T>(midOutResUB_[0]);
        }
        totalWeight_[0] = static_cast<T>(midWeightResUB_[0]);
    }
}

template <typename U, typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtMeanModeCompute2d(
    const uint32_t maxSize, const uint64_t offset, int64_t ignoreIndex_, uint32_t xDimC_, uint32_t isWeightPresent_,
    __ubuf__ float* outBuffer, __ubuf__ float* weightbBuffer, __gm__ U* targetGm_, __gm__ T* xGm_, __gm__ T* weightGm_)
{
    for (uint32_t ubIdx = Simt::GetThreadIdx(); ubIdx < maxSize; ubIdx += Simt::GetThreadNum()) {
        uint64_t i = ubIdx + offset;
        U targetIndex = targetGm_[i];
        if (targetIndex == ignoreIndex_) {
            outBuffer[ubIdx] = static_cast<float>(0);
            weightbBuffer[ubIdx] = static_cast<float>(0);
            continue;
        }
        TargetCheck(targetIndex, xDimC_);
        float curWeight = isWeightPresent_ == 1 ? static_cast<float>(weightGm_[targetIndex]) : static_cast<float>(1);
        weightbBuffer[ubIdx] = curWeight;
        outBuffer[ubIdx] = -curWeight * static_cast<float>(xGm_[i * xDimC_ + targetIndex]);
    }
}

template <typename U, typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtMeanModeCompute4d(
    const uint32_t maxSize, const uint64_t offset, int64_t ignoreIndex_, uint32_t xDimC_, int64_t xDimN_,
    int64_t xDimH_, int64_t xDimW_, uint32_t isWeightPresent_, __ubuf__ float* outBuffer, __ubuf__ float* weightbBuffer,
    __gm__ U* targetGm_, __gm__ T* xGm_, __gm__ T* weightGm_, int64_t productOfCHW_, int64_t productOfHW_)
{
    for (uint32_t ubIdx = Simt::GetThreadIdx(); ubIdx < maxSize; ubIdx += Simt::GetThreadNum()) {
        uint64_t i = ubIdx + offset;
        U targetIndex = targetGm_[i];
        auto n = i / productOfHW_;
        auto rem = i % productOfHW_;
        if (targetIndex == ignoreIndex_) {
            outBuffer[ubIdx] = static_cast<float>(0);
            weightbBuffer[ubIdx] = static_cast<float>(0);
            continue;
        }
        TargetCheck(targetIndex, xDimC_);
        float curWeight = isWeightPresent_ == 1 ? static_cast<float>(weightGm_[targetIndex]) : static_cast<float>(1);
        weightbBuffer[ubIdx] = curWeight;
        outBuffer[ubIdx] = -curWeight * static_cast<float>(xGm_[n * productOfCHW_ + targetIndex * productOfHW_ + rem]);
    }
}

template <typename T, typename U>
class KernelNLLLossSimd {
public:
    __aicore__ inline KernelNLLLossSimd()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR totalWeight, GM_ADDR workSpace,
        NLLLossACTilingData tilingData)
    {
        xGm_.SetGlobalBuffer((__gm__ T*)(x));
        yGm_.SetGlobalBuffer((__gm__ T*)(y));
        totalWeight_.SetGlobalBuffer((__gm__ T*)(totalWeight));
        targetGm_.SetGlobalBuffer((__gm__ U*)(target));
        isWeightPresent_ = tilingData.isWeightPresent;
        if (isWeightPresent_ == 1) {
            weightGm_.SetGlobalBuffer((__gm__ T*)(weight));
        }

        xDims_ = tilingData.xDims;
        ignoreIndex_ = tilingData.ignoreIndex;
        reduction_ = tilingData.reduction;
        xDimN_ = tilingData.xDimN;
        xDimC_ = tilingData.xDimC;
        xDimH_ = tilingData.xDimH;
        xDimW_ = tilingData.xDimW;
        coreNum_ = tilingData.coreNum;
        mainReduceSize_ = tilingData.mainReduceSize;
        usedLogicCore_ = tilingData.usedLogicCore;
        loopInCore_ = tilingData.loopInCore;
        fullIndex_ = tilingData.fullIndex;
        tailSize_ = tilingData.tailSize;
        padSize_ = tilingData.padSize;
        loopNum1_ = tilingData.loopNum1;
        tailNum1_ = tilingData.tailNum1;
        tailMoveSize_ = tilingData.tailMoveSize;
        tailMainReduceSize_ = tilingData.tailMainReduceSize;
        tailRemainSize_ = tilingData.tailRemainSize;
        loopNum3_ = tilingData.loopNum3;
        tailNum3_ = tilingData.tailNum3;
        vfFloatNum_ = VL_SIZE / sizeof(float);
        reduceOnceSize_ = vfFloatNum_ * DOUBLE;

        productOfCHW_ = xDimC_ * xDimH_ * xDimW_;
        productOfHW_ = xDimH_ * xDimW_;

        blockId_ = GetBlockIdx();
        blockNums_ = GetBlockNum();

        pipe_.InitBuffer(level1Out_, REDUCE_256 * sizeof(float));
        pipe_.InitBuffer(level2Out_, REDUCE_256 * sizeof(float));
        pipe_.InitBuffer(level3Out_, REDUCE_256 * sizeof(float));
        level1OutUB_ = level1Out_.Get<float>();
        level2OutUB_ = level2Out_.Get<float>();
        level3OutUB_ = level3Out_.Get<float>();

        pipe_.InitBuffer(out_, mainReduceSize_ * DOUBLE * sizeof(float));
        outUB_ = out_.Get<float>();

        pipe_.InitBuffer(midOutRes_, MID_RES_128 * sizeof(float));
        midOutResUB_ = midOutRes_.Get<float>();

        pipe_.InitBuffer(level1Weight_, REDUCE_256 * sizeof(float));
        pipe_.InitBuffer(level2Weight_, REDUCE_256 * sizeof(float));
        pipe_.InitBuffer(level3Weight_, REDUCE_256 * sizeof(float));
        level1WeightUB_ = level1Weight_.Get<float>();
        level2WeightUB_ = level2Weight_.Get<float>();
        level3WeightUB_ = level3Weight_.Get<float>();

        pipe_.InitBuffer(weight_, mainReduceSize_ * DOUBLE * sizeof(float));
        weightUB_ = weight_.Get<float>();

        pipe_.InitBuffer(midWeightRes_, MID_RES_128 * sizeof(float));
        midWeightResUB_ = midWeightRes_.Get<float>();

        workspaceOutGm_.SetGlobalBuffer((__gm__ float*)(workSpace), coreNum_);
        workspaceWeightGm_.SetGlobalBuffer((__gm__ float*)workSpace + coreNum_, coreNum_);
    }

    __aicore__ inline void ReduceSumInCore(
        const LocalTensor<float>& reduceBuf, const LocalTensor<float>& midRes, uint32_t mainReduceLength,
        uint32_t tailReduceLength)
    {
        __local_mem__ float* mainAddr = (__ubuf__ float*)reduceBuf.GetPhyAddr();
        __local_mem__ float* tailAddr = (__ubuf__ float*)reduceBuf.GetPhyAddr(mainReduceLength);
        __local_mem__ float* remainAddr = (__ubuf__ float*)reduceBuf.GetPhyAddr(tailReduceLength);
        __local_mem__ float* midResAddr = (__ubuf__ float*)midRes.GetPhyAddr();

        uint32_t tailLoop = tailReduceLength / reduceOnceSize_;
        uint32_t mainLoop = (mainReduceLength - tailReduceLength) / reduceOnceSize_;
        uint32_t restLoop = (tailLoop + mainLoop) / reduceOnceSize_;
        uint32_t lengthBeforeLastReduce = restLoop == 0 ? tailLoop + mainLoop : restLoop;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> main1, main2, tail1, tail2, res;
            AscendC::MicroAPI::MaskReg pregMain =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg pregLoop;

            for (uint16_t i = 0; i < static_cast<uint16_t>(tailLoop); ++i) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(tailReduceLength);
                DataCopy(main1, mainAddr + i * DOUBLE * vfFloatNum_);
                DataCopy(main2, mainAddr + (i * DOUBLE + 1) * vfFloatNum_);
                DataCopy(tail1, tailAddr + i * DOUBLE * vfFloatNum_);
                DataCopy(tail2, tailAddr + (i * DOUBLE + 1) * vfFloatNum_);

                Add(main1, main1, tail1, pregLoop);
                Add(main2, main2, tail2, pregLoop);
                Add(main1, main1, main2, pregLoop);
                ReduceSum(res, main1, pregLoop);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(midResAddr + i, res, pregMain);
            }
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < static_cast<uint16_t>(mainLoop); ++i) {
                uint32_t sreg0 = mainReduceLength - tailReduceLength;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(main1, remainAddr + i * DOUBLE * vfFloatNum_);
                DataCopy(main2, remainAddr + (i * DOUBLE + 1) * vfFloatNum_);
                Add(main1, main1, main2, pregLoop);
                ReduceSum(res, main1, pregLoop);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    midResAddr + tailLoop + i, res, pregMain);
            }
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < static_cast<uint16_t>(restLoop); ++i) {
                uint32_t sreg0 = tailLoop + mainLoop;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(main1, midResAddr + i * DOUBLE * vfFloatNum_);
                DataCopy(main2, midResAddr + (i * DOUBLE + 1) * vfFloatNum_);
                Add(main1, main1, main2, pregLoop);
                ReduceSum(res, main1, pregLoop);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(midResAddr + i, res, pregMain);
            }
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(lengthBeforeLastReduce);
                DataCopy(main1, midResAddr);
                ReduceSum(res, main1, pregLoop);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(mainAddr, res, pregMain);
            }
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }
    }

    __aicore__ inline void ReduceSum256(
        const LocalTensor<float>& reduceBuf, const LocalTensor<float>& nextLevelReduceBuf, uint32_t idx)
    {
        __local_mem__ float* mainAddr = (__ubuf__ float*)reduceBuf.GetPhyAddr();
        __local_mem__ float* outAddr = (__ubuf__ float*)nextLevelReduceBuf.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> main1, main2, main3, main4, res;
            uint32_t sreg0 = vfFloatNum_;
            AscendC::MicroAPI::MaskReg pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
            AscendC::MicroAPI::MaskReg pregMain =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            DataCopy(main1, mainAddr);
            DataCopy(main2, mainAddr + vfFloatNum_);
            DataCopy(main3, mainAddr + DOUBLE * vfFloatNum_);
            DataCopy(main4, mainAddr + THIRD * vfFloatNum_);

            Add(main1, main1, main2, pregLoop);
            Add(main3, main3, main4, pregLoop);
            Add(main1, main1, main3, pregLoop);
            ReduceSum(res, main1, pregLoop);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(outAddr + idx, res, pregMain);
            AscendC::MicroAPI::LocalMemBar<
                AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }
    }

    __aicore__ inline void FinalCoreRes2Worksapce(
        const LocalTensor<float>& outBuffer, const GlobalTensor<float>& workspace)
    {
        ReduceSum256(outBuffer, outBuffer, 0);
        PipeBarrier<PIPE_ALL>();

        AscendC::Simt::VF_CALL<GetReduceValueGm>(
            AscendC::Simt::Dim3{static_cast<uint32_t>(MIN_THREAD)}, blockId_, (__ubuf__ float*)outBuffer.GetPhyAddr(),
            (__gm__ volatile float*)workspace.GetPhyAddr());
    }

    __aicore__ inline void MeanModeSubProcess(uint32_t& level1Idx, uint32_t& level2Idx, uint32_t& level3Idx)
    {
        AscendC::Simt::VF_CALL<GetReduceValue>(
            AscendC::Simt::Dim3{static_cast<uint32_t>(MIN_THREAD)}, level1Idx, (__ubuf__ float*)outUB_.GetPhyAddr(),
            (__ubuf__ float*)level1OutUB_.GetPhyAddr());
        AscendC::Simt::VF_CALL<GetReduceValue>(
            AscendC::Simt::Dim3{static_cast<uint32_t>(MIN_THREAD)}, level1Idx, (__ubuf__ float*)weightUB_.GetPhyAddr(),
            (__ubuf__ float*)level1WeightUB_.GetPhyAddr());
        ++level1Idx;
        PipeBarrier<PIPE_ALL>();
        if (level1Idx == MAX_UINT8) {
            ReduceSum256(level1OutUB_, level2OutUB_, level2Idx);
            ReduceSum256(level1WeightUB_, level2WeightUB_, level2Idx);
            PipeBarrier<PIPE_ALL>();
            Duplicate(level1OutUB_, (float)0, REDUCE_256);
            Duplicate(level1WeightUB_, (float)0, REDUCE_256);
            PipeBarrier<PIPE_ALL>();
            ++level2Idx;
            level1Idx = 0;
        }
        if (level2Idx == MAX_UINT8) {
            ReduceSum256(level2OutUB_, level3OutUB_, level3Idx);
            ReduceSum256(level2WeightUB_, level3WeightUB_, level3Idx);
            PipeBarrier<PIPE_ALL>();
            Duplicate(level2OutUB_, (float)0, REDUCE_256);
            Duplicate(level2WeightUB_, (float)0, REDUCE_256);
            PipeBarrier<PIPE_ALL>();
            ++level3Idx;
            level2Idx = 0;
        }
    }

    __aicore__ inline void SumModeSubProcess(uint32_t& level1Idx, uint32_t& level2Idx, uint32_t& level3Idx)
    {
        AscendC::Simt::VF_CALL<GetReduceValue>(
            AscendC::Simt::Dim3{static_cast<uint32_t>(MIN_THREAD)}, level1Idx, (__ubuf__ float*)outUB_.GetPhyAddr(),
            (__ubuf__ float*)level1OutUB_.GetPhyAddr());
        ++level1Idx;
        PipeBarrier<PIPE_ALL>();
        if (level1Idx == MAX_UINT8) {
            ReduceSum256(level1OutUB_, level2OutUB_, level2Idx);
            PipeBarrier<PIPE_ALL>();
            Duplicate(level1OutUB_, (float)0, REDUCE_256);
            PipeBarrier<PIPE_ALL>();
            ++level2Idx;
            level1Idx = 0;
        }
        if (level2Idx == MAX_UINT8) {
            ReduceSum256(level2OutUB_, level3OutUB_, level3Idx);
            PipeBarrier<PIPE_ALL>();
            Duplicate(level2OutUB_, (float)0, REDUCE_256);
            PipeBarrier<PIPE_ALL>();
            ++level3Idx;
            level2Idx = 0;
        }
    }

    __aicore__ inline void MeanModeProcess2d()
    {
        uint32_t level1Idx = 0;
        uint32_t level2Idx = 0;
        uint32_t level3Idx = 0;
        uint32_t simtMoveSize = mainReduceSize_;
        Duplicate(level1OutUB_, (float)0, REDUCE_256);
        Duplicate(level1WeightUB_, (float)0, REDUCE_256);
        Duplicate(level2OutUB_, (float)0, REDUCE_256);
        Duplicate(level2WeightUB_, (float)0, REDUCE_256);
        Duplicate(level3OutUB_, (float)0, REDUCE_256);
        Duplicate(level3WeightUB_, (float)0, REDUCE_256);
        if (blockId_ == coreNum_ - 1) {
            uint64_t offset = (usedLogicCore_ - loopInCore_ - 1) * mainReduceSize_;
            for (uint64_t i = 0; i < loopNum1_; ++i) {
                if (i == loopNum1_ - tailNum1_) {
                    simtMoveSize += mainReduceSize_;
                }
                uint32_t mainReduceLen = simtMoveSize;
                uint32_t tailLen = 0;
                if (i == loopNum1_ - 1) {
                    simtMoveSize = tailMoveSize_;
                    mainReduceLen = tailMainReduceSize_;
                    tailLen = tailRemainSize_;
                    if (tailSize_ % MID_RES_128 != 0) {
                        AscendC::Simt::VF_CALL<SimtZeroPadTo256B>(
                            AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, simtMoveSize, padSize_,
                            (__ubuf__ float*)outUB_.GetPhyAddr());
                        AscendC::Simt::VF_CALL<SimtZeroPadTo256B>(
                            AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, simtMoveSize, padSize_,
                            (__ubuf__ float*)weightUB_.GetPhyAddr());
                    }
                }

                AscendC::Simt::VF_CALL<SimtMeanModeCompute2d<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, simtMoveSize, offset, ignoreIndex_, xDimC_,
                    isWeightPresent_, (__ubuf__ float*)outUB_.GetPhyAddr(), (__ubuf__ float*)weightUB_.GetPhyAddr(),
                    (__gm__ U*)targetGm_.GetPhyAddr(), (__gm__ T*)xGm_.GetPhyAddr(), (__gm__ T*)weightGm_.GetPhyAddr());

                offset += simtMoveSize;
                PipeBarrier<PIPE_ALL>();
                ReduceSumInCore(outUB_, midOutResUB_, mainReduceLen, tailLen);
                ReduceSumInCore(weightUB_, midWeightResUB_, mainReduceLen, tailLen);
                PipeBarrier<PIPE_ALL>();
                MeanModeSubProcess(level1Idx, level2Idx, level3Idx);
            }
        } else if (blockId_ < fullIndex_ - 1) {
            uint64_t offset = blockId_ * (loopInCore_ + 1) * mainReduceSize_;
            for (uint64_t i = 0; i < loopNum1_; ++i) {
                if (i == loopNum1_ - tailNum1_) {
                    simtMoveSize += mainReduceSize_;
                }

                AscendC::Simt::VF_CALL<SimtMeanModeCompute2d<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, simtMoveSize, offset, ignoreIndex_, xDimC_,
                    isWeightPresent_, (__ubuf__ float*)outUB_.GetPhyAddr(), (__ubuf__ float*)weightUB_.GetPhyAddr(),
                    (__gm__ U*)targetGm_.GetPhyAddr(), (__gm__ T*)xGm_.GetPhyAddr(), (__gm__ T*)weightGm_.GetPhyAddr());

                offset += simtMoveSize;
                PipeBarrier<PIPE_ALL>();
                ReduceSumInCore(outUB_, midOutResUB_, simtMoveSize, 0);
                ReduceSumInCore(weightUB_, midWeightResUB_, simtMoveSize, 0);
                PipeBarrier<PIPE_ALL>();
                MeanModeSubProcess(level1Idx, level2Idx, level3Idx);
            }
        } else {
            uint64_t offset =
                ((blockId_ + 1 - fullIndex_) * loopInCore_ + (fullIndex_ - 1) * (loopInCore_ + 1)) * mainReduceSize_;
            for (uint64_t i = 0; i < loopNum3_; ++i) {
                if (i == loopNum3_ - tailNum3_) {
                    simtMoveSize += mainReduceSize_;
                }

                AscendC::Simt::VF_CALL<SimtMeanModeCompute2d<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, simtMoveSize, offset, ignoreIndex_, xDimC_,
                    isWeightPresent_, (__ubuf__ float*)outUB_.GetPhyAddr(), (__ubuf__ float*)weightUB_.GetPhyAddr(),
                    (__gm__ U*)targetGm_.GetPhyAddr(), (__gm__ T*)xGm_.GetPhyAddr(), (__gm__ T*)weightGm_.GetPhyAddr());

                offset += simtMoveSize;
                PipeBarrier<PIPE_ALL>();
                ReduceSumInCore(outUB_, midOutResUB_, simtMoveSize, 0);
                ReduceSumInCore(weightUB_, midWeightResUB_, simtMoveSize, 0);
                PipeBarrier<PIPE_ALL>();
                MeanModeSubProcess(level1Idx, level2Idx, level3Idx);
            }
        }
        if (level3Idx != 0) {
            FinalCoreRes2Worksapce(level3OutUB_, workspaceOutGm_);
            FinalCoreRes2Worksapce(level3WeightUB_, workspaceWeightGm_);
        } else if (level2Idx != 0) {
            FinalCoreRes2Worksapce(level2OutUB_, workspaceOutGm_);
            FinalCoreRes2Worksapce(level2WeightUB_, workspaceWeightGm_);
        } else {
            FinalCoreRes2Worksapce(level1OutUB_, workspaceOutGm_);
            FinalCoreRes2Worksapce(level1WeightUB_, workspaceWeightGm_);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
    }

    __aicore__ inline void MeanModeProcess4d()
    {
        uint32_t level1Idx = 0;
        uint32_t level2Idx = 0;
        uint32_t level3Idx = 0;
        uint32_t simtMoveSize = mainReduceSize_;
        Duplicate(level1OutUB_, (float)0, REDUCE_256);
        Duplicate(level1WeightUB_, (float)0, REDUCE_256);
        Duplicate(level2OutUB_, (float)0, REDUCE_256);
        Duplicate(level2WeightUB_, (float)0, REDUCE_256);
        Duplicate(level3OutUB_, (float)0, REDUCE_256);
        Duplicate(level3WeightUB_, (float)0, REDUCE_256);
        if (blockId_ == coreNum_ - 1) {
            uint64_t offset = (usedLogicCore_ - loopInCore_ - 1) * mainReduceSize_;
            for (uint64_t i = 0; i < loopNum1_; ++i) {
                if (i == loopNum1_ - tailNum1_) {
                    simtMoveSize += mainReduceSize_;
                }
                uint32_t mainReduceLen = simtMoveSize;
                uint32_t tailLen = 0;
                if (i == loopNum1_ - 1) {
                    simtMoveSize = tailMoveSize_;
                    mainReduceLen = tailMainReduceSize_;
                    tailLen = tailRemainSize_;
                    if (tailSize_ % MID_RES_128 != 0) {
                        AscendC::Simt::VF_CALL<SimtZeroPadTo256B>(
                            AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, simtMoveSize, padSize_,
                            (__ubuf__ float*)outUB_.GetPhyAddr());
                        AscendC::Simt::VF_CALL<SimtZeroPadTo256B>(
                            AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, simtMoveSize, padSize_,
                            (__ubuf__ float*)weightUB_.GetPhyAddr());
                    }
                }

                AscendC::Simt::VF_CALL<SimtMeanModeCompute4d<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, simtMoveSize, offset, ignoreIndex_, xDimC_,
                    xDimN_, xDimH_, xDimW_, isWeightPresent_, (__ubuf__ float*)outUB_.GetPhyAddr(),
                    (__ubuf__ float*)weightUB_.GetPhyAddr(), (__gm__ U*)targetGm_.GetPhyAddr(),
                    (__gm__ T*)xGm_.GetPhyAddr(), (__gm__ T*)weightGm_.GetPhyAddr(), productOfCHW_, productOfHW_);

                offset += simtMoveSize;
                PipeBarrier<PIPE_ALL>();
                ReduceSumInCore(outUB_, midOutResUB_, mainReduceLen, tailLen);
                ReduceSumInCore(weightUB_, midWeightResUB_, mainReduceLen, tailLen);
                PipeBarrier<PIPE_ALL>();
                MeanModeSubProcess(level1Idx, level2Idx, level3Idx);
            }
        } else if (blockId_ < fullIndex_ - 1) {
            uint64_t offset = blockId_ * (loopInCore_ + 1) * mainReduceSize_;
            for (uint64_t i = 0; i < loopNum1_; ++i) {
                if (i == loopNum1_ - tailNum1_) {
                    simtMoveSize += mainReduceSize_;
                }

                AscendC::Simt::VF_CALL<SimtMeanModeCompute4d<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, simtMoveSize, offset, ignoreIndex_, xDimC_,
                    xDimN_, xDimH_, xDimW_, isWeightPresent_, (__ubuf__ float*)outUB_.GetPhyAddr(),
                    (__ubuf__ float*)weightUB_.GetPhyAddr(), (__gm__ U*)targetGm_.GetPhyAddr(),
                    (__gm__ T*)xGm_.GetPhyAddr(), (__gm__ T*)weightGm_.GetPhyAddr(), productOfCHW_, productOfHW_);

                offset += simtMoveSize;
                PipeBarrier<PIPE_ALL>();
                ReduceSumInCore(outUB_, midOutResUB_, simtMoveSize, 0);
                ReduceSumInCore(weightUB_, midWeightResUB_, simtMoveSize, 0);
                PipeBarrier<PIPE_ALL>();
                MeanModeSubProcess(level1Idx, level2Idx, level3Idx);
            }
        } else {
            uint64_t offset =
                ((blockId_ + 1 - fullIndex_) * loopInCore_ + (fullIndex_ - 1) * (loopInCore_ + 1)) * mainReduceSize_;
            for (uint64_t i = 0; i < loopNum3_; ++i) {
                if (i == loopNum3_ - tailNum3_) {
                    simtMoveSize += mainReduceSize_;
                }

                AscendC::Simt::VF_CALL<SimtMeanModeCompute4d<U, T>>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(THREAD_DIM)}, simtMoveSize, offset, ignoreIndex_, xDimC_,
                    xDimN_, xDimH_, xDimW_, isWeightPresent_, (__ubuf__ float*)outUB_.GetPhyAddr(),
                    (__ubuf__ float*)weightUB_.GetPhyAddr(), (__gm__ U*)targetGm_.GetPhyAddr(),
                    (__gm__ T*)xGm_.GetPhyAddr(), (__gm__ T*)weightGm_.GetPhyAddr(), productOfCHW_, productOfHW_);

                offset += simtMoveSize;
                PipeBarrier<PIPE_ALL>();
                ReduceSumInCore(outUB_, midOutResUB_, simtMoveSize, 0);
                ReduceSumInCore(weightUB_, midWeightResUB_, simtMoveSize, 0);
                PipeBarrier<PIPE_ALL>();
                MeanModeSubProcess(level1Idx, level2Idx, level3Idx);
            }
        }
        if (level3Idx != 0) {
            FinalCoreRes2Worksapce(level3OutUB_, workspaceOutGm_);
            FinalCoreRes2Worksapce(level3WeightUB_, workspaceWeightGm_);
        } else if (level2Idx != 0) {
            FinalCoreRes2Worksapce(level2OutUB_, workspaceOutGm_);
            FinalCoreRes2Worksapce(level2WeightUB_, workspaceWeightGm_);
        } else {
            FinalCoreRes2Worksapce(level1OutUB_, workspaceOutGm_);
            FinalCoreRes2Worksapce(level1WeightUB_, workspaceWeightGm_);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
    }

    __aicore__ inline void ReduceSumAmongCores(
        const LocalTensor<float>& outBuffer, const LocalTensor<float>& tmpBuffer, const GlobalTensor<float>& workspace)
    {
        DataCopyExtParams extParams{static_cast<uint16_t>(1), static_cast<uint32_t>(coreNum_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(tmpBuffer, workspace, extParams, padParams);
        event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        uint32_t srcShape[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(coreNum_)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(outBuffer, tmpBuffer, srcShape, false);
    }

    __aicore__ inline void Process()
    {
        if (xDims_ == 1 || xDims_ == TWO_NUMBER) {
            MeanModeProcess2d();
        } else if (xDims_ == FOUR_NUMBER) {
            MeanModeProcess4d();
        }
        if (blockId_ == 0) {
            ReduceSumAmongCores(midOutResUB_, level2OutUB_, workspaceOutGm_);
            ReduceSumAmongCores(midWeightResUB_, level3OutUB_, workspaceWeightGm_);
            PipeBarrier<PIPE_ALL>();
            AscendC::Simt::VF_CALL<GetRes<T>>(
                AscendC::Simt::Dim3{static_cast<uint32_t>(MIN_THREAD)}, reduction_, (__gm__ T*)yGm_.GetPhyAddr(),
                (__gm__ T*)totalWeight_.GetPhyAddr(), (__ubuf__ float*)midOutResUB_.GetPhyAddr(),
                (__ubuf__ float*)midWeightResUB_.GetPhyAddr());
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> out_;
    LocalTensor<float> outUB_;
    TBuf<TPosition::VECCALC> weight_;
    LocalTensor<float> weightUB_;

    TBuf<TPosition::VECCALC> level1Out_;
    LocalTensor<float> level1OutUB_;
    TBuf<TPosition::VECCALC> level2Out_;
    LocalTensor<float> level2OutUB_;
    TBuf<TPosition::VECCALC> level3Out_;
    LocalTensor<float> level3OutUB_;

    TBuf<TPosition::VECCALC> level1Weight_;
    LocalTensor<float> level1WeightUB_;
    TBuf<TPosition::VECCALC> level2Weight_;
    LocalTensor<float> level2WeightUB_;
    TBuf<TPosition::VECCALC> level3Weight_;
    LocalTensor<float> level3WeightUB_;

    TBuf<TPosition::VECCALC> midOutRes_;
    LocalTensor<float> midOutResUB_;

    TBuf<TPosition::VECCALC> midWeightRes_;
    LocalTensor<float> midWeightResUB_;

    AscendC::GlobalTensor<T> xGm_, weightGm_, yGm_, totalWeight_;
    AscendC::GlobalTensor<U> targetGm_;
    AscendC::GlobalTensor<float> workspaceOutGm_, workspaceWeightGm_;

    int64_t ignoreIndex_{0};
    uint32_t reduction_{0};
    uint32_t xDimN_{0};
    uint32_t xDimC_{1};
    int64_t xDimH_{1};
    int64_t xDimW_{1};
    uint32_t coreNum_{0};
    uint32_t isWeightPresent_{0};
    int64_t blockId_{0};
    int64_t blockNums_{0};
    int64_t xDims_{0};

    uint64_t mainReduceSize_{1};
    uint32_t usedLogicCore_{1};
    uint32_t loopInCore_{1};
    int32_t fullIndex_{0};
    uint32_t tailSize_{0};
    uint64_t padSize_{0};
    uint32_t loopNum1_{1};
    uint32_t tailNum1_{0};
    uint32_t tailMoveSize_{1};
    uint32_t tailMainReduceSize_{1};
    uint32_t tailRemainSize_{1};
    uint32_t loopNum3_{1};
    uint32_t tailNum3_{0};
    uint32_t reduceOnceSize_{1};
    uint32_t vfFloatNum_{1};
    int64_t productOfCHW_{0};
    int64_t productOfHW_{0};
};
#endif