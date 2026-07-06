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
 * \file clipped_swiglu_kernel.h
 * \brief Regbase kernel for ClippedSwiglu (Ascend 950 / DAV_3510)
 */

#ifndef CLIPPED_SWIGLU_KERNEL_H
#define CLIPPED_SWIGLU_KERNEL_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "clipped_swiglu_tiling_key.h"
#include "clipped_swiglu_tiling_data.h"

namespace ClippedSwigluOp {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t BLOCK_SIZE = Ops::Base::GetUbBlockSize(); // 32
constexpr int64_t DIM_HALVE = 2;
constexpr uint32_t VF_LEN_FP32 = Ops::Base::GetVRegSize() / sizeof(float);

static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_FP16_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr static AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

__aicore__ inline void ReduceAllVf(LocalTensor<int64_t>& reduceSumUb, LocalTensor<int64_t>& groupIndexUb,
                                   int64_t groupIndexNum)
{
    uint32_t vfTidx = Ops::Base::GetVRegSize() / sizeof(int64_t);
    uint16_t times = groupIndexNum / vfTidx;
    uint32_t tailNum = groupIndexNum % vfTidx;
    uint16_t tailTimes = tailNum != 0 ? 1 : 0;
    auto dstAddr = (__ubuf__ int64_t*)reduceSumUb.GetPhyAddr();
    auto srcAddr = (__ubuf__ int64_t*)groupIndexUb.GetPhyAddr();
    auto srcAddr1 = (__ubuf__ int64_t*)groupIndexUb[times * vfTidx].GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int64_t> addReg;
        AscendC::MicroAPI::RegTensor<int64_t> reduceSumReg;
        AscendC::MicroAPI::RegTensor<int64_t> reduceSumTReg;
        AscendC::MicroAPI::RegTensor<int64_t> srcReg;
        AscendC::MicroAPI::Duplicate(addReg, 0);
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<int64_t, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < times; i++) {
            AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<int64_t>(i, vfTidx);
            AscendC::MicroAPI::DataCopy(srcReg, srcAddr, srcIdxOffset);
            AscendC::MicroAPI::Add(addReg, addReg, srcReg, mask);
        }
        AscendC::MicroAPI::ReduceSum(reduceSumReg, addReg, mask);
        for (uint16_t j = 0; j < tailTimes; j++) {
            AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<int64_t>(tailNum);
            AscendC::MicroAPI::DataCopy(srcReg, srcAddr1);
            AscendC::MicroAPI::ReduceSum(reduceSumTReg, srcReg, maskT);
            AscendC::MicroAPI::Add(reduceSumReg, reduceSumTReg, reduceSumReg, maskT);
        }
        AscendC::MicroAPI::MaskReg maskOne = AscendC::MicroAPI::CreateMask<int64_t, MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::DataCopy(dstAddr, reduceSumReg, maskOne);
    }
}

template <typename T, bool isInterleaved, bool isGroup>
class ClippedSwigluKernel {
public:
    __aicore__ inline ClippedSwigluKernel(const ClippedSwigluArch35TilingData* tilingData, TPipe* pipe)
        : tiling_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeTiling();
    __aicore__ inline void UbDataCopy(__local_mem__ T* inAddr, __local_mem__ T* outAddr, int64_t onceNum);
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t count, int64_t blockLen);
    __aicore__ inline void ComputeVfSwiglu(__local_mem__ T* x1UbAddr, __local_mem__ T* x2UbAddr,
                                           __local_mem__ T* swigluUbAddr, int64_t dim0OnceSize, int64_t dim1OnceSize,
                                           int64_t alignDim1In);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t count, int64_t blockLen);

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<T> yGm_;
    const ClippedSwigluArch35TilingData* tiling_;
    TPipe* pipe_;

    TQue<QuePosition::VECIN, 1> inQueX_;
    TQue<QuePosition::VECOUT, 1> outQueY_;
    TBuf<QuePosition::VECCALC> vectorBuf_;
    TBuf<QuePosition::VECCALC> reduceSumBuf_;

    uint32_t blockIdx_ = 0;
    int64_t dimH_ = 0;
    int64_t dimB_ = 0;
    int64_t hUbFactor_ = 0;
    int64_t bUbFactor_ = 0;

    int64_t realCoreNum_ = 0;
    int64_t hPreBlockNum_ = 0; // 每个核处理多少个数，注意尾核
    int64_t hLoopTimes_ = 0;   // 每个核 h方向循环多少次
    int64_t hTailNum_ = 0;     // 最后一次ub循环处理的数据量
    int64_t bPreBlockNum_ = 0; // 每个核处理多少个数，注意尾核
    int64_t bLoopTimes_ = 0;   // 每个核 h方向循环多少次
    int64_t bTailNum_ = 0;     // 最后一次ub循环处理的数据量
    int64_t hBlockFactor_ = 0;
    int64_t bBlockFactor_ = 0;
    int64_t hCore_ = 1;
    int64_t bCore_ = 0;
    int64_t vfLenT_ = 0;
    int64_t bBlockIdx_ = 0;
    int64_t hBlockIdx_ = 0;
    float limit_ = 0.0f;
    float alpha_ = 0.0f;
    float bias_ = 0.0f;
};

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluKernel<T, isInterleaved, isGroup>::Init(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    realCoreNum_ = tiling_->realCoreNum;
    dimH_ = tiling_->dimH;
    hBlockFactor_ = dimH_;
    vfLenT_ = Ops::Base::GetVRegSize() / sizeof(T);
    hUbFactor_ = tiling_->hUbFactor;
    bUbFactor_ = tiling_->bUbFactor;
    limit_ = tiling_->gluLimit;
    alpha_ = tiling_->gluAlpha;
    bias_ = tiling_->gluBias;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    if constexpr (isGroup) {
        groupIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupIndex));
    }
    int64_t ub = hUbFactor_ * bUbFactor_ * sizeof(T);
    pipe_->InitBuffer(inQueX_, DB_BUFFER, ub * DIM_HALVE);
    pipe_->InitBuffer(outQueY_, DB_BUFFER, ub);
    pipe_->InitBuffer(vectorBuf_, ub);
    if constexpr (isGroup) {
        pipe_->InitBuffer(reduceSumBuf_, BLOCK_SIZE);
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluKernel<T, isInterleaved, isGroup>::ComputeTiling()
{
    int64_t dimBatchSize = tiling_->dimBatchSize;
    if constexpr (isGroup) {
        int64_t groupNum = tiling_->groupNum;
        LocalTensor<int64_t> reduceSumUb = reduceSumBuf_.Get<int64_t>();
        LocalTensor<int64_t> groupUb = inQueX_.AllocTensor<int64_t>();
        DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
        DataCopyPadExtParams<int64_t> padParams = {false, 0, 0, 0};
        copyParams.blockLen = groupNum * sizeof(int64_t);
        DataCopyPad(groupUb, groupIndexGm_, copyParams, padParams);
        inQueX_.EnQue(groupUb);
        groupUb = inQueX_.DeQue<int64_t>();
        ReduceAllVf(reduceSumUb, groupUb, groupNum);
        inQueX_.FreeTensor(groupUb);
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        dimB_ = static_cast<int64_t>(reduceSumUb.GetValue(0));
        dimB_ = dimB_ < dimBatchSize ? dimB_ : dimBatchSize;
    } else {
        dimB_ = dimBatchSize;
    }

    if constexpr (isInterleaved) {
        int64_t pairTotal = dimH_ * dimB_;
        hBlockFactor_ = (pairTotal + realCoreNum_ - 1) / realCoreNum_;
        realCoreNum_ = (pairTotal + hBlockFactor_ - 1) / hBlockFactor_;
        int64_t tailBlockNum = pairTotal - hBlockFactor_ * (realCoreNum_ - 1);
        hPreBlockNum_ = blockIdx_ == (realCoreNum_ - 1) ? tailBlockNum : hBlockFactor_;
        hLoopTimes_ = (hPreBlockNum_ + hUbFactor_ - 1) / hUbFactor_;
        hTailNum_ = hPreBlockNum_ - hUbFactor_ * (hLoopTimes_ - 1);
    } else {
        bBlockFactor_ = (dimB_ + realCoreNum_ - 1) / realCoreNum_;
        bCore_ = (dimB_ + bBlockFactor_ - 1) / bBlockFactor_;
        int64_t core = realCoreNum_ / bCore_;
        if (core > 1) {
            hBlockFactor_ = (dimH_ + core - 1) / core;
            hCore_ = (dimH_ + hBlockFactor_ - 1) / hBlockFactor_;
        }
        realCoreNum_ = bCore_ * hCore_;
        bBlockIdx_ = blockIdx_ / hCore_;
        hBlockIdx_ = blockIdx_ % hCore_;

        int64_t tailHcoreNum = dimH_ - hBlockFactor_ * (hCore_ - 1);
        hPreBlockNum_ = hBlockIdx_ == (hCore_ - 1) ? tailHcoreNum : hBlockFactor_;
        hLoopTimes_ = (hPreBlockNum_ + hUbFactor_ - 1) / hUbFactor_;
        hTailNum_ = hPreBlockNum_ - hUbFactor_ * (hLoopTimes_ - 1);

        int64_t tailBcoreNum = dimB_ - bBlockFactor_ * (bCore_ - 1);
        bPreBlockNum_ = bBlockIdx_ == (bCore_ - 1) ? tailBcoreNum : bBlockFactor_;
        bLoopTimes_ = (bPreBlockNum_ + bUbFactor_ - 1) / bUbFactor_;
        bTailNum_ = bPreBlockNum_ - bUbFactor_ * (bLoopTimes_ - 1);
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluKernel<T, isInterleaved, isGroup>::Process()
{
    ComputeTiling();
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    LocalTensor<T> swigluUb = vectorBuf_.Get<T>();
    int64_t oneBlockNum = BLOCK_SIZE / sizeof(T);
    if constexpr (isInterleaved) {
        int64_t blockOffset = blockIdx_ * hBlockFactor_;
        for (int64_t loopIdx = 0; loopIdx < hLoopTimes_; ++loopIdx) {
            int64_t onceNum = loopIdx == (hLoopTimes_ - 1) ? hTailNum_ : hUbFactor_;
            int64_t outGmOffset = blockOffset + loopIdx * hUbFactor_;
            int64_t inGmOffset = outGmOffset * DIM_HALVE;
            CopyIn(inGmOffset, 1, onceNum * DIM_HALVE);
            LocalTensor<T> inputUb = inQueX_.DeQue<T>();
            auto actAddr = (__ubuf__ T*)inputUb.GetPhyAddr();
            auto gateAddr = (__ubuf__ T*)inputUb[VF_LEN_FP32].GetPhyAddr();
            auto swigluAddr = (__ubuf__ T*)swigluUb.GetPhyAddr();
            int64_t alignDim1In = ((onceNum * DIM_HALVE + oneBlockNum - 1) / oneBlockNum) * oneBlockNum;
            ComputeVfSwiglu(actAddr, gateAddr, swigluAddr, 1, onceNum, alignDim1In);
            inQueX_.FreeTensor(inputUb);
            LocalTensor<T> outUb = outQueY_.AllocTensor<T>();
            auto outUbAddr = (__ubuf__ T*)outUb.GetPhyAddr();
            UbDataCopy(swigluAddr, outUbAddr, onceNum);
            outQueY_.EnQue(outUb);
            CopyOut(outGmOffset, 1, onceNum);
        }
    } else {
        int64_t bCoreOffset = bBlockFactor_ * bBlockIdx_;
        int64_t hCoreOffset = hBlockFactor_ * hBlockIdx_;
        for (int64_t bi = 0; bi < bLoopTimes_; bi++) {
            int64_t onceBNum = bi == (bLoopTimes_ - 1) ? bTailNum_ : bUbFactor_;
            for (int64_t hi = 0; hi < hLoopTimes_; hi++) {
                int64_t onceHNum = hi == (hLoopTimes_ - 1) ? hTailNum_ : hUbFactor_;
                int64_t alignDim1In = ((onceHNum + oneBlockNum - 1) / oneBlockNum) * oneBlockNum;
                int64_t inGmOffset = (bCoreOffset + bi * bUbFactor_) * dimH_ * DIM_HALVE + hCoreOffset +
                                     hi * hUbFactor_;
                CopyIn(inGmOffset, onceBNum, onceHNum);
                LocalTensor<T> inputUb = inQueX_.DeQue<T>();
                auto actAddr = (__ubuf__ T*)inputUb.GetPhyAddr();
                auto gateAddr = (__ubuf__ T*)inputUb[hUbFactor_ * bUbFactor_].GetPhyAddr();
                auto swigluAddr = (__ubuf__ T*)swigluUb.GetPhyAddr();
                ComputeVfSwiglu(actAddr, gateAddr, swigluAddr, onceBNum, onceHNum, alignDim1In);
                inQueX_.FreeTensor(inputUb);
                LocalTensor<T> outUb = outQueY_.AllocTensor<T>();
                auto outUbAddr = (__ubuf__ T*)outUb.GetPhyAddr();
                UbDataCopy(swigluAddr, outUbAddr, onceBNum * alignDim1In);
                outQueY_.EnQue(outUb);
                int64_t outGmOffset = (bCoreOffset + bi * bUbFactor_) * dimH_ + hCoreOffset + hi * hUbFactor_;
                CopyOut(outGmOffset, onceBNum, onceHNum);
            }
        }
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluKernel<T, isInterleaved, isGroup>::CopyIn(int64_t gmOffset, int64_t count,
                                                                              int64_t blockLen)
{
    LocalTensor<T> xDTypeUb = inQueX_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    if constexpr (isInterleaved) {
        copyParams.blockCount = 1;
        copyParams.blockLen = blockLen * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(xDTypeUb, xGm_[gmOffset], copyParams, padParams);
    } else {
        copyParams.blockCount = count;
        copyParams.blockLen = blockLen * sizeof(T);
        copyParams.srcStride = (dimH_ * DIM_HALVE - blockLen) * sizeof(T);
        copyParams.dstStride = 0;
        DataCopyPad(xDTypeUb, xGm_[gmOffset], copyParams, padParams);
        DataCopyPad(xDTypeUb[bUbFactor_ * hUbFactor_], xGm_[gmOffset + dimH_], copyParams, padParams);
    }
    inQueX_.EnQue(xDTypeUb);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluKernel<T, isInterleaved, isGroup>::UbDataCopy(__local_mem__ T* inAddr,
                                                                                  __local_mem__ T* outAddr,
                                                                                  int64_t onceNum)
{
    uint32_t size = onceNum;
    uint32_t vfLen = vfLenT_;
    uint16_t times = CeilDivision(size, vfLen);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xReg;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < times; i++) {
            mask = MicroAPI::UpdateMask<T>(size);
            AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<T>(i, vfLen);
            AscendC::MicroAPI::DataCopy(xReg, inAddr, srcIdxOffset);
            AscendC::MicroAPI::DataCopy(outAddr, xReg, srcIdxOffset, mask);
        }
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluKernel<T, isInterleaved, isGroup>::ComputeVfSwiglu(
    __local_mem__ T* x1UbAddr, __local_mem__ T* x2UbAddr, __local_mem__ T* swigluUbAddr, int64_t dim0OnceSize,
    int64_t dim1OnceSize, int64_t alignDim1In)
{
    float clampLimit = limit_;
    float negClampLimit = -limit_;
    float negAlpha = -alpha_;
    float gluBias = bias_;
    uint16_t dim0VfTimes = 1;
    float scalarOne = 1.0f;
    uint32_t vfLen = VF_LEN_FP32 * DIM_HALVE;
    if constexpr (!isInterleaved) {
        dim0VfTimes = dim0OnceSize;
        vfLen = VF_LEN_FP32;
    }
    int64_t oneBlockNum = BLOCK_SIZE / sizeof(T);
    int64_t alignDim1Out = 0;
    if constexpr (isInterleaved) {
        alignDim1Out = ((dim1OnceSize + oneBlockNum - 1) / oneBlockNum) * oneBlockNum;
    } else {
        alignDim1Out = alignDim1In;
    }
    uint16_t dim1VfTimes = dim1OnceSize / VF_LEN_FP32;
    uint32_t tail = dim1OnceSize % VF_LEN_FP32;
    uint16_t tailTimes = 0;
    if (tail > 0) {
        tailTimes = 1;
    }
    __local_mem__ T* x1UbAddrT = x1UbAddr + dim1VfTimes * vfLen;
    __local_mem__ T* x2UbAddrT = x2UbAddr + dim1VfTimes * vfLen;
    __local_mem__ T* swigluUbAddrT = swigluUbAddr + dim1VfTimes * VF_LEN_FP32;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vregX1;
        AscendC::MicroAPI::RegTensor<T> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;

        AscendC::MicroAPI::RegTensor<float> vregX1DeF;
        AscendC::MicroAPI::RegTensor<float> vregX2DeF;
        AscendC::MicroAPI::RegTensor<float> minsReg;
        AscendC::MicroAPI::RegTensor<float> mulsReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<T> outTReg;
        MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskT = MicroAPI::UpdateMask<float>(tail);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<T>(
                    dim0vfLoopIdx, alignDim1In, dim1vfLoopIdx, vfLen);
                if constexpr (isInterleaved) {
                    if constexpr (sizeof(T) == sizeof(half)) {
                        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddr,
                                                                                                     srcIdxOffset);
                        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddr,
                                                                                                     srcIdxOffset);
                        AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX1F, vregX1, mask);
                        AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX2F, vregX2, mask);
                    } else {
                        // float
                        AscendC::MicroAPI::DataCopy((MicroAPI::RegTensor<T>&)vregX1F, x1UbAddr, srcIdxOffset);
                        AscendC::MicroAPI::DataCopy((MicroAPI::RegTensor<T>&)vregX2F, x2UbAddr, srcIdxOffset);
                    }
                    AscendC::MicroAPI::DeInterleave(vregX1DeF, vregX2DeF, vregX1F, vregX2F);
                } else {
                    if constexpr (sizeof(T) == sizeof(half)) {
                        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddr,
                                                                                                     srcIdxOffset);
                        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddr,
                                                                                                     srcIdxOffset);
                        AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX1DeF, vregX1, mask);
                        AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX2DeF, vregX2, mask);
                    } else {
                        // float
                        AscendC::MicroAPI::DataCopy((MicroAPI::RegTensor<T>&)vregX1DeF, x1UbAddr, srcIdxOffset);
                        AscendC::MicroAPI::DataCopy((MicroAPI::RegTensor<T>&)vregX2DeF, x2UbAddr, srcIdxOffset);
                    }
                }
                AscendC::MicroAPI::Mins(minsReg, vregX1DeF, clampLimit, mask);
                AscendC::MicroAPI::Muls(mulsReg, minsReg, negAlpha, mask);
                AscendC::MicroAPI::Exp(expReg, mulsReg, mask);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);
                AscendC::MicroAPI::Div(sigmoidReg, minsReg, addsReg, mask);

                AscendC::MicroAPI::Mins(vregX2DeF, vregX2DeF, clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2DeF, vregX2DeF, negClampLimit, mask);
                AscendC::MicroAPI::Adds(vregX2DeF, vregX2DeF, gluBias, mask);

                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2DeF, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out,
                                                                                           dim1vfLoopIdx, VF_LEN_FP32);
                if constexpr (sizeof(T) == sizeof(half)) {
                    AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                    DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swigluUbAddr, outTReg, outOffset, mask);
                } else {
                    DataCopy(swigluUbAddr, (MicroAPI::RegTensor<T>&)outFReg, outOffset, mask);
                }
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1In);
            AscendC::MicroAPI::AddrReg outOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out);
            for (uint16_t ti = 0; ti < tailTimes; ti++) {
                if constexpr (isInterleaved) {
                    if constexpr (sizeof(T) == sizeof(half)) {
                        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddrT,
                                                                                                     srcIdxOffset1);
                        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddrT,
                                                                                                     srcIdxOffset1);
                        AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX1F, vregX1, mask);
                        AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX2F, vregX2, mask);
                    } else {
                        // float
                        AscendC::MicroAPI::DataCopy((MicroAPI::RegTensor<T>&)vregX1F, x1UbAddrT, srcIdxOffset1);
                        AscendC::MicroAPI::DataCopy((MicroAPI::RegTensor<T>&)vregX2F, x2UbAddrT, srcIdxOffset1);
                    }
                    AscendC::MicroAPI::DeInterleave(vregX1DeF, vregX2DeF, vregX1F, vregX2F);
                } else {
                    if constexpr (sizeof(T) == sizeof(half)) {
                        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddrT,
                                                                                                     srcIdxOffset1);
                        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddrT,
                                                                                                     srcIdxOffset1);
                        AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX1DeF, vregX1, mask);
                        AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX2DeF, vregX2, mask);
                    } else {
                        // float
                        AscendC::MicroAPI::DataCopy((MicroAPI::RegTensor<T>&)vregX1DeF, x1UbAddrT, srcIdxOffset1);
                        AscendC::MicroAPI::DataCopy((MicroAPI::RegTensor<T>&)vregX2DeF, x2UbAddrT, srcIdxOffset1);
                    }
                }
                AscendC::MicroAPI::Mins(minsReg, vregX1DeF, clampLimit, maskT);
                AscendC::MicroAPI::Muls(mulsReg, minsReg, negAlpha, maskT);
                AscendC::MicroAPI::Exp(expReg, mulsReg, maskT);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, maskT);
                AscendC::MicroAPI::Div(sigmoidReg, minsReg, addsReg, maskT);

                AscendC::MicroAPI::Mins(vregX2DeF, vregX2DeF, clampLimit, maskT);
                AscendC::MicroAPI::Maxs(vregX2DeF, vregX2DeF, negClampLimit, maskT);
                AscendC::MicroAPI::Adds(vregX2DeF, vregX2DeF, gluBias, maskT);

                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2DeF, maskT);
                if constexpr (sizeof(T) == sizeof(half)) {
                    AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, maskT);
                    DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swigluUbAddrT, outTReg, outOffset1, maskT);
                } else {
                    DataCopy(swigluUbAddrT, (MicroAPI::RegTensor<T>&)outFReg, outOffset1, maskT);
                }
            }
        }
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluKernel<T, isInterleaved, isGroup>::CopyOut(int64_t gmOffset, int64_t count,
                                                                               int64_t blockLen)
{
    LocalTensor<T> outputUb = outQueY_.DeQue<T>();
    outQueY_.EnQue(outputUb);

    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    if constexpr (isInterleaved) {
        copyParams.blockCount = 1;
        copyParams.dstStride = 0;
    } else {
        copyParams.blockCount = count;
        copyParams.dstStride = (dimH_ - blockLen) * sizeof(T);
    }
    copyParams.blockLen = blockLen * sizeof(T);
    copyParams.srcStride = 0;

    DataCopyPad(yGm_[gmOffset], outputUb, copyParams);
    outQueY_.FreeTensor(outputUb);
}
} // namespace ClippedSwigluOp

#endif // CLIPPED_SWIGLU_KERNEL_H
