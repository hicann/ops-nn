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
 * \file cla_gate_backward.h
 * \brief
 */

#ifndef CLA_GATE_BACKWARD_ARCH35_H_
#define CLA_GATE_BACKWARD_ARCH35_H_

#include "kernel_operator.h"
#include "cla_gate_backward_tiling_data.h"

namespace ClaGateBackwardOps {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t SINGLE_BUFFER = 1;
constexpr int64_t VEC_LANES_FP32 = 64;
constexpr int64_t MIN_REDUCE_TMP_BYTES = 32;

constexpr static Reg::CastTrait castTraitIn = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                               RoundMode::UNKNOWN};
constexpr static Reg::CastTrait castTraitOut = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                                RoundMode::CAST_RINT};

__aicore__ inline int64_t MaxI(int64_t a, int64_t b) { return a > b ? a : b; }

__aicore__ inline int64_t MinI(int64_t a, int64_t b) { return a < b ? a : b; }

__aicore__ inline int64_t AlignUpI(int64_t value, int64_t align) { return (value + align - 1) / align * align; }

template <typename T>
class ClaGateBackwardKernel {
public:
    __aicore__ inline ClaGateBackwardKernel() = default;

    __aicore__ inline void Init(GM_ADDR gradMerged, GM_ADDR globalAttn, GM_ADDR localAttn, GM_ADDR globalGateLogits,
                                GM_ADDR localGateLogits, GM_ADDR gradGlobalAttnOut, GM_ADDR gradLocalAttnOut,
                                GM_ADDR gradGlobalGateLogitsOut, GM_ADDR gradLocalGateLogitsOut,
                                const ClaGateBackwardTilingData* tilingData, TPipe* pipe)
    {
        tilingData_ = tilingData;
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        headDim_ = tilingData_->headDim;
        batch_ = tilingData_->batch;

        const int64_t tndTotal = tilingData_->totalHeads * headDim_;
        const int64_t gateTotal = tilingData_->totalHeads;

        gradGm_.SetGlobalBuffer((__gm__ T*)gradMerged, tndTotal);
        globalAttnGm_.SetGlobalBuffer((__gm__ T*)globalAttn, tndTotal);
        localAttnGm_.SetGlobalBuffer((__gm__ T*)localAttn, tndTotal);
        globalLogitsGm_.SetGlobalBuffer((__gm__ T*)globalGateLogits, gateTotal);
        localLogitsGm_.SetGlobalBuffer((__gm__ T*)localGateLogits, gateTotal);
        gradGlobalAttnOutGm_.SetGlobalBuffer((__gm__ T*)gradGlobalAttnOut, tndTotal);
        gradLocalAttnOutGm_.SetGlobalBuffer((__gm__ T*)gradLocalAttnOut, tndTotal);
        gradGlobalGateLogitsOutGm_.SetGlobalBuffer((__gm__ T*)gradGlobalGateLogitsOut, gateTotal);
        gradLocalGateLogitsOutGm_.SetGlobalBuffer((__gm__ T*)gradLocalGateLogitsOut, gateTotal);

        tndHalfElems_ = batch_ * headDim_;
        const int64_t tndHalfBytes = tndHalfElems_ * (int64_t)sizeof(T);
        const int64_t batchVecElems = AlignUpI(batch_, VEC_LANES_FP32);
        const int64_t zHalfBytes = batchVecElems * (int64_t)sizeof(T);
        zHalfElems_ = zHalfBytes / (int64_t)sizeof(T);
        const int64_t fp32BatchBytes = batchVecElems * (int64_t)sizeof(float);
        const int64_t tndFp32BatchBytes = batch_ * headDim_ * (int64_t)sizeof(float);

        pipe->InitBuffer(inQueGradMerge_, DB_BUFFER, tndHalfBytes);
        pipe->InitBuffer(inQueO_, DB_BUFFER, tndHalfBytes * 2);
        pipe->InitBuffer(inQueZ_, SINGLE_BUFFER, zHalfBytes * 2);
        pipe->InitBuffer(outQueGradO_, DB_BUFFER, tndHalfBytes * 2);
        pipe->InitBuffer(outQueGradZ_, SINGLE_BUFFER, zHalfBytes * 2);

        pipe->InitBuffer(gradMergeFp32_, tndFp32BatchBytes); // 整 batch cast 结果
        pipe->InitBuffer(outFp32_, tndFp32BatchBytes);       // Og/Ol cast 与 G*O 串行复用
        pipe->InitBuffer(sigmoidGBuf_, fp32BatchBytes);      // s_g
        pipe->InitBuffer(sigmoidLBuf_, fp32BatchBytes);      // s_l
        pipe->InitBuffer(reduceGBuf_, fp32BatchBytes);       // Σ_d G*Og
        pipe->InitBuffer(reduceLBuf_, fp32BatchBytes);       // Σ_d G*Ol
        reduceTmpBytes_ = AlignUpI(MaxI(tilingData_->reduceTmpSize, MIN_REDUCE_TMP_BYTES), MIN_REDUCE_TMP_BYTES);
        pipe->InitBuffer(reduceTmpBuf_, reduceTmpBytes_);
    }

    __aicore__ inline void Process()
    {
        const int64_t usedCoreNum = tilingData_->usedCoreNum;
        const int64_t baseCoreHeads = tilingData_->baseCoreHeads;
        const int64_t extraCoreCount = tilingData_->extraCoreCount;
        if (blockIdx_ >= usedCoreNum) {
            return;
        }
        const int64_t prefixExtra = MinI(blockIdx_, extraCoreCount);
        const int64_t coreStart = blockIdx_ * baseCoreHeads + prefixExtra;
        const int64_t coreHeads = baseCoreHeads + (blockIdx_ < extraCoreCount ? 1 : 0);
        if (coreHeads <= 0 || batch_ <= 0) {
            return;
        }

        const bool isHeadCore = blockIdx_ < extraCoreCount;
        const int64_t loopCount = isHeadCore ? tilingData_->headCoreLoopCount : tilingData_->tailCoreLoopCount;
        const int64_t perLoop = isHeadCore ? tilingData_->headCoreHeadsPerLoop : tilingData_->tailCoreHeadsPerLoop;
        if (loopCount <= 0 || perLoop <= 0) {
            return;
        }

        int64_t off = coreStart;
        for (int64_t i = 0; i < loopCount; ++i) {
            const int64_t cur = MinI(perLoop, coreHeads - i * perLoop);
            if (cur <= 0) {
                break;
            }
            CopyIn(off, cur);
            ComputeGradO(cur);
            CopyOutGradO(off, cur);
            ComputeGradZ(cur);
            CopyOutGradZ(off, cur);
            off += cur;
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t headOffset, int64_t curHeads)
    {
        CopyInTnd(headOffset, curHeads);
        CopyInZ(headOffset, curHeads);
    }

    __aicore__ inline void CopyInTnd(int64_t headOffset, int64_t curHeads)
    {
        const int64_t tndOffset = headOffset * headDim_;
        const uint32_t tndBytes = static_cast<uint32_t>(curHeads * headDim_ * (int64_t)sizeof(T));
        DataCopyExtParams tndParams{1, tndBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> gradLocal = inQueGradMerge_.AllocTensor<T>();
        LocalTensor<T> outInput = inQueO_.AllocTensor<T>();
        LocalTensor<T> globalAttnLocal = outInput;
        LocalTensor<T> localAttnLocal = outInput[tndHalfElems_];
        DataCopyPad(gradLocal, gradGm_[tndOffset], tndParams, padParams);
        DataCopyPad(globalAttnLocal, globalAttnGm_[tndOffset], tndParams, padParams);
        DataCopyPad(localAttnLocal, localAttnGm_[tndOffset], tndParams, padParams);
        inQueGradMerge_.EnQue(gradLocal);
        inQueO_.EnQue(outInput);
    }

    __aicore__ inline void CopyInZ(int64_t headOffset, int64_t curHeads)
    {
        const uint32_t zBytes = static_cast<uint32_t>(curHeads * (int64_t)sizeof(T));
        DataCopyExtParams zParams{1, zBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> zInput = inQueZ_.AllocTensor<T>();
        LocalTensor<T> globalLogitsLocal = zInput;
        LocalTensor<T> localLogitsLocal = zInput[zHalfElems_];
        DataCopyPad(globalLogitsLocal, globalLogitsGm_[headOffset], zParams, padParams);
        DataCopyPad(localLogitsLocal, localLogitsGm_[headOffset], zParams, padParams);
        inQueZ_.EnQue(zInput);
    }

    __aicore__ inline void ComputeGradO(int64_t curBatch)
    {
        LocalTensor<T> gradLocal = inQueGradMerge_.DeQue<T>();
        LocalTensor<T> zInput = inQueZ_.DeQue<T>();
        LocalTensor<T> globalLogitsLocal = zInput;
        LocalTensor<T> localLogitsLocal = zInput[zHalfElems_];

        LocalTensor<T> dOut = outQueGradO_.AllocTensor<T>();
        LocalTensor<T> gradGlobalAttnOutLocal = dOut;
        LocalTensor<T> gradLocalAttnOutLocal = dOut[tndHalfElems_];

        LocalTensor<float> gradFp32 = gradMergeFp32_.Get<float>();
        LocalTensor<float> sigmoidG = sigmoidGBuf_.Get<float>();
        LocalTensor<float> sigmoidL = sigmoidLBuf_.Get<float>();

        Cast(gradFp32, gradLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(curBatch * headDim_));
        inQueGradMerge_.FreeTensor(gradLocal);
        SigmoidImpl(sigmoidG, sigmoidL, globalLogitsLocal, localLogitsLocal, static_cast<uint32_t>(curBatch));
        inQueZ_.FreeTensor(zInput);

        BroadcastMulCast(gradGlobalAttnOutLocal, gradFp32, sigmoidG, curBatch);
        BroadcastMulCast(gradLocalAttnOutLocal, gradFp32, sigmoidL, curBatch);
        outQueGradO_.EnQue(dOut);
    }

    __aicore__ inline void ComputeGradZ(int64_t curBatch)
    {
        LocalTensor<T> outInput = inQueO_.DeQue<T>();
        LocalTensor<T> globalAttnLocal = outInput;
        LocalTensor<T> localAttnLocal = outInput[tndHalfElems_];

        LocalTensor<T> dZOut = outQueGradZ_.AllocTensor<T>();
        LocalTensor<T> gradGlobalGateLogitsOutLocal = dZOut;
        LocalTensor<T> gradLocalGateLogitsOutLocal = dZOut[zHalfElems_];

        LocalTensor<float> gradFp32 = gradMergeFp32_.Get<float>();
        LocalTensor<float> outFp32 = outFp32_.Get<float>();
        LocalTensor<float> sigmoidG = sigmoidGBuf_.Get<float>();
        LocalTensor<float> sigmoidL = sigmoidLBuf_.Get<float>();
        LocalTensor<float> reduceG = reduceGBuf_.Get<float>();
        LocalTensor<float> reduceL = reduceLBuf_.Get<float>();

        LocalTensor<uint8_t> workBuf = reduceTmpBuf_.Get<uint8_t>();
        const uint32_t cntElems = static_cast<uint32_t>(curBatch * headDim_);
        uint32_t srcShape[2] = {static_cast<uint32_t>(curBatch), static_cast<uint32_t>(headDim_)};

        // g 路
        Cast(outFp32, globalAttnLocal, RoundMode::CAST_NONE, cntElems);
        Mul(outFp32, outFp32, gradFp32, cntElems);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(reduceG, outFp32, workBuf, srcShape, false);

        // l 路
        Cast(outFp32, localAttnLocal, RoundMode::CAST_NONE, cntElems);
        inQueO_.FreeTensor(outInput);
        Mul(outFp32, outFp32, gradFp32, cntElems);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(reduceL, outFp32, workBuf, srcShape, false);

        WriteGateGrad(gradGlobalGateLogitsOutLocal, gradLocalGateLogitsOutLocal, sigmoidG, sigmoidL, reduceG, reduceL,
                      static_cast<uint32_t>(curBatch));
        outQueGradZ_.EnQue(dZOut);
    }

    __aicore__ inline void CopyOutGradO(int64_t headOffset, int64_t curHeads)
    {
        LocalTensor<T> dOut = outQueGradO_.DeQue<T>();
        const int64_t tndOffset = headOffset * headDim_;
        const uint32_t tndBytes = static_cast<uint32_t>(curHeads * headDim_ * (int64_t)sizeof(T));
        DataCopyExtParams tndParams{1, tndBytes, 0, 0, 0};
        DataCopyPad(gradGlobalAttnOutGm_[tndOffset], dOut, tndParams);
        DataCopyPad(gradLocalAttnOutGm_[tndOffset], dOut[tndHalfElems_], tndParams);
        outQueGradO_.FreeTensor(dOut);
    }

    __aicore__ inline void CopyOutGradZ(int64_t headOffset, int64_t curHeads)
    {
        LocalTensor<T> dZOut = outQueGradZ_.DeQue<T>();
        const uint32_t zBytes = static_cast<uint32_t>(curHeads * (int64_t)sizeof(T));
        DataCopyExtParams zParams{1, zBytes, 0, 0, 0};
        DataCopyPad(gradGlobalGateLogitsOutGm_[headOffset], dZOut, zParams);
        DataCopyPad(gradLocalGateLogitsOutGm_[headOffset], dZOut[zHalfElems_], zParams);
        outQueGradZ_.FreeTensor(dZOut);
    }

    __aicore__ inline void SigmoidImpl(LocalTensor<float>& sigmoidG, LocalTensor<float>& sigmoidL,
                                       const LocalTensor<T>& globalLogitsLocal, const LocalTensor<T>& localLogitsLocal,
                                       uint32_t logitsCount)
    {
        constexpr uint32_t vecLanes = VECTOR_REG_WIDTH / sizeof(float);
        const uint16_t vfLoopNum = CeilDivision(logitsCount, vecLanes);
        __ubuf__ T* gAddr = (__ubuf__ T*)globalLogitsLocal.GetPhyAddr();
        __ubuf__ T* lAddr = (__ubuf__ T*)localLogitsLocal.GetPhyAddr();
        __ubuf__ float* sgAddr = (__ubuf__ float*)sigmoidG.GetPhyAddr();
        __ubuf__ float* slAddr = (__ubuf__ float*)sigmoidL.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vregIn;
            Reg::RegTensor<float> vregCast;
            Reg::RegTensor<float> vregNeg;
            Reg::RegTensor<float> vregExp;
            Reg::RegTensor<float> vregDenom;
            Reg::RegTensor<float> vregOne;
            Reg::RegTensor<float> vregOut;
            Reg::MaskReg preg;

            uint32_t remain = logitsCount;
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg = Reg::UpdateMask<float>(remain);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(vregIn, gAddr + i * vecLanes);
                Reg::Cast<float, T, castTraitIn>(vregCast, vregIn, preg);
                Reg::Muls(vregNeg, vregCast, static_cast<float>(-1), preg);
                Reg::Exp(vregExp, vregNeg, preg);
                Reg::Adds(vregDenom, vregExp, static_cast<float>(1), preg);
                Reg::Duplicate(vregOne, static_cast<float>(1), preg);
                Reg::Div(vregOut, vregOne, vregDenom, preg);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(sgAddr + i * vecLanes, vregOut, preg);
            }

            remain = logitsCount;
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg = Reg::UpdateMask<float>(remain);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(vregIn, lAddr + i * vecLanes);
                Reg::Cast<float, T, castTraitIn>(vregCast, vregIn, preg);
                Reg::Muls(vregNeg, vregCast, static_cast<float>(-1), preg);
                Reg::Exp(vregExp, vregNeg, preg);
                Reg::Adds(vregDenom, vregExp, static_cast<float>(1), preg);
                Reg::Duplicate(vregOne, static_cast<float>(1), preg);
                Reg::Div(vregOut, vregOne, vregDenom, preg);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(slAddr + i * vecLanes, vregOut, preg);
            }
        }
    }

    __aicore__ inline void BroadcastMulCast(LocalTensor<T>& dst, const LocalTensor<float>& grad,
                                            const LocalTensor<float>& sig, int64_t headCount)
    {
        constexpr uint32_t vecLanes = VECTOR_REG_WIDTH / sizeof(float);
        const uint16_t headCountLocal = static_cast<uint16_t>(headCount);
        const uint32_t headDimLocal = static_cast<uint32_t>(headDim_);
        const uint16_t dLoop = CeilDivision(headDimLocal, vecLanes);

        __ubuf__ float* sAddr = (__ubuf__ float*)sig.GetPhyAddr();
        __ubuf__ float* gradAddr = (__ubuf__ float*)grad.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> sReg;
            Reg::RegTensor<float> gReg;
            Reg::RegTensor<float> oReg;
            Reg::RegTensor<T> tReg;
            Reg::MaskReg preg;
            for (uint16_t h = 0; h < headCountLocal; ++h) {
                const uint32_t rowBase = static_cast<uint32_t>(h) * headDimLocal;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(sReg, sAddr + h);
                uint32_t rem = headDimLocal;
                for (uint16_t vf = 0; vf < dLoop; ++vf) {
                    preg = Reg::UpdateMask<float>(rem);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(gReg, gradAddr + rowBase + vf * vecLanes);
                    Reg::Mul(oReg, gReg, sReg, preg);
                    Reg::Cast<T, float, castTraitOut>(tReg, oReg, preg);
                    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dstAddr + rowBase + vf * vecLanes, tReg, preg);
                }
            }
        }
    }

    __aicore__ inline void WriteGateGrad(LocalTensor<T>& gradGlobalGateLogitsOut,
                                         LocalTensor<T>& gradLocalGateLogitsOut, const LocalTensor<float>& sigmoidG,
                                         const LocalTensor<float>& sigmoidL, const LocalTensor<float>& reduceG,
                                         const LocalTensor<float>& reduceL, uint32_t logitsCount)
    {
        constexpr uint32_t vecLanes = VECTOR_REG_WIDTH / sizeof(float);
        const uint16_t loopNum = CeilDivision(logitsCount, vecLanes);
        __ubuf__ float* sgAddr = (__ubuf__ float*)sigmoidG.GetPhyAddr();
        __ubuf__ float* slAddr = (__ubuf__ float*)sigmoidL.GetPhyAddr();
        __ubuf__ float* rgAddr = (__ubuf__ float*)reduceG.GetPhyAddr();
        __ubuf__ float* rlAddr = (__ubuf__ float*)reduceL.GetPhyAddr();
        __ubuf__ T* dgAddr = (__ubuf__ T*)gradGlobalGateLogitsOut.GetPhyAddr();
        __ubuf__ T* dlAddr = (__ubuf__ T*)gradLocalGateLogitsOut.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> sReg;
            Reg::RegTensor<float> rReg;
            Reg::RegTensor<float> tReg;
            Reg::RegTensor<T> outReg;
            Reg::MaskReg preg;

            uint32_t rem = logitsCount;
            for (uint16_t i = 0; i < loopNum; ++i) {
                preg = Reg::UpdateMask<float>(rem);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(sReg, sgAddr + i * vecLanes);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(rReg, rgAddr + i * vecLanes);
                Reg::Adds(tReg, sReg, static_cast<float>(-1), preg);
                Reg::Muls(tReg, tReg, static_cast<float>(-1), preg);
                Reg::Mul(tReg, tReg, sReg, preg);
                Reg::Mul(tReg, tReg, rReg, preg);
                Reg::Cast<T, float, castTraitOut>(outReg, tReg, preg);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dgAddr + i * vecLanes, outReg, preg);
            }

            rem = logitsCount;
            for (uint16_t i = 0; i < loopNum; ++i) {
                preg = Reg::UpdateMask<float>(rem);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(sReg, slAddr + i * vecLanes);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(rReg, rlAddr + i * vecLanes);
                Reg::Adds(tReg, sReg, static_cast<float>(-1), preg);
                Reg::Muls(tReg, tReg, static_cast<float>(-1), preg);
                Reg::Mul(tReg, tReg, sReg, preg);
                Reg::Mul(tReg, tReg, rReg, preg);
                Reg::Cast<T, float, castTraitOut>(outReg, tReg, preg);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dlAddr + i * vecLanes, outReg, preg);
            }
        }
    }

private:
    const ClaGateBackwardTilingData* tilingData_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t headDim_ = 0;
    int64_t batch_ = 0;          // 一次搬运/计算的最大 TN 数
    int64_t tndHalfElems_ = 0;   // TND 队列每半段元素数（上/下），也是上/下半段数组步长
    int64_t zHalfElems_ = 0;     // logits 队列每半段元素数（上/下）
    int64_t reduceTmpBytes_ = 0; // reduceTmpBuf_ 分配字节数（g/l 串行复用同一份）

    GlobalTensor<T> gradGm_;
    GlobalTensor<T> globalAttnGm_;
    GlobalTensor<T> localAttnGm_;
    GlobalTensor<T> globalLogitsGm_;
    GlobalTensor<T> localLogitsGm_;
    GlobalTensor<T> gradGlobalAttnOutGm_;
    GlobalTensor<T> gradLocalAttnOutGm_;
    GlobalTensor<T> gradGlobalGateLogitsOutGm_;
    GlobalTensor<T> gradLocalGateLogitsOutGm_;

    TQue<QuePosition::VECIN, 1> inQueGradMerge_;
    TQue<QuePosition::VECIN, 1> inQueO_;
    TQue<QuePosition::VECIN, 1> inQueZ_;

    TQue<QuePosition::VECOUT, 1> outQueGradO_;
    TQue<QuePosition::VECOUT, 1> outQueGradZ_;

    TBuf<QuePosition::VECCALC> gradMergeFp32_;
    TBuf<QuePosition::VECCALC> outFp32_;
    TBuf<QuePosition::VECCALC> sigmoidGBuf_;
    TBuf<QuePosition::VECCALC> sigmoidLBuf_;
    TBuf<QuePosition::VECCALC> reduceGBuf_;
    TBuf<QuePosition::VECCALC> reduceLBuf_;
    TBuf<QuePosition::VECCALC> reduceTmpBuf_;
};
} // namespace ClaGateBackwardOps

#endif // CLA_GATE_BACKWARD_ARCH35_H_
