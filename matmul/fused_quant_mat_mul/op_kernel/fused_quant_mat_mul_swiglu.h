/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_quant_mat_mul_swiglu.h
 * \brief
 */

#ifndef FUSED_QUANT_MAT_MUL_SWIGLU_H
#define FUSED_QUANT_MAT_MUL_SWIGLU_H

#include "arch35/fused_quant_mat_mul_tiling_data.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_base.h"

#define FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS                                                                                    \
    template <class x1Type, class x2Type, class scaleType, class biasType, class yType, CubeFormat formatX1,           \
              CubeFormat formatX2, CubeFormat formatY, bool aTrans, bool bTrans>
#define FUSED_SWIGLU_TEMPLATE_FUNC_PARAMS                                                                                     \
    x1Type, x2Type, scaleType, biasType, yType, formatX1, formatX2, formatY, aTrans, bTrans

namespace AscendC {

/* 预加载deqQuantScale模板，增量case性能会恶化，先不使用 */
constexpr MatmulConfigMode cfgMode = MatmulConfigMode::CONFIG_NORM;
constexpr MatmulBatchParams cfgParams{false, BatchMode::BATCH_LESS_THAN_L1, false, BatchOutMode::SINGLE_BATCH};
constexpr MatmulConfig SWIGLU_MM_CFG = GetMMConfig<cfgMode>(cfgParams);

FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS
class FusedQuantMatmulSwiglu {
public:
    __aicore__ inline FusedQuantMatmulSwiglu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR deQuantScale, GM_ADDR quantScale,
                                GM_ADDR y, GM_ADDR workSpace, const FusedQuantMatmulSwigluTilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

protected:
    GlobalTensor<x1Type> xGm;
    GlobalTensor<x2Type> wGm;
    GlobalTensor<biasType> bGm;
    GlobalTensor<uint64_t> deQuantScaleGm;
    GlobalTensor<float> quantScaleGm;
    GlobalTensor<yType> yGm;

    const FusedQuantMatmulSwigluTilingData* __restrict tilingData;

    TQue<QuePosition::A1, 1> l1AQueue;
    TQue<QuePosition::VECIN, 1> matmulLeftOutQueue;
    TQue<QuePosition::VECIN, 1> matmulRightOutQueue;
    TQue<QuePosition::VECIN, 1> quantScaleQueue;
    TBuf<TPosition::VECCALC> outBuf;

    constexpr static AscendC::MicroAPI::CastTrait castTraitHalf2Float = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT,
    };

    constexpr static AscendC::MicroAPI::CastTrait castTraitFloat2Half = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT,
    };

    constexpr static AscendC::MicroAPI::CastTrait castTraitHalf2Int8 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT,
    };

    // 典型shape全载A，非全载添加需要添加模板，暂不处理
    using aType = matmul::MatmulType<AscendC::TPosition::TSCM, formatX1, x1Type, aTrans>;
    using bType = matmul::MatmulType<AscendC::TPosition::GM, formatX2, x2Type, bTrans>;
    using cType = matmul::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, half>;
    using biasMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
    matmul::MatmulImpl<aType, bType, cType, biasMatmulType, CFG_MDL> mm;

    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t baseM;
    uint32_t baseN;

    uint32_t curBlockIdx;
    uint32_t singleM;
    uint32_t singleMTail;
    uint32_t singleN;
    uint32_t singleNTail;
    uint32_t mLoops;
    uint32_t nLoops;

    // mm1 geglu
    uint64_t xCoreOffset;
    uint64_t wCoreOffset;
    uint32_t curSingleM;
    uint32_t curSingleN;
    uint32_t mInnerLoops;
    uint32_t nInnerLoops;
    uint32_t aicMtail;
    uint32_t aicNtail;
    uint32_t singleTimeM;
    uint32_t singleTimeN;
    uint32_t isQuant;

    __aicore__ inline void initTilingData();
    __aicore__ inline void CalcParams();
    __aicore__ inline void Iterate(LocalTensor<half> &result);
    __aicore__ inline void SwishCompute(uint32_t size);
    __aicore__ inline void MulCompute(uint64_t offset, uint32_t curAicM, uint32_t curAicN);
    __aicore__ inline void MulQuantCompute(uint64_t offset, uint32_t curAicM, uint32_t curAicN);
    __aicore__ inline void CopyIn(LocalTensor<x1Type> &aL1Local, uint32_t curAicM, uint64_t xOffset);
    __aicore__ inline void CopyOut(uint64_t xOffset, uint64_t wOffset, uint32_t curAicM, uint32_t curAicN);
};

using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::TBuf;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS
__aicore__ inline void FusedQuantMatmulSwiglu<FUSED_SWIGLU_TEMPLATE_FUNC_PARAMS>::Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR deQuantScale, GM_ADDR quantScale,
                                                                                GM_ADDR y, GM_ADDR workSpace, const FusedQuantMatmulSwigluTilingData *tiling, TPipe *pipe)
{
    curBlockIdx = GetBlockIdx();
    tilingData = tiling;
    initTilingData();

    xGm.SetGlobalBuffer((__gm__ x1Type *)x);
    wGm.SetGlobalBuffer((__gm__ x2Type *)w);
    bGm.SetGlobalBuffer((__gm__ biasType *)b);
    deQuantScaleGm.SetGlobalBuffer((__gm__ scaleType *)deQuantScale);
    quantScaleGm.SetGlobalBuffer((__gm__ float*)quantScale);
    yGm.SetGlobalBuffer((__gm__ yType *)y);

    // vector without db, not a performance bottleneck, and calc more data
    pipe->InitBuffer(l1AQueue, 1, baseM * k * sizeof(x1Type));
    pipe->InitBuffer(matmulLeftOutQueue, 1, baseM * baseN * sizeof(half));
    pipe->InitBuffer(matmulRightOutQueue, 1, baseM * baseN * sizeof(half));
    pipe->InitBuffer(quantScaleQueue, 1, baseN * sizeof(float));
    pipe->InitBuffer(outBuf, baseM * baseN * sizeof(half));

    mm.Init(&tilingData->mmTilingData, pipe);
}

FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS
__aicore__ inline void FusedQuantMatmulSwiglu<FUSED_SWIGLU_TEMPLATE_FUNC_PARAMS>::initTilingData()
{
    m = tilingData->mmTilingData.M;
    k = tilingData->mmTilingData.Ka;
    n = tilingData->mmTilingData.N;

    baseM = tilingData->mmTilingData.baseM;
    baseN = tilingData->mmTilingData.baseN;

    singleM = tilingData->baseParams.singleCoreM;
    singleN = tilingData->baseParams.singleCoreN;
    mLoops = tilingData->baseParams.mLoops;
    nLoops = tilingData->baseParams.nLoops;
    singleMTail = tilingData->baseParams.singleMTail;
    singleNTail = tilingData->baseParams.singleNTail;
    singleTimeM = tilingData->mmTilingData.baseM;
    singleTimeN = tilingData->mmTilingData.baseN;
    isQuant = tilingData->baseParams.isQuant;
}

FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS
__aicore__ inline void FusedQuantMatmulSwiglu<FUSED_SWIGLU_TEMPLATE_FUNC_PARAMS>::CalcParams()
{
    uint32_t mCoreIndx = curBlockIdx / nLoops;
    uint32_t nCoreIndx = curBlockIdx % nLoops;

    xCoreOffset = static_cast<uint64_t>(mCoreIndx) * singleM;
    wCoreOffset = static_cast<uint64_t>(nCoreIndx) * singleN;
    curSingleM = mCoreIndx == mLoops - 1 ? singleMTail : singleM;
    curSingleN = nCoreIndx == nLoops - 1 ? singleNTail : singleN;

    mInnerLoops = DequantBmm::CeilDiv(curSingleM, singleTimeM);
    nInnerLoops = DequantBmm::CeilDiv(curSingleN, singleTimeN);
    aicMtail = curSingleM - (mInnerLoops - 1) * singleTimeM;
    aicNtail = curSingleN - (nInnerLoops - 1) * singleTimeN;
}

FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS
__aicore__ inline void FusedQuantMatmulSwiglu<FUSED_SWIGLU_TEMPLATE_FUNC_PARAMS>::SwishCompute(uint32_t size)
{
    auto buffer = matmulLeftOutQueue.DeQue<half>();
    auto swishRes = outBuf.Get<half>();
    uint32_t calSize = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vfLoopNum = (size + calSize - 1) / calSize;

    __VEC_SCOPE__
    {
        RegTensor<half> vreg0;
        RegTensor<float> vreg1;
        RegTensor<float> vreg2;
        RegTensor<float> vreg3;
        RegTensor<float> vreg4;
        RegTensor<float> vreg5;
        RegTensor<half> vreg6;
        MaskReg preg0;

        __local_mem__ half *bufferAddr = (__local_mem__ half *)buffer.GetPhyAddr();
        __local_mem__ half *swishResAddr = (__local_mem__ half *)swishRes.GetPhyAddr();

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = AscendC::MicroAPI::UpdateMask<float>(size);
            AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, bufferAddr + i * calSize);
            AscendC::MicroAPI::Cast<float, half, castTraitHalf2Float>(vreg1, vreg0, preg0);
            AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg2, vreg1, static_cast<float>(-1.0), preg0);
            AscendC::MicroAPI::Exp<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg3, vreg2, preg0);
            AscendC::MicroAPI::Adds<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg4, vreg3, static_cast<float>(1.0), preg0);
            AscendC::MicroAPI::Div<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg5, vreg1, vreg4, preg0);
            AscendC::MicroAPI::Cast<half, float, castTraitFloat2Half>(vreg6, vreg5, preg0);
            AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swishResAddr + i * calSize, vreg6, preg0);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS
__aicore__ inline void FusedQuantMatmulSwiglu<FUSED_SWIGLU_TEMPLATE_FUNC_PARAMS>::MulQuantCompute(uint64_t offset, uint32_t curAicM, uint32_t curAicN)
{
    auto scale = quantScaleQueue.AllocTensor<float>();
    auto left = outBuf.Get<half>();
    auto right = matmulRightOutQueue.DeQue<half>();
    uint32_t calSize = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vfLoopNum = (curAicN + calSize - 1) / calSize;

    DataCopy(scale, quantScaleGm[offset], curAicN);
    quantScaleQueue.EnQue(scale);
    scale = quantScaleQueue.DeQue<float>();

    __VEC_SCOPE__
    {
        RegTensor<half> vreg0;
        RegTensor<float> vreg1;
        RegTensor<half> vreg2;
        RegTensor<float> vreg3;
        RegTensor<float> vreg4;
        RegTensor<float> vreg5;
        RegTensor<float> vreg6;
        RegTensor<half> vreg7;
        RegTensor<int8_t> vreg8;
        MaskReg preg0;

        __local_mem__ half *leftAddr = (__local_mem__ half *)left.GetPhyAddr();
        __local_mem__ half *rightAddr = (__local_mem__ half *)right.GetPhyAddr();
        __local_mem__ float *scaleAddr = (__local_mem__ float *)scale.GetPhyAddr();
        __local_mem__ int8_t *outAddr = (__local_mem__ int8_t *)left.GetPhyAddr(); // out复用swish计算结果内存
        for (uint16_t j = 0; j < static_cast<uint16_t>(curAicM); j++) {
            uint32_t count = curAicN;
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::MicroAPI::UpdateMask<float>(count);
                AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, leftAddr + j * curAicN + i * calSize);
                AscendC::MicroAPI::Cast<float, half, castTraitHalf2Float>(vreg1, vreg0, preg0);
                AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg2, rightAddr + j * curAicN + i * calSize);
                AscendC::MicroAPI::Cast<float, half, castTraitHalf2Float>(vreg3, vreg2, preg0);

                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg4, vreg1, vreg3, preg0);

                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg5, scaleAddr + i * calSize);
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg6, vreg4, vreg5, preg0);

                AscendC::MicroAPI::Cast<half, float, castTraitFloat2Half>(vreg7, vreg6, preg0);
                AscendC::MicroAPI::Cast<int8_t, half, castTraitHalf2Int8>(vreg8, vreg7, preg0);

                AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(outAddr + j * curAicN + i * calSize, vreg8, preg0);
            }
        }
    }
    AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventID);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventID);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(eventID);
    quantScaleQueue.FreeTensor(scale);
}

FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS
__aicore__ inline void FusedQuantMatmulSwiglu<FUSED_SWIGLU_TEMPLATE_FUNC_PARAMS>::MulCompute(uint64_t offset, uint32_t curAicM, uint32_t curAicN)
{
    auto left = outBuf.Get<half>();
    auto right = matmulRightOutQueue.DeQue<half>();
    auto size = curAicM * curAicN;
    uint32_t calSize = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vfLoopNum = (size + calSize - 1) / calSize;

    __VEC_SCOPE__
    {
        RegTensor<half> vreg0;
        RegTensor<float> vreg1;
        RegTensor<half> vreg2;
        RegTensor<float> vreg3;
        RegTensor<float> vreg4;
        RegTensor<half> vreg5;
        MaskReg preg0;

        __local_mem__ half *leftAddr = (__local_mem__ half *)left.GetPhyAddr();
        __local_mem__ half *rightAddr = (__local_mem__ half *)right.GetPhyAddr();
        __local_mem__ half *outAddr = (__local_mem__ half *)left.GetPhyAddr(); // out复用swish计算结果内存

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = AscendC::MicroAPI::UpdateMask<float>(size);
            AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, leftAddr + i * calSize);
            AscendC::MicroAPI::Cast<float, half, castTraitHalf2Float>(vreg1, vreg0, preg0);
            AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg2, rightAddr + i * calSize);
            AscendC::MicroAPI::Cast<float, half, castTraitHalf2Float>(vreg3, vreg2, preg0);

            AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg4, vreg1, vreg3, preg0);

            AscendC::MicroAPI::Cast<half, float, castTraitFloat2Half>(vreg5, vreg4, preg0);
            AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(outAddr + i * calSize, vreg5, preg0);
        }
    }
    AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventID);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventID);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(eventID);
}

FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS
__aicore__ inline void FusedQuantMatmulSwiglu<FUSED_SWIGLU_TEMPLATE_FUNC_PARAMS>::CopyIn(LocalTensor<x1Type> &aL1Local, uint32_t curAicM, uint64_t xOffset)
{
    uint64_t offsetA = 0;
    constexpr int64_t int4Factor = (AscendC::IsSameType<x1Type, AscendC::int4b_t>::value) ? 2 : 1;
    uint64_t nDim = curAicM;
    uint64_t dDim = k;
    uint64_t c0Size = DequantBmm::GetC0Size<x1Type>();

    Nd2NzParams params;

    params.ndNum = 1;
    if constexpr (aTrans) {
        offsetA = xOffset;
        nDim = k;
        dDim = curAicM;
        params.srcDValue = DequantBmm::CeilDiv(m, int4Factor);
        params.dstNzC0Stride = DequantBmm::Align(nDim, c0Size);
    } else {
        offsetA = xOffset * k;
        params.srcDValue =  DequantBmm::CeilDiv(k, int4Factor);
        params.dstNzC0Stride = DequantBmm::Align(nDim, BMM_BLOCK_NUM);
    }

    params.nValue = nDim;
    params.dValue = DequantBmm::CeilDiv(dDim, int4Factor);
    params.srcNdMatrixStride = 0;

    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    if constexpr (AscendC::IsSameType<x1Type, AscendC::int4b_t>::value) {
        GlobalTensor<int8_t> aGlobalInt8;
        aGlobalInt8.SetGlobalBuffer((__gm__ int8_t *)(xGm.GetPhyAddr()), (nDim * dDim) >> 1);
        auto al1LocalImpl = aL1Local.template ReinterpretCast<int8_t>();
        DataCopy(al1LocalImpl, aGlobalInt8[offsetA >> 1], params);
    } else {
        DataCopy(aL1Local, xGm[offsetA], params);
    }
}

FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS
__aicore__ inline void FusedQuantMatmulSwiglu<FUSED_SWIGLU_TEMPLATE_FUNC_PARAMS>::CopyOut(uint64_t xOffset, uint64_t wOffset, uint32_t curAicM, uint32_t curAicN)
{
    DataCopyParams dataCopyParams;

    dataCopyParams.blockLen = curAicN * sizeof(yType) / DATA_BLOCK;
    dataCopyParams.blockCount = curAicM;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = n * sizeof(yType) / DATA_BLOCK - curAicN * sizeof(yType) / DATA_BLOCK;

    auto out = outBuf.Get<yType>();
    DataCopy(yGm[xOffset * (uint64_t)n + wOffset], out, dataCopyParams);
}

FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS
__aicore__ inline void FusedQuantMatmulSwiglu<FUSED_SWIGLU_TEMPLATE_FUNC_PARAMS>::Iterate(LocalTensor<half> &result)
{
    while (mm.Iterate()) {
        mm.GetTensorC(result, 0, true);
    }
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_V));
    AscendC::SetFlag<AscendC::HardEvent::FIX_V>(eventID);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_V>(eventID);
}

FUSED_SWIGLU_TEMPLATE_CLASS_PARAMS
__aicore__ inline void FusedQuantMatmulSwiglu<FUSED_SWIGLU_TEMPLATE_FUNC_PARAMS>::Process()
{
    if (curBlockIdx >= mLoops * nLoops) {
        return;
    }

    CalcParams();
    uint64_t xCurOffset = xCoreOffset;
    auto l1a = l1AQueue.AllocTensor<x1Type>();
    auto leftResult = matmulLeftOutQueue.AllocTensor<half>();
    auto rightResult = matmulRightOutQueue.AllocTensor<half>();

    for (uint32_t mInnerIdx = 0; mInnerIdx < mInnerLoops; mInnerIdx++) {
        uint32_t curAicM = mInnerIdx == mInnerLoops - 1 ? aicMtail : singleTimeM;
        CopyIn(l1a, curAicM, xCurOffset);
        l1AQueue.EnQue(l1a);
        l1a = l1AQueue.DeQue<x1Type>();

        mm.SetOrgShape(curAicM, n, k);
        mm.SetTensorA(l1a);

        uint64_t nOffset = wCoreOffset;
        for (uint32_t nInnerIdx = 0; nInnerIdx < nInnerLoops; nInnerIdx++) {
            uint32_t curAicN = nInnerIdx == nInnerLoops - 1 ? aicNtail : singleTimeN;
            mm.SetTail(DequantBmm::Align(curAicM, BMM_BLOCK_NUM), DequantBmm::Align(curAicN, BMM_BLOCK_NUM), k);
            mm.SetTensorB(wGm[nOffset * (uint64_t)k]);
            mm.SetQuantVector(deQuantScaleGm[nOffset]);

            Iterate(leftResult);
            matmulLeftOutQueue.EnQue(leftResult);

            SwishCompute(curAicM * curAicN);

            mm.SetTensorB(wGm[(uint64_t)n * (uint64_t)k + nOffset * (uint64_t)k]);
            mm.SetQuantVector(deQuantScaleGm[n + nOffset]);

            Iterate(rightResult);
            matmulRightOutQueue.EnQue(rightResult);
            if (isQuant) {
                MulQuantCompute(nOffset, curAicM, curAicN);
            } else {
                MulCompute(nOffset, curAicM, curAicN);
            }

            CopyOut(xCurOffset, nOffset, curAicM, curAicN);

            nOffset += singleTimeN;
        }
        xCurOffset += singleTimeM;
        l1AQueue.FreeTensor(l1a);
    }
    matmulLeftOutQueue.FreeTensor(leftResult);
    matmulRightOutQueue.FreeTensor(rightResult);
    mm.End();
}
}
#endif // FUSED_QUANT_MAT_MUL_SWIGLU_H
