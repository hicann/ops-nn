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
 * \file flat_quant_high.h
 * \brief
 */
#ifndef FLAT_QUANT_HIGH_V2_H
#define FLAT_QUANT_HIGH_V2_H

#include "../tensor_utils.h"
#include "tensor_utils_v2.h"

namespace FlatQuantNS {
template <typename T>
class FlatQuantHighV2 {
public:
    aifunc FlatQuantHighV2() {}

    aifunc void Init(GM_ADDR xmtx_, GM_ADDR p1mtx_, GM_ADDR p2mtx_, GM_ADDR out_, GM_ADDR qscale_, GM_ADDR workspace_,
                     const FlatQuantTilingData* tilingData)
    {
        shape.M = tilingData->M;
        shape.N = tilingData->N;
        shape.K = tilingData->K;
        dstTypeMax_ = tilingData->dstTypeMax;
        invDstTypeMax_ = tilingData->invDstTypeMax;

        // Set addValueBit based on dstTypeMax
        if (dstTypeMax_ == SIX_FLOAT) {
            addValueBit_ = ADD_VALUE_FOR_BF16_MAN1;
        } else if (dstTypeMax_ == SEVEN_FLOAT) {
            addValueBit_ = ADD_VALUE_FOR_BF16_MAN2;
        }
        tiling();

        xGM.SetGlobalBuffer((__gm__ T*)xmtx_);
        p1GM.SetGlobalBuffer((__gm__ T*)p1mtx_);
        p2GM.SetGlobalBuffer((__gm__ T*)p2mtx_);
        x1GM.SetGlobalBuffer((__gm__ T*)workspace_ +
                             useAivNum * K_DOUBLE_VEC * shape.Mceil * shape.N * sizeof(float) / sizeof(T));
        x2GM.SetGlobalBuffer((__gm__ float*)workspace_);
        outGM.SetGlobalBuffer((__gm__ int8_t*)out_);
        qscaleGM.SetGlobalBuffer((__gm__ int8_t*)qscale_);
        pipe.InitBuffer(bufQueue, UB_SIZE);
        xF32Tensor = bufQueue.Get<float>();
        xTensor = xF32Tensor[MN_SIZE].template ReinterpretCast<bfloat16_t>();
        yTensor = xTensor[MN_SIZE].template ReinterpretCast<int8_t>();
        emaxTensor = yTensor[OUT_SIZE].template ReinterpretCast<uint16_t>();
        deqscaleTensor = emaxTensor[EMAX_SIZE];
        qscaleTensor = deqscaleTensor[EMAX_SIZE].template ReinterpretCast<int8_t>();
        qscaleBlockTensor = qscaleTensor[EMAX_SIZE];

        eventIdVToS = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
        eventIdVToMte2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
        eventIdMte2ToV = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        eventIdVToMte3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        eventIdMte3ToV = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_V));
        eventIdMte3ToS = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
    }

    aifunc void tiling()
    {
        aivNum = GetBlockNum() * DOUBLE;
        useAivNum = (shape.K + K_PER_VEC - 1) / K_PER_VEC;
        if (useAivNum > aivNum) {
            useAivNum = aivNum;
        }
        int k_per_core = ((shape.K + aivNum - 1) / aivNum + K_PER_VEC - 1) / (K_PER_VEC) * (K_PER_VEC);
        shape.K1 = k_per_core * GetBlockIdx();
        shape.K2 = ((k_per_core + shape.K1) > shape.K) ? shape.K : (k_per_core + shape.K1);
        shape.Mceil = (shape.M + CEIL_SIZE - 1) / CEIL_SIZE * CEIL_SIZE;
        shape.Nceil = VEC_N_LEN;
        AlignM = AscendC::CeilDiv((shape.M * shape.N), VEC_N_LEN);
        if (shape.M * shape.N > MN_SIZE) {
            splitM1 = DOUBLE_VEC_N_LEN;
            AlignM1 = shape.N * 2; // M1 = 2 * N
            splitM2 = shape.M - splitM1;
            AlignM2 = AscendC::CeilDiv((splitM2 * shape.N), VEC_N_LEN);
        }
        x1Offset = GetBlockIdx() * K_PER_VEC * shape.M * shape.N;
        x2Offset = GetBlockIdx() * K_DOUBLE_VEC * shape.Mceil * shape.N;
    }

    aifunc void Process()
    {
        clearTensor();

        int64_t scaleK = shape.K1;
        int64_t k = shape.K1;
        for (int64_t startK = shape.K1; startK < shape.K2; startK += K_PER_VEC) {
            int64_t endK = startK + K_PER_VEC > shape.K2 ? shape.K2 : startK + K_PER_VEC;
            ProcessHighK(startK, endK - startK);
            while (k < endK) {
                Quant(k);
                k++;
            }
        }
    }

    aifunc void Quant(int64_t k)
    {
        if ((shape.M * shape.N > MN_SIZE) && (AlignM1 > 0)) {
            SplitQuant(k);
        } else {
            uint64_t offset = x2Offset + (k % K_DOUBLE_VEC) * shape.Mceil * shape.N;
            uint64_t yOffset = k * shape.M * shape.N;
            uint64_t scaleOffset = k * AscendC::CeilDiv(shape.M * shape.N, MXFP_DIVISOR_SIZE) * 2;
            CopyInputFromGm2Ub(xF32Tensor, offset, shape.M, shape.N);
            int64_t totalDataInUB = shape.M * shape.N;
            computeMxQuant(xTensor, yTensor, emaxTensor, qscaleTensor, deqscaleTensor, totalDataInUB);
            computeTransLayout(qscaleTensor, qscaleBlockTensor, AlignM, shape.Nceil);
            AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            CopyOutputFromUb2Gm(shape.M, yOffset, yTensor);
            CopyScaleFromUb2Gm(AlignM, scaleOffset, qscaleBlockTensor);
            AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
    }

    aifunc void SplitQuant(int64_t k)
    {
        for (int64_t i = 0; i < M_SPLIT_COUNT; i++) { // Double progress
            uint64_t cmpM = i > 0 ? splitM2 : splitM1;
            uint64_t alignM = i > 0 ? AlignM2 : AlignM1;
            uint64_t offset = x2Offset + (k % K_DOUBLE_VEC) * shape.Mceil * shape.N + i * DOUBLE_VEC_N_LEN * shape.N;
            uint64_t yOffset = k * shape.M * shape.N + (DOUBLE_VEC_N_LEN * i) * shape.N;
            // align to 2 * 32
            uint64_t scaleOffset = k * 2 * ((shape.M * shape.N + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE) +
                                   AscendC::CeilDiv((DOUBLE_VEC_N_LEN * i * shape.N), MXFP_DIVISOR_SIZE) * 2;
            CopyInputFromGm2Ub(xF32Tensor, offset, cmpM, shape.N);
            int64_t totalDataInUB = cmpM * shape.N;
            computeMxQuant(xTensor, yTensor, emaxTensor, qscaleTensor, deqscaleTensor, totalDataInUB);
            computeTransLayout(qscaleTensor, qscaleBlockTensor, alignM, shape.Nceil);
            AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            CopyOutputFromUb2Gm(cmpM, yOffset, yTensor);
            CopyScaleFromUb2Gm(alignM, scaleOffset, qscaleBlockTensor);
            AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            clearTensor();
        }
    }

    aifunc void CopyInputFromGm2Ub(AscendC::LocalTensor<float>& xF32Tensor, uint64_t offset, int64_t Mcount,
                                   int64_t Nlength)
    {
        AscendC::DataCopyExtParams Gm2UbParams{1, 0, 0, 0, 0};
        Gm2UbParams.blockCount = 1;
        Gm2UbParams.blockLen = Mcount * Nlength * sizeof(float);
        Gm2UbParams.srcStride = 0;
        Gm2UbParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        AscendC::DataCopyPad(xF32Tensor, x2GM[offset], Gm2UbParams, padParams);
        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::Cast(xTensor, xF32Tensor, RoundMode::CAST_RINT, Mcount * Nlength);
    }

    aifunc void CopyOutputFromUb2Gm(uint64_t M, uint64_t offset, AscendC::LocalTensor<int8_t>& src)
    {
        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        ub2GmParams.blockCount = 1;
        ub2GmParams.blockLen = (M * shape.N * sizeof(int8_t)) >> 1;
        ub2GmParams.dstStride = 0;
        ub2GmParams.srcStride = 0;
        offset = offset >> 1;
        AscendC::DataCopyPad(outGM[offset], src, ub2GmParams);
    }

    aifunc void CopyScaleFromUb2Gm(uint64_t M, uint64_t offset, AscendC::LocalTensor<int8_t>& src)
    {
        uint64_t blockScaleN = 2;
        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        ub2GmParams.blockCount = AscendC::CeilDiv((M * shape.Nceil), MXFP_DIVISOR_SIZE);
        ub2GmParams.blockLen = blockScaleN * sizeof(int8_t);
        ub2GmParams.dstStride = 0;
        AscendC::DataCopyPad(qscaleGM[offset], src, ub2GmParams);
    }

    aifunc void computeMxQuant(LocalTensor<bfloat16_t>& xTensor, LocalTensor<int8_t>& yTensor,
                               LocalTensor<uint16_t>& emaxTensor, LocalTensor<int8_t>& qscaleTensor,
                               LocalTensor<uint16_t>& deqscaleTensor, int64_t totalDataInUB)
    {
        uint32_t MNCeil = totalDataInUB;
        uint32_t oneRepeateSize = AscendC::GetVecLen() / sizeof(T);
        uint16_t repeatCount = (MNCeil + oneRepeateSize * 2 - 1) / (oneRepeateSize * 2);
        uint32_t scaleNum = (MNCeil + GROUP_SIZE - 1) / GROUP_SIZE;
        uint16_t repeateScaleCount = (scaleNum + oneRepeateSize - 1) / oneRepeateSize;
        uint16_t repeateScaleHalfCount = (scaleNum + (oneRepeateSize / 2) - 1) / (oneRepeateSize / 2);

        __ubuf__ bfloat16_t* xAddr = (__ubuf__ bfloat16_t*)xTensor.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)emaxTensor.GetPhyAddr();

        if (dstTypeMax_ >= SIX_FLOAT && dstTypeMax_ <= TWELVE_FLOAT) {
            AscendC::VF_CALL<ExpMaxVfcuBLAS>(maxExpAddr, xAddr, MNCeil, repeatCount, oneRepeateSize);
        } else {
            AscendC::VF_CALL<ExpMaxVf>(maxExpAddr, xAddr, MNCeil, repeatCount, oneRepeateSize);
        }

        __ubuf__ uint16_t* deScaleAddr = (__ubuf__ uint16_t*)deqscaleTensor.GetPhyAddr();
        __ubuf__ uint16_t* scaleAddr = (__ubuf__ uint16_t*)qscaleTensor.GetPhyAddr();

        if (dstTypeMax_ == ZERO_FLOAT) {
            AscendC::VF_CALL<ScaleVf>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeateScaleCount);
        } else if (dstTypeMax_ == SIX_FLOAT || dstTypeMax_ == SEVEN_FLOAT) {
            AscendC::VF_CALL<ScaleVfDynamic>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeateScaleCount,
                                             addValueBit_);
        } else {
            AscendC::VF_CALL<ScaleVfcuBLAS>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeateScaleHalfCount,
                                            invDstTypeMax_);
        }

        __ubuf__ int8_t* yAddr = (__ubuf__ int8_t*)yTensor.GetPhyAddr();
        AscendC::VF_CALL<QuantVf>(yAddr, xAddr, deScaleAddr, MNCeil, repeatCount);
    }

    aifunc void computeTransLayout(LocalTensor<int8_t>& qscaleTensor, LocalTensor<int8_t>& qscaleBlockTensor,
                                   uint64_t M, uint64_t N)
    {
        uint16_t mSize = static_cast<uint16_t>(M);
        uint16_t scaleBlockN = static_cast<uint32_t>(AscendC::CeilDiv(N, MXFP_DIVISOR_SIZE) * 2);

        __ubuf__ int8_t* qscaleAddr = (__ubuf__ int8_t*)qscaleTensor.GetPhyAddr();
        __ubuf__ int8_t* qscaleBlkAddr = (__ubuf__ int8_t*)qscaleBlockTensor.GetPhyAddr();
        AscendC::VF_CALL<TransLayoutVf>(qscaleAddr, qscaleBlkAddr, mSize, scaleBlockN);
    }

    aifunc void clearTensor()
    {
        Duplicate<float>(xF32Tensor, (float)0, MN_SIZE);
        Duplicate<bfloat16_t>(xTensor, (bfloat16_t)0, MN_SIZE);
        Duplicate<uint16_t>(emaxTensor, (uint16_t)0, EMAX_SIZE);
        Duplicate<int8_t>(qscaleTensor, (int8_t)0, EMAX_SIZE);
        Duplicate<uint16_t>(deqscaleTensor, (uint16_t)0, EMAX_SIZE);
        Duplicate<int8_t>(qscaleBlockTensor, (int8_t)0, EMAX_SIZE);
        PipeBarrier<PIPE_V>();
    }

    aifunc void ProcessHighK(int64_t k, int64_t batch)
    {
        int64_t offset1 = x1Offset + (k % K_PER_VEC) * shape.M * shape.N;
        int64_t offset2 = x2Offset + (k % K_DOUBLE_VEC) * shape.Mceil * shape.N;
        matmulR.SetSingleShape(batch * shape.M, shape.N, shape.N);
        matmulR.SetTensorA(xGM[k * shape.M * shape.N], false);
        matmulR.SetTensorB(p2GM, false);
        matmulR.IterateAll(x1GM[offset1], false);
        PipeBarrier<PIPE_ALL>();

        matmulL.SetTensorA(p1GM, false);
        for (int64_t i = 0; i < batch; i++) {
            matmulL.SetTensorB(x1GM[offset1], false);
            matmulL.IterateAll(x2GM[offset2], false);
            offset1 += shape.M * shape.N;
            offset2 += shape.Mceil * shape.N;
        }
    }

    static __simd_vf__ inline void ExpMaxVf(__ubuf__ uint16_t* dstPtr, __ubuf__ bfloat16_t* srcPtr, uint32_t count,
                                            uint16_t repeatTimes, uint32_t oneRepeatSize)
    {
        AscendC::Reg::RegTensor<bfloat16_t> vSrcReg0;
        AscendC::Reg::RegTensor<bfloat16_t> vSrcReg1;
        AscendC::Reg::RegTensor<uint16_t> vExpExtract0;
        AscendC::Reg::RegTensor<uint16_t> vExpExtract1;
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;

        AscendC::Reg::RegTensor<uint16_t> expMaskBF16;
        AscendC::Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

        AscendC::Reg::MaskReg maskReg;
        AscendC::Reg::UnalignReg u1;
        AscendC::Reg::AddrReg aReg;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            aReg = AscendC::Reg::CreateAddrReg<uint32_t>(i, oneRepeatSize);
            maskReg = AscendC::Reg::UpdateMask<bfloat16_t>(count);

            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vSrcReg0, vSrcReg1, srcPtr,
                                                                                         aReg);
            AscendC::Reg::And(vExpExtract0, (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg0, expMaskBF16, maskReg);
            AscendC::Reg::And(vExpExtract1, (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg1, expMaskBF16, maskReg);
            AscendC::Reg::Max(vdMaxExp, vExpExtract0, vExpExtract1, maskReg);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(vdMaxExp, vdMaxExp, maskReg);
            AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                dstPtr, vdMaxExp, u1, STORE_UNALIGN_STRIDE_BYTES);
        }
        AscendC::Reg::StoreUnAlignPost(dstPtr, u1, 0);
    }

    static __simd_vf__ inline void ExpMaxVfcuBLAS(__ubuf__ uint16_t* dstPtr, __ubuf__ bfloat16_t* srcPtr,
                                                  uint32_t count, uint16_t repeatTimes, uint32_t oneRepeatSize)
    {
        AscendC::Reg::RegTensor<bfloat16_t> vSrcReg0;
        AscendC::Reg::RegTensor<bfloat16_t> vSrcReg1;
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;

        AscendC::Reg::RegTensor<uint16_t> absMask16Bit;
        AscendC::Reg::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);

        AscendC::Reg::MaskReg maskReg;
        AscendC::Reg::UnalignReg u1;
        AscendC::Reg::AddrReg aReg;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            aReg = AscendC::Reg::CreateAddrReg<uint32_t>(i, oneRepeatSize);
            maskReg = AscendC::Reg::UpdateMask<bfloat16_t>(count);

            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vSrcReg0, vSrcReg1, srcPtr,
                                                                                         aReg);
            AscendC::Reg::And((AscendC::Reg::RegTensor<uint16_t>&)vSrcReg0,
                              (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg0, absMask16Bit, maskReg);
            AscendC::Reg::And((AscendC::Reg::RegTensor<uint16_t>&)vSrcReg1,
                              (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg1, absMask16Bit, maskReg);
            AscendC::Reg::Max(vdMaxExp, (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg0,
                              (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg1, maskReg);

            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(vdMaxExp, vdMaxExp, maskReg);
            AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                dstPtr, vdMaxExp, u1, STORE_UNALIGN_STRIDE_BYTES);
        }
        AscendC::Reg::StoreUnAlignPost(dstPtr, u1, 0);
    }

    static __simd_vf__ inline void ScaleVf(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr,
                                           __ubuf__ uint16_t* srcPtr, uint32_t scaleNum, uint16_t repeatTimes)
    {
        AscendC::Reg::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::Reg::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::Reg::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
        AscendC::Reg::Duplicate(maxExpValue, FP4_E2M1_MAX_EXP);
        AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::Reg::Duplicate(zeroRegTensor, 0);
        AscendC::Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);

        AscendC::Reg::MaskReg invalidDataMask, specialDataMask;
        AscendC::Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(scaleNum);
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, srcPtr, 128);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK_B16>(dstPtr, scaleValue, 64, preMaskScale);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);

            AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dst2Ptr, halfScale, 128,
                                                                                            preMaskScale);
        }
    }

    static __simd_vf__ inline void ScaleVfDynamic(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr,
                                                  __ubuf__ uint16_t* srcPtr, uint32_t scaleNum, uint16_t repeatTimes,
                                                  uint16_t addValueBit)
    {
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::RegTensor<uint16_t> sharedExp;
        AscendC::Reg::RegTensor<uint16_t> scaleValue;
        AscendC::Reg::RegTensor<uint16_t> halfScale;
        AscendC::Reg::RegTensor<uint16_t> vdMaxExpAdd;
        AscendC::Reg::RegTensor<uint16_t> vdMaxExpOnly;

        AscendC::Reg::RegTensor<uint16_t> expMask;
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::Reg::RegTensor<uint16_t> addValue;
        AscendC::Reg::Duplicate(addValue, addValueBit);
        AscendC::Reg::RegTensor<uint16_t> maxExpValue;
        AscendC::Reg::Duplicate(maxExpValue, FP4_E2M1_MAX_EXP);
        AscendC::Reg::RegTensor<uint16_t> scaleBias;
        AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::Reg::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::Reg::RegTensor<uint16_t> zeroRegTensor;
        AscendC::Reg::Duplicate(zeroRegTensor, 0);
        AscendC::Reg::RegTensor<uint16_t> nanRegTensor;
        AscendC::Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::Reg::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);

        AscendC::Reg::MaskReg cmpResult;
        AscendC::Reg::MaskReg zeroMask;
        AscendC::Reg::MaskReg invalidDataMask;
        AscendC::Reg::MaskReg specialDataMask;
        AscendC::Reg::MaskReg preMaskScale;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(scaleNum);
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, srcPtr, vfLen16);
            AscendC::Reg::And(vdMaxExpOnly, vdMaxExp, expMask, preMaskScale); // Extract exponent bits
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExpOnly, expMask,
                                                                  preMaskScale); // Check INF/NAN
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExpOnly, zeroRegTensor, preMaskScale);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LT>(invalidDataMask, vdMaxExpOnly, maxExpValue,
                                                                  preMaskScale);

            AscendC::Reg::Add(vdMaxExpAdd, vdMaxExp, addValue, preMaskScale); // Result after carry
            AscendC::Reg::And(vdMaxExpAdd, vdMaxExpAdd, expMask,
                              preMaskScale); // Extract exponent bits from carry result
            AscendC::Reg::Select<uint16_t>(vdMaxExpAdd, maxExpValue, vdMaxExpAdd, invalidDataMask);
            AscendC::Reg::Sub(sharedExp, vdMaxExpAdd, maxExpValue, preMaskScale);

            AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK_B16>(dstPtr, scaleValue, vfLen32, preMaskScale);

            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_NORM>(dst2Ptr, halfScale, vfLen16, preMaskScale);
        }
    }

    static __simd_vf__ inline void ScaleVfcuBLAS(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr,
                                                 __ubuf__ uint16_t* srcPtr, uint32_t scaleNum, uint16_t repeatTimes,
                                                 float invDstTypeMax)
    {
        AscendC::Reg::RegTensor<uint16_t> max16;
        AscendC::Reg::RegTensor<uint32_t> max32;
        AscendC::Reg::RegTensor<uint32_t> exp32;
        AscendC::Reg::RegTensor<uint32_t> man32;
        AscendC::Reg::RegTensor<uint32_t> normalExp32;
        AscendC::Reg::RegTensor<uint32_t> expAddOne32;
        AscendC::Reg::RegTensor<uint32_t> extractExp;
        AscendC::Reg::RegTensor<uint16_t> expOut;
        AscendC::Reg::RegTensor<uint32_t> halfScale;
        AscendC::Reg::RegTensor<uint16_t> recExpOut;

        AscendC::Reg::RegTensor<uint32_t> manMaskFP32;
        AscendC::Reg::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        AscendC::Reg::RegTensor<uint32_t> expMask;
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_FP32);
        AscendC::Reg::RegTensor<uint32_t> zeroRegTensor32;
        AscendC::Reg::Duplicate(zeroRegTensor32, 0);
        AscendC::Reg::RegTensor<uint32_t> scaleBias;
        AscendC::Reg::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        AscendC::Reg::RegTensor<uint32_t> nanRegTensor;
        AscendC::Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        AscendC::Reg::RegTensor<uint32_t> fp4NanRegTensor;
        AscendC::Reg::Duplicate(fp4NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);
        AscendC::Reg::RegTensor<float> invMax;
        AscendC::Reg::Duplicate(invMax, invDstTypeMax);

        AscendC::Reg::MaskReg cmpResult;
        AscendC::Reg::MaskReg zeroMask;
        AscendC::Reg::MaskReg p0;
        AscendC::Reg::MaskReg p1;
        AscendC::Reg::MaskReg p2;
        AscendC::Reg::MaskReg preMaskScale;
        uint32_t SixtyFour = 64;
        AscendC::Reg::MaskReg dataMaskB16Half = AscendC::Reg::UpdateMask<uint16_t>(SixtyFour);

        static constexpr AscendC::Reg::CastTrait castTraitHalf2Float = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};

        for (uint16_t i = 0; i < repeatTimes; i++) {
            preMaskScale = AscendC::Reg::UpdateMask<uint32_t>(scaleNum);
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_UNPACK_B16>(max16, srcPtr, vfLen32);

            AscendC::Reg::Cast<float, bfloat16_t, castTraitHalf2Float>(
                (AscendC::Reg::RegTensor<float>&)max32, (AscendC::Reg::RegTensor<bfloat16_t>&)max16, preMaskScale);
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, preMaskScale);

            AscendC::Reg::Mul((AscendC::Reg::RegTensor<float>&)max32, (AscendC::Reg::RegTensor<float>&)max32, invMax,
                              preMaskScale);
            AscendC::Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, preMaskScale);
            AscendC::Reg::And(man32, max32, manMaskFP32, preMaskScale);

            AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::GT>(p0, exp32, NUMBER_ZERO, preMaskScale);
            AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::LT>(p1, exp32, NUMBER_TWO_FIVE_FOUR, preMaskScale);
            AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::GT>(p2, man32, NUMBER_ZERO, preMaskScale);
            AscendC::Reg::And(p0, p0, p1, preMaskScale);
            AscendC::Reg::And(p0, p0, p2, preMaskScale);

            AscendC::Reg::Adds(expAddOne32, exp32, 1, preMaskScale);
            AscendC::Reg::Select(extractExp, expAddOne32, exp32, p0);
            AscendC::Reg::Select<uint32_t>(extractExp, extractExp, fp4NanRegTensor, cmpResult);
            AscendC::Reg::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(expOut, extractExp);

            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_PACK_B16>(dstPtr + i * SCALE_STORE_STRIDE,
                                                                                       expOut, dataMaskB16Half);

            AscendC::Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::Reg::Sub(halfScale, scaleBias, extractExp, preMaskScale);
            AscendC::Reg::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::Reg::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(recExpOut, halfScale);

            AscendC::Reg::StoreAlign<uint16_t>(dst2Ptr + i * vfLen32, recExpOut, dataMaskB16Half);
        }
    }

    static __simd_vf__ inline void QuantVf(__ubuf__ int8_t* dstPtr, __ubuf__ bfloat16_t* srcPtr,
                                           __ubuf__ uint16_t* src2Ptr, uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::Reg::MaskReg dataMask1;
        AscendC::Reg::MaskReg dataMask2;
        AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp1;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0Convert;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp1Convert;

        AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp1Bf16;

        AscendC::Reg::RegTensor<fp4x2_e2m1_t> vdExp0FP4;
        AscendC::Reg::RegTensor<fp4x2_e2m1_t> vdExp1FP4;

        AscendC::Reg::RegTensor<bfloat16_t> vdBf16Exp0FP4;
        AscendC::Reg::RegTensor<bfloat16_t> vdBf16Exp1FP4;

        AscendC::Reg::AddrReg aReg;
        static constexpr AscendC::Reg::CastTrait castTrait = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < repeatTimes; i++) {
            aReg = AscendC::Reg::CreateAddrReg<uint16_t>(i, oneRepeatSize);
            dataMask1 = AscendC::Reg::UpdateMask<bfloat16_t>(oneRepeatSize);
            dataMask2 = AscendC::Reg::UpdateMask<bfloat16_t>(oneRepeatSize);

            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcPtr, 128 * 2); // copy two chunks from srcAddr to regbase
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, src2Ptr, 8);
            AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::Reg::Cast<fp4x2_e2m1_t, bfloat16_t, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            AscendC::Reg::Cast<fp4x2_e2m1_t, bfloat16_t, castTrait>(vdExp1FP4, vdExp1, dataMask2);

            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                dstPtr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP4, 64, dataMask1);
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                dstPtr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP4, 64, dataMask2);
        }
    }
    static __simd_vf__ inline void TransLayoutVf(__ubuf__ int8_t* scaleAddr, __ubuf__ int8_t* scaleBlkAddr,
                                                 uint16_t mSize, uint16_t scaleBlockN)
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t eleNum = scaleBlockN;
            AscendC::Reg::MaskReg maskScaleN = AscendC::Reg::UpdateMask<int8_t>(eleNum);
            AscendC::Reg::RegTensor<int8_t> vReg0;
            AscendC::Reg::UnalignReg u0, u1;
            auto srcUb = scaleAddr + mIdx * scaleBlockN;
            AscendC::Reg::LoadUnAlignPre(u0, srcUb);
            AscendC::Reg::LoadUnAlign(vReg0, u0, srcUb);
            auto dstUb = scaleBlkAddr + mIdx * 32;
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstUb, vReg0, maskScaleN);
        }
    }

public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
        matmulR;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>, MDL_CFG>
        matmulL;

private:
    FlatQuantShapeInfo shape;
    GlobalTensor<T> xGM;
    GlobalTensor<T> p1GM;
    GlobalTensor<T> p2GM;
    GlobalTensor<int8_t> outGM;
    GlobalTensor<int8_t> qscaleGM;
    GlobalTensor<T> x1GM;
    GlobalTensor<float> x2GM;

    TBuf<QuePosition::VECCALC> bufQueue;
    LocalTensor<float> xF32Tensor;
    LocalTensor<bfloat16_t> xTensor;
    LocalTensor<int8_t> yTensor;
    LocalTensor<uint16_t> emaxTensor;
    LocalTensor<int8_t> qscaleTensor;
    LocalTensor<uint16_t> deqscaleTensor;
    LocalTensor<int8_t> qscaleBlockTensor;

    event_t eventIdVToS;
    event_t eventIdVToMte2;
    event_t eventIdMte2ToV;
    event_t eventIdVToMte3;
    event_t eventIdMte3ToV;
    event_t eventIdMte3ToS;

    int64_t AlignM = 0;
    int64_t AlignM1 = 0;
    int64_t AlignM2 = 0;
    int64_t splitM1 = 0;
    int64_t splitM2 = 0;
    int64_t aivNum = 0;
    int64_t useAivNum = 0;
    int64_t x1Offset = 0;
    int64_t x2Offset = 0;
    uint32_t oneRepeatSize = 0;
    uint32_t vfForB16Number = 0;
    uint16_t elementAfterReduce = 0;

    float dstTypeMax_ = 0.0f;
    float invDstTypeMax_ = 0.0f;
    uint16_t addValueBit_ = 0;
};
} // namespace FlatQuantNS

#endif // FLAT_QUANT_HIGH_H
