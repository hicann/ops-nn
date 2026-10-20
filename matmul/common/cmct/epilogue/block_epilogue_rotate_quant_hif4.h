/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_rotate_quant_hif4.h
 * \brief HiFloat4 quant epilogue for rotate_quant. Ported from
 *        dynamic_hi_f4_quant_tail_axis.h and kept structurally identical.
 *        The only deviation: x comes from matmul Fixpipe to ubLocal_ (VECIN)
 *        instead of a TQue CopyIn from GM. Compute functions are declared as
 *        static __simd_vf__ and invoked via VF_CALL to prevent DCE under
 *        -O3 vec compilation (same pattern as block_epilogue_rotate_quant.h).
 */

#pragma once
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "fusion/default_fusion_op.h"
#ifndef FLOAT_OVERFLOW_MODE_CTRL
#define FLOAT_OVERFLOW_MODE_CTRL 60
#endif

namespace Cmct {
namespace Gemm {
namespace Block {
using namespace AscendC;

constexpr int64_t HIF4_DIGIT_TWO = 2;
constexpr int64_t HIF4_DIGIT_FOUR = 4;
constexpr int64_t HIF4_DIGIT_EIGHT = 8;
constexpr int64_t HIF4_BLOCK_SIZE = 64;
constexpr int64_t HIF4_GROUP_ELE_NUM = 64;
constexpr int64_t HIF4_GROUP_BYTE_NUM = 32; // 64 * 0.5 byte
constexpr int64_t HIF4_DB_BUFFER = 2;
constexpr int64_t HIF4_HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM = 8;
constexpr int64_t HIF4_HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM = 256;
constexpr int64_t HIF4_HIFP4_OUTPUT_REG_ONELOOP = 2;
constexpr int64_t HIF4_HIFP4_MAXPER8_REG_ONELOOP = 16;

struct RotateQuantHif4ShapeInfo {
    int64_t M;
    int64_t N;
    int64_t B;
};

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_,
          typename FusionOp_ = DefaultFusion<DataTypeOut_, DataTypeIn_>>
class BlockEpilogueRotateQuantHif4 {
public:
    using DataTypeIn = DataTypeIn_;
    using DataTypeOut = DataTypeOut_;
    using DataTypeScale = DataTypeScale_;
    using FusionOp = FusionOp_;

    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
    };

    __aicore__ inline BlockEpilogueRotateQuantHif4() {}

    __aicore__ inline void Init(Params const& params, const ProblemShape& problemShape, bfloat16_t alpha,
                                bool needClamp, int64_t axis, int64_t roundMode, int64_t scaleAlg, float dstTypeMax,
                                float invDstTypeMax)
    {
        (void)alpha;
        (void)needClamp;
        (void)axis;
        (void)roundMode;
        (void)scaleAlg;
        (void)dstTypeMax;
        (void)invDstTypeMax;
        SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.outGmAddr));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.scaleGmAddr));
        shape_.M = Get<MNK_M>(problemShape);
        shape_.N = Get<MNK_N>(problemShape);
        shape_.B = Get<MNK_B>(problemShape);

        pipe_.InitBuffer(bufQueue_, TOTAL_UB_SIZE);
        xTensor_ = bufQueue_.Get<int16_t>();
        e8Tensor_ = xTensor_[HIF4_X_SIZE / sizeof(int16_t)].template ReinterpretCast<uint16_t>();
        e1p8Tensor_ = e8Tensor_[HIF4_E8_SIZE / sizeof(uint16_t)].template ReinterpretCast<uint32_t>();
        e1p16Tensor_ = e1p8Tensor_[HIF4_E1P8_SIZE / sizeof(uint32_t)].template ReinterpretCast<uint8_t>();
        maxPer4Tensor_ = e1p16Tensor_[HIF4_E1P16_SIZE].template ReinterpretCast<uint16_t>();
        maxPer8Tensor_ = maxPer4Tensor_[HIF4_MAXPER4_SIZE / sizeof(uint16_t)].template ReinterpretCast<uint16_t>();
        tmpScaleTensor_ = maxPer8Tensor_[HIF4_MAXPER8_SIZE / sizeof(uint16_t)];
        yTensor_ = tmpScaleTensor_[HIF4_TMPSCALE_SIZE / sizeof(uint16_t)].template ReinterpretCast<uint8_t>();
        scaleTensor_ = yTensor_[HIF4_Y_SIZE].template ReinterpretCast<float>();

        dataCopyOutParams_.blockCount = 1;
        dataCopyOutParams_.blockLen = 0;
        dataCopyOutParams_.srcStride = 0;
        dataCopyOutParams_.dstStride = 0;
        dataCopyOutScaleParams_.blockCount = 1;
        dataCopyOutScaleParams_.blockLen = 0;
        dataCopyOutScaleParams_.srcStride = 0;
        dataCopyOutScaleParams_.dstStride = 0;

        eventIdVToMte3_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        eventIdMte3ToV_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void Quant(uint64_t offset, int64_t blkM, int64_t k)
    {
        int64_t mnSize = blkM * k;
        int64_t yOffset = offset * k / HIF4_DIGIT_TWO;
        int64_t groupNum = (mnSize + HIF4_GROUP_ELE_NUM - 1) / HIF4_GROUP_ELE_NUM;
        int64_t scaleGmOffset = offset * ((k + HIF4_GROUP_ELE_NUM - 1) / HIF4_GROUP_ELE_NUM);

        int64_t regLoop = (groupNum + HIF4_HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM - 1) /
                          HIF4_HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM;
        int64_t maxPer8RegLoop = (groupNum + HIF4_HIFP4_MAXPER8_REG_ONELOOP - 1) / HIF4_HIFP4_MAXPER8_REG_ONELOOP;
        int64_t scaleRegLoop = (groupNum + HIF4_HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM - 1) /
                               HIF4_HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM;
        int64_t outRegLoop = (groupNum + HIF4_HIFP4_OUTPUT_REG_ONELOOP - 1) / HIF4_HIFP4_OUTPUT_REG_ONELOOP;

        ComputeAll(regLoop, maxPer8RegLoop, scaleRegLoop, outRegLoop);

        dataCopyOutParams_.blockLen = mnSize / HIF4_DIGIT_TWO;
        dataCopyOutScaleParams_.blockLen = groupNum * sizeof(float);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        CopyOut(yOffset, scaleGmOffset);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
    }

    __aicore__ inline void ComputeAll(int64_t regLoop, int64_t maxPer8RegLoop, int64_t scaleRegLoop, int64_t outRegLoop)
    {
        auto e8Tensor = e8Tensor_;
        auto e1p8Tensor = e1p8Tensor_;
        auto e1p16Tensor = e1p16Tensor_;
        auto maxPer4Tensor = maxPer4Tensor_;
        auto maxPer8Tensor = maxPer8Tensor_;
        auto tmpScaleTensor = tmpScaleTensor_;
        auto yTensor = yTensor_;
        auto scaleTensor = scaleTensor_;

        auto xAddr1 = (__ubuf__ float*)xTensor_.GetPhyAddr();
        auto xAddr2 = (__ubuf__ DataTypeIn*)xTensor_.GetPhyAddr();
        auto yAddr = (__ubuf__ int32_t*)yTensor.GetPhyAddr();
        auto scaleAddr = (__ubuf__ uint16_t*)scaleTensor.GetPhyAddr();

        auto e8BufAddr = (__ubuf__ uint16_t*)e8Tensor.GetPhyAddr();
        auto e8BufAddr2 = (__ubuf__ uint8_t*)e8Tensor.GetPhyAddr();
        auto e1p8BufAddr = (__ubuf__ uint32_t*)e1p8Tensor.GetPhyAddr();
        auto e1p16BufAddr = (__ubuf__ uint8_t*)e1p16Tensor.GetPhyAddr();
        auto tmpScaleAddr = (__ubuf__ uint16_t*)tmpScaleTensor.GetPhyAddr();
        auto maxPer4BufAddr = (__ubuf__ uint16_t*)maxPer4Tensor.GetPhyAddr();
        auto maxPer8BufAddr = (__ubuf__ uint16_t*)maxPer8Tensor.GetPhyAddr();

        auto e8BufRead1Addr = e8BufAddr;
        auto e8BufRead2Addr = e8BufAddr2;
        auto e1p8BufReadAddr = (__ubuf__ uint8_t*)e1p8Tensor.GetPhyAddr();
        auto e1p16BufReadAddr = e1p16BufAddr;
        auto tmpScaleReadAddr = tmpScaleAddr;
        auto maxPer4BufReadAddr = maxPer4BufAddr;
        auto maxPer4BufRead2Addr = maxPer4BufAddr;
        auto maxPer8BufReadAddr = maxPer8BufAddr;

        VF_CALL<ComputeMaxPer4>(static_cast<uint16_t>(regLoop), xAddr1, maxPer4BufAddr);
        VF_CALL<ComputeMaxPer8>(static_cast<uint16_t>(maxPer8RegLoop), maxPer4BufReadAddr, e8BufAddr, maxPer8BufAddr);
        VF_CALL<ComputeTmpScale>(static_cast<uint16_t>(regLoop), maxPer4BufRead2Addr, maxPer8BufReadAddr,
                                 e8BufRead1Addr, e1p8BufAddr, e1p16BufAddr, tmpScaleAddr);
        VF_CALL<ComputeScale>(static_cast<uint16_t>(scaleRegLoop), e8BufRead2Addr, e1p8BufReadAddr, e1p16BufReadAddr,
                              scaleAddr);
        VF_CALL<ComputeOut>(static_cast<uint16_t>(outRegLoop), xAddr2, tmpScaleReadAddr, yAddr);
    }

    __aicore__ inline void CopyOut(const int64_t& yOffset, const int64_t& scaleGmOffset)
    {
        auto yTensor = yTensor_;
        auto scaleTensor = scaleTensor_;
        uint16_t yBlockCount = 1;
        uint32_t yDataSize = static_cast<uint32_t>(dataCopyOutParams_.blockLen);
        copy_ubuf_to_gm_align_v2(yGm_[yOffset].GetPhyAddr(), (__ubuf__ void*)yTensor.GetPhyAddr(), 0, yBlockCount,
                                 yDataSize, false, false, 0, 0, yDataSize, yDataSize, false);

        uint16_t scaleBlockCount = 1;
        uint32_t scaleDataSize = static_cast<uint32_t>(dataCopyOutScaleParams_.blockLen);
        copy_ubuf_to_gm_align_v2(scaleGm_[scaleGmOffset].GetPhyAddr(), (__ubuf__ void*)scaleTensor.GetPhyAddr(), 0,
                                 scaleBlockCount, scaleDataSize, false, false, 0, 0, scaleDataSize, scaleDataSize,
                                 false);
    }

    __aicore__ inline auto GetTensor() { return ubLocal_; }

    __aicore__ inline void operator()(uint64_t offsetM, int64_t blkM, int64_t k) { Quant(offsetM, blkM, k); }

    __host_aicore__ static Status CanImplement(Arguments const& args) { return Status::success; }

private:
    // Upper bound of groups processed in one tile on UB.
    static constexpr int64_t HIF4_GROUP_NUM_MAX = 1024;
    // Reserve space for x (output of matmul Fixpipe) at the front of UB.
    static constexpr int64_t HIF4_X_SIZE = HIF4_GROUP_NUM_MAX * HIF4_GROUP_ELE_NUM * sizeof(int16_t); // 131072
    // Intermediate buffer sizes (in bytes):
    static constexpr int64_t HIF4_E8_SIZE = HIF4_GROUP_NUM_MAX * sizeof(int16_t);            // 512
    static constexpr int64_t HIF4_E1P8_SIZE = HIF4_GROUP_NUM_MAX * sizeof(int32_t);          // 1024
    static constexpr int64_t HIF4_E1P16_SIZE = HIF4_GROUP_NUM_MAX * sizeof(int16_t);         // 512
    static constexpr int64_t HIF4_MAXPER4_SIZE = HIF4_GROUP_NUM_MAX * 16 * sizeof(int16_t);  // 8192
    static constexpr int64_t HIF4_MAXPER8_SIZE = HIF4_GROUP_NUM_MAX * 8 * sizeof(int16_t);   // 4096
    static constexpr int64_t HIF4_TMPSCALE_SIZE = HIF4_GROUP_NUM_MAX * 32 * sizeof(int16_t); // 16384
    static constexpr int64_t HIF4_Y_SIZE = HIF4_GROUP_NUM_MAX * HIF4_GROUP_BYTE_NUM;         // 8192
    static constexpr int64_t HIF4_SCALE_SIZE = HIF4_GROUP_NUM_MAX * sizeof(float);           // 1024

    static constexpr int64_t sysRegLen_ = 256;
    static constexpr int64_t gmCopyInRegStep_ = sysRegLen_ / sizeof(float) * HIF4_DIGIT_TWO;
    static constexpr int64_t gmCopyOutRegStep_ = sysRegLen_ / sizeof(int32_t) / HIF4_DIGIT_FOUR;
    static constexpr int64_t gmCopyOutScaleRegStep_ = sysRegLen_ / sizeof(float);
    static constexpr int64_t gmCopyInRegStep2_ = sysRegLen_ / sizeof(DataTypeIn);
    static constexpr int64_t gmCopyTmpBufE8Step_ = sysRegLen_ * HIF4_DIGIT_TWO;
    static constexpr int64_t gmCopyTmpBufOtherStep_ = sysRegLen_;

    RotateQuantHif4ShapeInfo shape_;
    TPipe pipe_;

    LocalTensor<DataTypeIn> ubLocal_{TPosition::VECIN, 0, TOTAL_UB_SIZE};
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<float> scaleGm_;

    TBuf<TPosition::VECCALC> bufQueue_;
    LocalTensor<int16_t> xTensor_;
    LocalTensor<uint16_t> e8Tensor_;
    LocalTensor<uint32_t> e1p8Tensor_;
    LocalTensor<uint8_t> e1p16Tensor_;
    LocalTensor<uint16_t> maxPer4Tensor_;
    LocalTensor<uint16_t> maxPer8Tensor_;
    LocalTensor<uint16_t> tmpScaleTensor_;
    LocalTensor<uint8_t> yTensor_;
    LocalTensor<float> scaleTensor_;

    event_t eventIdVToMte3_;
    event_t eventIdMte3ToV_;

    DataCopyExtParams dataCopyOutParams_;
    DataCopyExtParams dataCopyOutScaleParams_;

    constexpr static MicroAPI::CastTrait castTrait_ = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    static __simd_vf__ inline void ComputeMaxPer4(uint16_t regLoop, __ubuf__ float* xAddr1,
                                                  __ubuf__ uint16_t* maxPer4BufAddr)
    {
        MicroAPI::RegTensor<DataTypeIn> vregIn1;
        MicroAPI::RegTensor<DataTypeIn> vregIn2;
        MicroAPI::RegTensor<DataTypeIn> vregIn3;
        MicroAPI::RegTensor<DataTypeIn> vregIn4;
        MicroAPI::RegTensor<DataTypeIn> vregMax1;
        MicroAPI::RegTensor<DataTypeIn> vregMax2;
        MicroAPI::RegTensor<DataTypeIn> vregOut;
        MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, 0x7fff);
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                (MicroAPI::RegTensor<float>&)vregIn1, (MicroAPI::RegTensor<float>&)vregIn2, xAddr1, gmCopyInRegStep_);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1, expMaskBF16, maskAll);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2, expMaskBF16, maskAll);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                (MicroAPI::RegTensor<float>&)vregIn3, (MicroAPI::RegTensor<float>&)vregIn4, xAddr1, gmCopyInRegStep_);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3, expMaskBF16, maskAll);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4, expMaskBF16, maskAll);
            MicroAPI::Max(vregMax1, vregIn1, vregIn2, maskAll);
            MicroAPI::Max(vregMax2, vregIn3, vregIn4, maskAll);
            MicroAPI::DeInterleave(vregMax1, vregMax2, vregMax1, vregMax2);
            MicroAPI::Max(vregOut, vregMax1, vregMax2, maskAll);
            if constexpr (IsSameType<DataTypeIn, half>::value) {
                AscendC::MicroAPI::Cast<bfloat16_t, DataTypeIn, castTraitHalf2Bf16>(vregOutbf16, vregOut, maskAll);
                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxPer4BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregOutbf16, gmCopyInRegStep2_, maskAll);
            } else {
                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxPer4BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregOut, gmCopyInRegStep2_, maskAll);
            }
        }
    }

    static __simd_vf__ inline void ComputeMaxPer8(uint16_t maxPer8RegLoop, __ubuf__ uint16_t* maxPer4BufReadAddr,
                                                  __ubuf__ uint16_t* e8BufAddr, __ubuf__ uint16_t* maxPer8BufAddr)
    {
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper4;
        MicroAPI::RegTensor<bfloat16_t> vreg1Maxper4;
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper8;
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper64;
        MicroAPI::RegTensor<bfloat16_t> vreg1Maxper64;
        MicroAPI::RegTensor<uint16_t> vreg0E6m2;
        MicroAPI::RegTensor<uint16_t> vreg1E6m2;
        MicroAPI::RegTensor<uint16_t> vregBf16E6M2ScaleBias;
        MicroAPI::Duplicate(vregBf16E6M2ScaleBias, 0x3e12);
        MicroAPI::UnalignReg ureg1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitE6M2 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < maxPer8RegLoop; loopIdx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, (MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4,
                maxPer4BufReadAddr, gmCopyInRegStep2_ * HIF4_DIGIT_TWO);
            MicroAPI::Max(vreg0Maxper8, vreg0Maxper4, vreg1Maxper4, maskAll);
            MicroAPI::Interleave(vreg0Maxper4, vreg1Maxper4, vreg0Maxper4, vreg1Maxper4);

            MicroAPI::ReduceMaxWithDataBlock<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64,
                                                       (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, maskAll);
            MicroAPI::ReduceMaxWithDataBlock<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg1Maxper64,
                                                       (MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4, maskAll);

            MicroAPI::Mul(vreg0Maxper64, vreg0Maxper64, (MicroAPI::RegTensor<bfloat16_t>&)vregBf16E6M2ScaleBias,
                          maskAll);
            MicroAPI::Mul(vreg1Maxper64, vreg1Maxper64, (MicroAPI::RegTensor<bfloat16_t>&)vregBf16E6M2ScaleBias,
                          maskAll);

            MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0E6m2,
                                                                  vreg0Maxper64, maskAll);
            MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg1E6m2,
                                                                  vreg1Maxper64, maskAll);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxPer8BufAddr, (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper8, gmCopyInRegStep2_, maskAll);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                e8BufAddr, vreg0E6m2, ureg1, 8);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                e8BufAddr, vreg1E6m2, ureg1, 8);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(e8BufAddr, ureg1, 0);
    }

    static __simd_vf__ inline void ComputeTmpScale(uint16_t regLoop, __ubuf__ uint16_t* maxPer4BufRead2Addr,
                                                   __ubuf__ uint16_t* maxPer8BufReadAddr,
                                                   __ubuf__ uint16_t* e8BufRead1Addr, __ubuf__ uint32_t* e1p8BufAddr,
                                                   __ubuf__ uint8_t* e1p16BufAddr, __ubuf__ uint16_t* tmpScaleAddr)
    {
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper4;
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper8;
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper64;
        MicroAPI::RegTensor<bfloat16_t> vb16_Ea_rec_bf16;
        MicroAPI::RegTensor<bfloat16_t> vregTmpScale;

        float number1 = 1.0;
        MicroAPI::RegTensor<bfloat16_t> vregNum1;
        MicroAPI::Duplicate(vregNum1, number1);
        float numberhalf = 0.5;
        MicroAPI::RegTensor<bfloat16_t> vregNumhalf;
        MicroAPI::Duplicate(vregNumhalf, static_cast<bfloat16_t>(numberhalf));

        MicroAPI::MaskReg vb1E8Comp;
        MicroAPI::MaskReg vb1E16Comp;
        MicroAPI::UnalignReg ureg1;
        static constexpr MicroAPI::CastTrait castTraitReciprocalCast = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64, e8BufRead1Addr, 8);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper8, maxPer8BufReadAddr, 64);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, maxPer4BufRead2Addr, 128);

            MicroAPI::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast>(
                (MicroAPI::RegTensor<bfloat16_t>&)vb16_Ea_rec_bf16, (MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0Maxper64,
                maskAll);
            MicroAPI::Mul(vreg0Maxper8, vreg0Maxper8, vb16_Ea_rec_bf16, maskAll);
            MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vreg0Maxper8, 4, maskAll);
            MicroAPI::Select(vreg0Maxper8, vregNumhalf, vregNum1, vb1E8Comp);
            MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);
            MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vreg0Maxper4, 2, maskAll);
            MicroAPI::Select(vreg0Maxper4, vregNumhalf, vregNum1, vb1E16Comp);
            MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            MicroAPI::Mul(vregTmpScale, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);

            MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                tmpScaleAddr + loopIdx * gmCopyTmpBufOtherStep_, (MicroAPI::RegTensor<uint16_t>&)vregTmpScale,
                (MicroAPI::RegTensor<uint16_t>&)vregTmpScale, maskAll);

            MicroAPI::DataCopyUnAlign<uint32_t>(e1p8BufAddr, vb1E8Comp, ureg1);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_PACK>(
                e1p16BufAddr, vb1E16Comp, 16);
        }
        MicroAPI::DataCopyUnAlignPost<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(e1p8BufAddr, ureg1, 0);
    }

    static __simd_vf__ inline void ComputeScale(uint16_t scaleRegLoop, __ubuf__ uint8_t* e8BufRead2Addr,
                                                __ubuf__ uint8_t* e1p8BufReadAddr, __ubuf__ uint8_t* e1p16BufReadAddr,
                                                __ubuf__ uint16_t* scaleAddr)
    {
        MicroAPI::RegTensor<uint8_t> vregE8;
        MicroAPI::RegTensor<uint8_t> vreg1E8;
        MicroAPI::RegTensor<uint8_t> vreg1E16First;
        MicroAPI::RegTensor<uint8_t> vreg1E16Second;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < scaleRegLoop; loopIdx++) {
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DS_B8>(
                vregE8, e8BufRead2Addr, gmCopyTmpBufE8Step_);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vreg1E8, e1p8BufReadAddr, gmCopyTmpBufOtherStep_);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vreg1E16First, e1p16BufReadAddr, gmCopyTmpBufOtherStep_);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vreg1E16Second, e1p16BufReadAddr, gmCopyTmpBufOtherStep_);

            MicroAPI::Interleave<uint8_t>(vregE8, vreg1E8, vregE8, vreg1E8);
            MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                scaleAddr + loopIdx * gmCopyTmpBufE8Step_, (MicroAPI::RegTensor<uint16_t>&)vregE8,
                (MicroAPI::RegTensor<uint16_t>&)vreg1E16First, maskAll);
            MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                scaleAddr + loopIdx * gmCopyTmpBufE8Step_ + gmCopyTmpBufOtherStep_,
                (MicroAPI::RegTensor<uint16_t>&)vreg1E8, (MicroAPI::RegTensor<uint16_t>&)vreg1E16Second, maskAll);
        }
    }

    static __simd_vf__ inline void ComputeOut(uint16_t outRegLoop, __ubuf__ DataTypeIn* xAddr2,
                                              __ubuf__ uint16_t* tmpScaleReadAddr, __ubuf__ int32_t* yAddr)
    {
        MicroAPI::RegTensor<DataTypeIn> vregOut;
        MicroAPI::RegTensor<uint16_t> vregScaleTmp;
        MicroAPI::RegTensor<fp4x2_e1m2_t> vbOut;
        MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < outRegLoop; loopIdx++) {
            if constexpr (IsSameType<DataTypeIn, half>::value) {
                MicroAPI::DataCopy<DataTypeIn, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    vregOut, xAddr2, gmCopyInRegStep2_);
                AscendC::MicroAPI::Cast<bfloat16_t, DataTypeIn, castTraitHalf2Bf16>(
                    (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16, vregOut, maskAll);
            } else {
                MicroAPI::DataCopy<DataTypeIn, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    vregOutbf16, xAddr2, gmCopyInRegStep2_);
            }
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
                vregScaleTmp, tmpScaleReadAddr, gmCopyOutScaleRegStep_);
            MicroAPI::Mul((MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16, (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16,
                          (MicroAPI::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut, vregOutbf16, maskAll);
            MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                yAddr, (MicroAPI::RegTensor<int32_t>&)vbOut, gmCopyOutRegStep_, maskAll);
        }
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif // __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202
