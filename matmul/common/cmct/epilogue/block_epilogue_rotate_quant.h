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
 * \file block_epilogue_rotate_quant.h
 * \brief Block epilogue for rotate quantization with MX FP4/FP8 support
 */

#pragma once
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "fusion/default_fusion_op.h"
#define FLOAT_OVERFLOW_MODE_CTRL 60

struct RotateQuantShapeInfo {
    int64_t M;
    int64_t N;
    int64_t B;
};

namespace Cmct {
namespace Gemm {
namespace Block {
using namespace AscendC;
using namespace AscendC::MicroAPI;
template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_,
          typename FusionOp_ = DefaultFusion<DataTypeOut_, DataTypeIn_>>
class BlockEpilogueRotateQuant {
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

    __aicore__ inline BlockEpilogueRotateQuant() {}

    __aicore__ inline void Init(Params const& params, const ProblemShape& problemShape, bfloat16_t alpha,
                                bool needClamp, int64_t axis, int64_t roundMode, int64_t scaleAlg, float dstTypeMax,
                                float invDstTypeMax)
    {
        // 饱和模式由单指令设置生效
        SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
        InitGlobalBuffer(params);
        InitProblemShape(problemShape);
        InitQuantParams(alpha, needClamp, axis, roundMode, scaleAlg, dstTypeMax, invDstTypeMax);
        InitLocalBuffer();
        InitEventIds();
    }

    __aicore__ inline void Quant(uint64_t offset, int64_t blkM, int64_t k)
    {
        uint32_t mnSize = blkM * k;
        uint64_t yOffset = offset * static_cast<uint64_t>(k);
        ComputeMxQuant(xTensor_, yTensor_, maxTensor_, eMaxTensor_, scaleTensor_, deQuantScaleTensor_, mnSize,
                       needClamp_);

        uint32_t nScaleSize = RotateQuantCeilDiv(shape_.N, GROUP_SIZE);
        uint32_t isOdd = nScaleSize & 1;
        nScaleSize += isOdd;
        if (shape_.B > 1) {
            uint64_t mIdx = offset / shape_.B;
            uint64_t bIdx = offset % shape_.B;
            uint64_t scaleOffset = mIdx * nScaleSize + RotateQuantCeilDiv(bIdx * k, GROUP_SIZE);
            uint16_t scaleBlockN = RotateQuantCeilDiv(k, GROUP_SIZE);
            uint16_t dataSize = scaleBlockN + ((bIdx == shape_.B - 1) ? isOdd : 0);

            Duplicate(scaleBlockTensor_, (int8_t)0, blkM * dataSize);
            ComputeTransLayout(scaleTensor_, scaleBlockTensor_, blkM, scaleBlockN);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            CopyOutputFromUbToGm(yOffset, yTensor_, blkM, k, shape_.N);
            CopyScaleFromUbToGm(scaleOffset, scaleBlockTensor_, blkM, dataSize, nScaleSize);
        } else if (isOdd == 1) {
            uint32_t nKRatio = shape_.N / k;
            uint32_t mIdx = offset / nKRatio;
            uint32_t nkIdx = offset % nKRatio;
            uint64_t scaleOffset = mIdx * nScaleSize + nkIdx;
            Duplicate(scaleBlockTensor_, (int8_t)0, blkM * GROUP_SIZE);
            PadScale(scaleTensor_, scaleBlockTensor_, nKRatio, nkIdx, blkM);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            CopyOutputFromUbToGm(yOffset, yTensor_, 1, mnSize, 0);
            CopyPadScaleFromUbToGm(scaleOffset, scaleBlockTensor_, nKRatio, nkIdx, blkM);
        } else {
            uint64_t scaleOffset = offset * RotateQuantCeilDiv(k, GROUP_SIZE);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
            CopyOutputFromUbToGm(yOffset, yTensor_, 1, mnSize, 0);
            uint32_t scaleSize = RotateQuantCeilDiv(mnSize, GROUP_SIZE);
            CopyScaleFromUbToGm(scaleOffset, scaleTensor_, 1, scaleSize, 0);
        }

        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
    }

    __aicore__ inline auto GetTensor() { return ubLocal_; }

    __aicore__ inline void operator()(uint64_t offsetM, int64_t blkM, int64_t k) { Quant(offsetM, blkM, k); }

    __host_aicore__ static Status CanImplement(Arguments const& args) { return Status::success; }

private:
    static constexpr int64_t MODE_RINT = 4;
    static constexpr int64_t MODE_ROUND = 0;
    static constexpr int64_t MODE_FLOOR = 1;

    static constexpr int32_t GROUP_SIZE = 32;
    static constexpr int32_t MN_SIZE = 16 * 1024;
    static constexpr int32_t EMAX_SIZE = 512;

    static constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
    static constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
    static constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
    static constexpr int16_t SHR_NUM_FOR_BF16 = 7;
    static constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
    static constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
    static constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
    static constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
    static constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
    static constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
    static constexpr uint16_t SIGN_MASK_FOR_BF16 = 0x8000;
    static constexpr uint16_t ADD_VALUE_FOR_BF16_MAN1 = 0x003f;
    static constexpr uint16_t ADD_VALUE_FOR_BF16_MAN2 = 0x001f;

    static constexpr float SIX_FLOAT = 6.0f;
    static constexpr float SEVEN_FLOAT = 7.0f;
    static constexpr float TWELVE_FLOAT = 12.0f;
    static constexpr uint32_t STORE_UNALIGN_STRIDE_BYTES = 8;
    static constexpr uint32_t SCALE_STORE_STRIDE = 32;
    static constexpr int16_t SHR_NUM_FOR_FP32 = 23;
    static constexpr uint32_t NUMBER_ZERO = 0x00000000;
    static constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;
    static constexpr uint32_t NUMBER_HALF = 0x00400000;
    static constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
    static constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
    static constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
    static constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
    static constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
    static constexpr uint32_t FP8_E5M2_MAX = 0x37924925; // 1/57344的float32表示 57334是E5M2所能表示的最大值
    static constexpr uint32_t FP8_E4M3_MAX = 0x3b124925; // 1/448的float32表示 448是E4M3所能表示的最大值

    static constexpr uint32_t vfLen8 = GetVecLen() / sizeof(uint8_t);
    static constexpr uint32_t vfLen16 = GetVecLen() / sizeof(uint16_t);
    static constexpr uint32_t vfLen32 = GetVecLen() / sizeof(uint32_t);

    static constexpr CastTrait castTraitZero = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                RoundMode::UNKNOWN};
    static constexpr CastTrait castTraitOne = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                               RoundMode::UNKNOWN};
    static constexpr CastTrait castTrait32to80 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_RINT};
    static constexpr CastTrait castTrait32to81 = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_RINT};
    static constexpr CastTrait castTrait32to82 = {RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_RINT};
    static constexpr CastTrait castTrait32to83 = {RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_RINT};

    RotateQuantShapeInfo shape_;
    TPipe pipe_;

    LocalTensor<DataTypeIn> ubLocal_{TPosition::VECIN, 0, TOTAL_UB_SIZE};
    GlobalTensor<int8_t> cGlobal_;
    GlobalTensor<int8_t> scaleGlobal_;

    TBuf<QuePosition::VECCALC> bufQueue_;
    LocalTensor<bfloat16_t> xTensor_;
    LocalTensor<int8_t> yTensor_;
    LocalTensor<bfloat16_t> maxTensor_;
    LocalTensor<uint16_t> eMaxTensor_;
    LocalTensor<int8_t> scaleTensor_;
    LocalTensor<uint16_t> deQuantScaleTensor_;
    LocalTensor<int8_t> scaleBlockTensor_;

    event_t eventIdVToMte3_;
    event_t eventIdMte3ToV_;

    bfloat16_t alpha_ = 0.0f;
    bool needClamp_ = false;
    int64_t axis_ = 0;
    int64_t roundMode_ = 0;
    int64_t scaleAlg_ = 0;
    float dstTypeMax_ = 0.0f;
    float invDstTypeMax_ = 0.0f;
    uint16_t addValueBit_ = 0;
    uint16_t maxExpValue_ = 0;
    uint32_t dtypeMax_ = 0;

    template <typename T1, typename T2>
    __aicore__ inline auto RotateQuantCeilDiv(T1 a, T2 b) -> decltype(a + b)
    {
        return (b == 0) ? 0 : (a + b - 1) / b;
    }

    template <typename T1, typename T2>
    __aicore__ inline auto RotateQuantAlign(T1 a, T2 b) -> decltype(a + b)
    {
        return (b == 0) ? 0 : ((a + b - 1) / b) * b;
    }

    __aicore__ inline void InitGlobalBuffer(Params const& params)
    {
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.outGmAddr));
        scaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.scaleGmAddr));
    }

    __aicore__ inline void InitProblemShape(const ProblemShape& problemShape)
    {
        shape_.M = Get<MNK_M>(problemShape);
        shape_.N = Get<MNK_N>(problemShape);
        shape_.B = Get<MNK_B>(problemShape);
    }

    __aicore__ inline void InitQuantParams(bfloat16_t alpha, bool needClamp, int64_t axis, int64_t roundMode,
                                           int64_t scaleAlg, float dstTypeMax, float invDstTypeMax)
    {
        alpha_ = alpha;
        needClamp_ = needClamp;
        axis_ = axis;
        roundMode_ = roundMode;
        scaleAlg_ = scaleAlg;
        dstTypeMax_ = dstTypeMax;
        invDstTypeMax_ = invDstTypeMax;

        if (dstTypeMax_ == SIX_FLOAT) {
            addValueBit_ = ADD_VALUE_FOR_BF16_MAN1;
        } else if (dstTypeMax_ == SEVEN_FLOAT) {
            addValueBit_ = ADD_VALUE_FOR_BF16_MAN2;
        }

        if constexpr (IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
            maxExpValue_ = FP4_E2M1_MAX_EXP;
        } else if constexpr (IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
            maxExpValue_ = FP8_E4M3_MAX_EXP;
            dtypeMax_ = FP8_E4M3_MAX;
        } else if constexpr (IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
            maxExpValue_ = FP8_E5M2_MAX_EXP;
            dtypeMax_ = FP8_E5M2_MAX;
        }
    }

    __aicore__ inline void InitLocalBuffer()
    {
        pipe_.InitBuffer(bufQueue_, TOTAL_UB_SIZE);
        xTensor_ = bufQueue_.Get<bfloat16_t>();
        yTensor_ = xTensor_[MN_SIZE].template ReinterpretCast<int8_t>();
        maxTensor_ = yTensor_[MN_SIZE].template ReinterpretCast<bfloat16_t>();
        eMaxTensor_ = maxTensor_[EMAX_SIZE].template ReinterpretCast<uint16_t>();
        deQuantScaleTensor_ = eMaxTensor_[EMAX_SIZE];
        scaleTensor_ = deQuantScaleTensor_[EMAX_SIZE].template ReinterpretCast<int8_t>();
        scaleBlockTensor_ = scaleTensor_[EMAX_SIZE];
    }

    __aicore__ inline void InitEventIds()
    {
        eventIdVToMte3_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        eventIdMte3ToV_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void CopyOutputFromUbToGm(uint64_t offset, LocalTensor<int8_t>& src, uint16_t blockCount,
                                                uint32_t dataSize, uint32_t dstStride)
    {
        if constexpr (IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
            offset = offset >> 1;
            dataSize = dataSize >> 1;
            dstStride = dstStride >> 1;
        }
        copy_ubuf_to_gm_align_v2(cGlobal_[offset].GetPhyAddr(), (__ubuf__ void*)src.GetPhyAddr(), 0, blockCount,
                                 dataSize, 0, dstStride, dataSize);
    }

    __aicore__ inline void CopyScaleFromUbToGm(uint64_t offset, LocalTensor<int8_t>& src, uint16_t blockCount,
                                               uint32_t dataSize, uint32_t dstStride)
    {
        copy_ubuf_to_gm_align_v2(scaleGlobal_[offset].GetPhyAddr(), (__ubuf__ void*)src.GetPhyAddr(), 0, blockCount,
                                 dataSize, 0, dstStride, SCALE_STORE_STRIDE);
    }

    __aicore__ inline void CopyPadScaleFromUbToGm(uint64_t offset, LocalTensor<int8_t>& src, uint32_t nKRatio,
                                                  uint32_t nkIdx, uint32_t blkM)
    {
        uint16_t firstNum = (nKRatio - nkIdx <= blkM) ? nKRatio - nkIdx : blkM;
        uint16_t rowLoopNum = (firstNum < blkM) ? (blkM - firstNum) / nKRatio : 0;
        uint16_t lastNum = (firstNum < blkM) ? (blkM - firstNum) % nKRatio : 0;

        uint32_t dataSize = firstNum + ((nkIdx + firstNum == nKRatio) ? 1 : 0);
        copy_ubuf_to_gm_align_v2(scaleGlobal_[offset].GetPhyAddr(), (__ubuf__ void*)src.GetPhyAddr(), 0, 1, dataSize, 0,
                                 0, 0);
        offset += dataSize;
        uint32_t srcOffset = RotateQuantAlign(firstNum + 1, SCALE_STORE_STRIDE);

        if (rowLoopNum > 0) {
            dataSize = nKRatio + 1;
            uint32_t srcStride = RotateQuantAlign(dataSize, SCALE_STORE_STRIDE);
            copy_ubuf_to_gm_align_v2(scaleGlobal_[offset].GetPhyAddr(), (__ubuf__ void*)src[srcOffset].GetPhyAddr(), 0,
                                     rowLoopNum, dataSize, 0, dataSize, srcStride);
            offset += rowLoopNum * dataSize;
            srcOffset += rowLoopNum * srcStride;
        }

        if (lastNum > 0) {
            copy_ubuf_to_gm_align_v2(scaleGlobal_[offset].GetPhyAddr(), (__ubuf__ void*)src[srcOffset].GetPhyAddr(), 0,
                                     1, lastNum, 0, 0, 0);
        }
    }

    __aicore__ inline void ComputeMxQuant(LocalTensor<bfloat16_t>& xTensor, LocalTensor<int8_t>& yTensor,
                                          LocalTensor<bfloat16_t>& maxTensor, LocalTensor<uint16_t>& eMaxTensor,
                                          LocalTensor<int8_t>& scaleTensor, LocalTensor<uint16_t>& deQuantScaleTensor,
                                          uint32_t totalDataInUB, bool needClamp)
    {
        uint32_t oneRepeatSize = GetVecLen() / sizeof(DataTypeIn);
        uint16_t repeatCount = static_cast<uint16_t>(RotateQuantCeilDiv(totalDataInUB, oneRepeatSize * 2));
        uint16_t scaleNum = static_cast<uint16_t>(RotateQuantCeilDiv(totalDataInUB, GROUP_SIZE));
        uint16_t repeatScaleCount = static_cast<uint16_t>(RotateQuantCeilDiv(scaleNum, oneRepeatSize));
        uint16_t repeatScaleHalfCount = static_cast<uint16_t>(RotateQuantCeilDiv(scaleNum, oneRepeatSize / 2));

        __ubuf__ bfloat16_t* xAddr = (__ubuf__ bfloat16_t*)xTensor.GetPhyAddr();
        __ubuf__ bfloat16_t* maxAddr = (__ubuf__ bfloat16_t*)maxTensor.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)eMaxTensor.GetPhyAddr();
        ComputeExpMax(maxExpAddr, maxAddr, xAddr, totalDataInUB, repeatCount, oneRepeatSize, needClamp);

        __ubuf__ uint16_t* deScaleAddr = (__ubuf__ uint16_t*)deQuantScaleTensor.GetPhyAddr();
        __ubuf__ uint16_t* scaleAddr = (__ubuf__ uint16_t*)scaleTensor.GetPhyAddr();

        ComputeScale(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeatScaleCount, repeatScaleHalfCount);

        __ubuf__ int8_t* yAddr = (__ubuf__ int8_t*)yTensor.GetPhyAddr();
        if (needClamp) {
            ComputeQuantData<true>(yAddr, maxAddr, xAddr, deScaleAddr, totalDataInUB, repeatCount);
        } else {
            ComputeQuantData<false>(yAddr, maxAddr, xAddr, deScaleAddr, totalDataInUB, repeatCount);
        }
    }

    __aicore__ inline void ComputeExpMax(__ubuf__ uint16_t* maxExpAddr, __ubuf__ bfloat16_t* maxAddr,
                                         __ubuf__ bfloat16_t* xAddr, uint32_t totalDataInUB, uint16_t repeatCount,
                                         uint32_t oneRepeatSize, bool needClamp)
    {
        if (needClamp) {
            if (scaleAlg_ == 0) {
                VF_CALL<ClampExpMaxVf<true>>(maxExpAddr, maxAddr, xAddr, totalDataInUB, alpha_, repeatCount,
                                             oneRepeatSize);
            } else {
                VF_CALL<ClampExpMaxVf<false>>(maxExpAddr, maxAddr, xAddr, totalDataInUB, alpha_, repeatCount,
                                              oneRepeatSize);
            }
        } else {
            if (scaleAlg_ == 0) {
                VF_CALL<ExpMaxVf<true>>(maxExpAddr, xAddr, totalDataInUB, repeatCount, oneRepeatSize);
            } else {
                VF_CALL<ExpMaxVf<false>>(maxExpAddr, xAddr, totalDataInUB, repeatCount, oneRepeatSize);
            }
        }
    }

    __aicore__ inline void ComputeScale(__ubuf__ uint16_t* scaleAddr, __ubuf__ uint16_t* deScaleAddr,
                                        __ubuf__ uint16_t* maxExpAddr, uint16_t scaleNum, uint16_t repeatScaleCount,
                                        uint16_t repeatScaleHalfCount)
    {
        if (scaleAlg_ == 1) {
            VF_CALL<ScaleVfcuBLAS<true>>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeatScaleHalfCount,
                                         invDstTypeMax_, dtypeMax_);
        } else if (scaleAlg_ == 0) {
            VF_CALL<ScaleVf<false>>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeatScaleCount, addValueBit_,
                                    maxExpValue_);
        } else if (dstTypeMax_ == SIX_FLOAT || dstTypeMax_ == SEVEN_FLOAT) {
            VF_CALL<ScaleVf<true>>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeatScaleCount, addValueBit_,
                                   maxExpValue_);
        } else {
            VF_CALL<ScaleVfcuBLAS<false>>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeatScaleHalfCount,
                                          invDstTypeMax_, dtypeMax_);
        }
    }

    template <bool needClamp>
    __aicore__ inline void ComputeQuantData(__ubuf__ int8_t* yAddr, __ubuf__ bfloat16_t* maxAddr,
                                            __ubuf__ bfloat16_t* xAddr, __ubuf__ uint16_t* deScaleAddr,
                                            uint32_t totalDataInUB, uint16_t repeatCount)
    {
        if constexpr (IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
            if (roundMode_ == MODE_RINT) {
                VF_CALL<QuantVfFP4<RoundMode::CAST_RINT, needClamp>>(yAddr, xAddr, deScaleAddr, maxAddr, totalDataInUB,
                                                                     repeatCount);
            } else if (roundMode_ == MODE_ROUND) {
                VF_CALL<QuantVfFP4<RoundMode::CAST_ROUND, needClamp>>(yAddr, xAddr, deScaleAddr, maxAddr, totalDataInUB,
                                                                      repeatCount);
            } else if (roundMode_ == MODE_FLOOR) {
                VF_CALL<QuantVfFP4<RoundMode::CAST_FLOOR, needClamp>>(yAddr, xAddr, deScaleAddr, maxAddr, totalDataInUB,
                                                                      repeatCount);
            }
        } else {
            VF_CALL<QuantVfFP8<DataTypeOut, needClamp>>(yAddr, xAddr, deScaleAddr, maxAddr, totalDataInUB, repeatCount);
        }
    }

    __aicore__ inline void ComputeTransLayout(LocalTensor<int8_t>& scaleTensor, LocalTensor<int8_t>& scaleBlockTensor,
                                              uint16_t M, uint16_t scaleBlockN)
    {
        __ubuf__ int8_t* qscaleAddr = (__ubuf__ int8_t*)scaleTensor.GetPhyAddr();
        __ubuf__ int8_t* qscaleBlkAddr = (__ubuf__ int8_t*)scaleBlockTensor.GetPhyAddr();
        VF_CALL<TransLayoutVf>(qscaleAddr, qscaleBlkAddr, M, scaleBlockN);
    }

    __aicore__ inline void PadScale(LocalTensor<int8_t>& scaleTensor, LocalTensor<int8_t>& scaleBlockTensor,
                                    uint32_t nKRatio, uint32_t nkIdx, uint32_t blkM)
    {
        uint16_t firstNum = (nKRatio - nkIdx <= blkM) ? nKRatio - nkIdx : blkM;
        uint16_t rowLoopNum = (firstNum < blkM) ? (blkM - firstNum) / nKRatio : 0;
        uint16_t lastNum = (firstNum < blkM) ? (blkM - firstNum) % nKRatio : 0;
        __ubuf__ int8_t* qscaleAddr = (__ubuf__ int8_t*)scaleTensor.GetPhyAddr();
        __ubuf__ int8_t* qscaleBlkAddr = (__ubuf__ int8_t*)scaleBlockTensor.GetPhyAddr();
        VF_CALL<PadScaleVf>(qscaleAddr, qscaleBlkAddr, nkIdx, nKRatio, firstNum, lastNum, rowLoopNum);
    }

    template <bool needExp>
    static __simd_vf__ inline void ExpMaxVf(__ubuf__ uint16_t* dstPtr, __ubuf__ bfloat16_t* srcPtr, uint32_t count,
                                            uint16_t repeatTimes, uint32_t oneRepeatSize)
    {
        RegTensor<bfloat16_t> vSrcReg0;
        RegTensor<bfloat16_t> vSrcReg1;
        RegTensor<uint16_t> vdMaxExp;
        RegTensor<uint16_t> expMaskBF16;
        RegTensor<uint16_t> absMask16Bit;

        if constexpr (needExp) {
            Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        } else {
            Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);
        }
        MaskReg maskReg;
        UnalignReg u1;
        AddrReg aReg;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            aReg = CreateAddrReg<uint32_t>(i, oneRepeatSize);
            maskReg = UpdateMask<bfloat16_t>(count);

            LoadAlign<bfloat16_t, LoadDist::DIST_DINTLV_B16>(vSrcReg0, vSrcReg1, srcPtr, aReg);
            if constexpr (needExp) {
                And((RegTensor<uint16_t>&)vSrcReg0, (RegTensor<uint16_t>&)vSrcReg0, expMaskBF16, maskReg);
                And((RegTensor<uint16_t>&)vSrcReg1, (RegTensor<uint16_t>&)vSrcReg1, expMaskBF16, maskReg);
            } else {
                And((RegTensor<uint16_t>&)vSrcReg0, (RegTensor<uint16_t>&)vSrcReg0, absMask16Bit, maskReg);
                And((RegTensor<uint16_t>&)vSrcReg1, (RegTensor<uint16_t>&)vSrcReg1, absMask16Bit, maskReg);
            }
            Max(vdMaxExp, (RegTensor<uint16_t>&)vSrcReg0, (RegTensor<uint16_t>&)vSrcReg1, maskReg);
            ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, maskReg);
            StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(dstPtr, vdMaxExp, u1, STORE_UNALIGN_STRIDE_BYTES);
        }
        StoreUnAlignPost(dstPtr, u1, 0);
    }

    template <bool needExp>
    static __simd_vf__ inline void ClampExpMaxVf(__ubuf__ uint16_t* dstPtr, __ubuf__ bfloat16_t* dst2Ptr,
                                                 __ubuf__ bfloat16_t* srcPtr, uint32_t count, bfloat16_t alpha,
                                                 uint16_t repeatTimes, uint32_t oneRepeatSize)
    {
        RegTensor<bfloat16_t> vSrcReg0;
        RegTensor<bfloat16_t> vSrcReg1;
        RegTensor<bfloat16_t> alphaReg;
        RegTensor<bfloat16_t> vLimitReg;
        RegTensor<uint16_t> vdMaxExp;
        RegTensor<uint16_t> vExpExtract;
        RegTensor<uint16_t> expMaskBF16;
        RegTensor<uint16_t> absMask16Bit;

        if constexpr (needExp) {
            Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        }
        Duplicate(alphaReg, alpha);
        Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);
        MaskReg maskReg;
        UnalignReg u1;
        UnalignReg u2;
        AddrReg aReg;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            aReg = CreateAddrReg<uint32_t>(i, oneRepeatSize);
            maskReg = UpdateMask<bfloat16_t>(count);

            LoadAlign<bfloat16_t, LoadDist::DIST_DINTLV_B16>(vSrcReg0, vSrcReg1, srcPtr, aReg);
            And((RegTensor<uint16_t>&)vSrcReg0, (RegTensor<uint16_t>&)vSrcReg0, absMask16Bit, maskReg);
            And((RegTensor<uint16_t>&)vSrcReg1, (RegTensor<uint16_t>&)vSrcReg1, absMask16Bit, maskReg);
            Max(vdMaxExp, (RegTensor<uint16_t>&)vSrcReg0, (RegTensor<uint16_t>&)vSrcReg1, maskReg);
            ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, maskReg);
            Mul(vLimitReg, (RegTensor<bfloat16_t>&)vdMaxExp, alphaReg, maskReg);
            if constexpr (needExp) {
                And(vExpExtract, (RegTensor<uint16_t>&)vLimitReg, expMaskBF16, maskReg);
                StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(dstPtr, vExpExtract, u1,
                                                                      STORE_UNALIGN_STRIDE_BYTES);
            } else {
                StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(dstPtr, (RegTensor<uint16_t>&)vLimitReg, u1,
                                                                      STORE_UNALIGN_STRIDE_BYTES);
            }
            StoreUnAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE>(dst2Ptr, vLimitReg, u2, STORE_UNALIGN_STRIDE_BYTES);
        }
        StoreUnAlignPost(dstPtr, u1, 0);
        StoreUnAlignPost(dst2Ptr, u2, 0);
    }

    template <bool isDynamic>
    static __simd_vf__ inline void ScaleVf(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr,
                                           __ubuf__ uint16_t* srcPtr, uint32_t scaleNum, uint16_t repeatTimes,
                                           uint16_t addValueBit, uint16_t maxExpValue_)
    {
        RegTensor<uint16_t> vdMaxExp;
        RegTensor<uint16_t> vdMaxExpAdd;
        RegTensor<uint16_t> vdMaxExpOnly;
        RegTensor<uint16_t> expMask;
        RegTensor<uint16_t> sharedExp;
        RegTensor<uint16_t> scaleValue;
        RegTensor<uint16_t> halfScale;
        RegTensor<uint16_t> addValue;
        RegTensor<uint16_t> maxExpValue;
        RegTensor<uint16_t> scaleBias;
        RegTensor<uint16_t> zeroRegTensor;
        RegTensor<uint16_t> nanRegTensor;
        RegTensor<uint16_t> fp8NanRegTensor;
        RegTensor<uint16_t> specialExpRegTensor;

        Duplicate(expMask, MAX_EXP_FOR_BF16);
        Duplicate(maxExpValue, maxExpValue_);
        Duplicate(scaleBias, BF16_EXP_BIAS);
        Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        Duplicate(zeroRegTensor, 0);
        Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        if constexpr (isDynamic) {
            Duplicate(addValue, addValueBit);
        }

        MaskReg cmpResult;
        MaskReg zeroMask;
        MaskReg invalidDataMask;
        MaskReg specialDataMask;
        MaskReg preMaskScale;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preMaskScale = UpdateMask<uint16_t>(scaleNum);
            LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(vdMaxExp, srcPtr, vfLen16);

            if constexpr (isDynamic) {
                And(vdMaxExpOnly, vdMaxExp, expMask, preMaskScale);
                Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExpOnly, expMask, preMaskScale);
                Compare<uint16_t, CMPMODE::LT>(invalidDataMask, vdMaxExpOnly, maxExpValue, preMaskScale);
                Add(vdMaxExpAdd, vdMaxExp, addValue, preMaskScale);
                And(vdMaxExpAdd, vdMaxExpAdd, expMask, preMaskScale);
                Select<uint16_t>(vdMaxExpAdd, maxExpValue, vdMaxExpAdd, invalidDataMask);
                Sub(sharedExp, vdMaxExpAdd, maxExpValue, preMaskScale);
            } else {
                Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
                Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
                Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
                Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            }

            ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            StoreAlign<uint16_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>(dstPtr, scaleValue, vfLen32,
                                                                                          preMaskScale);
            Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroRegTensor, preMaskScale);
            Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
            StoreAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(dst2Ptr, halfScale, vfLen16, preMaskScale);
        }
    }

    template <bool isFP8>
    static __simd_vf__ inline void ScaleVfcuBLAS(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr,
                                                 __ubuf__ uint16_t* srcPtr, uint32_t scaleNum, uint16_t repeatTimes,
                                                 float invDstTypeMax, uint32_t dtypeMax)
    {
        RegTensor<uint16_t> max16;
        RegTensor<uint32_t> max32;
        RegTensor<uint32_t> exp32;
        RegTensor<uint32_t> man32;
        RegTensor<uint32_t> expAddOne32;
        RegTensor<uint32_t> extractExp;
        RegTensor<uint16_t> expOut;
        RegTensor<uint32_t> halfScale;
        RegTensor<uint16_t> recExpOut;
        RegTensor<uint32_t> manMaskFP32;
        RegTensor<uint32_t> expMask;
        RegTensor<uint32_t> zeroRegTensor32;
        RegTensor<uint32_t> scaleBias;
        RegTensor<uint32_t> nanRegTensor;
        RegTensor<uint32_t> fp4NanRegTensor;
        RegTensor<float> invMax;

        Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        Duplicate(expMask, MAX_EXP_FOR_FP32);
        Duplicate(zeroRegTensor32, 0);
        Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        Duplicate(fp4NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);
        if constexpr (isFP8) {
            Duplicate((RegTensor<uint32_t>&)invMax, dtypeMax);
        } else {
            Duplicate(invMax, invDstTypeMax);
        }

        MaskReg cmpResult;
        MaskReg zeroMask;
        MaskReg p0;
        MaskReg p1;
        MaskReg p2;
        MaskReg preMaskScale;
        uint32_t SixtyFour = 64;
        MaskReg dataMaskB16Half = UpdateMask<uint16_t>(SixtyFour);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preMaskScale = UpdateMask<uint32_t>(scaleNum);
            LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK_B16>(max16, srcPtr, vfLen32);

            Cast<float, bfloat16_t, castTraitZero>((RegTensor<float>&)max32, (RegTensor<bfloat16_t>&)max16,
                                                   preMaskScale);
            Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
            Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, preMaskScale);

            Mul((RegTensor<float>&)max32, (RegTensor<float>&)max32, invMax, preMaskScale);
            ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, preMaskScale);
            And(man32, max32, manMaskFP32, preMaskScale);

            CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, NUMBER_ZERO, preMaskScale);
            CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, NUMBER_TWO_FIVE_FOUR, preMaskScale);
            CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, NUMBER_ZERO, preMaskScale);
            MaskAnd(p0, p0, p1, preMaskScale);
            MaskAnd(p0, p0, p2, preMaskScale);
            if constexpr (isFP8) {
                CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32, NUMBER_ZERO, preMaskScale);
                CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, NUMBER_HALF, preMaskScale);
                MaskAnd(p1, p1, p2, preMaskScale);
                MaskOr(p0, p0, p1, preMaskScale);
            }

            Adds(expAddOne32, exp32, 1, preMaskScale);
            Select(extractExp, expAddOne32, exp32, p0);
            Select<uint32_t>(extractExp, extractExp, fp4NanRegTensor, cmpResult);
            Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(expOut, extractExp);
            StoreAlign<uint16_t, StoreDist::DIST_PACK_B16>(dstPtr + i * SCALE_STORE_STRIDE, expOut, dataMaskB16Half);

            ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, preMaskScale);
            Sub(halfScale, scaleBias, extractExp, preMaskScale);
            Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(recExpOut, halfScale);
            StoreAlign<uint16_t>(dst2Ptr + i * vfLen32, recExpOut, dataMaskB16Half);
        }
    }

    template <RoundMode roundMode, bool needClamp>
    static __simd_vf__ inline void QuantVfFP4(__ubuf__ int8_t* dstPtr, __ubuf__ bfloat16_t* srcPtr,
                                              __ubuf__ uint16_t* src2Ptr, __ubuf__ bfloat16_t* src3Ptr,
                                              uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        MaskReg dataMask1;
        MaskReg dataMask2;
        RegTensor<uint16_t> halfScaleForMul;
        RegTensor<bfloat16_t> vdExp0;
        RegTensor<bfloat16_t> vdExp1;
        RegTensor<fp4x2_e2m1_t> vdExp0FP4;
        RegTensor<fp4x2_e2m1_t> vdExp1FP4;
        RegTensor<uint16_t> signMaskBF16;
        RegTensor<bfloat16_t> vLimitReg;
        RegTensor<bfloat16_t> vnegLimitReg;
        AddrReg aReg;

        static constexpr CastTrait castTrait = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, roundMode};

        Duplicate(signMaskBF16, SIGN_MASK_FOR_BF16);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            aReg = CreateAddrReg<uint16_t>(i, oneRepeatSize);
            dataMask1 = UpdateMask<bfloat16_t>(oneRepeatSize);
            dataMask2 = UpdateMask<bfloat16_t>(oneRepeatSize);

            DataCopy<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcPtr,
                                                                                           128 * 2);
            DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(halfScaleForMul, src2Ptr, 8);

            if constexpr (needClamp) {
                DataCopy<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(vLimitReg, src3Ptr, 8);
                Xor((RegTensor<uint16_t>&)vnegLimitReg, (RegTensor<uint16_t>&)vLimitReg, signMaskBF16, dataMask1);
                Max(vdExp0, vdExp0, vnegLimitReg, dataMask1);
                Max(vdExp1, vdExp1, vnegLimitReg, dataMask1);
                Min(vdExp0, vdExp0, vLimitReg, dataMask1);
                Min(vdExp1, vdExp1, vLimitReg, dataMask1);
            }

            Mul(vdExp0, vdExp0, (RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
            Mul(vdExp1, vdExp1, (RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
            Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            Cast<fp4x2_e2m1_t, bfloat16_t, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            Cast<fp4x2_e2m1_t, bfloat16_t, castTrait>(vdExp1FP4, vdExp1, dataMask2);

            DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                dstPtr, (RegTensor<int8_t>&)vdExp0FP4, 64, dataMask1);
            DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                dstPtr, (RegTensor<int8_t>&)vdExp1FP4, 64, dataMask2);
        }
    }

    template <typename FP8Type, bool needClamp>
    static __simd_vf__ inline void QuantVfFP8(__ubuf__ int8_t* dstPtr, __ubuf__ bfloat16_t* srcPtr,
                                              __ubuf__ uint16_t* src2Ptr, __ubuf__ bfloat16_t* src3Ptr,
                                              uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        MaskReg dataMaskB16 = CreateMask<bfloat16_t>();
        MaskReg dataMaskB8 = CreateMask<FP8Type>();
        RegTensor<uint16_t> halfScaleForMul;
        RegTensor<bfloat16_t> vdExp0;
        RegTensor<bfloat16_t> vdExp1;
        RegTensor<float> vdExp0FP32Zero;
        RegTensor<float> vdExp0FP32One;
        RegTensor<float> vdExp1FP32Zero;
        RegTensor<float> vdExp1FP32One;
        RegTensor<FP8Type> vdExp0FP8Zero;
        RegTensor<FP8Type> vdExp0FP8One;
        RegTensor<FP8Type> vdExp1FP8Zero;
        RegTensor<FP8Type> vdExp1FP8One;
        RegTensor<uint16_t> signMaskBF16;
        RegTensor<bfloat16_t> vLimitReg;
        RegTensor<bfloat16_t> vnegLimitReg;
        AddrReg aReg;

        Duplicate(signMaskBF16, SIGN_MASK_FOR_BF16);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            aReg = CreateAddrReg<uint16_t>(i, oneRepeatSize);

            DataCopy<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcPtr,
                                                                                           128 * 2);
            DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(halfScaleForMul, src2Ptr, 8);

            if constexpr (needClamp) {
                DataCopy<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(vLimitReg, src3Ptr, 8);
                Xor((RegTensor<uint16_t>&)vnegLimitReg, (RegTensor<uint16_t>&)vLimitReg, signMaskBF16, dataMaskB16);
                Max(vdExp0, vdExp0, vnegLimitReg, dataMaskB16);
                Max(vdExp1, vdExp1, vnegLimitReg, dataMaskB16);
                Min(vdExp0, vdExp0, vLimitReg, dataMaskB16);
                Min(vdExp1, vdExp1, vLimitReg, dataMaskB16);
            }

            Mul(vdExp0, vdExp0, (RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);
            Mul(vdExp1, vdExp1, (RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);

            Cast<float, bfloat16_t, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMaskB16);
            Cast<float, bfloat16_t, castTraitOne>(vdExp0FP32One, vdExp0, dataMaskB16);
            Cast<float, bfloat16_t, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMaskB16);
            Cast<float, bfloat16_t, castTraitOne>(vdExp1FP32One, vdExp1, dataMaskB16);

            Cast<FP8Type, float, castTrait32to80>(vdExp0FP8Zero, vdExp0FP32Zero, dataMaskB16);
            Cast<FP8Type, float, castTrait32to82>(vdExp0FP8One, vdExp0FP32One, dataMaskB16);
            Cast<FP8Type, float, castTrait32to81>(vdExp1FP8Zero, vdExp1FP32Zero, dataMaskB16);
            Cast<FP8Type, float, castTrait32to83>(vdExp1FP8One, vdExp1FP32One, dataMaskB16);

            Add((RegTensor<uint8_t>&)vdExp0FP8Zero, (RegTensor<uint8_t>&)vdExp0FP8Zero,
                (RegTensor<uint8_t>&)vdExp0FP8One, dataMaskB8);
            Add((RegTensor<uint8_t>&)vdExp1FP8Zero, (RegTensor<uint8_t>&)vdExp1FP8Zero,
                (RegTensor<uint8_t>&)vdExp1FP8One, dataMaskB8);
            Add((RegTensor<uint8_t>&)vdExp0FP8Zero, (RegTensor<uint8_t>&)vdExp0FP8Zero,
                (RegTensor<uint8_t>&)vdExp1FP8Zero, dataMaskB8);

            DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_NORM_B8>(
                dstPtr, (RegTensor<int8_t>&)vdExp0FP8Zero, 128 * 2, dataMaskB8);
        }
    }

    static __simd_vf__ inline void TransLayoutVf(__ubuf__ int8_t* scaleAddr, __ubuf__ int8_t* scaleBlkAddr,
                                                 uint16_t mSize, uint16_t scaleBlockN)
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t eleNum = scaleBlockN;
            MaskReg maskScaleN = UpdateMask<int8_t>(eleNum);
            RegTensor<int8_t> vReg0;
            UnalignReg u0;
            auto srcUb = scaleAddr + mIdx * scaleBlockN;
            auto dstUb = scaleBlkAddr + mIdx * SCALE_STORE_STRIDE;

            DataCopyUnAlignPre(u0, srcUb);
            DataCopyUnAlign(vReg0, u0, srcUb);
            DataCopy<int8_t, StoreDist::DIST_NORM_B8>(dstUb, vReg0, maskScaleN);
        }
    }

    static __simd_vf__ inline void PadScaleVf(__ubuf__ int8_t* scaleAddr, __ubuf__ int8_t* scaleBlkAddr, uint32_t nkIdx,
                                              uint32_t nKRatio, uint16_t firstNum, uint16_t lastNum,
                                              uint16_t rowLoopNum)
    {
        DataCopyByCount(scaleBlkAddr, scaleAddr, firstNum);
        scaleAddr += firstNum;
        scaleBlkAddr += (firstNum + SCALE_STORE_STRIDE) / SCALE_STORE_STRIDE * SCALE_STORE_STRIDE;

        for (uint16_t mIdx = 0; mIdx < rowLoopNum; ++mIdx) {
            DataCopyByCount(scaleBlkAddr, scaleAddr, nKRatio);
            scaleAddr += nKRatio;
            scaleBlkAddr += (nKRatio + SCALE_STORE_STRIDE) / SCALE_STORE_STRIDE * SCALE_STORE_STRIDE;
        }

        if (lastNum > 0) {
            DataCopyByCount(scaleBlkAddr, scaleAddr, lastNum);
        }
    }

    static __simd_callee__ inline void DataCopyByCount(__ubuf__ int8_t* dstAddr, __ubuf__ int8_t* srcAddr,
                                                       uint32_t count)
    {
        uint16_t loopNum = (count + vfLen8 - 1) / vfLen8;
        RegTensor<int8_t> vReg0;
        MaskReg maskReg;
        UnalignReg u0;
        for (uint16_t i = 0; i < loopNum; ++i) {
            maskReg = UpdateMask<int8_t>(count);
            DataCopyUnAlignPre(u0, srcAddr + i * vfLen8);
            DataCopyUnAlign(vReg0, u0, srcAddr + i * vfLen8);
            DataCopy<int8_t, StoreDist::DIST_NORM_B8>(dstAddr + i * vfLen8, vReg0, maskReg);
        }
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif // __NPU_ARCH__ == 3510