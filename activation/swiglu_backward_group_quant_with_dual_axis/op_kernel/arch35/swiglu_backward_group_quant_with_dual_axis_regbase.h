/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swiglu_backward_group_quant_with_dual_axis_regbase.h
 * \brief Fused SwiGLU + grouped dual-axis MX quantization kernel for Ascend950 (regbase)
 */

#ifndef OPS_NN_SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_REGBASE_H
#define OPS_NN_SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_REGBASE_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../inc/platform.h"
#include "../../inc/kernel_utils.h"
#include "swiglu_backward_group_quant_with_dual_axis_tilingdata.h"
#include "swiglu_backward_group_quant_with_dual_axis_tiling_key.h"

namespace SwigluBackwardGroupQuantWithDualAxisMx {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_THREE = 3;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;

constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925;
constexpr uint32_t FP8_E5M2_MAX = 0x37924925;

constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t EXP_254 = 0x000000fe;
constexpr uint32_t HALF_FOR_MAN = 0x00400000;
constexpr uint32_t VF_LEN_FP32 = platform::GetVRegSize() / sizeof(float);
constexpr uint32_t VF_LEN_B16 = platform::GetVRegSize() / sizeof(half);
constexpr uint32_t VF_LEN_B16_DOUBLE = VF_LEN_B16 * DIGIT_TWO;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BLOCK_SIZE = 64;
constexpr int64_t TILE_N = 128;
constexpr int64_t UB_BLOCK_SIZE = platform::GetUbBlockSize();
constexpr int64_t GRAD_WEIGHT_PARTIAL_STRIDE = UB_BLOCK_SIZE / sizeof(float);

static constexpr Reg::CastTrait CAST_X_TO_FP32_ZERO = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                       Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr Reg::CastTrait CAST_X_TO_FP32_ONE = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

static constexpr Reg::CastTrait CAST_FP32_TO_FP16_BF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_80 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_81 = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_82 = {AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_83 = {AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
class SwigluBackwardGroupQuantWithDualAxisMxBase {
public:
    __aicore__ inline SwigluBackwardGroupQuantWithDualAxisMxBase(
        const SwigluBackwardGroupQuantWithDualAxisMxTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gradY, GM_ADDR weight, GM_ADDR yOrigin, GM_ADDR groupIndex,
                                GM_ADDR gradX, GM_ADDR gradWeight, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2,
                                GM_ADDR mxScale2);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    static __aicore__ inline void SplitBlocks(int64_t blockCount, int64_t coreIdx, int64_t coreCount,
                                              int64_t& loopCount, int64_t& blockOffset);
    __aicore__ inline void CopyIn(int64_t absRowStart, int64_t colOffsetAB, int64_t calcRow, int64_t calcColAB);
    __aicore__ inline void CopyInGradY(int64_t absRowStart, int64_t colOffsetAB, int64_t calcRow, int64_t calcColAB);
    __aicore__ inline void CopyInWeight(int64_t absRowStart, int64_t calcRow);
    __aicore__ inline void ComputeSwigluBackward(uint16_t dataLenAB, uint16_t blockCount, __ubuf__ xDtype* actAddr,
                                                 __ubuf__ xDtype* gateAddr, __ubuf__ xDtype* gradYAddr,
                                                 __ubuf__ xDtype* gradXAddr, uint32_t rowWidth,
                                                 __ubuf__ weightDtype* weightAddr);
    __aicore__ inline void ProcessGradWeight();
    __aicore__ inline void ProcessGradWeightQueuePipeline();
    static __aicore__ inline void ComputeGradWeightProduct(__ubuf__ xDtype* gradYAddr, __ubuf__ xDtype* yOriginAddr,
                                                           __ubuf__ float* productAddr, uint32_t count);
    static __aicore__ inline void StoreGradWeight(__ubuf__ float* reducedAddr, __ubuf__ weightDtype* outputAddr);
    template <HardEvent event>
    static __aicore__ inline void Synchronize()
    {
        TEventID eventId = GetTPipePtr()->AllocEventID<event>();
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
        GetTPipePtr()->ReleaseEventID<event>(eventId);
    }
    __aicore__ inline void PadZeroM(__ubuf__ xDtype* gradXAddr, uint32_t num);
    __aicore__ inline void ComputeScaleCuBLAS(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                              __ubuf__ uint8_t* mxScaleA1Addr, __ubuf__ uint8_t* mxScaleA2Addr,
                                              __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
                                              __ubuf__ uint16_t* mxScale2ReciprocalAddr, int64_t dataLenAB,
                                              __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeScaleCuBLASSecondLast(uint16_t dataLen, uint32_t localInvDtypeMax,
                                                        __ubuf__ uint16_t* mxScale2ReciprocalAddr,
                                                        __ubuf__ uint8_t* mxScale2Addr);
    __aicore__ inline void ComputeScaleCuBLASForSlot(
        __ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr, Reg::RegTensor<uint8_t>& scale8,
        Reg::RegTensor<uint32_t>& invMax, Reg::RegTensor<uint32_t>& manMaskReg, Reg::RegTensor<uint32_t>& expMaskReg,
        Reg::RegTensor<uint32_t>& zero32Reg, Reg::RegTensor<uint32_t>& scaleBiasReg, Reg::RegTensor<uint32_t>& nan32Reg,
        Reg::RegTensor<uint32_t>& fp8Nan32Reg, Reg::MaskReg& maskAll, Reg::MaskReg& maskAll32, Reg::MaskReg& maskB16);
    __aicore__ inline void ComputeY2ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void CopyQuantOutput(GlobalTensor<uint8_t>& output, LocalTensor<uint8_t>& local, int64_t offsetA,
                                           int64_t offsetB, int64_t blockCount, int64_t dataLenAB, int64_t rowWidth);
    __aicore__ inline void CopyOut(int64_t yOffsetA, int64_t yOffsetB, int64_t scale1OutOffset,
                                   int64_t scale2OutOffsetA, int64_t scale2OutOffsetB, int64_t blockCount,
                                   int64_t blockCountAlign, int64_t dataLenAB, int64_t rowWidth);

protected:
    static constexpr Reg::CastTrait castTraitFp32toYdtype = {
        Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castTraitFp32toYdtypeOne = {
        Reg::RegLayout::ONE, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castTraitFp32toYdtypeTwo = {
        Reg::RegLayout::TWO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castTraitFp32toYdtypeThree = {
        Reg::RegLayout::THREE, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

private:
    const SwigluBackwardGroupQuantWithDualAxisMxTilingData* tilingData_;

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECIN, 1> gradYQueue_;
    TQue<QuePosition::VECIN, 1> weightQueue_;
    TBuf<TPosition::VECCALC> gradXBuf_;
    TQue<QuePosition::VECOUT, 1> outQueue1_;
    TQue<QuePosition::VECOUT, 1> outQueue2_;
    TQue<QuePosition::VECOUT, 1> mxScaleQueue1_;
    TQue<QuePosition::VECOUT, 1> mxScaleQueue2_;
    TBuf<TPosition::VECCALC> mxScale1ReciprocalBuf_;
    TBuf<TPosition::VECCALC> mxScale2ReciprocalBuf_;

    GlobalTensor<xDtype> xGm_;
    GlobalTensor<xDtype> gradYGm_;
    GlobalTensor<xDtype> yOriginGm_;
    GlobalTensor<weightDtype> weightGm_;
    GlobalTensor<weightDtype> gradWeightGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<uint8_t> yGm1_;
    GlobalTensor<uint8_t> mxScaleGm1_;
    GlobalTensor<uint8_t> yGm2_;
    GlobalTensor<uint8_t> mxScaleGm2_;

    int64_t blockIdx_ = 0;
    int64_t ubRowLen_ = 0;
    int64_t ubRowLenTail_ = 0;
    int64_t ubRowCount_ = 0;
    int64_t dimNeg1ScaleNum_ = 0;
    float alpha_ = 1.702f;
    float limit_ = 7.0f;
    float bias_ = 1.0f;
    int64_t inHalfSize_ = 0;
    int64_t dimN_ = 0;
    int64_t dimGradX_ = 0;
    int64_t gradWeightTileH_ = 0;
    int64_t gradWeightTileTokens_ = 1;

    int64_t weightBlockCount_ = UB_BLOCK_SIZE / sizeof(weightDtype);
    int64_t oneBlockCountB16_ = UB_BLOCK_SIZE / sizeof(xDtype);
};

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<xDtype, y1Dtype, weightDtype, mode, hasGroupIndex,
                                                                  hasWeight, hasClampLimit>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    ubRowLen_ = tilingData_->tileN;
    ubRowLenTail_ = tilingData_->dimN % ubRowLen_ == 0 ? ubRowLen_ : tilingData_->dimN % ubRowLen_;
    ubRowCount_ = tilingData_->tileM;
    alpha_ = tilingData_->alpha;
    if constexpr (hasClampLimit) {
        limit_ = tilingData_->clampLimit;
    }
    bias_ = tilingData_->bias;
    dimN_ = tilingData_->dimN;
    dimGradX_ = dimN_ * DIGIT_TWO;
    if constexpr (hasWeight) {
        gradWeightTileH_ = tilingData_->gradWeightTileH;
        gradWeightTileTokens_ = tilingData_->gradWeightTileTokens == 0 ? 1 : tilingData_->gradWeightTileTokens;
    }
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void
SwigluBackwardGroupQuantWithDualAxisMxBase<xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight,
                                           hasClampLimit>::Init(GM_ADDR x, GM_ADDR gradY, GM_ADDR weight,
                                                                GM_ADDR yOrigin, GM_ADDR groupIndex, GM_ADDR gradX,
                                                                GM_ADDR gradWeight, GM_ADDR y1, GM_ADDR mxScale1,
                                                                GM_ADDR y2, GM_ADDR mxScale2)
{
    (void)gradX;
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif

    InitParams();

    xGm_.SetGlobalBuffer((__gm__ xDtype*)(x));
    gradYGm_.SetGlobalBuffer((__gm__ xDtype*)(gradY));
    if constexpr (hasWeight) {
        weightGm_.SetGlobalBuffer((__gm__ weightDtype*)(weight));
        weightGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        yOriginGm_.SetGlobalBuffer((__gm__ xDtype*)(yOrigin));
        yOriginGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        gradWeightGm_.SetGlobalBuffer((__gm__ weightDtype*)(gradWeight));
    }
    if constexpr (hasGroupIndex) {
        groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)(groupIndex));
        groupIndexGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    xGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    gradYGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    yGm1_.SetGlobalBuffer((__gm__ uint8_t*)(y1));
    mxScaleGm1_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale1));
    yGm2_.SetGlobalBuffer((__gm__ uint8_t*)(y2));
    mxScaleGm2_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale2));

    inHalfSize_ = ubRowLen_ * ubRowCount_;
    int64_t inBufferSize = inHalfSize_ * static_cast<int64_t>(sizeof(xDtype));
    int64_t gradXBufferSize = inHalfSize_ * DIGIT_TWO * static_cast<int64_t>(sizeof(xDtype));
    if constexpr (hasWeight) {
        const int64_t gradWeightBufferSize = gradWeightTileH_ * sizeof(float) + VF_LEN_FP32 * sizeof(float) +
                                             UB_BLOCK_SIZE;
        gradXBufferSize = gradWeightBufferSize > gradXBufferSize ? gradWeightBufferSize : gradXBufferSize;
    }
    // Four DIST_INTLV_B8 stores start at 0/128/256/384 and each may cover 512 bytes.
    int64_t mxScale2BufferSize = (ubRowLen_ * DIGIT_TWO) * ((ubRowCount_ / DOUBLE_BLOCK_SIZE) * DIGIT_THREE) +
                                 DIGIT_TWO * VF_LEN_FP32;
    int64_t mxScale1BufferSize = ubRowCount_ * UB_BLOCK_SIZE;
    int64_t tmpScale2BufferSize = (ubRowLen_ * DIGIT_TWO) * ((ubRowCount_ / DOUBLE_BLOCK_SIZE) * DIGIT_TWO) *
                                  static_cast<int64_t>(sizeof(xDtype));
    pipe_->InitBuffer(inQueue_, DB_BUFFER, inBufferSize * DIGIT_TWO);
    pipe_->InitBuffer(gradYQueue_, DB_BUFFER, inBufferSize);
    if constexpr (hasWeight) {
        const int64_t weightRowCount = ops::CeilDiv(ubRowCount_, weightBlockCount_) * weightBlockCount_;
        pipe_->InitBuffer(weightQueue_, DB_BUFFER, weightRowCount * static_cast<int64_t>(sizeof(weightDtype)));
    }
    pipe_->InitBuffer(gradXBuf_, gradXBufferSize);
    pipe_->InitBuffer(outQueue1_, DB_BUFFER, inHalfSize_ * DIGIT_TWO);
    pipe_->InitBuffer(outQueue2_, DB_BUFFER, inHalfSize_ * DIGIT_TWO);
    pipe_->InitBuffer(mxScaleQueue1_, DB_BUFFER, mxScale1BufferSize * DIGIT_TWO);
    pipe_->InitBuffer(mxScaleQueue2_, DB_BUFFER, mxScale2BufferSize);
    pipe_->InitBuffer(mxScale1ReciprocalBuf_, mxScale1BufferSize);
    pipe_->InitBuffer(mxScale2ReciprocalBuf_, tmpScale2BufferSize);
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<
    xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight, hasClampLimit>::SplitBlocks(int64_t blockCount,
                                                                                              int64_t coreIdx,
                                                                                              int64_t coreCount,
                                                                                              int64_t& loopCount,
                                                                                              int64_t& blockOffset)
{
    int64_t activeCores = blockCount < coreCount ? blockCount : coreCount;
    if (coreIdx >= activeCores) {
        return;
    }
    int64_t headCores = blockCount % activeCores;
    int64_t blocksPerCore = blockCount / activeCores;
    loopCount = blocksPerCore + (coreIdx < headCores ? 1 : 0);
    blockOffset = coreIdx * blocksPerCore + (coreIdx < headCores ? coreIdx : headCores);
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<xDtype, y1Dtype, weightDtype, mode, hasGroupIndex,
                                                                  hasWeight, hasClampLimit>::Process()
{
    const int64_t batchCount = tilingData_->dimBatch;
    const int64_t segmentCount = hasGroupIndex ? tilingData_->numGroups : batchCount;
    const int64_t totalRows = tilingData_->totalRows;
    const int64_t dimM = tilingData_->dimM;
    dimNeg1ScaleNum_ = ops::CeilDiv(dimGradX_, DOUBLE_BLOCK_SIZE) * DIGIT_TWO;
    const int64_t dimNBlockNum = tilingData_->nTiles;
    const int64_t totalCoreNum = tilingData_->usedCoreNum;
    if (blockIdx_ >= totalCoreNum) {
        return;
    }

    int64_t coreRotateOffset = 0;
    for (int64_t g = 0; g < segmentCount; ++g) {
        int64_t groupStart = 0;
        int64_t groupEnd = totalRows;
        if constexpr (hasGroupIndex) {
            groupStart = g > 0 ? groupIndexGm_.GetValue(g - 1) : 0;
            groupEnd = groupIndexGm_.GetValue(g);
        } else {
            groupStart = g * dimM;
            groupEnd = groupStart + dimM;
        }
        const int64_t groupRows = groupEnd - groupStart;
        if (groupRows <= 0) {
            continue;
        }
        int64_t scale2GmRowOffset = 0;
        if constexpr (hasGroupIndex) {
            scale2GmRowOffset = (groupStart / DOUBLE_BLOCK_SIZE + g) * DIGIT_TWO;
        } else {
            scale2GmRowOffset = g * ops::CeilDiv(dimM, DOUBLE_BLOCK_SIZE) * DIGIT_TWO;
        }
        const int64_t dimMSplitG = ops::CeilDiv(groupRows, ubRowCount_);
        const int64_t blockCountG = dimMSplitG * dimNBlockNum;

        int64_t loopPerCoreG = 0;
        int64_t blockOffsetG = 0;
        int64_t coreIdxInGroup = blockIdx_ - coreRotateOffset;
        if (coreIdxInGroup < 0) {
            coreIdxInGroup += totalCoreNum;
        }
        SplitBlocks(blockCountG, coreIdxInGroup, totalCoreNum, loopPerCoreG, blockOffsetG);
        coreRotateOffset = (coreRotateOffset + blockCountG) % totalCoreNum;
        const int64_t dimMTailG = groupRows % ubRowCount_ == 0 ? ubRowCount_ : groupRows % ubRowCount_;

        for (int64_t i = 0; i < loopPerCoreG; ++i) {
            const int64_t blockInGroup = blockOffsetG + i;
            const int64_t rowBlockIdx = blockInGroup / dimNBlockNum;
            const int64_t colBlockIdx = blockInGroup % dimNBlockNum;
            const int64_t calcColAB = colBlockIdx == dimNBlockNum - 1 ? ubRowLenTail_ : ubRowLen_;
            const int64_t calcRow = rowBlockIdx == dimMSplitG - 1 ? dimMTailG : ubRowCount_;
            const int64_t absRowStart = groupStart + rowBlockIdx * ubRowCount_;
            const int64_t colOffsetAB = colBlockIdx * ubRowLen_;
            const int64_t colOffsetGradX = colOffsetAB * DIGIT_TWO;

            CopyIn(absRowStart, colOffsetAB, calcRow, calcColAB);
            CopyInGradY(absRowStart, colOffsetAB, calcRow, calcColAB);
            if constexpr (hasWeight) {
                CopyInWeight(absRowStart, calcRow);
            }

            LocalTensor<xDtype> xLocal = inQueue_.template DeQue<xDtype>();
            LocalTensor<xDtype> gradYLocal = gradYQueue_.template DeQue<xDtype>();
            LocalTensor<xDtype> gradXLocal = gradXBuf_.template Get<xDtype>();
            auto actAddr = (__ubuf__ xDtype*)xLocal.GetPhyAddr();
            auto gateAddr = (__ubuf__ xDtype*)xLocal[inHalfSize_].GetPhyAddr();
            auto gradYAddr = (__ubuf__ xDtype*)gradYLocal.GetPhyAddr();
            LocalTensor<weightDtype> weightLocal;
            __ubuf__ weightDtype* weightAddr = nullptr;
            if constexpr (hasWeight) {
                weightLocal = weightQueue_.template DeQue<weightDtype>();
                weightAddr = (__ubuf__ weightDtype*)weightLocal.GetPhyAddr();
            }
            auto gradXAddr = (__ubuf__ xDtype*)gradXLocal.GetPhyAddr();
            const uint32_t calcPadRowAlign = ops::CeilDiv(calcRow, DOUBLE_BLOCK_SIZE) * DOUBLE_BLOCK_SIZE;
            const uint32_t rowWidth = ops::CeilDiv(calcColAB * DIGIT_TWO, static_cast<int64_t>(VF_LEN_B16)) *
                                      VF_LEN_B16;
            ComputeSwigluBackward(static_cast<uint16_t>(calcColAB), static_cast<uint16_t>(calcRow), actAddr, gateAddr,
                                  gradYAddr, gradXAddr, rowWidth, weightAddr);
            inQueue_.template FreeTensor(xLocal);
            gradYQueue_.template FreeTensor(gradYLocal);
            if constexpr (hasWeight) {
                weightQueue_.template FreeTensor(weightLocal);
            }

            const int64_t outOffsetA = absRowStart * dimGradX_ + colOffsetAB;
            const int64_t outOffsetB = absRowStart * dimGradX_ + dimN_ + colOffsetAB;
            if (calcRow % DOUBLE_BLOCK_SIZE != 0) {
                const uint32_t padRows = calcPadRowAlign - calcRow;
                PadZeroM(gradXAddr + calcRow * rowWidth, padRows * rowWidth);
            }
            LocalTensor<uint8_t> mxScale1 = mxScaleQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> mxScale2 = mxScaleQueue2_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> y1 = outQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> y2 = outQueue2_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> scale1Reciprocal = mxScale1ReciprocalBuf_.template Get<uint16_t>();
            LocalTensor<uint16_t> scale2Reciprocal = mxScale2ReciprocalBuf_.template Get<uint16_t>();
            auto y1Addr = (__ubuf__ uint8_t*)y1.GetPhyAddr();
            auto y2Addr = (__ubuf__ uint8_t*)y2.GetPhyAddr();
            auto scaleA1Addr = (__ubuf__ uint8_t*)mxScale1.GetPhyAddr();
            auto scaleB1Addr = (__ubuf__ uint8_t*)mxScale1[ubRowCount_ * UB_BLOCK_SIZE].GetPhyAddr();
            auto scale2Addr = (__ubuf__ uint8_t*)mxScale2.GetPhyAddr();
            auto scale1RecipAddr = (__ubuf__ uint16_t*)scale1Reciprocal.GetPhyAddr();
            auto scale2RecipAddr = (__ubuf__ uint16_t*)scale2Reciprocal.GetPhyAddr();

            ComputeScaleCuBLAS(static_cast<uint16_t>(rowWidth), static_cast<uint16_t>(calcPadRowAlign), gradXAddr,
                               scaleA1Addr, scaleB1Addr, scale1RecipAddr, scale2Addr, scale2RecipAddr, calcColAB,
                               y1Addr);
            for (int64_t blk = 0; blk < calcPadRowAlign / BLOCK_SIZE; ++blk) {
                const int64_t offset = blk * BLOCK_SIZE * rowWidth;
                ComputeY2ToFP8(static_cast<uint16_t>(rowWidth), static_cast<uint16_t>(BLOCK_SIZE), gradXAddr + offset,
                               scale2RecipAddr + blk * rowWidth, y2Addr + offset);
            }
            mxScaleQueue1_.template EnQue(mxScale1);
            mxScaleQueue2_.template EnQue(mxScale2);
            outQueue1_.template EnQue(y1);
            outQueue2_.template EnQue(y2);

            const int64_t scale1Offset = absRowStart * dimNeg1ScaleNum_ + colOffsetAB / BLOCK_SIZE;
            const int64_t scale2RowIdx = scale2GmRowOffset + rowBlockIdx * ubRowCount_ / BLOCK_SIZE;
            const int64_t scale2OffsetA = scale2RowIdx * dimGradX_ + colOffsetGradX;
            const int64_t scale2OffsetB = scale2OffsetA + dimN_ * DIGIT_TWO;
            CopyOut(outOffsetA, outOffsetB, scale1Offset, scale2OffsetA, scale2OffsetB, calcRow, calcPadRowAlign,
                    calcColAB, rowWidth);
        }
    }
    if constexpr (hasWeight) {
        ProcessGradWeight();
    }
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<xDtype, y1Dtype, weightDtype, mode, hasGroupIndex,
                                                                  hasWeight, hasClampLimit>::CopyIn(int64_t absRowStart,
                                                                                                    int64_t colOffsetAB,
                                                                                                    int64_t calcRow,
                                                                                                    int64_t calcColAB)
{
    int64_t xRowStride = DIGIT_TWO * dimN_;

    LocalTensor<xDtype> xLocal = inQueue_.template AllocTensor<xDtype>();

    DataCopyExtParams copyParams = {0, 0, 0, 0, 0};
    DataCopyPadExtParams<xDtype> padParams = {false, 0, 0, 0};
    copyParams.blockCount = static_cast<uint16_t>(calcRow);
    copyParams.blockLen = static_cast<uint32_t>(calcColAB * static_cast<int64_t>(sizeof(xDtype)));
    copyParams.srcStride = static_cast<uint32_t>((xRowStride - calcColAB) * static_cast<int64_t>(sizeof(xDtype)));

    int64_t leftGmOffset = absRowStart * xRowStride + colOffsetAB;
    DataCopyPad(xLocal, xGm_[leftGmOffset], copyParams, padParams);

    int64_t rightGmOffset = leftGmOffset + dimN_;
    DataCopyPad(xLocal[inHalfSize_], xGm_[rightGmOffset], copyParams, padParams);

    inQueue_.template EnQue(xLocal);
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<
    xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight, hasClampLimit>::CopyInGradY(int64_t absRowStart,
                                                                                              int64_t colOffsetAB,
                                                                                              int64_t calcRow,
                                                                                              int64_t calcColAB)
{
    int64_t gradYRowStride = dimN_;

    LocalTensor<xDtype> gradYLocal = gradYQueue_.template AllocTensor<xDtype>();

    DataCopyExtParams copyParams = {0, 0, 0, 0, 0};
    DataCopyPadExtParams<xDtype> padParams = {false, 0, 0, 0};
    copyParams.blockCount = static_cast<uint16_t>(calcRow);
    copyParams.blockLen = static_cast<uint32_t>(calcColAB * static_cast<int64_t>(sizeof(xDtype)));
    copyParams.srcStride = static_cast<uint32_t>((gradYRowStride - calcColAB) * static_cast<int64_t>(sizeof(xDtype)));

    int64_t gradYGmOffset = absRowStart * gradYRowStride + colOffsetAB;
    DataCopyPad(gradYLocal, gradYGm_[gradYGmOffset], copyParams, padParams);

    gradYQueue_.template EnQue(gradYLocal);
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<
    xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight, hasClampLimit>::CopyInWeight(int64_t absRowStart,
                                                                                               int64_t calcRow)
{
    LocalTensor<weightDtype> weightLocal = weightQueue_.template AllocTensor<weightDtype>();
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(calcRow * sizeof(weightDtype)), 0, 0, 0};
    const uint32_t weightBytes = calcRow * static_cast<uint32_t>(sizeof(weightDtype));
    const uint8_t rightPadding = static_cast<uint8_t>((UB_BLOCK_SIZE - weightBytes % UB_BLOCK_SIZE) % UB_BLOCK_SIZE /
                                                      sizeof(weightDtype));
    DataCopyPadExtParams<weightDtype> padParams = {true, 0, rightPadding, static_cast<weightDtype>(0)};
    DataCopyPad(weightLocal, weightGm_[absRowStart], copyParams, padParams);
    weightQueue_.template EnQue(weightLocal);
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<
    xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight,
    hasClampLimit>::ComputeSwigluBackward(uint16_t dataLenAB, uint16_t blockCount, __ubuf__ xDtype* actAddr,
                                          __ubuf__ xDtype* gateAddr, __ubuf__ xDtype* gradYAddr,
                                          __ubuf__ xDtype* gradXAddr, uint32_t rowWidth,
                                          __ubuf__ weightDtype* weightAddr)
{
    const uint32_t alignDim1In = ops::CeilDiv(static_cast<uint32_t>(dataLenAB),
                                              static_cast<uint32_t>(oneBlockCountB16_)) *
                                 oneBlockCountB16_;
    const uint16_t dim1VfTimes = static_cast<uint16_t>(dataLenAB / VF_LEN_FP32);
    const uint32_t dim1Tail = dataLenAB % VF_LEN_FP32;
    const uint16_t dim1TailTimes = dim1Tail > 0 ? 1 : 0;
    const uint32_t validRowWidth = dataLenAB * DIGIT_TWO;
    uint32_t paddingLen = rowWidth - validRowWidth;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> regHalf;
        Reg::RegTensor<float> regA;
        Reg::RegTensor<float> regB;
        Reg::RegTensor<float> regGrad;
        Reg::RegTensor<float> regWeight;
        Reg::RegTensor<weightDtype> regWeightInput;
        Reg::RegTensor<float> regSig;
        Reg::RegTensor<float> regTmp;
        Reg::RegTensor<float> regDa;
        Reg::RegTensor<float> regDb;
        Reg::RegTensor<float> regOne;
        Reg::RegTensor<float> regLimit;
        Reg::RegTensor<float> regNegLimit;
        Reg::RegTensor<float> regZero;
        Reg::MaskReg mask;
        Reg::MaskReg maskA;
        Reg::MaskReg maskB;
        Reg::MaskReg maskBn;

        if constexpr (hasClampLimit) {
            Reg::Duplicate(regLimit, limit_);
            Reg::Duplicate(regNegLimit, -limit_);
            Reg::Duplicate(regZero, 0.0f);
        }

        const uint16_t dim1LoopCount = static_cast<uint16_t>(dim1VfTimes + dim1TailTimes);
        for (uint16_t row = 0; row < blockCount; ++row) {
            if constexpr (hasWeight) {
                if constexpr (IsSameType<weightDtype, float>::value) {
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(regWeight, weightAddr + row);
                } else {
                    Reg::LoadAlign<weightDtype, Reg::LoadDist::DIST_BRC_B16>(regWeightInput, weightAddr + row);
                    Reg::MaskReg weightMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
                    Reg::Cast<float, weightDtype, CAST_X_TO_FP32_ZERO>(regWeight, regWeightInput, weightMask);
                }
            }
            uint32_t length = dataLenAB;
            for (uint16_t col = 0; col < dim1LoopCount; ++col) {
                mask = Reg::UpdateMask<float>(length);
                Reg::AddrReg inOffset = Reg::CreateAddrReg<xDtype>(row, alignDim1In, col, VF_LEN_FP32);
                Reg::DataCopy<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(regHalf, actAddr, inOffset);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(regA, regHalf, mask);
                Reg::DataCopy<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(regHalf, gateAddr, inOffset);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(regB, regHalf, mask);
                Reg::DataCopy<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(regHalf, gradYAddr, inOffset);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(regGrad, regHalf, mask);
                if constexpr (hasWeight) {
                    Reg::Mul(regGrad, regGrad, regWeight, mask);
                }

                if constexpr (hasClampLimit) {
                    Reg::Compare<float, CMPMODE::LE>(maskA, regA, regLimit, mask);
                    Reg::Compare<float, CMPMODE::LE>(maskB, regB, regLimit, mask);
                    Reg::Compare<float, CMPMODE::GE>(maskBn, regB, regNegLimit, mask);
                    Reg::Mins(regA, regA, limit_, mask);
                }
                Reg::Muls(regSig, regA, -alpha_, mask);
                Reg::Exp(regSig, regSig, mask);
                Reg::Adds(regSig, regSig, 1.0f, mask);
                Reg::Duplicate(regOne, 1.0f, mask);
                Reg::Div(regSig, regOne, regSig, mask);

                if constexpr (hasClampLimit) {
                    Reg::Mins(regB, regB, limit_, mask);
                    Reg::Maxs(regB, regB, -limit_, mask);
                }
                Reg::Adds(regB, regB, bias_, mask);

                // Match the Ascend950 clipped_swiglu_grad order: gradY * A * sigmoid(A).
                Reg::Mul(regDb, regGrad, regA, mask);
                Reg::Mul(regDb, regDb, regSig, mask);
                if constexpr (hasClampLimit) {
                    Reg::Select<float>(regDb, regDb, regZero, maskB);
                    Reg::Select<float>(regDb, regDb, regZero, maskBn);
                }

                Reg::Muls(regTmp, regSig, -1.0f, mask);
                Reg::Adds(regTmp, regTmp, 1.0f, mask);
                Reg::Mul(regTmp, regTmp, regA, mask);
                Reg::Muls(regTmp, regTmp, alpha_, mask);
                Reg::Adds(regTmp, regTmp, 1.0f, mask);
                // clipped_swiglu_grad computes ((1 - sigmoid) * A * alpha + 1)
                // * sigmoid * (B + bias) * upstream_grad in this order.
                Reg::Mul(regTmp, regTmp, regSig, mask);
                Reg::Mul(regTmp, regTmp, regB, mask);
                Reg::Mul(regDa, regTmp, regGrad, mask);
                if constexpr (hasClampLimit) {
                    Reg::Select<float>(regDa, regDa, regZero, maskA);
                }

                Reg::AddrReg outOffset = Reg::CreateAddrReg<xDtype>(row, rowWidth, col, VF_LEN_FP32);
                Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(regHalf, regDa, mask);
                Reg::DataCopy<xDtype, Reg::StoreDist::DIST_PACK_B32>(gradXAddr, regHalf, outOffset, mask);
                Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(regHalf, regDb, mask);
                Reg::DataCopy<xDtype, Reg::StoreDist::DIST_PACK_B32>(gradXAddr + dataLenAB, regHalf, outOffset, mask);
            }
        }

        if (paddingLen > 0) {
            Reg::Duplicate(regHalf, static_cast<xDtype>(0));
            mask = Reg::UpdateMask<xDtype>(paddingLen);
            for (uint16_t row = 0; row < blockCount; ++row) {
                Reg::AddrReg paddingOffset = Reg::CreateAddrReg<xDtype>(row, rowWidth);
                Reg::DataCopy<xDtype>(gradXAddr + validRowWidth, regHalf, paddingOffset, mask);
            }
        }
    }
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void
SwigluBackwardGroupQuantWithDualAxisMxBase<xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight,
                                           hasClampLimit>::ComputeGradWeightProduct(__ubuf__ xDtype* gradYAddr,
                                                                                    __ubuf__ xDtype* yOriginAddr,
                                                                                    __ubuf__ float* productAddr,
                                                                                    uint32_t count)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> packed;
        Reg::RegTensor<float> gradY;
        Reg::RegTensor<float> yOrigin;
        Reg::RegTensor<float> product;
        Reg::MaskReg mask;
        const uint16_t repeats = static_cast<uint16_t>((count + VF_LEN_FP32 - 1) / VF_LEN_FP32);
        uint32_t remaining = count;
        for (uint16_t i = 0; i < repeats; ++i) {
            mask = Reg::UpdateMask<float>(remaining);
            Reg::AddrReg offset = Reg::CreateAddrReg<xDtype>(i, VF_LEN_FP32);
            Reg::DataCopy<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(packed, gradYAddr, offset);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(gradY, packed, mask);
            Reg::DataCopy<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(packed, yOriginAddr, offset);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(yOrigin, packed, mask);
            Reg::Mul(product, gradY, yOrigin, mask);
            Reg::StoreAlign(productAddr + i * VF_LEN_FP32, product, mask);
        }
    }
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void
SwigluBackwardGroupQuantWithDualAxisMxBase<xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight,
                                           hasClampLimit>::StoreGradWeight(__ubuf__ float* reducedAddr,
                                                                           __ubuf__ weightDtype* outputAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> sum;
        uint32_t oneElement = 1;
        Reg::MaskReg oneMask = Reg::UpdateMask<float>(oneElement);
        Reg::LoadAlign(sum, reducedAddr);
        if constexpr (IsSameType<weightDtype, float>::value) {
            Reg::StoreAlign((__ubuf__ float*)outputAddr, sum, oneMask);
        } else {
            Reg::RegTensor<weightDtype> output;
            Reg::Cast<weightDtype, float, CAST_FP32_TO_FP16_BF16>(output, sum, oneMask);
            Reg::DataCopy<weightDtype, Reg::StoreDist::DIST_PACK_B32>(outputAddr, output, oneMask);
        }
    }
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<
    xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight, hasClampLimit>::ProcessGradWeightQueuePipeline()
{
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> productLocal = gradXBuf_.template Get<float>();
    LocalTensor<float> partialLocal = productLocal[gradWeightTileH_];
    LocalTensor<float> reducedLocal = partialLocal[VF_LEN_FP32];
    auto productAddr = (__ubuf__ float*)productLocal.GetPhyAddr();
    const int64_t outputStride = UB_BLOCK_SIZE / sizeof(weightDtype);
    const int64_t totalRows = tilingData_->totalRows;
    const int64_t batchTokens = gradWeightTileTokens_;
    const int64_t chunks = ops::CeilDiv(dimN_, gradWeightTileH_);
    DataCopyPadExtParams<xDtype> noPad = {false, 0, 0, 0};

    for (int64_t rowStart = blockIdx_ * batchTokens; rowStart < totalRows;
         rowStart += tilingData_->usedCoreNum * batchTokens) {
        const int64_t batchRows = (totalRows - rowStart > batchTokens) ? batchTokens : totalRows - rowStart;
        LocalTensor<weightDtype> outputLocal = outQueue1_.template AllocTensor<weightDtype>();
        auto outputAddr = (__ubuf__ weightDtype*)outputLocal.GetPhyAddr();
        float gradWeight = -0.0f;

        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            const int64_t offset = chunk * gradWeightTileH_;
            const int64_t count = (dimN_ - offset > gradWeightTileH_) ? gradWeightTileH_ : dimN_ - offset;
            LocalTensor<xDtype> gradYLocal = inQueue_.template AllocTensor<xDtype>();
            LocalTensor<xDtype> yOriginLocal = gradYQueue_.template AllocTensor<xDtype>();
            DataCopyExtParams copyParams = {
                static_cast<uint16_t>(batchRows), static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(xDtype))),
                static_cast<uint32_t>((dimN_ - count) * static_cast<int64_t>(sizeof(xDtype))),
                static_cast<uint32_t>((gradWeightTileH_ - count) * static_cast<int64_t>(sizeof(xDtype)) / BLOCK_SIZE),
                0};
            const int64_t gmOffset = rowStart * dimN_ + offset;
            DataCopyPad(gradYLocal, gradYGm_[gmOffset], copyParams, noPad);
            DataCopyPad(yOriginLocal, yOriginGm_[gmOffset], copyParams, noPad);
            inQueue_.template EnQue(gradYLocal);
            gradYQueue_.template EnQue(yOriginLocal);

            gradYLocal = inQueue_.template DeQue<xDtype>();
            yOriginLocal = gradYQueue_.template DeQue<xDtype>();
            auto gradYAddr = (__ubuf__ xDtype*)gradYLocal.GetPhyAddr();
            auto yOriginAddr = (__ubuf__ xDtype*)yOriginLocal.GetPhyAddr();
            for (int64_t token = 0; token < batchRows; ++token) {
                ComputeGradWeightProduct(gradYAddr + token * gradWeightTileH_, yOriginAddr + token * gradWeightTileH_,
                                         productAddr, static_cast<uint32_t>(count));
                PipeBarrier<PIPE_V>();
                ReduceSum<float>(reducedLocal, productLocal, partialLocal, static_cast<uint32_t>(count));
                PipeBarrier<PIPE_V>();
                if (chunks == 1) {
                    StoreGradWeight((__ubuf__ float*)reducedLocal.GetPhyAddr(), outputAddr + token * outputStride);
                } else {
                    Synchronize<HardEvent::V_S>();
                    gradWeight += reducedLocal.GetValue(0);
                }
            }
            inQueue_.template FreeTensor(gradYLocal);
            gradYQueue_.template FreeTensor(yOriginLocal);
        }

        if (chunks > 1) {
            reducedLocal.SetValue(0, gradWeight);
            Synchronize<HardEvent::S_V>();
            StoreGradWeight((__ubuf__ float*)reducedLocal.GetPhyAddr(), outputAddr);
        }
        outQueue1_.template EnQue(outputLocal);

        outputLocal = outQueue1_.template DeQue<weightDtype>();
        DataCopyExtParams copyOut = {1, static_cast<uint32_t>(sizeof(weightDtype)), 0, 0, 0};
        for (int64_t token = 0; token < batchRows; ++token) {
            DataCopyPad(gradWeightGm_[rowStart + token], outputLocal[token * outputStride], copyOut);
        }
        outQueue1_.template FreeTensor(outputLocal);
    }
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<xDtype, y1Dtype, weightDtype, mode, hasGroupIndex,
                                                                  hasWeight, hasClampLimit>::ProcessGradWeight()
{
    ProcessGradWeightQueuePipeline();
}
template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void
SwigluBackwardGroupQuantWithDualAxisMxBase<xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight,
                                           hasClampLimit>::PadZeroM(__ubuf__ xDtype* swigluOutAddr, uint32_t num)
{
    uint16_t times = CeilDivision(num, 128);
    uint32_t size = num;
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> zeroReg;
        AscendC::Reg::MaskReg mask;
        Reg::Duplicate(zeroReg, 0);
        for (uint16_t i = 0; i < times; i++) {
            mask = AscendC::Reg::UpdateMask<xDtype>(size);
            Reg::AddrReg offset = Reg::CreateAddrReg<xDtype>(i, 128);
            AscendC::Reg::DataCopy(swigluOutAddr, zeroReg, offset, mask);
        }
    }
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<
    xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight,
    hasClampLimit>::ComputeScaleCuBLASForSlot(__ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr,
                                              Reg::RegTensor<uint8_t>& scale8, Reg::RegTensor<uint32_t>& invMax,
                                              Reg::RegTensor<uint32_t>& manMaskReg,
                                              Reg::RegTensor<uint32_t>& expMaskReg, Reg::RegTensor<uint32_t>& zero32Reg,
                                              Reg::RegTensor<uint32_t>& scaleBiasReg,
                                              Reg::RegTensor<uint32_t>& nan32Reg, Reg::RegTensor<uint32_t>& fp8Nan32Reg,
                                              Reg::MaskReg& maskAll, Reg::MaskReg& maskAll32, Reg::MaskReg& maskB16)
{
    Reg::RegTensor<uint16_t> max16Reg;
    Reg::RegTensor<uint32_t> max32Reg;
    Reg::RegTensor<uint32_t> exp32Reg;
    Reg::RegTensor<uint32_t> man32Reg;
    Reg::RegTensor<uint32_t> expOne32Reg;
    Reg::RegTensor<uint32_t> extractExp;
    Reg::RegTensor<uint32_t> halfScale;
    Reg::RegTensor<uint16_t> scale16;
    Reg::RegTensor<uint16_t> recip16;
    Reg::MaskReg cmpResult;
    Reg::MaskReg zeroMask;
    Reg::MaskReg p0;
    Reg::MaskReg p1;

    Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(max16Reg, maxReadAddr,
                                                                                                VF_LEN_FP32);
    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<xDtype>&)max16Reg,
                                                  maskAll);
    Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32Reg, expMaskReg, maskAll32);
    Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32Reg, zero32Reg, maskAll32);
    Reg::Mul((Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<float>&)invMax,
             maskAll32);
    Reg::ShiftRights(exp32Reg, max32Reg, SHR_NUM_FOR_FP32, maskAll32);
    Reg::And(man32Reg, max32Reg, manMaskReg, maskAll32);
    Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    Reg::CompareScalar<uint32_t, CMPMODE::LT>(p0, exp32Reg, EXP_254, p0);
    Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, man32Reg, static_cast<uint32_t>(0), p0);
    Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, man32Reg, HALF_FOR_MAN, p1);
    Reg::MaskOr(p0, p0, p1, maskAll32);
    Reg::Adds(expOne32Reg, exp32Reg, 1, maskAll32);
    Reg::Select(extractExp, expOne32Reg, exp32Reg, p0);
    Reg::Select<uint32_t>(extractExp, extractExp, fp8Nan32Reg, cmpResult);
    Reg::Select<uint32_t>(extractExp, extractExp, zero32Reg, zeroMask);
    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale16, extractExp);
    Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scale8, scale16);

    Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
    Reg::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
    Reg::Select<uint32_t>(halfScale, halfScale, nan32Reg, cmpResult);
    Reg::Select<uint32_t>(halfScale, halfScale, zero32Reg, zeroMask);
    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recip16, halfScale);
    Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(reciprocalWriteAddr, recip16, VF_LEN_FP32, maskB16);
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<
    xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight,
    hasClampLimit>::ComputeScaleCuBLASSecondLast(uint16_t dataLen, uint32_t localInvDtypeMax,
                                                 __ubuf__ uint16_t* mxScale2ReciprocalAddr,
                                                 __ubuf__ uint8_t* mxScale2Addr)
{
    uint16_t times = dataLen / VF_LEN_FP32;
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint32_t> invMax;
        Reg::RegTensor<uint32_t> manMaskReg;
        Reg::RegTensor<uint32_t> expMaskReg;
        Reg::RegTensor<uint32_t> zero32Reg;
        Reg::RegTensor<uint32_t> scaleBiasReg;
        Reg::RegTensor<uint32_t> nan32Reg;
        Reg::RegTensor<uint32_t> fp8Nan32Reg;
        Reg::RegTensor<uint8_t> scale8Slot0;
        Reg::RegTensor<uint8_t> scale8Slot1;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::VL64>();
        Reg::MaskReg interleaveMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        Reg::Duplicate(zero32Reg, static_cast<uint32_t>(0));
        Reg::Duplicate(invMax, localInvDtypeMax);
        Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        Reg::Duplicate(fp8Nan32Reg, MAX_EXP_FOR_FP8_IN_FP32);
        Reg::Duplicate(nan32Reg, static_cast<uint32_t>(NAN_CUSTOMIZATION));

        for (uint16_t i = 0; i < times; i++) {
            uint16_t colOffset = i * VF_LEN_FP32;
            __ubuf__ uint16_t* slot0Addr = mxScale2ReciprocalAddr + colOffset;
            __ubuf__ uint16_t* slot1Addr = mxScale2ReciprocalAddr + dataLen + colOffset;
            ComputeScaleCuBLASForSlot(slot0Addr, slot0Addr, scale8Slot0, invMax, manMaskReg, expMaskReg, zero32Reg,
                                      scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
            ComputeScaleCuBLASForSlot(slot1Addr, slot1Addr, scale8Slot1, invMax, manMaskReg, expMaskReg, zero32Reg,
                                      scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
            Reg::DataCopy<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(mxScale2Addr + DIGIT_TWO * colOffset, scale8Slot0,
                                                                  scale8Slot1, interleaveMask);
        }
    }
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void SwigluBackwardGroupQuantWithDualAxisMxBase<
    xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight,
    hasClampLimit>::ComputeScaleCuBLAS(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                       __ubuf__ uint8_t* mxScaleA1Addr, __ubuf__ uint8_t* mxScaleA2Addr,
                                       __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
                                       __ubuf__ uint16_t* mxScale2ReciprocalAddr, int64_t dataLenAB,
                                       __ubuf__ uint8_t* y1Addr)
{
    uint32_t localInvDtypeMax = FP8_E4M3_MAX;
    if constexpr (IsSameType<y1Dtype, fp8_e5m2_t>::value) {
        localInvDtypeMax = FP8_E5M2_MAX;
    }
    uint32_t scale1BIdx = ops::CeilDiv(dataLenAB, BLOCK_SIZE);

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> regA;
        Reg::RegTensor<xDtype> regB;
        Reg::RegTensor<uint16_t> regU16_0;
        Reg::RegTensor<uint16_t> regU16_1;
        Reg::RegTensor<uint16_t> regU16_2;
        Reg::RegTensor<uint16_t> absMax1Dim2;
        Reg::RegTensor<uint16_t> absMax2Dim2;

        Reg::RegTensor<uint32_t> expAddOne32;
        Reg::RegTensor<uint32_t> extractExp;
        Reg::RegTensor<uint32_t> halfScale;
        Reg::RegTensor<uint32_t> invMax;
        Reg::RegTensor<uint32_t> manMaskReg;
        Reg::RegTensor<uint32_t> expMaskReg;
        Reg::RegTensor<uint32_t> zeroReg32;
        Reg::RegTensor<uint32_t> scaleBiasReg;
        Reg::RegTensor<uint32_t> nanReg32;
        Reg::RegTensor<uint32_t> fp8NanReg32;

        Reg::RegTensor<uint8_t> scale8Reg;
        Reg::RegTensor<uint8_t> scale8Row;
        Reg::RegTensor<uint8_t> scale8Swapped;
        Reg::RegTensor<uint16_t> recip16Row;
        Reg::RegTensor<int8_t> indexShiftS8;
        Reg::RegTensor<int8_t> extractIdx;
        Reg::RegTensor<uint16_t> absMask;

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::VL16>();
        Reg::UnalignReg ureg;

        Reg::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        Reg::Duplicate(invMax, localInvDtypeMax);
        Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        Reg::Duplicate(zeroReg32, static_cast<uint32_t>(0));
        Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        Reg::Duplicate(nanReg32, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        Reg::Duplicate(fp8NanReg32, MAX_EXP_FOR_FP8_IN_FP32);
        Reg::Arange(indexShiftS8, static_cast<int8_t>(scale1BIdx));

        constexpr uint16_t blockSize = static_cast<uint16_t>(BLOCK_SIZE);
        const uint16_t slotCount = blockCount / blockSize;
        for (uint16_t slotIdx = 0; slotIdx < slotCount; slotIdx++) {
            __ubuf__ uint16_t* slotScale2Addr = mxScale2ReciprocalAddr + slotIdx * dataLen;
            __ubuf__ uint16_t* tempMaxAddr = slotScale2Addr;
            __ubuf__ xDtype* xAddrBase = xAddr;
            __ubuf__ uint16_t* recipBase = mxScale1ReciprocalAddr;

            Reg::Duplicate(absMax1Dim2, static_cast<uint16_t>(0));
            Reg::Duplicate(absMax2Dim2, static_cast<uint16_t>(0));

            for (uint16_t i = 0; i < blockSize; i++) {
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                    regA, regB, xAddr, dataLen);

                Reg::And(regU16_0, (Reg::RegTensor<uint16_t>&)regA, absMask, maskAll);
                Reg::And(regU16_1, (Reg::RegTensor<uint16_t>&)regB, absMask, maskAll);

                Reg::Max(regU16_2, regU16_0, regU16_1, maskAll);
                Reg::ReduceMaxWithDataBlock(regU16_2, regU16_2, maskAll);
                // 8: 每次循环scale1偏移8个元素
                Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(tempMaxAddr, regU16_2, ureg, 8);

                Reg::Max(absMax1Dim2, absMax1Dim2, regU16_0, maskAll);
                Reg::Max(absMax2Dim2, absMax2Dim2, regU16_1, maskAll);
            }
            Reg::StoreUnAlignPost(tempMaxAddr, ureg, 0);

            __ubuf__ uint16_t* readMaxAddr = slotScale2Addr;
            // 8: 每次处理的数据元素个数=32 * 8
            uint16_t batchCount = ops::CeilDiv(static_cast<uint16_t>(BLOCK_SIZE * 8),
                                               static_cast<uint16_t>(VF_LEN_FP32));

            for (uint16_t j = 0; j < batchCount; j++) {
                Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                    regU16_0, readMaxAddr, VF_LEN_FP32);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float>&)regA,
                                                              (Reg::RegTensor<xDtype>&)regU16_0, maskAll);

                Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, (Reg::RegTensor<uint32_t>&)regA, expMaskReg, maskAll32);
                Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, (Reg::RegTensor<uint32_t>&)regA, zeroReg32, maskAll32);
                Reg::Mul((Reg::RegTensor<float>&)regA, (Reg::RegTensor<float>&)regA, (Reg::RegTensor<float>&)invMax,
                         maskAll32);
                Reg::ShiftRights((Reg::RegTensor<uint32_t>&)regB, (Reg::RegTensor<uint32_t>&)regA, SHR_NUM_FOR_FP32,
                                 maskAll32);
                Reg::And((Reg::RegTensor<uint32_t>&)regU16_1, (Reg::RegTensor<uint32_t>&)regA, manMaskReg, maskAll32);

                Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, (Reg::RegTensor<uint32_t>&)regB, static_cast<uint32_t>(0),
                                                          maskAll32);
                Reg::CompareScalar<uint32_t, CMPMODE::LT>(p0, (Reg::RegTensor<uint32_t>&)regB, EXP_254, p0);
                Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, (Reg::RegTensor<uint32_t>&)regU16_1,
                                                          static_cast<uint32_t>(0), p0);
                Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, (Reg::RegTensor<uint32_t>&)regB, static_cast<uint32_t>(0),
                                                          maskAll32);
                Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, (Reg::RegTensor<uint32_t>&)regU16_1, HALF_FOR_MAN, p1);
                Reg::MaskOr(p0, p0, p1, maskAll32);

                Reg::Adds(expAddOne32, (Reg::RegTensor<uint32_t>&)regB, 1, maskAll32);
                Reg::Select(extractExp, expAddOne32, (Reg::RegTensor<uint32_t>&)regB, p0);
                Reg::Select<uint32_t>(extractExp, extractExp, fp8NanReg32, cmpResult);
                Reg::Select<uint32_t>(extractExp, extractExp, zeroReg32, zeroMask);

                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(regU16_2, extractExp);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scale8Reg, regU16_2);

                Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
                Reg::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
                Reg::Select<uint32_t>(halfScale, halfScale, nanReg32, cmpResult);
                Reg::Select<uint32_t>(halfScale, halfScale, zeroReg32, zeroMask);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(regU16_2, halfScale);

                // 8: 一个RegTensor可容纳64个fp32元素，每次循环处理8个元素
                for (uint16_t k = 0; k < 8; k++) {
                    // 8: scale1的搬运, 每次循环索引值的更新步长为8
                    Reg::Arange(extractIdx, static_cast<int8_t>(k * 8));
                    Reg::Gather(scale8Row, scale8Reg, (Reg::RegTensor<uint8_t>&)extractIdx);
                    Reg::Gather(scale8Swapped, scale8Row, (Reg::RegTensor<uint8_t>&)indexShiftS8);

                    // 32: 偏移32字节
                    Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScaleA1Addr, scale8Row, 32,
                                                                               maskReduceB8);
                    Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScaleA2Addr, scale8Swapped, 32,
                                                                               maskReduceB8);

                    // 16: 1/scale1的搬运, 每次循环索引值的更新步长为16
                    Reg::Arange(extractIdx, static_cast<int8_t>(k * 16));
                    Reg::Gather((Reg::RegTensor<uint8_t>&)recip16Row, (Reg::RegTensor<uint8_t>&)regU16_2,
                                (Reg::RegTensor<uint8_t>&)extractIdx);
                    Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1ReciprocalAddr, recip16Row, 16,
                                                                                maskReduceB16);
                }
            }

            Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
            Reg::RegTensor<uint16_t> scaleForMulFP16;
            Reg::RegTensor<float> scaleForMulFP32;
            Reg::RegTensor<float> x0ZeroFP32;
            Reg::RegTensor<float> x0OneFP32;
            Reg::RegTensor<float> x1ZeroFP32;
            Reg::RegTensor<float> x1OneFP32;
            Reg::RegTensor<y1Dtype> fp8Part0;
            Reg::RegTensor<y1Dtype> fp8Part1;
            Reg::RegTensor<y1Dtype> fp8Part2;
            Reg::RegTensor<y1Dtype> fp8Part3;

            __ubuf__ xDtype* xReadAddr = xAddrBase;
            __ubuf__ uint16_t* recipReadAddr = recipBase;

            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < blockSize; i++) {
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                    regA, regB, xReadAddr, dataLen);
                Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                    scaleForMulFP16, recipReadAddr, 16); // 1/scale1每次偏移16个元素

                if constexpr (IsSameType<xDtype, half>::value) {
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, regA, maskAll);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, regA, maskAll);
                    Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                        scaleForMulFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
                    Reg::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                    Reg::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, regB, maskAll);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, regB, maskAll);
                    Reg::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                    Reg::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
                } else {
                    Reg::Mul(regA, regA, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                    Reg::Mul(regB, regB, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, regA, maskAll);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, regA, maskAll);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, regB, maskAll);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, regB, maskAll);
                }
                AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_80>(fp8Part0, x0ZeroFP32, maskAll);
                AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_82>(fp8Part1, x0OneFP32, maskAll);
                AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_81>(fp8Part2, x1ZeroFP32, maskAll);
                AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_83>(fp8Part3, x1OneFP32, maskAll);
                AscendC::Reg::Add((Reg::RegTensor<uint8_t>&)fp8Part0, (Reg::RegTensor<uint8_t>&)fp8Part0,
                                  (Reg::RegTensor<uint8_t>&)fp8Part1, maskAllB8);
                AscendC::Reg::Add((Reg::RegTensor<uint8_t>&)fp8Part0, (Reg::RegTensor<uint8_t>&)fp8Part0,
                                  (Reg::RegTensor<uint8_t>&)fp8Part2, maskAllB8);
                AscendC::Reg::Add((Reg::RegTensor<uint8_t>&)fp8Part0, (Reg::RegTensor<uint8_t>&)fp8Part0,
                                  (Reg::RegTensor<uint8_t>&)fp8Part3, maskAllB8);
                AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_NORM_B8>(
                    y1Addr, (Reg::RegTensor<uint8_t>&)fp8Part0, dataLen, maskAllB8);
            }

            Reg::DataCopy<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(slotScale2Addr, absMax1Dim2, absMax2Dim2, maskAll);
        }
    }
    ComputeScaleCuBLASSecondLast(dataLen, localInvDtypeMax, mxScale2ReciprocalAddr, mxScale2Addr);
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void
SwigluBackwardGroupQuantWithDualAxisMxBase<xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight,
                                           hasClampLimit>::ComputeY2ToFP8(uint16_t dataLen, uint16_t blockCount,
                                                                          __ubuf__ xDtype* xAddr,
                                                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr,
                                                                          __ubuf__ uint8_t* y2Addr)
{
    int64_t localUbRowLen = dataLen;

    if (dataLen == VF_LEN_B16_DOUBLE) {
        __VEC_SCOPE__
        {
            Reg::RegTensor<xDtype> xEven;
            Reg::RegTensor<xDtype> xOdd;
            Reg::RegTensor<uint16_t> reciprocalEven;
            Reg::RegTensor<uint16_t> reciprocalOdd;
            Reg::RegTensor<float> reciprocalEvenFP32Layout0;
            Reg::RegTensor<float> reciprocalEvenFP32Layout1;
            Reg::RegTensor<float> reciprocalOddFP32Layout0;
            Reg::RegTensor<float> reciprocalOddFP32Layout1;
            Reg::RegTensor<float> xEvenFP32Layout0;
            Reg::RegTensor<float> xEvenFP32Layout1;
            Reg::RegTensor<float> xOddFP32Layout0;
            Reg::RegTensor<float> xOddFP32Layout1;
            Reg::RegTensor<y1Dtype> yEvenFP8Layout0;
            Reg::RegTensor<y1Dtype> yEvenFP8Layout2;
            Reg::RegTensor<y1Dtype> yOddFP8Layout1;
            Reg::RegTensor<y1Dtype> yOddFP8Layout3;

            Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
            Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
            Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

            __ubuf__ uint16_t* reciprocalCursor = mxScale2ReciprocalAddr;
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                reciprocalEven, reciprocalOdd, reciprocalCursor, VF_LEN_B16_DOUBLE);
            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    reciprocalEvenFP32Layout0, (Reg::RegTensor<bfloat16_t>&)reciprocalEven, pregAll16);
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(
                    reciprocalEvenFP32Layout1, (Reg::RegTensor<bfloat16_t>&)reciprocalEven, pregAll16);
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    reciprocalOddFP32Layout0, (Reg::RegTensor<bfloat16_t>&)reciprocalOdd, pregAll16);
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalOddFP32Layout1,
                                                                 (Reg::RegTensor<bfloat16_t>&)reciprocalOdd, pregAll16);
            }

            __ubuf__ xDtype* xCursor = xAddr;
            for (uint16_t j = 0; j < blockCount; j++) {
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                    xEven, xOdd, xCursor, VF_LEN_B16_DOUBLE);
                if constexpr (IsSameType<xDtype, half>::value) {
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, pregAll16);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, pregAll16);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, pregAll16);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, pregAll16);
                    Reg::Mul(xEvenFP32Layout0, xEvenFP32Layout0, reciprocalEvenFP32Layout0, pregAll32);
                    Reg::Mul(xEvenFP32Layout1, xEvenFP32Layout1, reciprocalEvenFP32Layout1, pregAll32);
                    Reg::Mul(xOddFP32Layout0, xOddFP32Layout0, reciprocalOddFP32Layout0, pregAll32);
                    Reg::Mul(xOddFP32Layout1, xOddFP32Layout1, reciprocalOddFP32Layout1, pregAll32);
                } else {
                    Reg::Mul(xEven, xEven, (Reg::RegTensor<xDtype>&)reciprocalEven, pregAll16);
                    Reg::Mul(xOdd, xOdd, (Reg::RegTensor<xDtype>&)reciprocalOdd, pregAll16);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, pregAll16);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, pregAll16);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, pregAll16);
                    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, pregAll16);
                }

                Reg::Cast<y1Dtype, float, castTraitFp32toYdtype>(yEvenFP8Layout0, xEvenFP32Layout0, pregAll32);
                Reg::Cast<y1Dtype, float, castTraitFp32toYdtypeTwo>(yEvenFP8Layout2, xEvenFP32Layout1, pregAll32);
                Reg::Cast<y1Dtype, float, castTraitFp32toYdtypeOne>(yOddFP8Layout1, xOddFP32Layout0, pregAll32);
                Reg::Cast<y1Dtype, float, castTraitFp32toYdtypeThree>(yOddFP8Layout3, xOddFP32Layout1, pregAll32);
                Reg::Add((Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0,
                         (Reg::RegTensor<uint8_t>&)yEvenFP8Layout2, pregAll8);
                Reg::Add((Reg::RegTensor<uint8_t>&)yOddFP8Layout1, (Reg::RegTensor<uint8_t>&)yOddFP8Layout1,
                         (Reg::RegTensor<uint8_t>&)yOddFP8Layout3, pregAll8);
                Reg::Add((Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0,
                         (Reg::RegTensor<uint8_t>&)yOddFP8Layout1, pregAll8);
                Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                    y2Addr, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, VF_LEN_B16_DOUBLE, pregAll8);
            }
        }
        return;
    }
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<bfloat16_t> xBF16;
        Reg::RegTensor<float> x0FP32;
        Reg::RegTensor<float> x1FP32;
        Reg::RegTensor<uint16_t> reversedShareExp;
        Reg::RegTensor<float> reversedShareExp0FP32;
        Reg::RegTensor<float> reversedShareExp1FP32;
        Reg::RegTensor<y1Dtype> yZeroFP8;
        Reg::RegTensor<y1Dtype> yOneFP8;

        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>(reversedShareExp, mxScale2ReciprocalAddr);
        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reversedShareExp0FP32,
                                                              (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reversedShareExp1FP32,
                                                             (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
        }

        for (uint16_t j = 0; j < blockCount; j++) {
            Reg::DataCopy<xDtype, Reg::LoadDist::DIST_NORM>(x, xAddr + j * localUbRowLen);
            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0FP32, x, pregAll16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1FP32, x, pregAll16);
                Reg::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
                Reg::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);
            } else {
                Reg::Mul(xBF16, x, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(x0FP32, xBF16, pregAll16);
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(x1FP32, xBF16, pregAll16);
            }

            Reg::Cast<y1Dtype, float, castTraitFp32toYdtype>(yZeroFP8, (Reg::RegTensor<float>&)x0FP32, pregAll32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)yZeroFP8,
                                                                    (Reg::RegTensor<uint32_t>&)yZeroFP8);

            Reg::Cast<y1Dtype, float, castTraitFp32toYdtype>(yOneFP8, (Reg::RegTensor<float>&)x1FP32, pregAll32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)yOneFP8,
                                                                    (Reg::RegTensor<uint32_t>&)yOneFP8);

            Reg::Interleave((Reg::RegTensor<uint16_t>&)yZeroFP8, (Reg::RegTensor<uint16_t>&)yOneFP8,
                            (Reg::RegTensor<uint16_t>&)yZeroFP8, (Reg::RegTensor<uint16_t>&)yOneFP8);

            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t>&)yZeroFP8,
                                                                   (Reg::RegTensor<uint16_t>&)yZeroFP8);

            DataCopy(y2Addr + (j * localUbRowLen), (Reg::RegTensor<uint8_t>&)yZeroFP8, pregAll8);
        }
    }
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void
SwigluBackwardGroupQuantWithDualAxisMxBase<xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight,
                                           hasClampLimit>::CopyQuantOutput(GlobalTensor<uint8_t>& output,
                                                                           LocalTensor<uint8_t>& local, int64_t offsetA,
                                                                           int64_t offsetB, int64_t blockCount,
                                                                           int64_t dataLenAB, int64_t rowWidth)
{
    uint16_t burst = static_cast<uint16_t>(blockCount);
    uint32_t blockLen = static_cast<uint32_t>(dataLenAB);
    uint32_t rowPitch = static_cast<uint32_t>(rowWidth);
    uint32_t gradBOffset = static_cast<uint32_t>(dataLenAB);
    if ((rowPitch - blockLen) % UB_BLOCK_SIZE != 0) {
        DataCopyExtParams rowParams = {1, blockLen, 0, 0, 0};
        for (int64_t row = 0; row < blockCount; ++row) {
            DataCopyPad(output[offsetA + row * dimGradX_], local[row * rowPitch], rowParams);
            DataCopyPad(output[offsetB + row * dimGradX_], local[row * rowPitch + gradBOffset], rowParams);
        }
        return;
    }
    uint32_t srcStride = (rowPitch - blockLen) / UB_BLOCK_SIZE;
    uint32_t dstStride = static_cast<uint32_t>(dimGradX_ - dataLenAB);
    DataCopyExtParams copyParams = {burst, blockLen, srcStride, dstStride, 0};
    DataCopyPad(output[offsetA], local, copyParams);
    DataCopyPad(output[offsetB], local[gradBOffset], copyParams);
}

template <typename xDtype, typename y1Dtype, typename weightDtype, uint64_t mode, bool hasGroupIndex, bool hasWeight,
          bool hasClampLimit>
__aicore__ inline void
SwigluBackwardGroupQuantWithDualAxisMxBase<xDtype, y1Dtype, weightDtype, mode, hasGroupIndex, hasWeight,
                                           hasClampLimit>::CopyOut(int64_t yOffsetA, int64_t yOffsetB,
                                                                   int64_t scale1OutOffset, int64_t scale2OutOffsetA,
                                                                   int64_t scale2OutOffsetB, int64_t blockCount,
                                                                   int64_t blockCountAlign, int64_t dataLenAB,
                                                                   int64_t rowWidth)
{
    uint16_t outBurst = static_cast<uint16_t>(blockCount);

    uint32_t scale1OutLen = ops::CeilDiv(dataLenAB, BLOCK_SIZE);
    DataCopyExtParams scale1CopyOutParams = {
        outBurst, static_cast<uint32_t>(scale1OutLen * sizeof(uint8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(ops::CeilAlign(dimGradX_, DOUBLE_BLOCK_SIZE) / BLOCK_SIZE - scale1OutLen),
        static_cast<uint32_t>(0)};

    uint32_t scale2BlockLen = dataLenAB * DIGIT_TWO * sizeof(uint8_t);
    uint32_t scale2GroupSize = scale2BlockLen * DIGIT_TWO;
    uint32_t scale2SrcStride = (scale2GroupSize - scale2BlockLen) / UB_BLOCK_SIZE;
    uint32_t scale2DstStride = (dimGradX_ - dataLenAB * DIGIT_TWO) * sizeof(uint8_t);

    DataCopyExtParams scale2CopyOutParamsA = {static_cast<uint16_t>(blockCountAlign / DOUBLE_BLOCK_SIZE),
                                              scale2BlockLen, scale2SrcStride, scale2DstStride,
                                              static_cast<uint32_t>(0)};

    LocalTensor<uint8_t> y1Local = outQueue1_.template DeQue<uint8_t>();
    CopyQuantOutput(yGm1_, y1Local, yOffsetA, yOffsetB, blockCount, dataLenAB, rowWidth);
    outQueue1_.FreeTensor(y1Local);

    LocalTensor<uint8_t> y2Local = outQueue2_.template DeQue<uint8_t>();
    CopyQuantOutput(yGm2_, y2Local, yOffsetA, yOffsetB, blockCount, dataLenAB, rowWidth);
    outQueue2_.FreeTensor(y2Local);

    LocalTensor<uint8_t> mxScale1Local = mxScaleQueue1_.template DeQue<uint8_t>();
    int64_t scale1BOffset = ubRowCount_ * UB_BLOCK_SIZE;
    DataCopyPad(mxScaleGm1_[scale1OutOffset], mxScale1Local, scale1CopyOutParams);
    DataCopyPad(mxScaleGm1_[scale1OutOffset + dimN_ / BLOCK_SIZE], mxScale1Local[scale1BOffset], scale1CopyOutParams);
    mxScaleQueue1_.FreeTensor(mxScale1Local);

    LocalTensor<uint8_t> mxScale2Local = mxScaleQueue2_.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm2_[scale2OutOffsetA], mxScale2Local, scale2CopyOutParamsA);
    DataCopyPad(mxScaleGm2_[scale2OutOffsetB], mxScale2Local[scale2BlockLen], scale2CopyOutParamsA);

    mxScaleQueue2_.FreeTensor(mxScale2Local);
}
} // namespace SwigluBackwardGroupQuantWithDualAxisMx

#endif // OPS_NN_SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_REGBASE_H
