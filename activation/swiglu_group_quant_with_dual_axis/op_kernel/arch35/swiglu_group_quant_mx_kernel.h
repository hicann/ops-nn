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
 * \file swiglu_group_quant_mx_kernel.h
 * \brief Dual-axis-owned SwiGLU MX kernel shared with the single-axis operator.
 *
 * Input x is [M, 2N] (left half [M, N] and right half [M, N] for SwiGLU).
 * group_index is [numGroups] cumsum int64 defining group boundaries.
 * Outputs: y1, mx_scale1 (axis=-1 quantization), y2, mx_scale2 (axis=-2 quantization).
 *
 * Each task = (group, columnBlock). Within a task, iterate over splitBlockH-row chunks.
 */

#ifndef OPS_NN_SWIGLU_GROUP_QUANT_MX_KERNEL_H
#define OPS_NN_SWIGLU_GROUP_QUANT_MX_KERNEL_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../inc/platform.h"
#include "../../inc/kernel_utils.h"
#include "swiglu_group_quant_axis1_common.h"
#include "swiglu_group_quant_mx_tiling_data.h"
#include "swiglu_group_quant_flags.h"

#ifndef TPL_SCALE_ALG_1
#define TPL_SCALE_ALG_1 1
#endif
#ifndef TPL_MODE_ROTATE
#define TPL_MODE_ROTATE 0
#endif
#ifndef TPL_MODE_BLOCK
#define TPL_MODE_BLOCK 1
#endif
#ifndef TPL_NO_GROUP_INDEX
#define TPL_NO_GROUP_INDEX 0
#endif
#ifndef TPL_GROUP_INDEX
#define TPL_GROUP_INDEX 1
#endif

namespace SwigluGroupQuantMx {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;

constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint32_t FP8_E5M2_MAX = 0x37924925;
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925;

struct SingleAxisPolicy {
    static constexpr bool ENABLE_SECOND_AXIS = false;
};

struct DualAxisPolicy {
    static constexpr bool ENABLE_SECOND_AXIS = true;
};

// CuBLAS-compatible scaleAlg=1 constants
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t EXP_254 = 0x000000fe;
constexpr uint32_t HALF_FOR_MAN = 0x00400000;
constexpr uint32_t VF_LEN_FP32 = platform::GetVRegSize() / sizeof(float);
constexpr uint32_t VF_LEN_B16 = platform::GetVRegSize() / sizeof(half);
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BLOCK_SIZE = 64;
constexpr int64_t ONCE_ROW_LEN = 256;
constexpr int64_t PACKED_COLS = 384;
constexpr int64_t PACKED_ROWS = 64;
// The shared 64x384 tile uses two 32-row input buffers and single-buffered outputs.
// The last 128-column axis-2 vector load has an allocated 128-element suffix,
// so its unused lanes stay within the allocated UB buffer.
constexpr int64_t PACKED_SINGLE_UB_BYTES = 7 * PACKED_ROWS * PACKED_COLS +
                                           2 * (PACKED_ROWS * PACKED_COLS / ONCE_ROW_LEN) * platform::GetUbBlockSize() +
                                           4 * PACKED_COLS + 2 * platform::GetVRegSize();
constexpr int64_t PACKED_DUAL_UB_BYTES = PACKED_SINGLE_UB_BYTES + PACKED_ROWS * PACKED_COLS +
                                         2 * platform::GetVRegSize() +
                                         (PACKED_COLS / VF_LEN_FP32 - 1) * DIGIT_TWO * VF_LEN_FP32;
static_assert(PACKED_SINGLE_UB_BYTES <= 203776, "Packed UB exceeds the host budget");
static_assert(PACKED_DUAL_UB_BYTES <= 238336, "Packed dual-axis UB exceeds the host budget");
constexpr int64_t UB_BLOCK_SIZE = platform::GetUbBlockSize();
constexpr uint32_t SCALE1_RECIPROCAL_ROW_ELEMS = UB_BLOCK_SIZE / sizeof(uint16_t);
// DAV_3510 BrcbCommonImpl advances a DIST_E2B_B16 source by this many elements.
constexpr uint32_t E2B_B16_SOURCE_ELEMS = BRCB_BROADCAST_NUMBER;
constexpr uint32_t CUBLAS_SCALE2_STORE_COUNT = ONCE_ROW_LEN / VF_LEN_FP32;
constexpr uint32_t CUBLAS_SCALE2_STORE_STRIDE_BYTES = DIGIT_TWO * VF_LEN_FP32;
constexpr uint32_t CUBLAS_SCALE2_ONE_STORE_BYTES = DIGIT_TWO * platform::GetVRegSize();
constexpr uint32_t CUBLAS_SCALE2_BUFFER_BYTES = CUBLAS_SCALE2_ONE_STORE_BYTES +
                                                (CUBLAS_SCALE2_STORE_COUNT - 1U) * CUBLAS_SCALE2_STORE_STRIDE_BYTES;

static_assert(ONCE_ROW_LEN % VF_LEN_FP32 == 0, "A full row must contain an integral number of vector registers");
static_assert(SCALE1_RECIPROCAL_ROW_ELEMS * sizeof(uint16_t) == UB_BLOCK_SIZE,
              "Each scale1 reciprocal row must occupy exactly one UB block");
static_assert(E2B_B16_SOURCE_ELEMS * sizeof(uint16_t) <= UB_BLOCK_SIZE,
              "DIST_E2B_B16 must not read beyond one scale1 reciprocal row");
static_assert(CUBLAS_SCALE2_BUFFER_BYTES == 896U,
              "Four overlapping DIST_INTLV_B8 stores must fit in the scale2 buffer");

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

template <typename xDtype, typename weightDtype, bool hasWeight, bool outputOrigin>
__simd_vf__ inline void ApplyWeightAndSaveOriginVf(__ubuf__ xDtype* activationAddr, __ubuf__ xDtype* originAddr,
                                                   __ubuf__ weightDtype* weightAddr, uint16_t rows)
{
    using namespace AscendC::Reg;
    constexpr uint32_t rowStride = ONCE_ROW_LEN;
    constexpr uint32_t loadElems = platform::GetVRegSize() / sizeof(xDtype);
    constexpr uint16_t loops = rowStride / loadElems;
    RegTensor<xDtype> packed;
    RegTensor<weightDtype> weightPacked;
    RegTensor<float> layout0;
    RegTensor<float> layout1;
    RegTensor<float> interleaved0;
    RegTensor<float> interleaved1;
    RegTensor<float> weight;
    MaskReg allB16 = CreateMask<xDtype, MaskPattern::ALL>();
    MaskReg allF32 = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        if constexpr (hasWeight) {
            SwigluGroupQuantAxis1::LoadWeight<weightDtype>(weight, weightPacked, weightAddr + row, allF32);
        }
        for (uint16_t loop = 0; loop < loops; ++loop) {
            const uint32_t inputOffset = row * rowStride + loop * loadElems;
            LoadAlign(packed, activationAddr + inputOffset);
            if constexpr (outputOrigin) {
                StoreAlign<xDtype>(originAddr + inputOffset, packed, allB16);
            }
            Cast<float, xDtype, SwigluGroupQuantAxis1::CAST_B16_TO_B32_LAYOUT0>(layout0, packed, allB16);
            Cast<float, xDtype, SwigluGroupQuantAxis1::CAST_B16_TO_B32_LAYOUT1>(layout1, packed, allB16);
            if constexpr (hasWeight) {
                SwigluGroupQuantAxis1::ApplyWeight<xDtype, true>(layout0, weight, allF32);
                SwigluGroupQuantAxis1::ApplyWeight<xDtype, true>(layout1, weight, allF32);
                Interleave(interleaved0, interleaved1, layout0, layout1);
                SwigluGroupQuantAxis1::StoreInput<xDtype>(activationAddr, interleaved0, allF32,
                                                          row * rowStride + (2 * loop) * VF_LEN_FP32);
                SwigluGroupQuantAxis1::StoreInput<xDtype>(activationAddr, interleaved1, allF32,
                                                          row * rowStride + (2 * loop + 1) * VF_LEN_FP32);
            }
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs = false, bool hasClamp = false, typename SecondAxisPolicy = DualAxisPolicy>
class SwigluGroupQuantMxKernel {
public:
    __aicore__ inline SwigluGroupQuantMxKernel(const SwigluGroupQuantMxTilingData* tilingData, TPipe* pipe,
                                               float clampLimit = 0.0F, float alpha = 1.0F, float bias = 0.0F,
                                               uint32_t flags = 0U, uint32_t weightType = 2U)
        : tilingData_(tilingData),
          pipe_(pipe),
          clampLimit_(clampLimit),
          alpha_(alpha),
          bias_(bias),
          flags_(flags),
          weightType_(weightType){};
    __aicore__ inline ~SwigluGroupQuantMxKernel()
    {
        if (shareOrigin_) {
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(originDone_);
        }
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2,
                                GM_ADDR mxScale2, GM_ADDR yOrigin);
    __aicore__ inline void Process();

private:
    // One real activation tile and its output/scratch buffers. Axis-2 outputs
    // are null for the single-axis policy and are never accessed in that policy.
    struct QuantizationBuffers {
        __ubuf__ xDtype* activation;
        __ubuf__ uint8_t* y1;
        __ubuf__ uint8_t* scale1;
        __ubuf__ uint16_t* reciprocal1;
        __ubuf__ uint8_t* y2;
        __ubuf__ uint8_t* scale2;
        __ubuf__ uint16_t* reciprocal2;
    };
    __aicore__ inline void PrepareRegularActivation(int64_t absRowStart, int64_t colOffset, int64_t realRows,
                                                    int64_t validCols, uint32_t alignedCols,
                                                    __ubuf__ xDtype* swigluAddr);
    // nextTileRows == 0 ends prefetch at this core's assigned group boundary.
    __aicore__ inline void PreparePacked384Activation(int64_t absRowStart, int64_t realRows, bool firstTile,
                                                      int64_t nextTileRows, __ubuf__ xDtype* swigluAddr);
    __aicore__ inline void QuantizeRegularTile(uint16_t paddedRows, const QuantizationBuffers& buffers);
    __aicore__ inline void QuantizePacked384Tile(uint16_t paddedRows, const QuantizationBuffers& buffers);
    __aicore__ inline void InitParams();
    __aicore__ inline void CopyInSwiglu(int64_t absRowStart, int64_t colOffset, int64_t calcRow, int64_t calcCol);
    __aicore__ inline void ComputeSwiglu(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* actAddr,
                                         __ubuf__ xDtype* gateAddr, __ubuf__ xDtype* swigluOutAddr,
                                         uint32_t alignDim1Out);
    template <bool packedTile = false>
    __aicore__ inline void ComputeSwigluFullTile(__ubuf__ xDtype* actAddr, __ubuf__ xDtype* gateAddr,
                                                 __ubuf__ xDtype* swigluOutAddr, uint16_t rows = DOUBLE_BLOCK_SIZE);
    template <typename weightDtype, bool hasWeight, bool outputOrigin>
    __aicore__ inline void ApplyWeightAndSaveOrigin(const LocalTensor<xDtype>& activation,
                                                    const LocalTensor<xDtype>& origin,
                                                    const LocalTensor<uint8_t>& weight, uint16_t rows);
    __aicore__ inline void DispatchWeightAndOrigin(const LocalTensor<xDtype>& activation,
                                                   const LocalTensor<xDtype>& origin,
                                                   const LocalTensor<uint8_t>& weight, uint16_t rows);
    __aicore__ inline void PadZeroM(__ubuf__ xDtype* swigluOutAddr, uint32_t num);
    // Packed axis 1 stores scales contiguously and leaves axis 2 to a separate real-row scan.
    template <bool packedAxis1>
    __aicore__ inline void ComputeScaleCuBLAS(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                              __ubuf__ uint8_t* y1Addr, __ubuf__ uint8_t* mxScale1Addr,
                                              __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
                                              __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    __aicore__ inline void ComputeScaleCuBLASSecondLast(uint16_t dataLen, uint32_t localInvDtypeMax,
                                                        __ubuf__ uint16_t* mxScale2ReciprocalAddr,
                                                        __ubuf__ uint8_t* mxScale2Addr);
    // DAV_3510 Reg models source RegTensor/MaskReg operands as mutable references;
    // these parameters are semantically read-only but cannot be const-qualified.
    __aicore__ inline void ComputeScaleCuBLASForSlot(
        __ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr, Reg::RegTensor<uint8_t>& scale8,
        Reg::RegTensor<uint32_t>& invMax, Reg::RegTensor<uint32_t>& manMaskReg, Reg::RegTensor<uint32_t>& expMaskReg,
        Reg::RegTensor<uint32_t>& zero32Reg, Reg::RegTensor<uint32_t>& scaleBiasReg, Reg::RegTensor<uint32_t>& nan32Reg,
        Reg::RegTensor<uint32_t>& fp8Nan32Reg, Reg::MaskReg& maskAll, Reg::MaskReg& maskAll32, Reg::MaskReg& maskB16);
    template <uint16_t validCols = ONCE_ROW_LEN>
    __aicore__ inline void ComputeY2ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputePackedSecondAxis(__ubuf__ xDtype* activation, __ubuf__ uint16_t* reciprocal,
                                                   __ubuf__ uint8_t* scale, __ubuf__ uint8_t* output);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t scale1OutOffset, int64_t scale2OutOffset,
                                   int64_t blockCount, int64_t blockCountAlign, int64_t dataLen, int64_t dataLenAlign);

private:
    // tiling data
    const SwigluGroupQuantMxTilingData* tilingData_;

    // pipe & queue & buf
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TBuf<TPosition::VECCALC> swigluBuf_;
    TQue<QuePosition::VECOUT, 1> outQueue1_;
    TQue<QuePosition::VECOUT, 1> outQueue2_;
    TQue<QuePosition::VECOUT, 1> mxScaleQueue1_;
    TQue<QuePosition::VECOUT, 1> mxScaleQueue2_;
    TQue<QuePosition::VECIN, 1> weightQueue_;
    TQue<QuePosition::VECOUT, 1> originQueue_;
    TBuf<TPosition::VECCALC> mxScale1ReciprocalBuf_;
    TBuf<TPosition::VECCALC> mxScale2ReciprocalBuf_;

    // gm
    GlobalTensor<xDtype> xGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<uint8_t> yGm1_;
    GlobalTensor<uint8_t> mxScaleGm1_;
    GlobalTensor<uint8_t> yGm2_;
    GlobalTensor<uint8_t> mxScaleGm2_;
    GlobalTensor<uint8_t> weightGm_;
    GlobalTensor<xDtype> yOriginGm_;

    // base variables
    int64_t blockIdx_ = 0;
    bool packed384_ = false;
    int64_t ubRowLen_ = 0;        // 256 normally; 384 for packed tiles
    int64_t ubRowLenTail_ = 0;    // dimNTail
    int64_t ubRowCount_ = 0;      // 64 real rows for either axis policy
                                  // MX block size, fixed
    int64_t dimNeg1ScaleNum_ = 0; // ceil(dimN / blockSize)
    uint32_t invDtypeMax_ = 0;
    int64_t activateLeft_ = 1;
    int64_t inHalfSize_ = 0; // per-half buffer size in elements (splitBlockH * blockW)
    int64_t dimN_ = 0;
    float clampLimit_ = 0.0F;
    float alpha_ = 1.0F;
    float bias_ = 0.0F;
    uint32_t flags_ = 0U;
    uint32_t weightType_ = 2U;
    bool hasWeight_ = false;
    bool outputOrigin_ = false;
    bool shareOrigin_ = false;
    bool ownWholeGroups_ = false;
    event_t originDone_;

    int64_t oneBlockCountB16_ = UB_BLOCK_SIZE / sizeof(xDtype);
};

// InitParams — cache tiling parameters, set dtype constants
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs,
                                                hasClamp, SecondAxisPolicy>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    ubRowLen_ = ONCE_ROW_LEN; // 固定256
    ubRowLenTail_ = tilingData_->dimNTail;
    ubRowCount_ = DOUBLE_BLOCK_SIZE; // 固定64
    activateLeft_ = tilingData_->activateLeft;
    dimN_ = tilingData_->dimN;
    // Both axis policies use exactly the same 768-column activation/axis-1 path.
    // Origin sharing is guaranteed by the two-dimensional host policies.
    packed384_ = dimN_ == PACKED_COLS && (flags_ & MX_HAS_WEIGHT) == 0 &&
                 ((flags_ & MX_OUTPUT_ORIGIN) == 0 || (flags_ & MX_SHARE_ORIGIN) != 0);
    if (packed384_) {
        ubRowLen_ = PACKED_COLS;
        ubRowLenTail_ = PACKED_COLS;
        ubRowCount_ = PACKED_ROWS;
    }

    static_assert(scaleAlg == TPL_SCALE_ALG_1, "The shared MX kernel only supports CuBLAS MX scale");
    static_assert(IsSameType<y1Dtype, fp8_e4m3fn_t>::value || IsSameType<y1Dtype, fp8_e5m2_t>::value,
                  "The shared MX kernel only supports FP8 output");
    if constexpr (IsSameType<y1Dtype, fp8_e4m3fn_t>::value) {
        invDtypeMax_ = FP8_E4M3_MAX;
    } else {
        invDtypeMax_ = FP8_E5M2_MAX;
    }
}

// Init — set up GlobalTensors, allocate UB buffers
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs,
                                                hasClamp, SecondAxisPolicy>::Init(GM_ADDR x, GM_ADDR weight,
                                                                                  GM_ADDR groupIndex, GM_ADDR y1,
                                                                                  GM_ADDR mxScale1, GM_ADDR y2,
                                                                                  GM_ADDR mxScale2, GM_ADDR yOrigin)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif

    InitParams();

    xGm_.SetGlobalBuffer((__gm__ xDtype*)(x));
    if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {
        groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)(groupIndex));
        groupIndexGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    xGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    yGm1_.SetGlobalBuffer((__gm__ uint8_t*)(y1));
    mxScaleGm1_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale1));
    if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS) {
        yGm2_.SetGlobalBuffer((__gm__ uint8_t*)(y2));
        mxScaleGm2_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale2));
    }
    hasWeight_ = (flags_ & MX_HAS_WEIGHT) != 0;
    outputOrigin_ = (flags_ & MX_OUTPUT_ORIGIN) != 0;
    // Origin writeback uses the common activation buffer on either axis policy.
    shareOrigin_ = (flags_ & MX_SHARE_ORIGIN) != 0;
    if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS) {
        ownWholeGroups_ = (flags_ & MX_OWN_WHOLE_GROUPS) != 0;
    }
    if (shareOrigin_) {
        // Reserve the event so queues cannot reuse the origin-completion signal.
        originDone_ = static_cast<event_t>(pipe_->AllocEventID<HardEvent::MTE3_V>());
    }
    if (hasWeight_) {
        weightGm_.SetGlobalBuffer((__gm__ uint8_t*)(weight));
    }
    if (outputOrigin_) {
        yOriginGm_.SetGlobalBuffer((__gm__ xDtype*)(yOrigin));
    }

    inHalfSize_ = ubRowLen_ * ubRowCount_;
    int64_t inBufferSize = inHalfSize_ * static_cast<int64_t>(sizeof(xDtype));

    const int64_t mxScale2BufferSize = packed384_ ?
                                           CUBLAS_SCALE2_ONE_STORE_BYTES +
                                               (PACKED_COLS / VF_LEN_FP32 - 1) * CUBLAS_SCALE2_STORE_STRIDE_BYTES :
                                           CUBLAS_SCALE2_BUFFER_BYTES;

    // axis=-1 scale buffer
    int64_t mxScale1BufferSize = (packed384_ ? ubRowCount_ * ubRowLen_ / ONCE_ROW_LEN : ubRowCount_) * UB_BLOCK_SIZE;

    // axis=-2 1/scale (xDtype sized for bf16 reciprocal storage)
    int64_t tmpScale2BufferSize = (ubRowLen_ * DIGIT_TWO + (packed384_ ? VF_LEN_B16 : 0)) *
                                  static_cast<int64_t>(sizeof(xDtype));

    // Regular input: two 64-row buffers with separate left/right halves.
    // Packed input: two 32-row buffers of contiguous [rows, 768] input.
    // Each packed input chunk occupies as many bytes as the [64, 384] activation tile.
    pipe_->InitBuffer(inQueue_, DB_BUFFER, packed384_ ? inBufferSize : inBufferSize * DIGIT_TWO);
    pipe_->InitBuffer(swigluBuf_, inBufferSize + (packed384_ ? platform::GetVRegSize() : 0));
    const uint32_t outputQueueDepth = packed384_ || ((hasWeight_ || outputOrigin_) && !shareOrigin_) ? 1U : DB_BUFFER;
    pipe_->InitBuffer(outQueue1_, outputQueueDepth, inHalfSize_);
    pipe_->InitBuffer(mxScaleQueue1_, outputQueueDepth, mxScale1BufferSize);
    if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS) {
        pipe_->InitBuffer(outQueue2_, outputQueueDepth, inHalfSize_);
        pipe_->InitBuffer(mxScaleQueue2_, outputQueueDepth, mxScale2BufferSize);
    }
    pipe_->InitBuffer(mxScale1ReciprocalBuf_, mxScale1BufferSize);
    pipe_->InitBuffer(mxScale2ReciprocalBuf_, tmpScale2BufferSize);
    if (hasWeight_) {
        pipe_->InitBuffer(weightQueue_, 1, ubRowCount_ * sizeof(float));
    }
    if (outputOrigin_ && !shareOrigin_) {
        pipe_->InitBuffer(originQueue_, 1, inBufferSize);
    }
}

// Process — outer loop over groups; two core-distribution scenarios:
//   Scenario 1 (nSplitNum < usedCoreNum): groups rotate across core ranges.
//     Each group's blocks are assigned starting from a rotating core offset.
//     e.g. group0: core 0..31, group1: core 32..63, group2: core 0..31, ...
//   Scenario 2 (nSplitNum >= usedCoreNum): each group distributes all its
//     blocks across all usedCoreNum cores (same as dynamic_mx_quant_with_dual_axis).
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs,
                                                hasClamp, SecondAxisPolicy>::Process()
{
    int64_t numGroups = 1;
    if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {
        numGroups = tilingData_->numGroups;
    }
    int64_t dimM = tilingData_->dimM;
    // Scale column count per row: ceil(dimN / blockSize)
    dimNeg1ScaleNum_ = (ops::CeilDiv(dimN_, DOUBLE_BLOCK_SIZE)) * DIGIT_TWO;
    int64_t dimNBlockNum = packed384_ ? 1 : tilingData_->dimNBlockNum;
    int64_t totalCoreNum = tilingData_->usedCoreNum; // kernel 侧 usedCoreNum 即 totalCoreNum
    if (blockIdx_ >= totalCoreNum) {
        return;
    }

    if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {
        // Check coverage on every active core before any group can write output.
        // Trap is intentional: a debug-only assert can disappear in release builds.
        if (numGroups <= 0 || groupIndexGm_.GetValue(numGroups - 1) != dimM) {
            AscendC::Trap();
            return;
        }
    }

    // scale2 row offset (computed per-group, matching grouped_dynamic_mx_quant formula)
    // Rotating core offset for mode=ROTATE
    [[maybe_unused]] int64_t coreRotateOffset = 0; // 每个group从哪个物理核开始用

    // If groups fill all cores, keep each group's row and column tiles on one owner.
    const bool ownWholeGroups = ownWholeGroups_;
    const int64_t firstGroup = ownWholeGroups ? blockIdx_ : 0;
    const int64_t groupStride = ownWholeGroups ? totalCoreNum : 1;
    for (int64_t g = firstGroup; g < numGroups; g += groupStride) {
        // Read group boundaries from cumsum group_index
        int64_t groupStart = 0;
        int64_t groupEnd = dimM;
        if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {
            groupStart = (g > 0) ? groupIndexGm_.GetValue(g - 1) : 0;
            groupEnd = groupIndexGm_.GetValue(g);
            // Validate before subtraction or GM offset arithmetic. Equal endpoints
            // are legal empty groups. Each owner checks its own boundary pair.
            if (groupStart < 0 || groupEnd < groupStart || groupEnd > dimM) {
                AscendC::Trap();
                return;
            }
        }
        int64_t groupRows = groupEnd - groupStart; // 每个group处理行数
        if (groupRows <= 0) {
            continue;
        }
        // Compute scale2 row offset: each group's scale2 occupies 2 rows in GM
        int64_t scale2GmRowOffset = (groupStart / DOUBLE_BLOCK_SIZE + g) * DIGIT_TWO;

        int64_t dimMSplitG = ops::CeilDiv(groupRows, ubRowCount_);
        int64_t blockCountG = dimMSplitG * dimNBlockNum; // 每个group的所有块个数

        // Determine this core's work for this group
        int64_t loopPerCoreG = 0;
        int64_t blockOffsetG = 0;

        if (ownWholeGroups) {
            loopPerCoreG = blockCountG;
        } else if constexpr (mode == TPL_MODE_ROTATE) { // N方向块数 < totalCoreNum
            // mode=0: dimNBlockNum < totalCoreNum — groups rotate across core ranges
            int64_t usedCoreNumG = (blockCountG < totalCoreNum) ? blockCountG : totalCoreNum; // 这次group用多少个核
            int64_t myCoreInGroup = blockIdx_ - coreRotateOffset;
            if (myCoreInGroup < 0) {
                myCoreInGroup += totalCoreNum; // wrap around
            }

            if (myCoreInGroup < usedCoreNumG) {
                int64_t headCoreNumG = blockCountG % usedCoreNumG;
                int64_t blockPerHeadCoreG = ops::CeilDiv(blockCountG, usedCoreNumG);
                int64_t blockPerTailCoreG = blockCountG / usedCoreNumG;
                if (myCoreInGroup < headCoreNumG) {
                    loopPerCoreG = blockPerHeadCoreG;
                    blockOffsetG = myCoreInGroup * loopPerCoreG;
                } else {
                    loopPerCoreG = blockPerTailCoreG;
                    blockOffsetG = headCoreNumG * blockPerHeadCoreG + (myCoreInGroup - headCoreNumG) * loopPerCoreG;
                }
            }
            // Advance rotating offset for next group
            coreRotateOffset = (coreRotateOffset + blockCountG) % totalCoreNum;
            if (loopPerCoreG == 0) {
                continue;
            }
        } else {
            // mode=1: dimNBlockNum >= totalCoreNum — all cores share each group's blocks
            int64_t usedCoreNumG = totalCoreNum;
            int64_t headCoreNumG = blockCountG % usedCoreNumG;
            int64_t blockPerHeadCoreG = ops::CeilDiv(blockCountG, usedCoreNumG);
            int64_t blockPerTailCoreG = blockCountG / usedCoreNumG;
            if (blockIdx_ < headCoreNumG) {
                loopPerCoreG = blockPerHeadCoreG;
                blockOffsetG = blockIdx_ * loopPerCoreG;
            } else {
                loopPerCoreG = blockPerTailCoreG;
                blockOffsetG = headCoreNumG * blockPerHeadCoreG + (blockIdx_ - headCoreNumG) * loopPerCoreG;
            }
        }

        int64_t dimMTailG = groupRows % ubRowCount_ == 0 ? ubRowCount_ : groupRows % ubRowCount_;

        // Process assigned blocks (same as dynamic_mx_quant_with_dual_axis::Process)
        for (int64_t i = 0; i < loopPerCoreG; i++) { // -2轴循环多少次
            int64_t blockInGroup = blockOffsetG + i;
            int64_t rowBlockIdx = blockInGroup / dimNBlockNum; // M方向
            int64_t colBlockIdx = blockInGroup % dimNBlockNum; // N方向

            int64_t calcCol = (colBlockIdx == dimNBlockNum - 1) ? ubRowLenTail_ : ubRowLen_;
            int64_t calcRow = (rowBlockIdx == dimMSplitG - 1) ? dimMTailG : ubRowCount_;
            int64_t absRowStart = groupStart + rowBlockIdx * ubRowCount_;
            int64_t colOffset = colBlockIdx * ubRowLen_;
            LocalTensor<xDtype> swigluLocal = swigluBuf_.template Get<xDtype>();
            auto swigluAddr = (__ubuf__ xDtype*)swigluLocal.GetPhyAddr();
            uint32_t alignDim1OutAlgin = ops::CeilDiv(calcCol, DOUBLE_BLOCK_SIZE) * DOUBLE_BLOCK_SIZE;
            uint32_t calcPadRowAlgin = ops::CeilDiv(calcRow, DOUBLE_BLOCK_SIZE) * DOUBLE_BLOCK_SIZE;
            if (packed384_) {
                const int64_t nextRowBlock = rowBlockIdx + 1;
                const int64_t nextTileRows = i + 1 < loopPerCoreG ?
                                                 (nextRowBlock == dimMSplitG - 1 ? dimMTailG : ubRowCount_) :
                                                 0;
                PreparePacked384Activation(absRowStart, calcRow, i == 0, nextTileRows, swigluAddr);
            } else {
                PrepareRegularActivation(absRowStart, colOffset, calcRow, calcCol, alignDim1OutAlgin, swigluAddr);
            }
            if (calcRow % DOUBLE_BLOCK_SIZE != 0) {
                uint32_t calcPadRow = calcPadRowAlgin - calcRow;
                uint32_t allNumZero = calcPadRow * ubRowLen_;
                auto swigluAddrPadZero = (__ubuf__ xDtype*)swigluLocal[calcRow * ubRowLen_].GetPhyAddr();
                PadZeroM(swigluAddrPadZero, allNumZero); // M 方向补0
            }

            LocalTensor<uint8_t> weightLocal;
            if (hasWeight_) {
                const uint32_t weightBytes = weightType_ == 2 ? sizeof(float) : sizeof(uint16_t);
                weightLocal = weightQueue_.template AllocTensor<uint8_t>();
                DataCopyExtParams weightCopy = {1, static_cast<uint32_t>(calcRow * weightBytes), 0, 0, 0};
                DataCopyPadExtParams<uint8_t> weightPad = {false, 0, 0, 0};
                DataCopyPad(weightLocal, weightGm_[absRowStart * weightBytes], weightCopy, weightPad);
                weightQueue_.EnQue(weightLocal);
                weightLocal = weightQueue_.template DeQue<uint8_t>();
            }
            LocalTensor<xDtype> originLocal;
            if (outputOrigin_ && !shareOrigin_) {
                originLocal = originQueue_.template AllocTensor<xDtype>();
            }
            if (hasWeight_ || (outputOrigin_ && !shareOrigin_)) {
                DispatchWeightAndOrigin(swigluLocal, originLocal, weightLocal, static_cast<uint16_t>(calcRow));
            }
            if (hasWeight_) {
                weightQueue_.FreeTensor(weightLocal);
            }
            if (outputOrigin_ && !shareOrigin_) {
                originQueue_.EnQue(originLocal);
            }
            if (shareOrigin_) {
                // DMA and quantization only read this activation until the tile finishes.
                const auto ready = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(ready);
                WaitFlag<HardEvent::V_MTE3>(ready);
                DataCopyExtParams originParams = {
                    static_cast<uint16_t>(calcRow), static_cast<uint32_t>(calcCol * sizeof(xDtype)),
                    static_cast<uint32_t>((ubRowLen_ - calcCol) * sizeof(xDtype) / UB_BLOCK_SIZE),
                    (dimN_ - calcCol) * static_cast<int64_t>(sizeof(xDtype)), 0};
                if (packed384_) {
                    originParams.blockCount = 1;
                    originParams.blockLen = calcRow * calcCol * sizeof(xDtype);
                    originParams.dstStride = 0;
                }
                DataCopyPad(yOriginGm_[absRowStart * dimN_ + colOffset], swigluLocal, originParams);
                SetFlag<HardEvent::MTE3_V>(originDone_);
            }
            // ---- ComputeAll (scale + quantize) — same as dynamic_mx_quant_with_dual_axis ----
            LocalTensor<uint8_t> mxScale1 = mxScaleQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> y1 = outQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> mxScale1Reciprocal = mxScale1ReciprocalBuf_.template Get<uint16_t>();
            LocalTensor<uint16_t> mxScale2Reciprocal = mxScale2ReciprocalBuf_.template Get<uint16_t>();

            auto y1Addr = (__ubuf__ uint8_t*)y1.GetPhyAddr();
            auto ms1Addr = (__ubuf__ uint8_t*)mxScale1.GetPhyAddr();
            auto ms1RecipAddr = (__ubuf__ uint16_t*)mxScale1Reciprocal.GetPhyAddr();
            auto ms2RecipAddr = (__ubuf__ uint16_t*)mxScale2Reciprocal.GetPhyAddr();
            __ubuf__ uint8_t* y2Addr = nullptr;
            __ubuf__ uint8_t* ms2Addr = nullptr;
            LocalTensor<uint8_t> mxScale2;
            LocalTensor<uint8_t> y2;
            if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS) {
                mxScale2 = mxScaleQueue2_.template AllocTensor<uint8_t>();
                y2 = outQueue2_.template AllocTensor<uint8_t>();
                y2Addr = (__ubuf__ uint8_t*)y2.GetPhyAddr();
                ms2Addr = (__ubuf__ uint8_t*)mxScale2.GetPhyAddr();
            }

            const QuantizationBuffers quantBuffers = {swigluAddr, y1Addr,  ms1Addr,     ms1RecipAddr,
                                                      y2Addr,     ms2Addr, ms2RecipAddr};
            if (packed384_) {
                QuantizePacked384Tile(static_cast<uint16_t>(calcPadRowAlgin), quantBuffers);
            } else {
                QuantizeRegularTile(static_cast<uint16_t>(calcPadRowAlgin), quantBuffers);
            }
            mxScaleQueue1_.template EnQue(mxScale1);
            outQueue1_.template EnQue(y1);
            if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS) {
                mxScaleQueue2_.template EnQue(mxScale2);
                outQueue2_.template EnQue(y2);
            }

            int64_t yGmOffset = absRowStart * dimN_ + colOffset;
            int64_t scale1GmOffset = absRowStart * dimNeg1ScaleNum_ + colOffset / BLOCK_SIZE;
            int64_t scale2RowIdx = scale2GmRowOffset + rowBlockIdx * ubRowCount_ / BLOCK_SIZE;
            int64_t scale2GmOffset = scale2RowIdx * dimN_ + colOffset * DIGIT_TWO;

            CopyOut(yGmOffset, scale1GmOffset, scale2GmOffset, calcRow, calcPadRowAlgin, calcCol, alignDim1OutAlgin);
            if (shareOrigin_) {
                // Protect activation reuse without waiting for later quantized-output DMA.
                WaitFlag<HardEvent::MTE3_V>(originDone_);
            }
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::PrepareRegularActivation(int64_t absRowStart, int64_t colOffset,
                                                                     int64_t realRows, int64_t validCols,
                                                                     uint32_t alignedCols, __ubuf__ xDtype* swigluAddr)
{
    CopyInSwiglu(absRowStart, colOffset, realRows, validCols);
    LocalTensor<xDtype> xLocal = inQueue_.template DeQue<xDtype>();
    auto actAddr = (__ubuf__ xDtype*)xLocal.GetPhyAddr();
    auto gateAddr = (__ubuf__ xDtype*)xLocal[inHalfSize_].GetPhyAddr();
    if (activateLeft_ == 0) {
        actAddr = (__ubuf__ xDtype*)xLocal[inHalfSize_].GetPhyAddr();
        gateAddr = (__ubuf__ xDtype*)xLocal.GetPhyAddr();
    }
    if (validCols == ubRowLen_ && realRows == ubRowCount_) {
        ComputeSwigluFullTile<false>(actAddr, gateAddr, swigluAddr);
    } else {
        ComputeSwiglu(static_cast<uint16_t>(validCols), static_cast<uint16_t>(realRows), actAddr, gateAddr, swigluAddr,
                      alignedCols);
    }
    inQueue_.template FreeTensor(xLocal);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::PreparePacked384Activation(int64_t absRowStart, int64_t realRows,
                                                                       bool firstTile, int64_t nextTileRows,
                                                                       __ubuf__ xDtype* swigluAddr)
{
    // Two 32-row input buffers overlap DMA and activation without increasing UB.
    // Single and dual axis use this identical schedule and 64-row activation layout.
    if (firstTile) {
        CopyInSwiglu(absRowStart, 0, realRows < BLOCK_SIZE ? realRows : BLOCK_SIZE, ubRowLen_);
    }
    for (int64_t row = 0; row < realRows; row += BLOCK_SIZE) {
        LocalTensor<xDtype> xLocal = inQueue_.template DeQue<xDtype>();
        if (row + BLOCK_SIZE < realRows) {
            CopyInSwiglu(absRowStart + BLOCK_SIZE, 0, realRows - BLOCK_SIZE, ubRowLen_);
        } else if (nextTileRows > 0) {
            CopyInSwiglu(absRowStart + PACKED_ROWS, 0, nextTileRows < BLOCK_SIZE ? nextTileRows : BLOCK_SIZE,
                         ubRowLen_);
        }
        auto actAddr = (__ubuf__ xDtype*)xLocal.GetPhyAddr();
        auto gateAddr = actAddr + ubRowLen_;
        if (activateLeft_ == 0) {
            gateAddr = actAddr;
            actAddr += ubRowLen_;
        }
        const uint16_t chunkRows = static_cast<uint16_t>(realRows - row < BLOCK_SIZE ? realRows - row : BLOCK_SIZE);
        ComputeSwigluFullTile<true>(actAddr, gateAddr, swigluAddr + row * ubRowLen_, chunkRows);
        inQueue_.template FreeTensor(xLocal);
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::QuantizeRegularTile(uint16_t paddedRows, const QuantizationBuffers& buffers)
{
    // The regular layout accumulates axis-2 maxima during the axis-1 scan.
    ComputeScaleCuBLAS<false>(static_cast<uint16_t>(ubRowLen_), paddedRows, buffers.activation, buffers.y1,
                              buffers.scale1, buffers.reciprocal1, buffers.scale2, buffers.reciprocal2);
    if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS) {
        const int64_t rowGroupCount = paddedRows / BLOCK_SIZE;
        for (int64_t group = 0; group < rowGroupCount; ++group) {
            const int64_t activationOffset = group * BLOCK_SIZE * ubRowLen_;
            const int64_t reciprocalOffset = group * ubRowLen_;
            ComputeY2ToFP8<>(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                             buffers.activation + activationOffset, buffers.reciprocal2 + reciprocalOffset,
                             buffers.y2 + activationOffset);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::QuantizePacked384Tile(uint16_t paddedRows,
                                                                  const QuantizationBuffers& buffers)
{
    // No reshape or extra allocation: 64 real rows of 384 elements are scanned
    // as 96 contiguous 256-element segments. Every 32-element MX group is intact.
    const uint16_t axis1SegmentCount = paddedRows * PACKED_COLS / ONCE_ROW_LEN;
    ComputeScaleCuBLAS<true>(ONCE_ROW_LEN, axis1SegmentCount, buffers.activation, buffers.y1, buffers.scale1,
                             buffers.reciprocal1, buffers.scale2, buffers.reciprocal2);
    if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS) {
        // Axis 2 must use real rows, not the contiguous axis-1 segment view.
        ComputePackedSecondAxis(buffers.activation, buffers.reciprocal2, buffers.scale2, buffers.y2);
    }
}

// CopyInSwiglu — load left and right halves of x into a single inQueue_ buffer
// x layout: [M, 2N] where left = x[row, 0..N-1], right = x[row, N..2N-1]
// Default layout: [0, inHalfSize_) = left half, [inHalfSize_, 2*inHalfSize_) = right half
// Packed tiles retain the contiguous GM [rows, 2N] layout instead.
// NO padding applied here — zero-padding is done in ComputeSwiglu via masking.
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs,
                                                hasClamp, SecondAxisPolicy>::CopyInSwiglu(int64_t absRowStart,
                                                                                          int64_t colOffset,
                                                                                          int64_t calcRow,
                                                                                          int64_t calcCol)
{
    int64_t xRowStride = DIGIT_TWO * dimN_; // total columns in x (2N)

    LocalTensor<xDtype> xLocal = inQueue_.template AllocTensor<xDtype>();

    if (packed384_) {
        // Whole rows are contiguous in GM; keep [left, right] interleaved in UB.
        DataCopyExtParams packedCopy = {1, static_cast<uint32_t>(calcRow * xRowStride * sizeof(xDtype)), 0, 0, 0};
        DataCopyPadExtParams<xDtype> packedPad = {false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[absRowStart * xRowStride], packedCopy, packedPad);
        inQueue_.EnQue(xLocal);
        return;
    }

    DataCopyExtParams copyParams = {0, 0, 0, 0, 0};
    DataCopyPadExtParams<xDtype> padParams = {false, 0, 0, 0};
    copyParams.blockCount = static_cast<uint16_t>(calcRow);
    copyParams.blockLen = static_cast<uint32_t>(calcCol * static_cast<int64_t>(sizeof(xDtype)));
    copyParams.srcStride = (xRowStride - calcCol) * static_cast<int64_t>(sizeof(xDtype));

    // Left half: x[absRowStart, colOffset .. colOffset + calcCol - 1] → xLocal[0..]
    int64_t leftGmOffset = absRowStart * xRowStride + colOffset;
    DataCopyPad(xLocal, xGm_[leftGmOffset], copyParams, padParams);

    // Right half: x[absRowStart, dimN + colOffset .. dimN + colOffset + calcCol - 1] → xLocal[inHalfSize_..]
    int64_t rightGmOffset = leftGmOffset + dimN_;
    DataCopyPad(xLocal[inHalfSize_], xGm_[rightGmOffset], copyParams, padParams);

    inQueue_.template EnQue(xLocal);
}

// ComputeSwiglu — SwiGLU activation: output = SiLU(act) * gate
// SiLU(x) = x / (1 + exp(-x))
//
// actAddr:      activation input (left or right half based on activateLeft)
// gateAddr:     gate input (the other half)
// swigluOutAddr: output buffer with row stride ubRowLen_ (256, or packed 384)
//
// Zero-padding of tail columns is done via masking (zero-mode), NOT in CopyIn.
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::ComputeSwiglu(uint16_t dataLen, uint16_t blockCount,
                                                          __ubuf__ xDtype* actAddr, __ubuf__ xDtype* gateAddr,
                                                          __ubuf__ xDtype* swigluOutAddr, uint32_t alignDim1Out)
{
    uint32_t localOneBlockNum = oneBlockCountB16_;
    uint32_t outAllNum = ubRowLen_;

    // Tail handling: compute masks for zero-padding
    uint16_t dim0VfTimes = blockCount;
    uint32_t localVfLenFp32 = VF_LEN_FP32;
    uint16_t dim1VfTimes = static_cast<uint16_t>(dataLen / VF_LEN_FP32);
    uint32_t dim1Tail = dataLen % localVfLenFp32;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    uint32_t alignDim1In = ((dataLen + localOneBlockNum - 1) / localOneBlockNum) * localOneBlockNum;
    // Quantization reads the full UB row, not just the 64-column aligned valid region.
    const uint32_t paddingCols = outAllNum - alignDim1Out;
    const uint16_t paddingLoops = CeilDivision(paddingCols, VF_LEN_B16);

    __ubuf__ xDtype* actAddr1 = actAddr;
    __ubuf__ xDtype* gateAddr1 = gateAddr;
    __ubuf__ xDtype* swigluAddr1 = swigluOutAddr;
    __ubuf__ xDtype* swigluAddr2 = swigluOutAddr;

    xDtype numZero = 0;
    if (dim1Tail > 0) {
        mask1Num = dim1Tail;
        dim1TailTimes = 1;
        uint32_t padNum = alignDim1Out - dim1VfTimes * localVfLenFp32;
        if (padNum <= localVfLenFp32) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = localVfLenFp32;
            mask3Num = padNum - localVfLenFp32;
        }
        int32_t offsetAlign = dim1VfTimes * localVfLenFp32;
        actAddr1 = actAddr + offsetAlign;
        gateAddr1 = gateAddr + offsetAlign;
        swigluAddr1 = swigluOutAddr + offsetAlign;
        swigluAddr2 = swigluOutAddr + offsetAlign + dim1TailTimes * localVfLenFp32;
    }
    float clampLimit = clampLimit_;
    float alpha = alpha_;
    float bias = bias_;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> vregAct;
        Reg::RegTensor<xDtype> vregGate;
        Reg::RegTensor<float> vregActF;
        Reg::RegTensor<float> vregGateF;
        Reg::RegTensor<float> negReg;
        Reg::RegTensor<float> expReg;
        Reg::RegTensor<float> addsReg;
        Reg::RegTensor<float> sigmoidReg;
        Reg::RegTensor<float> outFReg;
        Reg::RegTensor<xDtype> outTReg;

        Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg mask1 = Reg::UpdateMask<float>(mask1Num);
        Reg::MaskReg mask2 = Reg::UpdateMask<float>(mask2Num);
        Reg::MaskReg mask3 = Reg::UpdateMask<xDtype>(mask3Num);
        Reg::RegTensor<xDtype> paddingZero;
        Reg::Duplicate(paddingZero, 0);

        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            // Full VF iterations (no tail)
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                Reg::AddrReg srcIdxOffset = Reg::CreateAddrReg<xDtype>(dim0vfLoopIdx, alignDim1In, dim1vfLoopIdx,
                                                                       VF_LEN_FP32);
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregAct, actAddr, srcIdxOffset);
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregGate, gateAddr, srcIdxOffset);

                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregActF, vregAct, mask);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregGateF, vregGate, mask);

                SwigluGroupQuantAxis1::ActivationCoreWithScratch<xDtype, hasClamp, hasAttrs>(
                    outFReg, vregActF, vregGateF, negReg, expReg, addsReg, sigmoidReg, vregGateF, mask, clampLimit,
                    alpha, bias);

                Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                Reg::AddrReg outOffset = Reg::CreateAddrReg<xDtype>(dim0vfLoopIdx, outAllNum, dim1vfLoopIdx,
                                                                    VF_LEN_FP32);
                Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(swigluOutAddr, outTReg, outOffset, mask);
            }

            // Tail VF iteration with mask-based zero-padding
            Reg::AddrReg srcIdxOffset1 = Reg::CreateAddrReg<xDtype>(dim0vfLoopIdx, alignDim1In);
            Reg::AddrReg outOffset1 = Reg::CreateAddrReg<xDtype>(dim0vfLoopIdx, outAllNum);

            for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregAct, actAddr1, srcIdxOffset1);
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregGate, gateAddr1, srcIdxOffset1);

                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregActF, vregAct, mask1);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregGateF, vregGate, mask1);

                SwigluGroupQuantAxis1::ActivationCoreWithScratch<xDtype, hasClamp, hasAttrs>(
                    outFReg, vregActF, vregGateF, negReg, expReg, addsReg, sigmoidReg, vregGateF, mask1, clampLimit,
                    alpha, bias);

                // mask2 writes zeros for positions beyond valid data (zero-mode mask)
                Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(swigluAddr1, outTReg, outOffset1, mask2);
            }
            // Additional zero-fill for extra padding positions
            for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
                Duplicate<xDtype>(vregAct, numZero);
                Reg::StoreAlign<xDtype>(swigluAddr2, vregAct, outOffset1, mask3);
            }
            // Preserve valid lanes and zero every remaining column before scale/weight reads.
            uint32_t remainingPadding = paddingCols;
            for (uint16_t padLoop = 0; padLoop < paddingLoops; padLoop++) {
                Reg::MaskReg padMask = Reg::UpdateMask<xDtype>(remainingPadding);
                Reg::AddrReg padOffset = Reg::CreateAddrReg<xDtype>(dim0vfLoopIdx, outAllNum, padLoop, VF_LEN_B16);
                Reg::StoreAlign<xDtype>(swigluOutAddr + alignDim1Out, paddingZero, padOffset, padMask);
            }
        }
    }
}

// Fixed-column activation paths remove runtime tail-mask setup while
// preserving the common path's SwiGLU arithmetic and output layout.
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
template <bool packedTile>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::ComputeSwigluFullTile(__ubuf__ xDtype* actAddr, __ubuf__ xDtype* gateAddr,
                                                                  __ubuf__ xDtype* swigluOutAddr, uint16_t rows)
{
    const uint16_t fullTileRows = packedTile ? rows : static_cast<uint16_t>(DOUBLE_BLOCK_SIZE);
    constexpr uint32_t fullTileCols = packedTile ? PACKED_COLS : ONCE_ROW_LEN;
    constexpr uint32_t inputRowStride = packedTile ? DIGIT_TWO * fullTileCols : fullTileCols;
    static_assert(fullTileCols % VF_LEN_FP32 == 0,
                  "The full-tile column count must contain an integral number of vector registers");
    constexpr uint16_t fullTileVfs = static_cast<uint16_t>(fullTileCols / VF_LEN_FP32);
    float clampLimit = clampLimit_;
    float alpha = alpha_;
    float bias = bias_;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> vregAct;
        Reg::RegTensor<xDtype> vregGate;
        Reg::RegTensor<float> vregActF;
        Reg::RegTensor<float> vregGateF;
        Reg::RegTensor<float> negReg;
        Reg::RegTensor<float> expReg;
        Reg::RegTensor<float> addsReg;
        Reg::RegTensor<float> sigmoidReg;
        Reg::RegTensor<float> outFReg;
        Reg::RegTensor<xDtype> outTReg;
        Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

        for (uint16_t row = 0; row < fullTileRows; row++) {
            for (uint16_t vf = 0; vf < fullTileVfs; vf++) {
                Reg::AddrReg srcOffset = Reg::CreateAddrReg<xDtype>(row, inputRowStride, vf, VF_LEN_FP32);
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregAct, actAddr, srcOffset);
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregGate, gateAddr, srcOffset);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregActF, vregAct, mask);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregGateF, vregGate, mask);
                SwigluGroupQuantAxis1::ActivationCoreWithScratch<xDtype, hasClamp, hasAttrs>(
                    outFReg, vregActF, vregGateF, negReg, expReg, addsReg, sigmoidReg, vregGateF, mask, clampLimit,
                    alpha, bias);
                Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                Reg::AddrReg outOffset = Reg::CreateAddrReg<xDtype>(row, fullTileCols, vf, VF_LEN_FP32);
                Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(swigluOutAddr, outTReg, outOffset, mask);
            }
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
template <typename weightDtype, bool hasWeight, bool outputOrigin>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::ApplyWeightAndSaveOrigin(const LocalTensor<xDtype>& activation,
                                                                     const LocalTensor<xDtype>& origin,
                                                                     const LocalTensor<uint8_t>& weight, uint16_t rows)
{
    __ubuf__ xDtype* originAddr = nullptr;
    __ubuf__ weightDtype* weightAddr = nullptr;
    if constexpr (outputOrigin) {
        originAddr = (__ubuf__ xDtype*)origin.GetPhyAddr();
    }
    if constexpr (hasWeight) {
        weightAddr = (__ubuf__ weightDtype*)weight.GetPhyAddr();
    }
    AscendC::VF_CALL<ApplyWeightAndSaveOriginVf<xDtype, weightDtype, hasWeight, outputOrigin>>(
        (__ubuf__ xDtype*)activation.GetPhyAddr(), originAddr, weightAddr, rows);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::DispatchWeightAndOrigin(const LocalTensor<xDtype>& activation,
                                                                    const LocalTensor<xDtype>& origin,
                                                                    const LocalTensor<uint8_t>& weight, uint16_t rows)
{
    if (!hasWeight_) {
        ApplyWeightAndSaveOrigin<float, false, true>(activation, origin, weight, rows);
    } else if (weightType_ == 0) {
        if (outputOrigin_) {
            ApplyWeightAndSaveOrigin<half, true, true>(activation, origin, weight, rows);
        } else {
            ApplyWeightAndSaveOrigin<half, true, false>(activation, origin, weight, rows);
        }
    } else if (weightType_ == 1) {
        if (outputOrigin_) {
            ApplyWeightAndSaveOrigin<bfloat16_t, true, true>(activation, origin, weight, rows);
        } else {
            ApplyWeightAndSaveOrigin<bfloat16_t, true, false>(activation, origin, weight, rows);
        }
    } else if (outputOrigin_) {
        ApplyWeightAndSaveOrigin<float, true, true>(activation, origin, weight, rows);
    } else {
        ApplyWeightAndSaveOrigin<float, true, false>(activation, origin, weight, rows);
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs,
                                                hasClamp, SecondAxisPolicy>::PadZeroM(__ubuf__ xDtype* swigluOutAddr,
                                                                                      uint32_t num)
{
    uint16_t times = CeilDivision(num, VF_LEN_B16);
    uint32_t size = num;
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> zeroReg;
        AscendC::Reg::MaskReg mask;
        Reg::Duplicate(zeroReg, 0);
        for (uint16_t i = 0; i < times; i++) {
            mask = AscendC::Reg::UpdateMask<xDtype>(size);
            Reg::AddrReg offset = Reg::CreateAddrReg<xDtype>(i, VF_LEN_B16);
            AscendC::Reg::StoreAlign(swigluOutAddr, zeroReg, offset, mask);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp, SecondAxisPolicy>::
    ComputeScaleCuBLASForSlot(__ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr,
                              Reg::RegTensor<uint8_t>& scale8, Reg::RegTensor<uint32_t>& invMax,
                              Reg::RegTensor<uint32_t>& manMaskReg, Reg::RegTensor<uint32_t>& expMaskReg,
                              Reg::RegTensor<uint32_t>& zero32Reg, Reg::RegTensor<uint32_t>& scaleBiasReg,
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

    Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(max16Reg, maxReadAddr,
                                                                                                 VF_LEN_FP32);
    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<xDtype>&)max16Reg,
                                                  maskAll);
    Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32Reg, expMaskReg, maskAll32);
    Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32Reg, zero32Reg, maskAll32);
    Reg::Mul((Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<float>&)invMax,
             maskAll32);
    Reg::ShiftRights(exp32Reg, max32Reg, SHR_NUM_FOR_FP32, maskAll32);
    Reg::And(man32Reg, max32Reg, manMaskReg, maskAll32);
    Reg::Compares<uint32_t, CMPMODE::GT>(p0, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    Reg::Compares<uint32_t, CMPMODE::LT>(p0, exp32Reg, EXP_254, p0);
    Reg::Compares<uint32_t, CMPMODE::GT>(p0, man32Reg, static_cast<uint32_t>(0), p0);
    Reg::Compares<uint32_t, CMPMODE::EQ>(p1, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    Reg::Compares<uint32_t, CMPMODE::GT>(p1, man32Reg, HALF_FOR_MAN, p1);
    Reg::Or(p0, p0, p1, maskAll32);
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
    Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(reciprocalWriteAddr, recip16, VF_LEN_FP32, maskB16);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::ComputeScaleCuBLASSecondLast(uint16_t dataLen, uint32_t localInvDtypeMax,
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
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(mxScale2Addr + DIGIT_TWO * colOffset, scale8Slot0,
                                                                    scale8Slot1, interleaveMask);
        }
    }
}

// ComputeScaleCuBLAS — CUBLAS (scaleAlg=1) scale computation for both axes
// Uses absolute value max + FP32 multiply by 1/dtype_max + mantissa rounding
// Only supported for FP8 output types
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
template <bool packedAxis1>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::ComputeScaleCuBLAS(uint16_t dataLen, uint16_t blockCount,
                                                               __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* y1Addr,
                                                               __ubuf__ uint8_t* mxScale1Addr,
                                                               __ubuf__ uint16_t* mxScale1ReciprocalAddr,
                                                               __ubuf__ uint8_t* mxScale2Addr,
                                                               __ubuf__ uint16_t* mxScale2ReciprocalAddr)
{
    uint32_t localInvDtypeMax = invDtypeMax_;
    __ubuf__ uint16_t* tempMaxAddr = mxScale2ReciprocalAddr;
    __ubuf__ xDtype* xAddrBase = xAddr;
    __ubuf__ uint16_t* reciprocalBase = mxScale1ReciprocalAddr;

    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;
        Reg::RegTensor<uint16_t> x0Abs;
        Reg::RegTensor<uint16_t> x1Abs;
        Reg::RegTensor<uint16_t> absMaxDim1;
        Reg::RegTensor<uint16_t> scale2Slot0Part0;
        Reg::RegTensor<uint16_t> scale2Slot0Part1;
        Reg::RegTensor<uint16_t> scale2Slot1Part0;
        Reg::RegTensor<uint16_t> scale2Slot1Part1;
        Reg::RegTensor<uint32_t> max32;
        Reg::RegTensor<uint32_t> extractExp;
        Reg::RegTensor<uint32_t> halfScale;
        Reg::RegTensor<uint16_t> scale16;
        Reg::RegTensor<uint8_t> scale8Reg;
        Reg::RegTensor<uint8_t> scale8Row;
        Reg::RegTensor<uint16_t> recip16Reg;
        Reg::RegTensor<uint16_t> recip16Row;
        Reg::RegTensor<int8_t> extractIdx;
        Reg::RegTensor<uint16_t> absMask;
        Reg::RegTensor<uint32_t> invMax;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();
        Reg::UnalignReg ureg;

        Reg::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS && !packedAxis1) {
            Reg::Duplicate(scale2Slot0Part0, static_cast<uint16_t>(0));
            Reg::Duplicate(scale2Slot0Part1, static_cast<uint16_t>(0));
            Reg::Duplicate(scale2Slot1Part0, static_cast<uint16_t>(0));
            Reg::Duplicate(scale2Slot1Part1, static_cast<uint16_t>(0));
        }

        // Phase 1: collect scale1 maxima compactly while preserving perf/all's
        // two independent 32-row scale2 slots.
        uint16_t slot0Rows = blockCount < BLOCK_SIZE ? blockCount : BLOCK_SIZE;
        for (uint16_t i = 0; i < slot0Rows; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       dataLen);
            Reg::And(x0Abs, (Reg::RegTensor<uint16_t>&)x0, absMask, maskAll);
            Reg::And(x1Abs, (Reg::RegTensor<uint16_t>&)x1, absMask, maskAll);
            Reg::Max(absMaxDim1, x0Abs, x1Abs, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(tempMaxAddr, absMaxDim1, ureg,
                                                                            dataLen / BLOCK_SIZE);
            if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS && !packedAxis1) {
                Reg::Max(scale2Slot0Part0, scale2Slot0Part0, x0Abs, maskAll);
                Reg::Max(scale2Slot0Part1, scale2Slot0Part1, x1Abs, maskAll);
            }
        }
        for (uint16_t i = slot0Rows; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       dataLen);
            Reg::And(x0Abs, (Reg::RegTensor<uint16_t>&)x0, absMask, maskAll);
            Reg::And(x1Abs, (Reg::RegTensor<uint16_t>&)x1, absMask, maskAll);
            Reg::Max(absMaxDim1, x0Abs, x1Abs, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(tempMaxAddr, absMaxDim1, ureg,
                                                                            dataLen / BLOCK_SIZE);
            if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS && !packedAxis1) {
                Reg::Max(scale2Slot1Part0, scale2Slot1Part0, x0Abs, maskAll);
                Reg::Max(scale2Slot1Part1, scale2Slot1Part1, x1Abs, maskAll);
            }
        }
        Reg::StoreUnAlignPost(tempMaxAddr, ureg, 0);
        if constexpr (packedAxis1) {
            // Packed tails can contain only three virtual rows: maxima stores
            // must finish before phase 2 reads the same UB, regardless of latency.
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }

        Reg::Duplicate(invMax, localInvDtypeMax);

        // Phase 2: convert 64 cached maxima per VF, then scatter eight rows
        // back to the original 32-byte-aligned scale/reciprocal layout.
        __ubuf__ uint16_t* readMaxAddr = mxScale2ReciprocalAddr;
        uint16_t scaleCount = dataLen / BLOCK_SIZE;
        uint16_t batchCount = ops::CeilDiv(static_cast<uint16_t>(blockCount * scaleCount),
                                           static_cast<uint16_t>(VF_LEN_FP32));
        for (uint16_t j = 0; j < batchCount; j++) {
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                absMaxDim1, readMaxAddr, VF_LEN_FP32);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float>&)max32,
                                                          (Reg::RegTensor<xDtype>&)absMaxDim1, maskAll);
            SwigluGroupQuantAxis1::EncodeMxScale<true>(extractExp, halfScale, (Reg::RegTensor<float>&)max32, invMax,
                                                       maskAll32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale16, extractExp);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scale8Reg, scale16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recip16Reg, halfScale);

            if constexpr (packedAxis1) {
                Reg::MaskReg compactMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL64>();
                Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1Addr, scale8Reg, VF_LEN_FP32,
                                                                             compactMask);
            }
            for (uint16_t k = 0; k < 8; k++) {
                if constexpr (!packedAxis1) {
                    Reg::Arange(extractIdx, static_cast<int8_t>(k * 8));
                    Reg::Gather(scale8Row, scale8Reg, (Reg::RegTensor<uint8_t>&)extractIdx);
                    Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1Addr, scale8Row, UB_BLOCK_SIZE,
                                                                                 maskReduceB8);
                }
                Reg::Arange(extractIdx, static_cast<int8_t>(k * 16));
                Reg::Gather((Reg::RegTensor<uint8_t>&)recip16Row, (Reg::RegTensor<uint8_t>&)recip16Reg,
                            (Reg::RegTensor<uint8_t>&)extractIdx);
                Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    mxScale1ReciprocalAddr, recip16Row, SCALE1_RECIPROCAL_ROW_ELEMS, maskReduceB16);
            }
        }

        if constexpr (packedAxis1) {
            // The quantizer consumes reciprocals written by phase 2.
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
        // Generate y1 immediately after scale1 while staying in the same vector scope.
        Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<float> scaleForMulFP32;
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<y1Dtype> x0ZeroFP8;
        Reg::RegTensor<y1Dtype> x0OneFP8;
        Reg::RegTensor<y1Dtype> x1ZeroFP8;
        Reg::RegTensor<y1Dtype> x1OneFP8;
        __ubuf__ xDtype* xReadAddr = xAddrBase;
        __ubuf__ uint16_t* recipReadAddr = reciprocalBase;

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xReadAddr, dataLen);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, recipReadAddr, SCALE1_RECIPROCAL_ROW_ELEMS);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    scaleForMulFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
                Reg::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
                Reg::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
            } else {
                Reg::Mul(x0, x0, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                Reg::Mul(x1, x1, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
            }
            SwigluGroupQuantAxis1::CastQuantized<y1Dtype, 0>(x0ZeroFP8, x0ZeroFP32, maskAll);
            SwigluGroupQuantAxis1::CastQuantized<y1Dtype, 2>(x0OneFP8, x0OneFP32, maskAll);
            SwigluGroupQuantAxis1::CastQuantized<y1Dtype, 1>(x1ZeroFP8, x1ZeroFP32, maskAll);
            SwigluGroupQuantAxis1::CastQuantized<y1Dtype, 3>(x1OneFP8, x1OneFP32, maskAll);
            Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x0OneFP8, maskAllB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x1ZeroFP8, maskAllB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x1OneFP8, maskAllB8);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x0ZeroFP8, dataLen, maskAllB8);
        }

        if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS && !packedAxis1) {
            // Reuse the temporary area for the normal axis=-2 maxima output.
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, scale2Slot0Part0,
                                                                      scale2Slot0Part1, maskAll);
            if (blockCount > BLOCK_SIZE) {
                Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr + dataLen,
                                                                          scale2Slot1Part0, scale2Slot1Part1, maskAll);
            }
        }
    }
    if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS && !packedAxis1) {
        ComputeScaleCuBLASSecondLast(dataLen, localInvDtypeMax, mxScale2ReciprocalAddr, mxScale2Addr);
    }
}
// Axis 2 is appended after the identical packed activation/axis-1 computation.
// Its 32-row groups stay in the real [64,384] layout, never the virtual [96,256] layout.
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::ComputePackedSecondAxis(__ubuf__ xDtype* activation,
                                                                    __ubuf__ uint16_t* reciprocal,
                                                                    __ubuf__ uint8_t* scale, __ubuf__ uint8_t* output)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> even;
        Reg::RegTensor<xDtype> odd;
        Reg::RegTensor<uint16_t> evenAbs;
        Reg::RegTensor<uint16_t> oddAbs;
        Reg::RegTensor<uint16_t> evenMax;
        Reg::RegTensor<uint16_t> oddMax;
        Reg::RegTensor<uint16_t> absMask;
        Reg::MaskReg mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        for (uint16_t slot = 0; slot < static_cast<uint16_t>(DIGIT_TWO); ++slot) {
            for (uint16_t part = 0; part < static_cast<uint16_t>(DIGIT_TWO); ++part) {
                Reg::Duplicate(evenMax, static_cast<uint16_t>(0));
                Reg::Duplicate(oddMax, static_cast<uint16_t>(0));
                for (uint16_t row = 0; row < static_cast<uint16_t>(BLOCK_SIZE); ++row) {
                    __ubuf__ xDtype* cursor = activation + (slot * BLOCK_SIZE + row) * PACKED_COLS +
                                              part * ONCE_ROW_LEN;
                    Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                        even, odd, cursor, ONCE_ROW_LEN);
                    Reg::And(evenAbs, (Reg::RegTensor<uint16_t>&)even, absMask, mask);
                    Reg::And(oddAbs, (Reg::RegTensor<uint16_t>&)odd, absMask, mask);
                    Reg::Max(evenMax, evenMax, evenAbs, mask);
                    Reg::Max(oddMax, oddMax, oddAbs, mask);
                }
                // The last vector's unused half overlaps the next slot, which is
                // produced later, or the allocated suffix after the final slot.
                Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(
                    reciprocal + slot * PACKED_COLS + part * ONCE_ROW_LEN, evenMax, oddMax, mask);
            }
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
    ComputeScaleCuBLASSecondLast(PACKED_COLS, invDtypeMax_, reciprocal, scale);
    for (uint16_t slot = 0; slot < static_cast<uint16_t>(DIGIT_TWO); ++slot) {
        const uint32_t rowOffset = slot * BLOCK_SIZE * PACKED_COLS;
        ComputeY2ToFP8<256>(256, BLOCK_SIZE, activation + rowOffset, reciprocal + slot * PACKED_COLS,
                            output + rowOffset);
        ComputeY2ToFP8<128>(128, BLOCK_SIZE, activation + rowOffset + ONCE_ROW_LEN,
                            reciprocal + slot * PACKED_COLS + ONCE_ROW_LEN, output + rowOffset + ONCE_ROW_LEN);
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
template <uint16_t validCols>
__aicore__ inline void
SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs, hasClamp,
                         SecondAxisPolicy>::ComputeY2ToFP8(uint16_t dataLen, uint16_t blockCount,
                                                           __ubuf__ xDtype* xAddr,
                                                           __ubuf__ uint16_t* mxScale2ReciprocalAddr,
                                                           __ubuf__ uint8_t* y2Addr)
{
    int64_t localUbRowLen = ubRowLen_;
    constexpr uint32_t dualLoadLen = VF_LEN_B16 * DIGIT_TWO;

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

        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        if constexpr (validCols == 128) {
            maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL128>();
        }
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        // Load all 256 reciprocal values once and reuse them for all 32 rows in this scale2 slot.
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
            reciprocalEven, reciprocalOdd, mxScale2ReciprocalAddr, dualLoadLen);
        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reciprocalEvenFP32Layout0,
                                                              (Reg::RegTensor<bfloat16_t>&)reciprocalEven, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalEvenFP32Layout1,
                                                             (Reg::RegTensor<bfloat16_t>&)reciprocalEven, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reciprocalOddFP32Layout0,
                                                              (Reg::RegTensor<bfloat16_t>&)reciprocalOdd, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalOddFP32Layout1,
                                                             (Reg::RegTensor<bfloat16_t>&)reciprocalOdd, maskB16);
        }

        for (uint16_t row = 0; row < blockCount; row++) {
            __ubuf__ xDtype* xCursor = xAddr + row * localUbRowLen;
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xEven, xOdd, xCursor, dualLoadLen);
            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, maskB16);

                Reg::Mul(xEvenFP32Layout0, xEvenFP32Layout0, reciprocalEvenFP32Layout0, maskB32);
                Reg::Mul(xEvenFP32Layout1, xEvenFP32Layout1, reciprocalEvenFP32Layout1, maskB32);
                Reg::Mul(xOddFP32Layout0, xOddFP32Layout0, reciprocalOddFP32Layout0, maskB32);
                Reg::Mul(xOddFP32Layout1, xOddFP32Layout1, reciprocalOddFP32Layout1, maskB32);
            } else {
                Reg::Mul(xEven, xEven, (Reg::RegTensor<xDtype>&)reciprocalEven, maskB16);
                Reg::Mul(xOdd, xOdd, (Reg::RegTensor<xDtype>&)reciprocalOdd, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, maskB16);
            }

            Reg::Cast<y1Dtype, float, CAST_32_TO_80>(yEvenFP8Layout0, xEvenFP32Layout0, maskB32);
            Reg::Cast<y1Dtype, float, CAST_32_TO_82>(yEvenFP8Layout2, xEvenFP32Layout1, maskB32);
            Reg::Cast<y1Dtype, float, CAST_32_TO_81>(yOddFP8Layout1, xOddFP32Layout0, maskB32);
            Reg::Cast<y1Dtype, float, CAST_32_TO_83>(yOddFP8Layout3, xOddFP32Layout1, maskB32);

            Reg::Add((Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0,
                     (Reg::RegTensor<uint8_t>&)yEvenFP8Layout2, maskB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)yOddFP8Layout1, (Reg::RegTensor<uint8_t>&)yOddFP8Layout1,
                     (Reg::RegTensor<uint8_t>&)yOddFP8Layout3, maskB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0,
                     (Reg::RegTensor<uint8_t>&)yOddFP8Layout1, maskB8);

            __ubuf__ uint8_t* yCursor = y2Addr + row * localUbRowLen;
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                yCursor, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, dualLoadLen, maskB8);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx, bool hasAttrs, bool hasClamp, typename SecondAxisPolicy>
__aicore__ inline void SwigluGroupQuantMxKernel<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx, hasAttrs,
                                                hasClamp, SecondAxisPolicy>::CopyOut(int64_t yOffset,
                                                                                     int64_t scale1OutOffset,
                                                                                     int64_t scale2OutOffset,
                                                                                     int64_t blockCount,
                                                                                     int64_t blockCountAlign,
                                                                                     int64_t dataLen,
                                                                                     int64_t dataLenAlign)
{
    uint16_t outBurst = static_cast<uint16_t>(blockCount);
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    int64_t dstStride = 0; // GM strides are byte counts and must retain all 64 bits.

    outBlockLen = static_cast<uint32_t>(dataLen * sizeof(uint8_t));
    srcStride = static_cast<uint32_t>((ubRowLen_ - dataLen) * sizeof(y1Dtype) / UB_BLOCK_SIZE);
    dstStride = (dimN_ - dataLen) * static_cast<int64_t>(sizeof(uint8_t));

    DataCopyExtParams yCopyOutParams = {outBurst, outBlockLen, srcStride, dstStride, 0};
    if (packed384_) {
        yCopyOutParams.blockCount = 1;
        yCopyOutParams.blockLen = blockCount * dataLen;
        yCopyOutParams.dstStride = 0;
    }

    // axis=-1 scale output: shape [M, ceil(N/blockSize)]
    uint32_t scale1OutLen = dataLenAlign / BLOCK_SIZE;

    DataCopyExtParams scale1CopyOutParams = {
        outBurst, static_cast<uint32_t>(scale1OutLen * sizeof(uint8_t)), static_cast<uint32_t>(0),
        ops::CeilAlign(dimN_, DOUBLE_BLOCK_SIZE) / BLOCK_SIZE - scale1OutLen, static_cast<uint32_t>(0)};

    if (packed384_) {
        // Packed scale batches are contiguous, matching the original [M, 12] tensor.
        scale1CopyOutParams.blockCount = 1;
        scale1CopyOutParams.blockLen = blockCount * PACKED_COLS / BLOCK_SIZE;
        scale1CopyOutParams.dstStride = 0;
    }

    LocalTensor<uint8_t> y1Local = outQueue1_.template DeQue<uint8_t>();
    DataCopyPad(yGm1_[yOffset], y1Local, yCopyOutParams);
    outQueue1_.FreeTensor(y1Local);

    LocalTensor<uint8_t> mxScale1Local = mxScaleQueue1_.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm1_[scale1OutOffset], mxScale1Local, scale1CopyOutParams);
    mxScaleQueue1_.FreeTensor(mxScale1Local);

    if constexpr (SecondAxisPolicy::ENABLE_SECOND_AXIS) {
        LocalTensor<uint8_t> y2Local = outQueue2_.template DeQue<uint8_t>();
        DataCopyPad(yGm2_[yOffset], y2Local, yCopyOutParams);
        outQueue2_.FreeTensor(y2Local);

        uint32_t scaleSrcStride = DIGIT_TWO * ops::CeilDiv(dataLen, UB_BLOCK_SIZE) -
                                  ops::CeilDiv(DIGIT_TWO * dataLen, UB_BLOCK_SIZE);
        DataCopyExtParams scale2CopyOutParams = {
            static_cast<uint16_t>(blockCountAlign / DOUBLE_BLOCK_SIZE),
            static_cast<uint32_t>(dataLen * DIGIT_TWO * sizeof(uint8_t)), static_cast<uint32_t>(scaleSrcStride),
            DIGIT_TWO * (dimN_ - dataLen) * static_cast<int64_t>(sizeof(uint8_t)), static_cast<uint32_t>(0)};
        LocalTensor<uint8_t> mxScale2Local = mxScaleQueue2_.template DeQue<uint8_t>();
        DataCopyPad(mxScaleGm2_[scale2OutOffset], mxScale2Local, scale2CopyOutParams);
        mxScaleQueue2_.FreeTensor(mxScale2Local);
    }

    if (outputOrigin_ && !shareOrigin_) {
        DataCopyExtParams originCopyOutParams = {
            outBurst, static_cast<uint32_t>(dataLen * sizeof(xDtype)),
            static_cast<uint32_t>((ubRowLen_ - dataLen) * sizeof(xDtype) / UB_BLOCK_SIZE),
            (dimN_ - dataLen) * static_cast<int64_t>(sizeof(xDtype)), 0};
        LocalTensor<xDtype> originLocal = originQueue_.template DeQue<xDtype>();
        DataCopyPad(yOriginGm_[yOffset], originLocal, originCopyOutParams);
        originQueue_.FreeTensor(originLocal);
    }
}
} // namespace SwigluGroupQuantMx

#endif // OPS_NN_SWIGLU_GROUP_QUANT_MX_KERNEL_H
