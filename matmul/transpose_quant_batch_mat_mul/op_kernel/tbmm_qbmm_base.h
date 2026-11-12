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
 * \file tbmm_qbmm_base.h
 * \brief
 */
#ifndef TBMM_QBMM_BASE_H
#define TBMM_QBMM_BASE_H

#include "../inc/macro.h"
#include <cstdint>
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "quantization/ascend_dequant_utils.h"
#include "pad/broadcast.h"
#include "quantization/ascend_dequant.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "kernel_type.h"
#include "lib/matmul_intf.h"
#include "transpose_batch_mat_mul_tiling_data.h"

#define TemplateBasicTypeForClass                                                                                  \
    typename x1Type, typename x2Type, typename scaleType, typename yType, int x1Format, int x2Format, bool aTrans, \
        bool bTrans, class UPDATE_TYPE, class EpilogueType = EpilogueDequant
#define TemplateBasicType                                                                                          \
    typename x1Type, typename x2Type, typename scaleType, typename yType, int x1Format, int x2Format, bool aTrans, \
        bool bTrans, class UPDATE_TYPE, class EpilogueType
#define TemplateBasicValue \
    x1Type, x2Type, scaleType, yType, x1Format, x2Format, aTrans, bTrans, UPDATE_TYPE, EpilogueType

constexpr uint32_t BMM_BLOCK_NUM = 16;
constexpr uint32_t K0_INT4 = 64;
constexpr uint32_t K0_INT8 = 32;
constexpr uint32_t k0_FLOAT16 = 16;
constexpr uint32_t k0_FLOAT32 = 8;
constexpr int FORMAT_FRACTAL_NZ_INT = 29;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t M_N_TWO_DIMS = 2;

const uint32_t ROW_FIRST = 1;
const uint32_t COL_FIRST = 2;
constexpr uint16_t C2V_PING_FLAG = 0x4;
constexpr uint16_t C2V_PONG_FLAG = 0x5;
constexpr uint16_t V2C_PING_FLAG = 0x6;
// asw const
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint8_t AIC_SYNC_AIV_MODE = 4;
constexpr uint8_t CV_RATIO = 2;
constexpr int32_t MXFP_GROUP_SIZE = 32;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int32_t ONE_BLOCK_SIZE = 32;
constexpr uint16_t DATA_BLOCK = 32;
const uint32_t FP32_OUTPUT_TIMES = 4;
const uint64_t INT1_X2_OFFSET_FACTOR_8 = 8;

// iterBatch const
constexpr uint64_t L0C_SIZE_256K = 256 * 1024UL;
constexpr uint8_t A_NEED_BROADCAST = 1;
constexpr uint8_t B_NEED_BROADCAST = 2;
constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
constexpr MatmulBatchParams batchParams{false, BatchMode::BATCH_LESS_THAN_L1, false, BatchOutMode::MULTI_BATCH};
constexpr MatmulConfig MM_CFG_MULTI_BATCH = GetMMConfig<configMode>(batchParams);
constexpr MatmulBatchParams batchParamsNoBatchOut{false, BatchMode::BATCH_LESS_THAN_L1, false,
                                                  BatchOutMode::SINGLE_BATCH};
constexpr MatmulConfig MM_CFG_MULTI_BATCH_NO_BATCH_OUT = GetMMConfig<configMode>(batchParamsNoBatchOut);

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
constexpr MatmulConfig MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG = GetMDLConfig(false, false, 0, false, false, false, true, true,
                                                                       false, false, false, true);
constexpr MatmulConfig MM_DEFAULT_MDL_CFG = GetMDLConfig(true, false, 0, false, false, false, true, true, true, false,
                                                         false, true);
#else
constexpr MatmulConfig MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG = GetMDLConfig(false, false, 0, false, false, false, true, true,
                                                                       false, false, false);
constexpr MatmulConfig MM_DEFAULT_MDL_CFG = GetMDLConfig();
#endif

// Keep ordinary MDL/K scheduling and all other baseline options. Turning off
// unit_flag selects the SDK's queue-backed CubeOutBuffer implementation.
constexpr MatmulConfig TBMM_TRANSB_MDL_CFG = []() {
    MatmulConfig cfg = MM_DEFAULT_MDL_CFG;
    cfg.enUnitFlag = false;
    return cfg;
}();

struct L2CacheParam {
    uint32_t l2MCnt;
    uint32_t l2NCnt;
    uint32_t l2MCntTail;
    uint32_t l2NCntTail;
    uint32_t l2TotalTileCnt;
    uint32_t l2MCntUse;
    uint32_t l2NCntUse;
};

// 量化mm和mc2都用的输入输出地址结构体，不可随意修改
struct QBmmBlockOffset {
    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetScale = 0;
    uint64_t offsetBias = 0;
    uint64_t offsetPertoken = 0;
    uint64_t offsetC = 0;
};

// 量化mm和mc2都用的block输入信息结构体，不可随意修改
struct QBmmBaseBlockArgs {
    uint64_t index;
    uint64_t totalTileCnt;
    uint64_t singleCoreM; // 当前基本块计算大小
    uint64_t singleCoreN;
    uint64_t mTileCntL2;
    uint64_t nTileCntL2;
    uint64_t mTotalCnt;
    uint64_t nTotalCnt;
    uint64_t mCntUse;
    uint64_t nCntUse;
    uint64_t mTileAddrOffset;
    uint64_t nTileAddrOffset;
};

enum class FusedOpType : uint32_t { NONE = 0U, RELU = 1U, SWIGLU = 2U, GELU_TANH = 3UL, GELU_ERF = 4UL };

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
template <typename yType, typename scaleType>
struct EpilogueParams {
    AscendC::GlobalTensor<int32_t>& curMmOutGm;
    AscendC::GlobalTensor<yType>& yGm;
    AscendC::GlobalTensor<bfloat16_t>& biasGmBf16;
    AscendC::GlobalTensor<half>& biasGmFp16;
    AscendC::GlobalTensor<float>& biasGmFp32;
    AscendC::GlobalTensor<scaleType>& scaleGm;
    AscendC::GlobalTensor<float>& pertokenScaleGm;

    AscendC::TBuf<AscendC::TPosition::VECCALC>& outFp32Tmp;
    AscendC::TBuf<AscendC::TPosition::VECCALC>& vecQueTmp;
    AscendC::TBuf<AscendC::TPosition::VECCALC>& biasFp32Tmp;
    AscendC::TBuf<AscendC::TPosition::VECCALC>& broadcastFp32Tmp;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1>& vecQueSrc;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1>& vecQueOut;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1>& vecQueBias;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1>& vecQueScale;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1>& vecQuePertokenScale;

    uint32_t curAicM;
    uint32_t curAicN;
    uint32_t subBlockIdx;
    uint32_t ubCalcM;
    uint32_t ubCalcN;
    uint64_t offsetWorkspaceC;
    uint32_t biasDtype;
    uint32_t biasDtypeSize;
    bool isPerTensor;
    scaleType scaleScalar;
    QBmmBlockOffset offset;
    uint32_t n;
    uint32_t batch; // TBMM delta: token_scale[M,B] 的 stride=B, Y[M,B,N] 的 M 维 stride=B*N

    __aicore__ EpilogueParams(AscendC::GlobalTensor<int32_t>& curMmOutGm, AscendC::GlobalTensor<yType>& yGm,
                              AscendC::GlobalTensor<bfloat16_t>& biasGmBf16, AscendC::GlobalTensor<half>& biasGmFp16,
                              AscendC::GlobalTensor<float>& biasGmFp32, AscendC::GlobalTensor<scaleType>& scaleGm,
                              AscendC::GlobalTensor<float>& pertokenScaleGm,

                              AscendC::TBuf<AscendC::TPosition::VECCALC>& outFp32Tmp,
                              AscendC::TBuf<AscendC::TPosition::VECCALC>& vecQueTmp,
                              AscendC::TBuf<AscendC::TPosition::VECCALC>& biasFp32Tmp,
                              AscendC::TBuf<AscendC::TPosition::VECCALC>& broadcastFp32Tmp,

                              AscendC::TQue<AscendC::QuePosition::VECIN, 1>& vecQueSrc,
                              AscendC::TQue<AscendC::QuePosition::VECOUT, 1>& vecQueOut,
                              AscendC::TQue<AscendC::QuePosition::VECIN, 1>& vecQueBias,
                              AscendC::TQue<AscendC::QuePosition::VECIN, 1>& vecQueScale,
                              AscendC::TQue<AscendC::QuePosition::VECIN, 1>& vecQuePertokenScale,

                              uint32_t curAicM, uint32_t curAicN, uint32_t subBlockIdx, uint32_t ubCalcM,
                              uint32_t ubCalcN, uint64_t offsetWorkspaceC, uint32_t biasDtype, uint32_t biasDtypeSize,
                              bool isPerTensor, scaleType scaleScalar, QBmmBlockOffset offset, uint32_t n,
                              uint32_t batch)
        : curMmOutGm(curMmOutGm),
          yGm(yGm),
          biasGmBf16(biasGmBf16),
          biasGmFp16(biasGmFp16),
          biasGmFp32(biasGmFp32),
          scaleGm(scaleGm),
          pertokenScaleGm(pertokenScaleGm),
          outFp32Tmp(outFp32Tmp),
          vecQueTmp(vecQueTmp),
          biasFp32Tmp(biasFp32Tmp),
          broadcastFp32Tmp(broadcastFp32Tmp),
          vecQueSrc(vecQueSrc),
          vecQueOut(vecQueOut),
          vecQueBias(vecQueBias),
          vecQueScale(vecQueScale),
          vecQuePertokenScale(vecQuePertokenScale),
          curAicM(curAicM),
          curAicN(curAicN),
          subBlockIdx(subBlockIdx),
          ubCalcM(ubCalcM),
          ubCalcN(ubCalcN),
          offsetWorkspaceC(offsetWorkspaceC),
          biasDtype(biasDtype),
          biasDtypeSize(biasDtypeSize),
          isPerTensor(isPerTensor),
          scaleScalar(scaleScalar),
          offset(offset),
          n(n),
          batch(batch)
    {}
};
#endif

enum class BasicQuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
    PERGROUP_MODE = 0x1U << 5,
};

namespace DequantBmm {
template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b = 16) { return (a + b - 1) / b * b; }

__aicore__ inline uint64_t FloorAlign(uint64_t a, uint64_t b = 16)
{
    if (b == 0) {
        return a;
    }
    return a / b * b;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t Align16(uint64_t num) { return (num + 15UL) & 0xFFFFFFFFFFFFFFF0UL; }

__aicore__ inline uint64_t Align32(uint64_t num) { return (num + 31UL) & 0xFFFFFFFFFFFFFFE0UL; }

__aicore__ inline uint64_t Align64(uint64_t num) { return (num + 63UL) & 0xFFFFFFFFFFFFFFC0UL; }

__aicore__ inline constexpr CubeFormat GetFormat(int format)
{
    if (format == FORMAT_FRACTAL_NZ_INT) {
        return CubeFormat::NZ;
    }
    return CubeFormat::ND;
}

template <typename T, bool isLut = false>
__aicore__ inline constexpr uint32_t GetC0Size()
{
    // lut查表逻辑: 原始数据DT_INT2和DT_UINT1，查表后转DT_INT4; 原始数据DT_INT4, 查表后转DT_INT8
    if constexpr (isLut && AscendC::IsSameType<T, AscendC::int4b_t>::value) {
        return K0_INT8;
#if __LUT_SUPPORT__
    } else if constexpr (isLut && (AscendC::IsSameType<T, AscendC::int2b_t>::value ||
                                   AscendC::IsSameType<T, AscendC::uint1b_t>::value)) {
        return K0_INT4;
#endif
    } else if constexpr (AscendC::IsSameType<T, AscendC::int4b_t>::value) {
        return K0_INT4;
#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)) || \
    __CUBE_S8S4_S4S4__
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return K0_INT4;
#endif
    } else if constexpr (sizeof(T) == sizeof(float)) {
        return k0_FLOAT32;
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        return K0_INT8;
    } else {
        return k0_FLOAT16;
    }
}

template <typename T, bool isLut = false>
__aicore__ inline constexpr uint64_t GetSizeWithDataType(uint64_t shapeSize)
{
    bool is4BitInput = false;
    bool is8BitInput = false;
    if constexpr (isLut) {
        // lut查表逻辑: 原始数据DT_INT2和DT_UINT1，查表后转DT_INT4; 原始数据DT_INT4, 查表后转DT_INT8
#if __LUT_SUPPORT__
        is4BitInput = (AscendC::IsSameType<T, AscendC::int2b_t>::value ||
                       AscendC::IsSameType<T, AscendC::uint1b_t>::value);
#endif
        is8BitInput = (AscendC::IsSameType<T, AscendC::int4b_t>::value);
    } else {
#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)) || \
    __CUBE_S8S4_S4S4__
        is4BitInput = (AscendC::IsSameType<T, AscendC::int4b_t>::value || AscendC::IsSameType<T, fp4x2_e2m1_t>::value);
#else
        is4BitInput = (AscendC::IsSameType<T, AscendC::int4b_t>::value);
#endif
    }
    if (is4BitInput) {
        return shapeSize / 2;
    } else if (is8BitInput) {
        return shapeSize;
    } else {
        return shapeSize * sizeof(T);
    }
}

template <typename T, bool trans>
__aicore__ inline uint64_t CalcAL1Size(uint64_t mL1, uint64_t kL1)
{
    uint64_t innerAlignedBlock = ONE_BLOCK_SIZE / sizeof(T);
    uint64_t mAligned = 0;
    uint64_t kAligned = 0;
    if constexpr (trans) {
        mAligned = Align(mL1, innerAlignedBlock);
        kAligned = Align(kL1, BMM_BLOCK_NUM);
    } else {
        mAligned = Align(mL1, BMM_BLOCK_NUM);
        kAligned = Align(kL1, innerAlignedBlock);
    }
    return mAligned * kAligned * sizeof(T);
}

/**
 * Get the aiv corenum in different platforms
 */
__aicore__ inline constexpr uint32_t GetTaskRation()
{
#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113))
    return 2U; // aiv corenum is 2 in C310 platform
#else
    return 1U;
#endif
}

#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)) || \
    __CUBE_S8S4_S4S4__
template <typename T>
__aicore__ inline constexpr bool IsMxType()
{
    return AscendC::IsSameType<T, AscendC::fp8_e8m0_t>::value;
}

template <typename T>
__aicore__ inline constexpr bool IsFp4()
{
    return AscendC::IsSameType<T, fp4x2_e2m1_t>::value;
}
#endif

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
__aicore__ inline void CalcDequantParams(uint32_t curAivM, uint32_t curAivN, AscendC::DequantParams& dequantParams,
                                         bool needUpdate = true)
{
    if (!needUpdate) {
        return;
    }
    uint32_t computedAivN = Align(curAivN, 8U); // 8: 32B aligned for int32_t
    uint32_t ubResAlignedN = Align(curAivN);    // 16: sizeof(yType) is 2, 32B / 2
    if (computedAivN == ubResAlignedN) {
        // choose ddequat high performance
        dequantParams.m = 1;
        dequantParams.n = curAivM * computedAivN;
        dequantParams.calCount = computedAivN;
    } else {
        // general
        dequantParams.m = curAivM;
        dequantParams.n = computedAivN;
        dequantParams.calCount = curAivN;
    }
}

__aicore__ inline void SetGm2UbParams(AscendC::DataCopyParams& gm2UbParams, uint32_t curAivM, uint32_t curAivN)
{
    gm2UbParams.blockLen = curAivN * sizeof(int32_t);
    gm2UbParams.blockCount = curAivM;
    gm2UbParams.srcStride = 0;
}

template <typename yType>
__aicore__ inline void SetUb2GmParams(AscendC::DataCopyExtParams& ub2GmParams, uint32_t curAivM, uint32_t curAivN,
                                      uint32_t n)
{
    ub2GmParams.blockLen = curAivN * sizeof(yType);
    ub2GmParams.blockCount = curAivM;
    ub2GmParams.dstStride = (n - curAivN) * sizeof(yType);
}

__aicore__ inline void CopyMmOutToLocal(AscendC::LocalTensor<int32_t>& srcLocal,
                                        AscendC::GlobalTensor<int32_t>& curMmOutGm,
                                        AscendC::DataCopyParams& gm2UbParams, AscendC::DataCopyPadParams& padParams,
                                        uint32_t curAicAivOffset)
{
    DataCopyPad(srcLocal, curMmOutGm[curAicAivOffset], gm2UbParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
}

template <typename yType>
__aicore__ inline void CopyUbToGm(uint64_t yGmOffset, AscendC::DataCopyExtParams& ub2GmParams,
                                  AscendC::LocalTensor<yType>& dstLocal, AscendC::GlobalTensor<yType>& yGm,
                                  AscendC::TQue<AscendC::QuePosition::VECOUT, 1>& vecQueOut)
{
    DataCopyPad(yGm[yGmOffset], dstLocal, ub2GmParams);
    vecQueOut.FreeTensor(dstLocal);
}

template <typename scaleType>
__aicore__ inline void Bf16ScaleGm2Ub(AscendC::LocalTensor<scaleType>& scaleLocal,
                                      AscendC::GlobalTensor<scaleType>& scaleGm, AscendC::DataCopyPadParams& padParams,
                                      uint32_t curAivN, uint64_t offsetScale)
{
    AscendC::DataCopyParams scale2UbParams{1, 0, 0, 0};
    scale2UbParams.blockLen = curAivN * sizeof(scaleType);
    DataCopyPad(scaleLocal, scaleGm[offsetScale], scale2UbParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
}
#endif
} // namespace DequantBmm

namespace AscendC {
class QuantBatchMatmulV3BaseBlock {
public:
    __aicore__ inline QuantBatchMatmulV3BaseBlock() {}
    __aicore__ inline void Init(const QuantBatchMatmulV3TilingData* tilingData);
    __aicore__ inline void Init(const L2cacheTileParam* tilingL2, const TCubeTiling* matmulTilingData);
    __aicore__ inline void UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void InitFirstTileBlockIndex();
    __aicore__ inline void InitBlockIndex();
    __aicore__ inline void UpdateBlockIndex();

public:
    QBmmBaseBlockArgs params_;
    bool indexInit_ = false;
    uint32_t rowOrder_;
    uint64_t mCnt_;
    uint64_t nCnt_;
    uint64_t mCntTail_;
    uint64_t nCntTail_;
    uint64_t round_;
    uint64_t realRound_;
    uint64_t preCoreNum_;
    uint64_t blockIdxStart_;
    uint64_t blockIdxEnd_;
    uint32_t blockIdx_;
    const TCubeTiling* matmulTilingData_;
};

__aicore__ inline void QuantBatchMatmulV3BaseBlock::Init(const QuantBatchMatmulV3TilingData* tilingData)
{
    this->Init(&(tilingData->tileL2cacheTiling), &(tilingData->matmulTiling));
}

__aicore__ inline void QuantBatchMatmulV3BaseBlock::Init(const L2cacheTileParam* tilingL2,
                                                         const TCubeTiling* matmulTilingData)
{
    blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
    if ASCEND_IS_AIV {
        blockIdx_ = blockIdx_ / GetTaskRation();
    }
    matmulTilingData_ = matmulTilingData;
    params_.mTileCntL2 = static_cast<uint64_t>(tilingL2->mTileCntL2);
    params_.nTileCntL2 = static_cast<uint64_t>(tilingL2->nTileCntL2);
    params_.mTotalCnt = (static_cast<uint64_t>(matmulTilingData_->M) + matmulTilingData_->singleCoreM - 1) /
                        matmulTilingData_->singleCoreM;
    params_.nTotalCnt = (static_cast<uint64_t>(matmulTilingData_->N) + matmulTilingData_->singleCoreN - 1) /
                        matmulTilingData_->singleCoreN;
    mCnt_ = tilingL2->mTileBlock;
    nCnt_ = tilingL2->nTileBlock;
    params_.totalTileCnt = mCnt_ * nCnt_;

    mCntTail_ = params_.mTotalCnt - (params_.mTileCntL2 - 1) * mCnt_;
    nCntTail_ = params_.nTotalCnt - (params_.nTileCntL2 - 1) * nCnt_;
    round_ = (params_.totalTileCnt + matmulTilingData_->usedCoreNum - 1) / matmulTilingData_->usedCoreNum;

    preCoreNum_ = params_.totalTileCnt % matmulTilingData_->usedCoreNum;
    params_.mCntUse = mCnt_;
    params_.nCntUse = nCnt_;
    rowOrder_ = tilingL2->calOrder > 0 ? tilingL2->calOrder : ROW_FIRST;
}

__aicore__ inline void QuantBatchMatmulV3BaseBlock::UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex)
{
    params_.mTileAddrOffset = mTileIndex * mCnt_ * matmulTilingData_->singleCoreM;
    params_.nTileAddrOffset = nTileIndex * nCnt_ * matmulTilingData_->singleCoreN;

    if ((mTileIndex == (params_.mTileCntL2 - 1)) && (nTileIndex == (params_.nTileCntL2 - 1))) {
        params_.totalTileCnt = mCntTail_ * nCntTail_;
        params_.mCntUse = mCntTail_;
        params_.nCntUse = nCntTail_;
    } else if (mTileIndex == (params_.mTileCntL2 - 1)) {
        params_.totalTileCnt = mCntTail_ * nCnt_;
        params_.mCntUse = mCntTail_;
        params_.nCntUse = nCnt_;
    } else if (nTileIndex == (params_.nTileCntL2 - 1)) {
        params_.totalTileCnt = mCnt_ * nCntTail_;
        params_.mCntUse = mCnt_;
        params_.nCntUse = nCntTail_;
    } else {
        params_.totalTileCnt = mCnt_ * nCnt_;
        params_.mCntUse = mCnt_;
        params_.nCntUse = nCnt_;
    }

    round_ = DequantBmm::CeilDiv(params_.totalTileCnt, static_cast<uint64_t>(matmulTilingData_->usedCoreNum));
    preCoreNum_ = params_.totalTileCnt - (round_ - 1) * matmulTilingData_->usedCoreNum;
    if (preCoreNum_ == 0) {
        preCoreNum_ = static_cast<uint64_t>(matmulTilingData_->usedCoreNum);
    }
}

__aicore__ inline void QuantBatchMatmulV3BaseBlock::InitFirstTileBlockIndex()
{
    params_.mTileAddrOffset = 0;
    params_.nTileAddrOffset = 0;
    indexInit_ = true;
    blockIdxEnd_ = preCoreNum_;
    if (preCoreNum_ == 0) {
        preCoreNum_ = matmulTilingData_->usedCoreNum;
    }
    uint64_t preTotalBlock = 0U;
    if (blockIdx_ < preCoreNum_) {
        if (rowOrder_ == ROW_FIRST) {
            params_.index = blockIdx_ * round_;
        } else {
            preTotalBlock = blockIdx_ * round_;
            params_.index = preTotalBlock / mCnt_ + (preTotalBlock % mCnt_) * nCnt_;
        }
        realRound_ = round_;
    } else {
        if (rowOrder_ == ROW_FIRST) {
            params_.index = blockIdx_ * (round_ - 1) + preCoreNum_;
        } else {
            preTotalBlock = blockIdx_ * (round_ - 1) + preCoreNum_;
            params_.index = preTotalBlock / mCnt_ + (preTotalBlock % mCnt_) * nCnt_;
        }
        realRound_ = round_ - 1;
    }
}

__aicore__ inline void QuantBatchMatmulV3BaseBlock::InitBlockIndex()
{
    if (indexInit_) {
        blockIdxStart_ = blockIdxEnd_;
    } else {
        blockIdxStart_ = 0;
        indexInit_ = true;
    }
    blockIdxEnd_ = (blockIdxStart_ + preCoreNum_) % matmulTilingData_->usedCoreNum;
    uint64_t indexStart = blockIdxStart_;
    uint64_t indexEnd = blockIdxEnd_;

    if (indexStart < indexEnd) {
        if (blockIdx_ < indexStart) {
            params_.index = blockIdx_ * (round_ - 1);
            realRound_ = round_ - 1;
        } else if (blockIdx_ < indexEnd) {
            params_.index = indexStart * (round_ - 1) + (blockIdx_ - indexStart) * round_;
            realRound_ = round_;
        } else {
            params_.index = (indexStart * (round_ - 1) + preCoreNum_ * round_ + (blockIdx_ - indexEnd) * (round_ - 1));
            realRound_ = round_ - 1;
        }
        if (rowOrder_ == COL_FIRST) {
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    } else if (indexEnd < indexStart) {
        if (blockIdx_ < indexEnd) {
            params_.index = blockIdx_ * round_;
            realRound_ = round_;
        } else if (blockIdx_ < indexStart) {
            params_.index = indexEnd * round_ + (blockIdx_ - indexEnd) * (round_ - 1);
            realRound_ = round_ - 1;
        } else {
            params_.index = (indexEnd * round_ + (indexStart - indexEnd) * (round_ - 1) +
                             (blockIdx_ - indexStart) * round_);
            realRound_ = round_;
        }
        if (rowOrder_ == COL_FIRST) {
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    } else {
        params_.index = blockIdx_ * round_;
        realRound_ = round_;
        if (rowOrder_ == COL_FIRST) {
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    }
}

__aicore__ inline void QuantBatchMatmulV3BaseBlock::UpdateBlockIndex()
{
    if (rowOrder_ == ROW_FIRST) {
        params_.index += 1;
    } else if (rowOrder_ == COL_FIRST) {
        params_.index += params_.nCntUse;
        if (params_.index >= params_.totalTileCnt) {
            params_.index = params_.index % params_.totalTileCnt + 1;
        }
    }
}

struct QBmmUpdateInfo {
    uint64_t mBaseTail;
    uint64_t nBaseTail;
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;
};

class QuantBatchMatmulV3Update {
public:
    __aicore__ inline QuantBatchMatmulV3Update() {}
    template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type = int8_t, typename x2Type = int8_t>
    __aicore__ inline void Init(const TCubeTiling* mmTiling, const QBmmBaseBlockArgs& params);
    template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type = int8_t, typename x2Type = int8_t>
    __aicore__ inline void UpdateBlockParamsAndCalcGmOffset(QBmmBaseBlockArgs& params, QBmmBlockOffset& offset,
                                                            uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void UpdateBlockParams(QBmmBaseBlockArgs& params, uint64_t mTileIndex, uint64_t nTileIndex);
    template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type = int8_t, typename x2Type = int8_t>
    __aicore__ inline void CalcGMOffset(QBmmBaseBlockArgs& params, QBmmBlockOffset& offset);
    __aicore__ inline void SetBatch(uint32_t batch) { batch_ = batch; }

private:
    QBmmUpdateInfo info_;
    const TCubeTiling* mmTiling_;
    uint32_t batch_ = 1;
};

template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type, typename x2Type>
__aicore__ inline void QuantBatchMatmulV3Update::Init(const TCubeTiling* mmTiling, const QBmmBaseBlockArgs& params)
{
    mmTiling_ = mmTiling;
    info_.nBaseTail = static_cast<uint64_t>(mmTiling_->N) - (params.nTotalCnt - 1) * mmTiling_->singleCoreN;
    info_.mBaseTail = static_cast<uint64_t>(mmTiling_->M) - (params.mTotalCnt - 1) * mmTiling_->singleCoreM;
    if constexpr (aTrans) {
        info_.alignedKaSize = DequantBmm::Align(mmTiling_->Ka, BMM_BLOCK_NUM);
    } else {
        info_.alignedKaSize = DequantBmm::Align(mmTiling_->Ka, DequantBmm::GetC0Size<x1Type>());
    }
    if constexpr (bTrans) {
        info_.alignedKbSize = DequantBmm::Align(mmTiling_->Kb, DequantBmm::GetC0Size<x2Type>());
    } else if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::NZ) {
        info_.alignedKbSize = DequantBmm::Align(mmTiling_->Kb, BMM_BLOCK_NUM);
    } else {
        info_.alignedKbSize = DequantBmm::Align(mmTiling_->Kb, BMM_BLOCK_NUM);
    }
}

template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type, typename x2Type>
__aicore__ inline void QuantBatchMatmulV3Update::UpdateBlockParamsAndCalcGmOffset(QBmmBaseBlockArgs& params,
                                                                                  QBmmBlockOffset& offset,
                                                                                  uint64_t mTileIndex,
                                                                                  uint64_t nTileIndex)
{
    UpdateBlockParams(params, mTileIndex, nTileIndex);
    CalcGMOffset<x1Format, x2Format, aTrans, bTrans, x1Type, x2Type>(params, offset);
}

__aicore__ inline void QuantBatchMatmulV3Update::UpdateBlockParams(QBmmBaseBlockArgs& params, uint64_t mTileIndex,
                                                                   uint64_t nTileIndex)
{
    if ((mTileIndex == (params.mTileCntL2 - 1)) && (nTileIndex == (params.nTileCntL2 - 1)) &&
        (params.index == (params.totalTileCnt - 1))) {
        params.singleCoreM = info_.mBaseTail;
        params.singleCoreN = info_.nBaseTail;
    } else if ((mTileIndex == (params.mTileCntL2 - 1)) && (params.index >= (params.mCntUse - 1) * params.nCntUse)) {
        params.singleCoreM = info_.mBaseTail;
        params.singleCoreN = mmTiling_->baseN;
    } else if ((nTileIndex == (params.nTileCntL2 - 1)) && ((params.index + 1) % params.nCntUse == 0)) {
        params.singleCoreM = mmTiling_->baseM;
        params.singleCoreN = info_.nBaseTail;
    } else {
        params.singleCoreM = mmTiling_->baseM;
        params.singleCoreN = mmTiling_->baseN;
    }
}

template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type, typename x2Type>
__aicore__ inline void QuantBatchMatmulV3Update::CalcGMOffset(QBmmBaseBlockArgs& params, QBmmBlockOffset& offset)
{
    uint64_t mCntIndex = params.index / params.nCntUse;
    uint64_t nCntIndex = params.index - mCntIndex * params.nCntUse;
    uint64_t mOffset = mCntIndex * mmTiling_->singleCoreM + params.mTileAddrOffset;
    if constexpr (DequantBmm::GetFormat(x1Format) == CubeFormat::ND) {
        if constexpr (aTrans) {
            offset.offsetA = mOffset;
        } else {
            offset.offsetA = mOffset * static_cast<uint64_t>(batch_) * mmTiling_->Ka;
        }
    } else if constexpr (DequantBmm::GetFormat(x1Format) == CubeFormat::NZ) {
        if constexpr (aTrans) {
            offset.offsetA = mOffset * info_.alignedKaSize;
        } else {
            offset.offsetA = mOffset * DequantBmm::GetC0Size<x1Type>();
        }
    }
    uint64_t nOffset = nCntIndex * mmTiling_->singleCoreN + params.nTileAddrOffset;
    if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::ND) {
        if constexpr (bTrans) {
            offset.offsetB = nOffset * mmTiling_->Kb;
        } else {
            offset.offsetB = nOffset;
        }
    } else if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::NZ) {
        if constexpr (bTrans) {
            offset.offsetB = nOffset * DequantBmm::GetC0Size<x2Type>();
        } else {
            offset.offsetB = nOffset * info_.alignedKbSize;
        }
    }
    offset.offsetC = mOffset * static_cast<uint64_t>(batch_) * mmTiling_->N + nOffset;
    offset.offsetScale = nOffset;
    offset.offsetBias = nOffset;
    offset.offsetPertoken = mOffset;
}
} // namespace AscendC

#endif // TBMM_QBMM_BASE_H
