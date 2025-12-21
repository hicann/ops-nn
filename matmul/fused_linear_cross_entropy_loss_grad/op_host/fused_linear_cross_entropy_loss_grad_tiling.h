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
 * \file fused_matmul_celoss_grad_tiling.h
 * \brief
 */
#ifndef FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_TILING_H
#define FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr uint64_t HIGH_PERF_KEY = 100;  // 高性能分支tiling key
constexpr uint64_t MEM_FRIENDLY_KEY = 101;  // 省显存分支tiling key
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t INDEX_0 = 0;
constexpr uint32_t INDEX_1 = 1;
constexpr uint32_t INDEX_2 = 2;
constexpr uint32_t INDEX_3 = 3;
constexpr uint32_t INDEX_4 = 4;
constexpr uint32_t INDEX_5 = 5;
constexpr uint32_t INDEX_6 = 6;
constexpr uint32_t INDEX_7 = 7;
constexpr uint32_t IN_BYTE_SIZE = 4;
constexpr uint32_t OUT_BYTE_SIZE = 2;
constexpr uint32_t TBUF_BYTE_SIZE = 32;
constexpr uint32_t SIZE_2 = 2;
constexpr uint32_t SIZE_4 = 4;
constexpr uint32_t SIZE_8 = 8;
constexpr uint32_t SIZE_16 = 16;
constexpr uint32_t SIZE_32 = 32;
constexpr uint32_t SIZE_64 = 64;
constexpr uint32_t SIZE_128 = 128;
constexpr uint32_t SIZE_256 = 256;
constexpr uint32_t SIZE_512 = 512;
constexpr uint32_t SIZE_65535 = 65535;
constexpr uint32_t MAX_LEFT_BASE_BYTE_SIZE = SIZE_256 * SIZE_64 * OUT_BYTE_SIZE;
constexpr uint32_t BT_BOUNDARY = 23296;  // 交换baseM/baseN收益边界
uint32_t baseM = SIZE_128;
uint32_t baseN = SIZE_256;
constexpr uint32_t baseK = SIZE_64;

// default register, only for compile
BEGIN_TILING_DATA_DEF(FusedLinearCrossEntropyLossGradDummyTilingData)
  TILING_DATA_FIELD_DEF(int8_t, dummy);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FusedLinearCrossEntropyLossGrad, FusedLinearCrossEntropyLossGradDummyTilingData);

// 高性能分支
BEGIN_TILING_DATA_DEF(FusedLinearCrossEntropyLossGradHighPerfTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1Tiling);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2Tiling);
  TILING_DATA_FIELD_DEF(uint64_t, BT);
  TILING_DATA_FIELD_DEF(uint64_t, V);
  TILING_DATA_FIELD_DEF(uint64_t, H);
  TILING_DATA_FIELD_DEF(uint64_t, BT_16ALIGNED);
  TILING_DATA_FIELD_DEF(uint64_t, btPerCore);
  TILING_DATA_FIELD_DEF(uint64_t, btLastCore);
  TILING_DATA_FIELD_DEF(uint64_t, V64BAlignedSize);
  TILING_DATA_FIELD_DEF(uint64_t, VOut512BAlignedSize);
  TILING_DATA_FIELD_DEF(uint64_t, VOut32BAlignedSize);
  TILING_DATA_FIELD_DEF(uint64_t, targetMaskSize);
  TILING_DATA_FIELD_DEF(uint64_t, userWorkspaceSize);
  TILING_DATA_FIELD_DEF(uint64_t, taskCntTotal);
  TILING_DATA_FIELD_DEF(uint32_t, aicNum);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, btLastCoreId);
  TILING_DATA_FIELD_DEF(uint32_t, copyOutDstStride);
  TILING_DATA_FIELD_DEF(uint32_t, btPerEpochSingle);
  TILING_DATA_FIELD_DEF(uint32_t, additionalEndEpochId);
  TILING_DATA_FIELD_DEF(uint32_t, additionalEndCoreId);
  TILING_DATA_FIELD_DEF(uint32_t, additionalBtTaskNum);
  TILING_DATA_FIELD_DEF(uint32_t, btPerEpochTotal);
  TILING_DATA_FIELD_DEF(uint32_t, quebindSoftmaxByteSize);
  TILING_DATA_FIELD_DEF(uint32_t, copyInDstStride);
  TILING_DATA_FIELD_DEF(uint32_t, epochCnt);
  TILING_DATA_FIELD_DEF(uint32_t, lastEpochCnt);
  TILING_DATA_FIELD_DEF(uint32_t, epochOutCoreId);
  TILING_DATA_FIELD_DEF(uint32_t, using32BlockCnt);
  TILING_DATA_FIELD_DEF(uint32_t, sizePerTask);
  TILING_DATA_FIELD_DEF(uint32_t, taskCntPerV);
  TILING_DATA_FIELD_DEF(uint32_t, sizeLastTask);
  TILING_DATA_FIELD_DEF(uint32_t, mm1BaseMCnt);
  TILING_DATA_FIELD_DEF(uint32_t, mm1BaseNCnt);
  TILING_DATA_FIELD_DEF(uint32_t, mm1TailM);
  TILING_DATA_FIELD_DEF(uint32_t, mm1TailN);
  TILING_DATA_FIELD_DEF(uint32_t, mm2BaseMCnt);
  TILING_DATA_FIELD_DEF(uint32_t, mm2BaseNCnt);
  TILING_DATA_FIELD_DEF(uint32_t, mm2BaseKCnt);
  TILING_DATA_FIELD_DEF(uint32_t, mm2TailM);
  TILING_DATA_FIELD_DEF(uint32_t, mm2TailN);
  TILING_DATA_FIELD_DEF(uint32_t, mm2TailK);
  TILING_DATA_FIELD_DEF(uint32_t, mm2BasePerVec);
  TILING_DATA_FIELD_DEF(uint32_t, mm2BaseLastVec);
  TILING_DATA_FIELD_DEF(uint32_t, mm2BaseLastVecId);
  TILING_DATA_FIELD_DEF(uint32_t, mm2LastBaseK);
  TILING_DATA_FIELD_DEF(uint32_t, wsNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedLinearCrossEntropyLossGrad_100, FusedLinearCrossEntropyLossGradHighPerfTilingData);

// 省显存分支
BEGIN_TILING_DATA_DEF(FusedLinearCrossEntropyLossGradMemFriendlyTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm0Tiling);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1Tiling);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2Tiling);
  TILING_DATA_FIELD_DEF(uint32_t, aicNum);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, BT);
  TILING_DATA_FIELD_DEF(uint32_t, V);
  TILING_DATA_FIELD_DEF(uint32_t, H);
  TILING_DATA_FIELD_DEF(uint32_t, targetMaskSize);
  TILING_DATA_FIELD_DEF(uint32_t, baseBT);
  TILING_DATA_FIELD_DEF(uint32_t, baseV);
  TILING_DATA_FIELD_DEF(uint32_t, baseBTTail);
  TILING_DATA_FIELD_DEF(uint32_t, baseVTail);
  TILING_DATA_FIELD_DEF(uint32_t, mm0ResultSize);
  TILING_DATA_FIELD_DEF(uint32_t, mm0ResultSizePerEpoch);
  TILING_DATA_FIELD_DEF(uint32_t, V1MainSize);
  TILING_DATA_FIELD_DEF(uint32_t, mm1ResultSize);
  TILING_DATA_FIELD_DEF(uint32_t, mm1ResultSizePerEpoch);
  TILING_DATA_FIELD_DEF(uint32_t, mm2ResultSize);
  TILING_DATA_FIELD_DEF(uint32_t, mm2ResultSizePerEpoch);
  TILING_DATA_FIELD_DEF(uint32_t, V1InUBStride);
  TILING_DATA_FIELD_DEF(uint32_t, V1OutUBStride);
  TILING_DATA_FIELD_DEF(uint32_t, mmAccumRowOnetime);
  TILING_DATA_FIELD_DEF(uint32_t, mmAccumCyclePerRow);
  TILING_DATA_FIELD_DEF(uint32_t, mmAccumCntPerCycle);
  TILING_DATA_FIELD_DEF(uint32_t, mmAccumCntLastCycle);
  TILING_DATA_FIELD_DEF(uint32_t, V2MainSize);
  TILING_DATA_FIELD_DEF(uint32_t, mmCastRowOnetime);
  TILING_DATA_FIELD_DEF(uint32_t, mmCastCyclePerRow);
  TILING_DATA_FIELD_DEF(uint32_t, mmCastCntPerCycle);
  TILING_DATA_FIELD_DEF(uint32_t, mmCastCntLastCycle);
  TILING_DATA_FIELD_DEF(uint32_t, castMainSize);
  TILING_DATA_FIELD_DEF(uint32_t, poolByteSize);
  TILING_DATA_FIELD_DEF(uint32_t, mm1CastRowPerVec);
  TILING_DATA_FIELD_DEF(uint32_t, mm1CastExtraRowCnt);
  TILING_DATA_FIELD_DEF(uint32_t, mm2CastRowPerVec);
  TILING_DATA_FIELD_DEF(uint32_t, mm2CastExtraRowCnt);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedLinearCrossEntropyLossGrad_101, FusedLinearCrossEntropyLossGradMemFriendlyTilingData);

struct FusedLinearCrossEntropyLossGradCompileInfo {};
}  // namespace optiling

#endif // FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_TILING_H