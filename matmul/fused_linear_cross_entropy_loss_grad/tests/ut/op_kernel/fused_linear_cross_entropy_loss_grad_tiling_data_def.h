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
 * \file fused_linear_cross_entropy_loss_grad_tiling_data_def.h
 * \brief
 */
#ifndef TILING_DATA_DEF_H
#define TILING_DATA_DEF_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__
#define DTYPE_INPUT bfloat16_t
#define DTYPE_TARGET_MASK bool
#define DTYPE_MASKED_TARGET int32_t

namespace {
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t INDEX_0 = 0;
    constexpr uint32_t INDEX_1 = 1;
    constexpr uint32_t INDEX_2 = 2;
    constexpr uint32_t INDEX_3 = 3;
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
    constexpr uint32_t SIZE_BASE_TASK_MAX = 65536 * 3;
}

struct FusedLinearCrossEntropyLossGradMemFriendlyTilingData {
    TCubeTiling mm0Tiling;
    TCubeTiling mm1Tiling;
    TCubeTiling mm2Tiling;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t BT;
    uint32_t V;
    uint32_t H;
    uint32_t targetMaskSize;
    uint32_t baseBT;
    uint32_t baseV;
    uint32_t baseBTTail;
    uint32_t baseVTail;
    uint32_t mm0ResultSize;
    uint32_t mm0ResultSizePerEpoch;
    uint32_t V1MainSize;
    uint32_t mm1ResultSize;
    uint32_t mm1ResultSizePerEpoch;
    uint32_t mm2ResultSize;
    uint32_t mm2ResultSizePerEpoch;
    uint32_t V1InUBStride;
    uint32_t V1OutUBStride;
    uint32_t mmAccumRowOnetime;
    uint32_t mmAccumCyclePerRow;
    uint32_t mmAccumCntPerCycle;
    uint32_t mmAccumCntLastCycle;
    uint32_t V2MainSize;
    uint32_t mmCastRowOnetime;
    uint32_t mmCastCyclePerRow;
    uint32_t mmCastCntPerCycle;
    uint32_t mmCastCntLastCycle;
    uint32_t castMainSize;
    uint32_t poolByteSize;
    uint32_t mm1CastRowPerVec;
    uint32_t mm1CastExtraRowCnt;
    uint32_t mm2CastRowPerVec;
    uint32_t mm2CastExtraRowCnt;
};

struct FusedLinearCrossEntropyLossGradHighPerfTilingData {
    TCubeTiling mm1Tiling;
    TCubeTiling mm2Tiling;
    uint64_t BT;
    uint64_t V;
    uint64_t H;
    uint64_t BT_16ALIGNED;
    uint64_t btPerCore;
    uint64_t btLastCore;
    uint64_t V64BAlignedSize;
    uint64_t VOut512BAlignedSize;
    uint64_t VOut32BAlignedSize;
    uint64_t targetMaskSize;
    uint64_t userWorkspaceSize;
    uint64_t taskCntTotal;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t btLastCoreId;
    uint32_t copyOutDstStride;
    uint32_t btPerEpochSingle;
    uint32_t additionalEndEpochId;
    uint32_t additionalEndCoreId;
    uint32_t additionalBtTaskNum;
    uint32_t btPerEpochTotal;
    uint32_t quebindSoftmaxByteSize;
    uint32_t copyInDstStride;
    uint32_t epochCnt;
    uint32_t lastEpochCnt;
    uint32_t epochOutCoreId;
    uint32_t using32BlockCnt;
    uint32_t sizePerTask;
    uint32_t taskCntPerV;
    uint32_t sizeLastTask;
    uint32_t mm1BaseMCnt;
    uint32_t mm1BaseNCnt;
    uint32_t mm1TailM;
    uint32_t mm1TailN;
    uint32_t mm2BaseMCnt;
    uint32_t mm2BaseNCnt;
    uint32_t mm2BaseKCnt;
    uint32_t mm2TailM;
    uint32_t mm2TailN;
    uint32_t mm2TailK;
    uint32_t mm2BasePerVec;
    uint32_t mm2BaseLastVec;
    uint32_t mm2BaseLastVecId;
    uint32_t mm2LastBaseK;
    uint32_t wsNum;
};

#endif