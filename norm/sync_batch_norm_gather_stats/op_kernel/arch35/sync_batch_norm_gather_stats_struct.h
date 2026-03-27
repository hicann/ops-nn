/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_batch_norm_gather_stats_struct.h
 * \brief
 */
#ifndef __SYNC_BATCH_NORM_GATHER_STATS_STRUCT__
#define __SYNC_BATCH_NORM_GATHER_STATS_STRUCT__

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace SyncBatchNormGatherStats {
constexpr static uint16_t VECTOR_LENGTH = platform::GetVRegSize();
constexpr static uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
constexpr static uint16_t VL_INT64 = VECTOR_LENGTH / sizeof(int64_t);
constexpr static int16_t DOUBLE_BUFFER = 2;
constexpr static int16_t TRIPLE_BUFFER = 3;
constexpr static int16_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr static int16_t BS_INT64 = BLOCK_SIZE / sizeof(int64_t);

constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB32B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB32B64 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
};
}
#endif

