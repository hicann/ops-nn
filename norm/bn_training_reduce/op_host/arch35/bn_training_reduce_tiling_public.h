/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_TRAINING_REDUCE_TILING_PUBLIC_H_
#define BN_TRAINING_REDUCE_TILING_PUBLIC_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../../op_kernel/arch35/bn_training_reduce_tiling_data.h"

namespace optiling {

enum class BNTrainingReducePublicStatus : int32_t {
    SUCCESS = 0,
    SHAPE_MISMATCH,
    DTYPE_NOT_SUPPORTED,
    NULL_INPUT,
    TILING_FAILED,
    UNIMPLEMENTED,
};

enum class BNTrainingReducePublicFormat : int32_t {
    NCHW = 0,
    NHWC,
    NCDHW,
};

enum class BNTrainingReducePublicDType : int32_t {
    FLOAT16 = 0,
    BFLOAT16,
    FLOAT32,
    INT32,
};

enum class BNTrainingReduceEmptyKind : int32_t {
    NORMAL = 0,
    EMPTY_A,
    EMPTY_R,
};

enum class BNTrainingReduceTilingKey : int64_t {
    NORMAL_TAIL_A = 0,
    GROUP_TAIL_A = 1,
    EMPTY = 2,
    NORMAL_TAIL_R = 4,
    GROUP_TAIL_R = 5,
    SMALL_R = 6,
    DETERMINISTIC_GROUP_TAIL_A = 9,
    DETERMINISTIC_GROUP_TAIL_R = 13,
};

struct BNTrainingReducePublicInputs {
    bool inputPresent = true;
    int32_t rank = 4;
    std::array<int64_t, 5> shape = {1, 1, 1, 1, 1};
    BNTrainingReducePublicFormat format = BNTrainingReducePublicFormat::NCHW;
    BNTrainingReducePublicDType inputDtype = BNTrainingReducePublicDType::FLOAT32;

    int32_t sumRank = 1;
    int64_t sumDim0 = 1;
    BNTrainingReducePublicDType sumDtype = BNTrainingReducePublicDType::FLOAT32;
    int32_t squareSumRank = 1;
    int64_t squareSumDim0 = 1;
    BNTrainingReducePublicDType squareSumDtype = BNTrainingReducePublicDType::FLOAT32;

    int64_t coreNum = 64;
    int64_t ubSize = 0;
    int64_t blockSize = 32;
    int64_t cacheLineSize = 256;
    int64_t vectorSize = 256;
    size_t systemWorkspaceSize = 0;
    bool deterministic = false;
};

struct BNTrainingReducePublicResult {
    BNTrainingReducePublicStatus status = BNTrainingReducePublicStatus::UNIMPLEMENTED;
    int64_t tilingKey = std::numeric_limits<int64_t>::min();
    uint32_t blockDim = std::numeric_limits<uint32_t>::max();
    size_t workspaceSize = std::numeric_limits<size_t>::max();
    int32_t scheduleMode = -1;
    BNTrainingReduceTilingData tilingData = {};
};

BNTrainingReducePublicStatus ValidateBNTrainingReducePublicInputs(const BNTrainingReducePublicInputs& inputs);

// Side-effect-free Host Tiling implementation used by the runtime glue.
BNTrainingReducePublicResult ComputeBNTrainingReducePublicTiling(const BNTrainingReducePublicInputs& inputs);

} // namespace optiling

#endif // BN_TRAINING_REDUCE_TILING_PUBLIC_H_
