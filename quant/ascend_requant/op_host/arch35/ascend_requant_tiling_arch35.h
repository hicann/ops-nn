/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_ASCEND_REQUANT_OP_HOST_ARCH35_ASCEND_REQUANT_TILING_ARCH35_H
#define QUANT_ASCEND_REQUANT_OP_HOST_ARCH35_ASCEND_REQUANT_TILING_ARCH35_H

#include <cstdint>
#include <vector>

#include "graph/types.h"

#include "../../op_kernel/arch35/ascend_requant_tiling_data.h"

namespace optiling {

struct AscendRequantCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

namespace ascend_requant {
namespace public_api {

constexpr int64_t kErrShapeMismatch = -1;
constexpr int64_t kErrDtypeNotSupported = -2;
constexpr int64_t kErrFormatNotSupported = -3;
constexpr int64_t kErrDimOutOfRange = -4;
constexpr int64_t kErrRankExceedsX = -5;
constexpr int64_t kOk = 0;

int64_t PadAndSqueeze(const std::vector<std::vector<int64_t>>& inputShapes,
                      const std::vector<std::vector<int64_t>>& outputShapes, std::vector<int64_t>& maximumBroShape,
                      std::vector<std::vector<int64_t>>& normalInputShapes,
                      std::vector<std::vector<int64_t>>& normalOutputShapes);

int64_t CheckBroadcastShape(const std::vector<std::vector<int64_t>>& paddedIn,
                            const std::vector<std::vector<int64_t>>& paddedOut, int64_t maxRank);

int64_t FindSplitAxis(const std::vector<int64_t>& maxBroShape, int64_t maxDtypeSize, int64_t ubPerCore,
                      int64_t physNodes, SplitResult& out);

int64_t MultiCoreSplit(const std::vector<int64_t>& maxBroShape, const SplitResult& ubSplit, int64_t maxCores,
                       MultiCoreResult& out);

int64_t MapRankToTemplate(int64_t rank);

int64_t ValidateDtype(ge::DataType xDtype, ge::DataType scaleDtype);

int64_t ValidateFormat(ge::Format xFmt, ge::Format scaleFmt, ge::Format yFmt);

int64_t ValidateDimensions(int64_t xRank, int64_t scaleRank);

int64_t ValidateAttr(bool reluFlag);

} // namespace public_api

namespace branch_api {

template <int64_t RANK>
struct BranchInputs {
    std::vector<int64_t> maxBroShape;
    std::vector<std::vector<int64_t>> normalInputShapes;
    std::vector<std::vector<int64_t>> normalOutputShapes;
    int64_t maxCores = 0;
    int64_t ubPerCore = 0;
    int64_t reluFlag = 0;
};

template <int64_t RANK>
int64_t ComputeBranchTiling(const BranchInputs<RANK>& in, AscendRequantTilingData<RANK>& out);

} // namespace branch_api
} // namespace ascend_requant
} // namespace optiling

#endif // QUANT_ASCEND_REQUANT_OP_HOST_ARCH35_ASCEND_REQUANT_TILING_ARCH35_H
