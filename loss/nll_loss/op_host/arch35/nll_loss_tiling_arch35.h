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
 * \file nll_loss_tiling_arch35.h
 * \brief nll_loss_tiling_arch35
 */

#ifndef OPS_LOSS_NLL_LOSS_OP_HOST_NLL_LOSS_TILING_ARCH35_H_
#define OPS_LOSS_NLL_LOSS_OP_HOST_NLL_LOSS_TILING_ARCH35_H_

#include <cstdint>
#include "tiling_base/tiling_base.h"
#include "register/tilingdata_base.h"

namespace optiling {
using namespace Ops::NN::Optiling;

struct NLLLossCompileInfo {
    int64_t ubSize{1};
    int64_t coreNum{0};
    std::string reduction{"sum"};
    int64_t reductionInt{1};
    bool isAscendc{false};
    uint32_t maxCoreNum{0};
    uint32_t maxThread{0};
};

BEGIN_TILING_DATA_DEF(NLLLossACTilingData)
TILING_DATA_FIELD_DEF(int64_t, ignoreIndex);
TILING_DATA_FIELD_DEF(int64_t, xDims);
TILING_DATA_FIELD_DEF(uint32_t, reduction);
TILING_DATA_FIELD_DEF(int64_t, xDimN);
TILING_DATA_FIELD_DEF(uint32_t, xDimC);
TILING_DATA_FIELD_DEF(int64_t, xDimH);
TILING_DATA_FIELD_DEF(int64_t, xDimW);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, isWeightPresent);
TILING_DATA_FIELD_DEF(uint32_t, coreLoopCnt);
TILING_DATA_FIELD_DEF(int32_t, calNumInterBlock);
TILING_DATA_FIELD_DEF(int64_t, mainReduceSize);
TILING_DATA_FIELD_DEF(uint32_t, usedLogicCore);
TILING_DATA_FIELD_DEF(int32_t, loopInCore);
TILING_DATA_FIELD_DEF(int32_t, fullIndex);
TILING_DATA_FIELD_DEF(int32_t, tailSize);
TILING_DATA_FIELD_DEF(int32_t, padSize);
TILING_DATA_FIELD_DEF(int32_t, loopNum1);
TILING_DATA_FIELD_DEF(int32_t, tailNum1);
TILING_DATA_FIELD_DEF(int32_t, tailMoveSize);
TILING_DATA_FIELD_DEF(int32_t, tailMainReduceSize);
TILING_DATA_FIELD_DEF(int32_t, tailRemainSize);
TILING_DATA_FIELD_DEF(int32_t, loopNum3);
TILING_DATA_FIELD_DEF(int32_t, tailNum3);
TILING_DATA_FIELD_DEF(int64_t, dealingNumOneCore);
TILING_DATA_FIELD_DEF(int32_t, frontCore);
TILING_DATA_FIELD_DEF(int64_t, dealingNumOnce);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NLLLoss, NLLLossACTilingData)

struct NLLLossACTilingParam {
    int64_t ignoreIndex{0};
    uint32_t reduction{0};
    int64_t xDimN{0};
    uint32_t xDimC{1};
    int64_t xDimH{1};
    int64_t xDimW{1};
    uint32_t coreNum{0};
    uint32_t maxCoreNum{0};
    uint32_t maxThread{0};
    uint32_t usedThread{0};
    int64_t tilingKey{0};
    uint32_t isWeightPresent{0};
    uint32_t coreLoopCnt{0};
    uint32_t coreNumSum{0};
    int64_t xDims{0};
    int32_t calNumInterBlock{0};
    int64_t mainReduceSize{1};
    uint32_t usedLogicCore{1};
    int32_t loopInCore{1};
    int32_t fullIndex{1};
    int32_t tailSize{0};
    int32_t padSize{0};
    int32_t loopNum1{1};
    int32_t tailNum1{0};
    int32_t tailMoveSize{1};
    int32_t tailMainReduceSize{1};
    int32_t tailRemainSize{1};
    int32_t loopNum3{1};
    int32_t tailNum3{0};
    uint64_t ubSize{0};
    uint32_t dtypeSize{0};
    uint64_t dealingNumOneCore{0};
    uint32_t frontCore{0};
    uint64_t dealingNumOnce{0};
};

} // namespace optiling
#endif // OPS_LOSS_NLL_LOSS_OP_HOST_NLL_LOSS_TILING_ARCH35_H_W