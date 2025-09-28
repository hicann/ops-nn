/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file cross_entropy_loss_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CROSS_ENTROPY_LOSS_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CROSS_ENTROPY_LOSS_H

#include <iostream>
#include <cstring>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_util.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CrossEntropyLossTilingData)
TILING_DATA_FIELD_DEF(uint64_t, targetNum);
TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, frontBatchNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, tailBatchNum);
TILING_DATA_FIELD_DEF(uint64_t, inputUbSize);
TILING_DATA_FIELD_DEF(uint64_t, castTmpBufByte);
TILING_DATA_FIELD_DEF(uint64_t, lnTmpBufSize);
TILING_DATA_FIELD_DEF(uint64_t, weightTmpBufSize);
TILING_DATA_FIELD_DEF(uint64_t, weight4SmoothingBufSize);
TILING_DATA_FIELD_DEF(uint64_t, totalTmpBufByte);
TILING_DATA_FIELD_DEF(uint64_t, ubLoopNum);
TILING_DATA_FIELD_DEF(uint64_t, ubTailNum);
TILING_DATA_FIELD_DEF(uint64_t, vecLoopNum);
TILING_DATA_FIELD_DEF(uint64_t, vecTailNum);
TILING_DATA_FIELD_DEF(uint64_t, tailVecLoopNum);
TILING_DATA_FIELD_DEF(uint64_t, tailVecTailNum);
TILING_DATA_FIELD_DEF(uint64_t, reduction);
TILING_DATA_FIELD_DEF(int64_t, ignoreIndex);
TILING_DATA_FIELD_DEF(float, labelSmoothing);
TILING_DATA_FIELD_DEF(uint32_t, defaultWeight);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossEntropyLoss, CrossEntropyLossTilingData)
ge::graphStatus Tiling4CrossEntropyLossRegbase(gert::TilingContext* context);
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_CROSS_ENTROPY_LOSS_H