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
 * \file swiglu_group_quant_tiling_base.h
 * \brief
 */

#ifndef SWIGLU_GROUP_QUANT_TILING_BASE_H
#define SWIGLU_GROUP_QUANT_TILING_BASE_H

#include <cstdint>
#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "log/log.h"

namespace optiling {

struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct SwigluGroupQuantCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

constexpr int64_t MAX_CORE_COUNT = 64;

BEGIN_TILING_DATA_DEF(SwigluGroupQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, splitD);
TILING_DATA_FIELD_DEF(int64_t, scaleCol);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowFactor);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dLoop);
TILING_DATA_FIELD_DEF(int64_t, dFactor);
TILING_DATA_FIELD_DEF(int64_t, tailDFactor);
TILING_DATA_FIELD_DEF(int64_t, roundScale);
TILING_DATA_FIELD_DEF(int64_t, outputOrigin);
TILING_DATA_FIELD_DEF(float, clampLimit);
TILING_DATA_FIELD_DEF(int64_t, hasClampLimit);
TILING_DATA_FIELD_DEF(int64_t, g);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, gLoop);
TILING_DATA_FIELD_DEF(int64_t, gFactor);
TILING_DATA_FIELD_DEF(int64_t, tailGFactor);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwigluGroupQuant, SwigluGroupQuantTilingData)

BEGIN_TILING_DATA_DEF(SwigluGroupQuantHifp8TilingData)
TILING_DATA_FIELD_DEF(int64_t, totalTokens);
TILING_DATA_FIELD_DEF(int64_t, dim2H);
TILING_DATA_FIELD_DEF(int64_t, dimH);
TILING_DATA_FIELD_DEF(int64_t, isGroup);
TILING_DATA_FIELD_DEF(int64_t, hasWeight);
TILING_DATA_FIELD_DEF(int64_t, hasClamp);
TILING_DATA_FIELD_DEF(int64_t, outputOrigin);
TILING_DATA_FIELD_DEF(float, clampLimit);
TILING_DATA_FIELD_DEF(float, dstTypeMax);
TILING_DATA_FIELD_DEF(int64_t, tileTokens);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, tokensPerCore);
TILING_DATA_FIELD_DEF(int64_t, groupNum);
TILING_DATA_FIELD_DEF(int64_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwigluGroupQuant_4000, SwigluGroupQuantHifp8TilingData)

REGISTER_TILING_DATA_CLASS(SwigluGroupQuant_4100, SwigluGroupQuantHifp8TilingData)

}  // namespace optiling

#endif  // SWIGLU_GROUP_QUANT_TILING_BASE_H
