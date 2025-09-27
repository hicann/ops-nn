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
 * \file fake_quant_affine_cachemask_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_FAKE_QUANT_AFFINE_CACHEMASK_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_FAKE_QUANT_AFFINE_CACHEMASK_H
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FakeQuantAffineCachemaskTilingData)
TILING_DATA_FIELD_DEF(int64_t, quantMin);
TILING_DATA_FIELD_DEF(int64_t, quantMax);
TILING_DATA_FIELD_DEF(uint32_t, loopNum);
TILING_DATA_FIELD_DEF(uint32_t, remainNum);
TILING_DATA_FIELD_DEF(uint32_t, calcLength);
TILING_DATA_FIELD_DEF(uint32_t, headNum);
TILING_DATA_FIELD_DEF(uint32_t, dataPerRepeat);
TILING_DATA_FIELD_DEF(uint32_t, totalLengthAligned);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FakeQuantAffineCachemask, FakeQuantAffineCachemaskTilingData)
struct FakeQuantAffineCachemaskCompileInfo {
    int32_t totalCoreNum = 30;
    uint64_t ubSizePlatForm = 0;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_FAKE_QUANT_AFFINE_CACHEMASK_H