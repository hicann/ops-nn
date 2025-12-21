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
 * \file modulate_tiling.h
 * \brief
 */
#ifndef MODULATE_TILING_H
#define MODULATE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ModulateTilingData)
TILING_DATA_FIELD_DEF(int64_t, inputB);
TILING_DATA_FIELD_DEF(int64_t, inputL);
TILING_DATA_FIELD_DEF(int64_t, inputD);
TILING_DATA_FIELD_DEF(int64_t, ubLength);
TILING_DATA_FIELD_DEF(int64_t, frontNum);
TILING_DATA_FIELD_DEF(int64_t, frontLength);
TILING_DATA_FIELD_DEF(int64_t, tailNum);
TILING_DATA_FIELD_DEF(int64_t, tailLength);
TILING_DATA_FIELD_DEF(int64_t, useDTiling);
TILING_DATA_FIELD_DEF(int64_t, parameterStatus);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Modulate, ModulateTilingData)

struct ModulateCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};
} // namespace optiling
#endif