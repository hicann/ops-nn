/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * \file cross_v2_tiling.h
 * \brief
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 */
#ifndef CROSS_V2_TILING_H
#define CROSS_V2_TILING_H
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CrossV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, stepSize);
TILING_DATA_FIELD_DEF(int64_t, tileNum);
TILING_DATA_FIELD_DEF(int64_t, tileNumPerBatch);
TILING_DATA_FIELD_DEF(int64_t, tileDataNum);
TILING_DATA_FIELD_DEF(int64_t, tailDataNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossV2, CrossV2TilingData)
} // namespace optiling
#endif // CROSS_V2_TILING_H