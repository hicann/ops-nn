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
 * \file hard_swish_grad_v2_tiling_def.h
 * \brief hard_swish_grad_v2_tiling_def
 */
#ifndef HARD_SWISH_GRAD_V2_TILING_DEF_H
#define HARD_SWISH_GRAD_V2_TILING_DEF_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct HardSwishGradV2CompileInfo {};

BEGIN_TILING_DATA_DEF(HardSwishGradV2TilingData)

TILING_DATA_FIELD_DEF(int64_t, elementNum);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, elementNumEachCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HardSwishGradV2, HardSwishGradV2TilingData)
}  // namespace optiling

#endif