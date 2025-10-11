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
 * \file edge_softmax_tiling.h
 * \brief
 */
#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EdgeSoftmaxTilingData)
TILING_DATA_FIELD_DEF(int32_t, E);
TILING_DATA_FIELD_DEF(int16_t, F);
TILING_DATA_FIELD_DEF(int16_t, N);
TILING_DATA_FIELD_DEF(int8_t, blockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EdgeSoftmax, EdgeSoftmaxTilingData)
}  // namespace optiling
