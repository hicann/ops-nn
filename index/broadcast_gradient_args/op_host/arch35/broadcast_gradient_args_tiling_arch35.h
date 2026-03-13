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
 * \file broadcast_gradient_args_tiling_arch35.h
 * \brief
 */

#ifndef BROADCAST_GRADIENT_ARGS_TILING_ARCH35_H_
#define BROADCAST_GRADIENT_ARGS_TILING_ARCH35_H_

#include "register/tilingdata_base.h"

namespace optiling {

struct BroadcastGradientArgsCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

BEGIN_TILING_DATA_DEF(BroadcastGradientArgsTilingData)
TILING_DATA_FIELD_DEF(int64_t, x1Len);
TILING_DATA_FIELD_DEF(int64_t, x2Len);
TILING_DATA_FIELD_DEF(int64_t, maxRank);
TILING_DATA_FIELD_DEF(int64_t, ubMaxRank);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BroadcastGradientArgs, BroadcastGradientArgsTilingData)
} // namespace optiling

#endif // BROADCAST_GRADIENT_ARGS_TILING_ARCH35_H_