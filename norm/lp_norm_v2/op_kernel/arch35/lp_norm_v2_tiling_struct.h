/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file lp_norm_v2_tiling.h
 * \brief
 */
#ifndef OPS_LP_NORM_V2_TILING_STRUCT_H_
#define OPS_LP_NORM_V2_TILING_STRUCT_H_

#include "atvoss/reduce/reduce_tiling_data.h"

using namespace Ops::Base;

namespace optiling
{
struct LpNormV2TilingData {
    ReduceOpTilingData reduceTiling;
    float epsilon;
    float p;
    float recp;
};
} // namespace optiling

#endif // OPS_LP_NORM_V2_TILING_STRUCT_H_