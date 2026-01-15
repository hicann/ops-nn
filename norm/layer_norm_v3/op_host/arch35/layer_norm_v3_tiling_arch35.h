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
 * \file layer_norm_v3.h
 * \brief
 */
#ifndef LAYER_NORM_V3_H_
#define LAYER_NORM_V3_H_
#include <string>
#include <vector>

#include "layer_norm_v3_tiling.h"

namespace optiling {
struct LayerNormV3OpInfo {
    std::vector<int32_t> ori_reduce_axis;
    std::string reduce_mean_cof_dtype;
    bool is_unknown_mode;
    ge::DataType reduce_mean_cof_ge_dtype;
    uint32_t ci_key;
    LayerNormV3CompileInfo regbaseCompileInfo;
    bool is_regbase = false;
};
} // namespace optiling
#endif // LAYER_NORM_V3_H_
