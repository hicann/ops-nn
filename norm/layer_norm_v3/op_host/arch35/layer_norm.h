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
 * /file layer_norm.h
 * \brief
 */
#ifndef LAYER_NORM_H_
#define LAYER_NORM_H_
#include <string>
#include <vector>

#include "layer_norm_v3_tiling.h"

namespace optiling {
struct LayerNormOpInfo {
    bool is_support_vexp_pattern;
    std::vector<int32_t> ori_reduce_axis;
    std::string input_format;
    int32_t core_num;
    uint32_t ci_key;
    int32_t begin_norm_axis;
    int32_t begin_params_axis;
    bool is_tik_support;
    std::string tik_mode;
    int32_t ub_max_byte;
    bool atomic_clean_diff_shape;
    bool is_support_vexp;
    std::string reduce_mean_cof_dtype;
    std::vector<int32_t> common_info;
    std::vector<int32_t> pattern_info;
    std::vector<int32_t> ub_info;
    std::vector<int32_t> reduce_axis;
    int32_t max_ub_size_normal_fp16;
    int32_t max_ub_size_normal_fp32;
    std::string mode;
    bool is_unknown_mode;
    ge::DataType reduce_mean_cof_ge_dtype;
    LayerNormV3CompileInfo regbaseCompileInfo;
    bool is_regbase = true;
};
} // namespace optiling
#endif // LAYER_NORM_H_
