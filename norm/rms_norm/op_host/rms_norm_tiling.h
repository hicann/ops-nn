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
 * \file rms_norm_tiling.h
 * \brief RmsNorm Op Tiling
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_H_

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_templates_registry.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(RMSNormTilingData)
TILING_DATA_FIELD_DEF(uint64_t, num_row);
TILING_DATA_FIELD_DEF(uint64_t, num_col);
TILING_DATA_FIELD_DEF(uint64_t, num_col_align);
TILING_DATA_FIELD_DEF(uint64_t, block_factor);
TILING_DATA_FIELD_DEF(uint32_t, row_factor);
TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
TILING_DATA_FIELD_DEF(uint32_t, reduce_mask);
TILING_DATA_FIELD_DEF(uint32_t, left_num);
TILING_DATA_FIELD_DEF(uint32_t, last_reduce_mask);
TILING_DATA_FIELD_DEF(uint32_t, last_left_num);
TILING_DATA_FIELD_DEF(uint32_t, rstd_size);
TILING_DATA_FIELD_DEF(uint32_t, ub_loop);
TILING_DATA_FIELD_DEF(uint32_t, col_buffer_length);
TILING_DATA_FIELD_DEF(uint32_t, multi_n_num);
TILING_DATA_FIELD_DEF(uint32_t, is_nddma);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avg_factor);
TILING_DATA_FIELD_DEF(uint8_t, is_gemma);

TILING_DATA_FIELD_DEF(uint64_t, last_block_factor);
TILING_DATA_FIELD_DEF(uint64_t, row_loop);
TILING_DATA_FIELD_DEF(uint64_t, last_block_row_loop);
TILING_DATA_FIELD_DEF(uint64_t, row_tail);
TILING_DATA_FIELD_DEF(uint64_t, last_block_row_tail);
TILING_DATA_FIELD_DEF(uint32_t, mul_loop);
TILING_DATA_FIELD_DEF(uint32_t, mul_tail);
TILING_DATA_FIELD_DEF(uint8_t, dst_rep_stride);
TILING_DATA_FIELD_DEF(uint8_t, is_performance);
TILING_DATA_FIELD_DEF(uint8_t, normal_flag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNorm, RMSNormTilingData)
REGISTER_TILING_DATA_CLASS(GemmaRmsNorm, RMSNormTilingData)

struct Tiling4RmsNormCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
};

struct ComputeTotalBufSizeParam {
    uint32_t bufferNum = 0U;
    ge::DataType dtype;
    uint32_t dtypeSizeX = 0U;
    uint32_t dtypeSizeGamma = 0U;
    uint32_t length = 0U;
    bool split = false;
};

class RMSNormTilingInfo {
public:
    uint64_t ubSize{0};
    uint64_t numCol{0};
    uint64_t numRow{0};
    uint32_t numCore{0};
    uint64_t numColAlign{0};
    uint32_t xDtypeKey{0};

    bool isSoc910B{false};
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_H_
