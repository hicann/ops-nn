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
 * \file test_batch_norm_grad_v3_tiling.h
 * \brief test BatchNormGradV3CompileInfo Op Tiling
 */
#ifndef OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BATCH_NORM_GRAD_V3_H_
#define OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BATCH_NORM_GRAD_V3_H_

#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
namespace optiling {

struct BatchNormGradV3CompileInfo {
    int32_t coreNum;
    int64_t ubSize;
    int64_t blockSize;
    int64_t vlFp32;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BATCH_NORM_GRAD_V3_H_
