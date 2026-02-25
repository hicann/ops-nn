/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BATCH_NORM_V3_H_
#define OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BATCH_NORM_V3_H_

#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
namespace optiling {

struct InstanceNormCompileInfo {
    uint64_t coreNum;      // 系统核数
    uint64_t ubSize;       // UB空间
    uint32_t vectorLength; // 256
    uint64_t ubBlockSize;  // 32B，UB的字节对齐单位
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BATCH_NORM_V3_H_
