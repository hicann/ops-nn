/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_IN_TRAINING_REDUCE_V2_H_
#define OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_IN_TRAINING_REDUCE_V2_H_

#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
namespace optiling {

// 与 op_host/in_training_reduce_v2_tiling.h 的 INTrainingReduceV2CompileInfo 布局一致。
// tiling_parse_func 把解析结果写入此结构；测试侧独立声明以避免与 op_host 头重复包含冲突
// （对齐 instance_norm 的 test_instance_norm_tiling.h 做法）。
struct INTrainingReduceV2CompileInfo {
    uint64_t coreNum;      // 系统核数
    uint64_t ubSize;       // UB 空间
    uint32_t vectorLength; // 256
    uint64_t ubBlockSize;  // 32B，UB 的字节对齐单位
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_IN_TRAINING_REDUCE_V2_H_
