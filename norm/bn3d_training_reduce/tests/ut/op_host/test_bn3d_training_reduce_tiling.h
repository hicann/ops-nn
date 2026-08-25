/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BN3D_TRAINING_REDUCE_H_
#define OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BN3D_TRAINING_REDUCE_H_

#include <cstdint>
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {

// 与 op_host/bn3d_training_reduce_tiling.h 的 BN3DTrainingReduceCompileInfo 布局一致。
// tiling_parse_func 把解析结果写入此结构；测试侧独立声明以避免与 op_host 头重复包含冲突
// （对齐 in_training_reduce_v2 的 test_in_training_reduce_v2_tiling.h 做法）。
struct BN3DTrainingReduceCompileInfo {
    uint64_t coreNum;      // AIV 核数
    uint64_t ubSize;       // UB 空间
    uint32_t vectorLength; // 向量寄存器字节宽度
    uint64_t ubBlockSize;  // 32B，UB 的字节对齐单位
};

// 与 op_kernel/arch35/bn3d_training_reduce_tiling_data.h 的
// BN3DTrainingReduceDenseChannelTilingData 布局一致。用例据此断言字段值，
// 而不是只断言 TilingKey——三条路线（100000 / 200000 / 300000）内部还有 R0 全载 /
// sub-R 分块 / 空通道等分支，仅断言 key 区分不出来。
struct BN3DTrainingReduceDenseChannelTilingData {
    uint64_t numN;
    uint64_t numC;
    uint64_t numR0;
    uint64_t r0Align;
    uint64_t usedCoreNum;
    uint64_t cPerCore;
    uint64_t cRound;
    uint64_t nTile;
    uint64_t isSubR;
    uint64_t r0Factor;
    uint64_t numChunks;
    uint64_t tailLen;
    uint64_t numC0; // 0：NCDHW/NCHW，每通道 1 个标量；>0：NDC1HWC0 的 C0
    uint64_t numAccSlots;
    uint64_t foldPasses;
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BN3D_TRAINING_REDUCE_H_
