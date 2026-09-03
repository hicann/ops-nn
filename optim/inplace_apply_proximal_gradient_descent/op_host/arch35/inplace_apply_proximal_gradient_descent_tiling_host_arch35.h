/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * =============================================================================
 * optim/inplace_apply_proximal_gradient_descent/op_host/arch35/
 * inplace_apply_proximal_gradient_descent_tiling_host_arch35.h
 * =============================================================================
 * Role: DESIGN §9 Host 侧 Tiling 的纯计算公式接口声明。TilingFunc 的
 *       TilingContext glue（§9.9）与 tests/ut 的 gtest 独立 oracle 共用本声明：
 *       UT 以 same-source 方式编译 tiling_host_arch35.cpp，保证 UT 测的就是 Task 25
 *       将实现的真实公式。所有函数为纯公式/纯提交逻辑，不读 context。
 *
 * 函数清单（DESIGN §9.3 / §9.4 / §9.5 / §9.7）：
 *   - CalcDim0 / ExactShapeEqual / IsSharedScalarShape   §9.3 展平与 shape 校验
 *   - CalcUbFactor                                       §9.4 UB 预算（26/36 B/elem）
 *   - CeilDivPositive / AlignUpPositive / CalcLoopTail / CalcMultiCore  §9.5 多核切分
 *   - CommitTilingData                                   §9.7 typed 提交（副作用契约见下）
 *
 * 失败语义约定（§9.9 失败点表）：bool 返回 false 时，各 out 参数保持调用前取值
 * 不变（部分乘积场景见 CalcDim0 注释）；Submit 辅助在 td/workspace 为空时不做
 * 任何写入也不调 SetBlockDim 回调。
 * =============================================================================
 */

#ifndef INPLACE_APPLY_PROXIMAL_GRADIENT_DESCENT_TILING_HOST_ARCH35_H_
#define INPLACE_APPLY_PROXIMAL_GRADIENT_DESCENT_TILING_HOST_ARCH35_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include "exe_graph/runtime/shape.h"
#include "../../op_kernel/arch35/inplace_apply_proximal_gradient_descent_tiling_data.h"

namespace optiling {

// =============================================================================
// §9.3 输入预处理 / 合轴 / 场景预处理
// =============================================================================

// 展平元素总数。rank-0 返回 1；任一维为 0 得 dim0=0（仍 true）；任一维为负
// （Tiling 时仍未知或非法）返回 false 且 dim0 不变；逐维连乘前检查
// dim0 > INT64_MAX / dim，溢出返回 false（此时 dim0 停在上一个已完成的部分积）。
bool CalcDim0(const gert::Shape& shape, int64_t& dim0);

// 逐维完全同形：GetDimNum() 相等且每一维 GetDim(i) 相等。仅 numel 相等不算同形。
bool ExactShapeEqual(const gert::Shape& lhs, const gert::Shape& rhs);

// 标量载体合法 shape：仅 0-D（GetDimNum()==0）或一维 [1]；[1,1]、[0]、[2] 等均 false。
bool IsSharedScalarShape(const gert::Shape& shape);

// =============================================================================
// §9.4 UB 切分
// =============================================================================

// 按 BUFFER_MODE 统一最坏情况预算（0=26 B/elem、1=36 B/elem）计算完整 UB 块
// 有效元素数：ubFactor = AlignDown((ubSize-8192)/bytesPerElemWorst, 128)。
// ubSize<=8192、ubSize>INT64_MAX、bufferMode>1 或对齐后 ubFactor==0 返回 false。
bool CalcUbFactor(uint64_t ubSize, uint8_t bufferMode, int64_t& ubFactor);

// =============================================================================
// §9.5 多核切分（正整数安全算术）
// =============================================================================

// result = CeilDiv(value, divisor)，仅对 value>0 且 divisor>0；否则 false。
bool CeilDivPositive(int64_t value, int64_t divisor, int64_t& result);

// result = 不小于 value 的最小 align 倍数；加法前检查 value <= INT64_MAX-(align-rem)。
bool AlignUpPositive(int64_t value, int64_t align, int64_t& result);

// 对长度 length（blockFactor 或 blockTail）与 factor（ubFactor）求
// loop=CeilDiv(length,factor)、tail=length-(loop-1)*factor，保证 0<tail<=factor。
bool CalcLoopTail(int64_t length, int64_t factor, int64_t& loop, int64_t& tail);

// 候选核 candidateCoreNum=min(availableCoreNum, max(1, CeilDiv(dim0, 2048)))，
// blockFactor=AlignUp(CeilDiv(dim0, candidateCoreNum), 512)，实际核
// usedCoreNum=CeilDiv(dim0, blockFactor)，blockTail=dim0-(usedCoreNum-1)*blockFactor。
// 保证 usedCoreNum<=candidate、0<blockTail<=blockFactor；empty（dim0<=0）或
// availableCoreNum==0 返回 false。维度单位：元素数。
bool CalcMultiCore(int64_t dim0, uint32_t availableCoreNum, int32_t& usedCoreNum, int64_t& blockFactor,
                   int64_t& blockTail);

// =============================================================================
// §9.7 填 TilingData + 设 TilingKey（typed 提交辅助）
// =============================================================================

// SetBlockDim 回调：对应 glue 中的 context->SetBlockDim(usedCoreNum) == GRAPH_SUCCESS。
using SetBlockDimFn = std::function<bool(uint32_t usedCoreNum)>;

// 按 §9.9 顺序提交：先校验 td/workspace 非空（不满足则返回 false 且无任何副作用，
// 包括不调 setBlockDim）；随后写全 10 个 TilingData 字段（含 reserved=0）与
// workspace[0]=0；最后调 setBlockDim(usedCoreNum)，失败返回 false。返回 true
// 仅当全部提交动作成功；调用方（TilingFunc glue）在返回 false 时不得继续执行
// BUFFER_MODE selector 提交。
bool CommitTilingData(InplaceApplyProximalGradientDescentTilingData* td, size_t* workspace,
                      const SetBlockDimFn& setBlockDim, int64_t dim0, int32_t usedCoreNum, int64_t blockFactor,
                      int64_t blockTail, int64_t ubFactor, int64_t ubLoopOfFormerBlock, int64_t ubTailOfFormerBlock,
                      int64_t ubLoopOfTailBlock, int64_t ubTailOfTailBlock);

// =============================================================================
// DESIGN-BRANCH-0 §2 · 历史 Branch ID 0，当前 runtime key 0
//   进入条件（§0）：input 0 datatype=C_DT_FLOAT 且 dim0<=1024（BUFFER_MODE=0）。
//   分支常量（§2.2/§4）：候选核阈值 2048、blockFactor 对齐 512、
//   ubFactor=AlignDown((ubSize-8192)/26,128)（ubSize=253952 时 9344）。
//   公式（§2）：empty 短路 usedCoreNum=1 其余全 0；非空恒单核、
//   blockFactor∈{512,1024}、blockTail=ubTailOfTailBlock=dim0、
//   ubLoopOfTailBlock=1。失败语义同 §9.3–§9.5 out 参数约定：返回 false 时
//   不写 td。
// =============================================================================

// §2 分支输入：dim0=展平元素总数（0=empty；1..1024=非空小数据）、
// availableCoreNum=可用 AIV 核数、ubSize=UB 字节预算。
struct Branch0TilingInputs {
    int64_t dim0;              // 展平后的有效元素总数（元素数）
    uint32_t availableCoreNum; // 可用 AIV 核数（核数）
    uint64_t ubSize;           // UB 字节预算（字节）
};

// 按 DESIGN-BRANCH-0 §2.1（empty）/§2.2（非空小数据）填充 TilingData 的
// 全部 9 个切分字段（dim0/usedCoreNum/reserved/blockFactor/blockTail/ubFactor/
// ubLoopOfFormerBlock/ubTailOfFormerBlock/ubLoopOfTailBlock/ubTailOfTailBlock）。
bool ComputeBranch0Tiling(const Branch0TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td);

// =============================================================================
// DESIGN-BRANCH-1 §2 · 历史 Branch ID 1，当前 runtime key 0
//   进入条件（§0）：input 0 datatype=C_DT_FLOAT16 且 dim0<=1024
//   （BUFFER_MODE=0）。切分常量与 Branch-0 的 SB 统一预算相同；FP16 搬运
//   以 16 元素对齐到 32B。empty 固定 usedCoreNum=1 且其余切分字段全 0；
//   非空恒单核，blockFactor∈{512,1024}、blockTail=dim0，尾核仅一个 tile。
// =============================================================================

struct Branch1TilingInputs {
    int64_t dim0;              // 展平后的有效元素总数（元素数）
    uint32_t availableCoreNum; // 可用 AIV 核数（核数）
    uint64_t ubSize;           // UB 字节预算（字节）
};

// 按 DESIGN-BRANCH-1 §2.1（empty）/§2.2（非空小数据）填充 §7 TilingData。
// 失败时返回 false 且不写 td。
bool ComputeBranch1Tiling(const Branch1TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td);

// =============================================================================
// DESIGN-BRANCH-27 §2 · 历史 Branch ID 27，当前 runtime key 0
//   进入条件（§0）：input 0 datatype=C_DT_BF16 且 dim0<=1024
//   （BUFFER_MODE=0）。切分采用 SB 统一 26 B/elem 预算；empty 固定
//   usedCoreNum=1 且其余切分字段全 0，非空恒单核且尾核仅一个 tile。
// =============================================================================

struct Branch27TilingInputs {
    int64_t dim0;              // 展平后的有效元素总数（元素数）
    uint32_t availableCoreNum; // 可用 AIV 核数（核数）
    uint64_t ubSize;           // UB 字节预算（字节）
};

// 按 DESIGN-BRANCH-27 §2.1（empty）/§2.2（非空小数据）填充 §7 TilingData。
// BF16 由外层 binary 承载；本接口不接收 dtype 编码或 scalar payload。
bool ComputeBranch27Tiling(const Branch27TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td);

// =============================================================================
// DESIGN-BRANCH-256 §2 · 历史 Branch ID 256，当前 runtime key 1
//   进入条件（§0）：input 0 datatype=C_DT_FLOAT 且 dim0>1024
//   （BUFFER_MODE=1）。候选核按每核至少 2048 元素取得，blockFactor 向上对齐
//   512 元素后重算实际核数；DB 统一按 36 B/elem 与 8192B 预留计算 UB tile。
// =============================================================================

struct Branch256TilingInputs {
    int64_t dim0;              // 展平后的有效元素总数（元素数，恒大于 1024）
    uint32_t availableCoreNum; // 可用 AIV 核数（核数）
    uint64_t ubSize;           // UB 字节预算（字节）
};

// 按 DESIGN-BRANCH-256 §2 填充 §7 TilingData。通过 §9 公共 CalcUbFactor(模式 1)、
// CalcMultiCore 与 CalcLoopTail 生产 helper 实现；失败时返回 false 且不写 td。
bool ComputeBranch256Tiling(const Branch256TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td);

// =============================================================================
// DESIGN-BRANCH-257 §2 · 历史 Branch ID 257，当前 runtime key 1
//   进入条件（§0）：input 0 datatype=C_DT_FLOAT16 且 dim0>1024
//   （BUFFER_MODE=1）。候选核按每核至少 2048 元素取得，blockFactor 向上对齐
//   512 元素后重算实际核数；DB 统一按 36 B/elem 与 8192B 预留计算 UB tile。
// =============================================================================

struct Branch257TilingInputs {
    int64_t dim0;              // 展平后的有效元素总数（元素数，恒大于 1024）
    uint32_t availableCoreNum; // 可用 AIV 核数（核数）
    uint64_t ubSize;           // UB 字节预算（字节）
};

// 按 DESIGN-BRANCH-257 §2 填充 §7 TilingData。通过 §9 公共 CalcUbFactor(模式 1)、
// CalcMultiCore 与 CalcLoopTail 生产 helper 实现；失败时返回 false 且不写 td。
bool ComputeBranch257Tiling(const Branch257TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td);

// =============================================================================
// DESIGN-BRANCH-283 §2 · 历史 Branch ID 283，当前 runtime key 1
//   进入条件（§0）：input 0 datatype=C_DT_BF16 且 dim0>1024
//   （BUFFER_MODE=1）。候选核按每核至少 2048 元素取得，blockFactor 向上对齐
//   512 元素后重算实际核数；DB 统一按 36 B/elem 与 8192B 预留计算 UB tile。
// =============================================================================

struct Branch283TilingInputs {
    int64_t dim0;              // 展平后的有效元素总数（元素数，恒大于 1024）
    uint32_t availableCoreNum; // 可用 AIV 核数（核数）
    uint64_t ubSize;           // UB 字节预算（字节）
};

// 按 DESIGN-BRANCH-283 §2 填充 §7 TilingData。通过 §9 公共 CalcUbFactor(模式 1)、
// CalcMultiCore 与 CalcLoopTail 生产 helper 实现；失败时返回 false 且不写 td。
bool ComputeBranch283Tiling(const Branch283TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td);

} // namespace optiling

#endif
