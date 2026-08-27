/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_BN3_D_TRAINING_REDUCE_GRAD_OP_HOST_ARCH35_BN3_D_TRAINING_REDUCE_GRAD_TILING_ARCH35_H
#define OPS_MATH_BN3_D_TRAINING_REDUCE_GRAD_OP_HOST_ARCH35_BN3_D_TRAINING_REDUCE_GRAD_TILING_ARCH35_H

// Include the tiling data struct definition — this is shared between
// host-side tiling and device-side kernel (same struct, different compilation).
#include "../../op_kernel/arch35/bn3d_training_reduce_grad_tiling_data.h"

#include <cstdint>

#include "graph/types.h" // ge::DataType / ge::Format（公共 Tiling 纯公式接口入参）

namespace optiling {

// ---------------------------------------------------------------------------
// BN3DTrainingReduceGradCompileInfo — platform information for tiling compile phase
//
// This struct is populated by TilingPrepareForBN3DTrainingReduceGrad() with platform
// hardware information (number of AIV cores, UB memory size). It is then
// passed to the tiling function so it can make decisions about data
// partitioning and parallelism.
//
// Fields:
//   coreNum — number of available AIV (AI Vector) cores on the NPU
//   ubSize  — size of Unified Buffer (UB) in bytes per core
//             UB is the fast on-chip memory used for kernel computation
//
// To create FooBar: rename to FooBarCompileInfo.
//   Most operators need the same fields (coreNum, ubSize).
//   Add fields here if your operator needs additional compile-time info.
// ---------------------------------------------------------------------------
struct BN3DTrainingReduceGradCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

// =============================================================================
// 公共 Tiling 纯公式接口
// =============================================================================
//
// 与 TilingContext 解耦的纯计算接口：TilingFunc 胶水把 gert 上下文读成
// PublicTilingInputs 后调用 ComputePublicTiling；公共 TilingUT
// （tests/tiling/test_tiling_public.cpp）直接调同一函数、以独立 oracle 交叉验证。
//
// 覆盖口径：
//   - 前置校验顺序：dtype → format → 维度 → attr → shape
//   - 通道轴重塑（NCDHW → dim1 / NDHWC → dim4）与 num = N·D·H·W
//   - PadAndSqueeze（补 1 → 去 1 → 归一）与 CheckBroadcastShape
//   - 路由：有效 rank ≤ 4 → mapped=4 / tilingKey=0；= 5 → mapped=8 / tilingKey=1

// 前置校验错误码

enum class PublicTilingError : int32_t {
    kOk = 0,            // 校验通过，进入预处理与路由
    kNullInput = 1,     // T14：空 tensor（grads / x 任一维 == 0）
    kShapeMismatch = 2, // T12（参数长度 != C）/ T13（grads 与 x shape 不一致）/ T15（rank 非法）
    kDtypeError = 3,    // T16（grads 与 x dtype 不一致）/ T17（参数张量非 f32）/ 白名单外 dtype
    kFormatError = 4, // T18：format 非 NCDHW/NDHWC（含 5D FORMAT_ND）或三者不同 format（1-D 参数不校验 format）
    kAttrError = 5, // T19：epsilon <= 0（值域须 > 0）
};

// 公共 Tiling 输入：8 个张量 + attr epsilon
//   张量 index 约定：
//     0 = grads（5D）、1 = x（5D）、2..6 = diff_scale / diff_offset / scale /
//         batch_mean / batch_variance（1D，长度 = C）、7 = y（5D）
struct PublicTilingInputs {
    int32_t ranks[8];       // 各张量 rank：grads/x/y == 5；5 个参数 == 1
    int64_t shapes[8][5];   // 各张量 shape（仅前 ranks[i] 维有效）
    ge::DataType dtypes[8]; // 组合表口径：grads/x/y 同 dtype ∈ {f16, f32, bf16}；参数 f32
    ge::Format formats[8];  // grads/x/y ∈ {FORMAT_NCDHW, FORMAT_NDHWC}（严格双格式，不做 ND
                            //   参数长度推断）且三者一致；1-D 参数不校验 format
    float epsilon;          // attr epsilon（默认 0.0001；值域须 > 0）
};

// 公共 Tiling 输出：仅 error == kOk 时其余字段有效（错误时无意义、内容未定义）
struct PublicTilingOutput {
    int32_t channelAxis;              // 通道轴：NCDHW → 1 / NDHWC → 4
    int64_t num;                      // num = N·D·H·W
    int32_t rank;                     // PadAndSqueeze 去 1 后有效 rank ∈ [1, 5]
    int64_t maxBroShape[5];           // maximumBroShape（仅前 rank 维有效）
    int32_t mapped;                   // 分档：rank ≤ 4 → 4；rank = 5 → 8
    int32_t tilingKey;                // 路由：0 = RANK_4；1 = RANK_8
    int64_t normalInputShapes[7][5];  // 各输入补 1 → 去 1 → 归一后 shape（仅前 rank 维有效）
    int64_t normalOutputShapes[1][5]; // 输出 y 归一后 shape（仅前 rank 维有效）
};

// 公共 tiling 公式（纯函数，无 TilingContext 依赖）：
//   校验（dtype → format → 维度 → attr → shape）→ 通道轴重塑 → num →
//   PadAndSqueeze → CheckBroadcastShape → 路由（有效 rank ≤ 4 → key 0；= 5 → key 1）
// 返回错误码；kOk 时填 out。
PublicTilingError ComputePublicTiling(const PublicTilingInputs& in, PublicTilingOutput& out);

// =============================================================================
// Branch-0（tilingKey=0 / RANK=4）纯公式接口
// =============================================================================
//
// 与 TilingContext 解耦的纯计算接口：TilingFunc 胶水（DoTilingAndSet<4>）把公共
// Tiling 的预处理结果（归一化 shape / maxBroShape / rank / epsilon / num）与平台量
// （perBufBytes / coreNum）读成 Branch0TilingInputs 后调用 ComputeBranch0Tiling；
// 分支 TilingUT（tests/tiling/test_tiling_branch0.cpp）直接调同一函数、以独立
// oracle 交叉验证。
//
// 覆盖口径：
//   - UB 切分（FindSplitAxis，perBufElems 按 f32 口径 = perBufBytes / 4）
//   - 多核切分（MultiCoreSplit：numCores = min(totalTiles, coreNum) 动态计算）
//   - 字段逐项填充 + RANK=4 前补右移（delta = 4 − rank，split.axis += delta，
//     前 delta 维 shape=1 / stride=0）

// Branch-0 Tiling 输入：公共预处理结果（有效 rank ∈ [1,4]，effective 坐标系）+ 平台量
struct Branch0TilingInputs {
    int32_t rank;                     // 有效 rank ∈ [1,4]
    int64_t maxBroShape[4];           // maximumBroShape（仅前 rank 维有效）
    int64_t normalInputShapes[7][4];  // 各输入归一 shape（仅前 rank 维有效）
    int64_t normalOutputShapes[1][4]; // 输出 y 归一 shape（仅前 rank 维有效）
    int64_t perBufBytes;              // per_buf_bytes = (ubSize/P) & ~31
    int64_t coreNum;                  // GetCoreNumAiv()
    float epsilon;                    // attr epsilon
    int64_t num;                      // num = N·D·H·W
};

// 分支 tiling 纯公式（无 TilingContext 依赖，无失败路径——输入已由公共校验
//   保证非空合法）：FindSplitAxis → MultiCoreSplit → stride 预计算（广播轴 stride=0）

void ComputeBranch0Tiling(const Branch0TilingInputs& in, BN3DTrainingReduceGradTilingData<4>& out);

// =============================================================================
// Branch-1（tilingKey=1 / RANK=8）纯公式接口
// =============================================================================
//
// 与 TilingContext 解耦的纯计算接口：TilingFunc 胶水（DoTilingAndSet<8>）把公共
// Tiling 的预处理结果（归一化 shape / maxBroShape / rank / epsilon / num）与平台量
// （perBufBytes / coreNum）读成 Branch1TilingInputs 后调用 ComputeBranch1Tiling；
// 分支 TilingUT（tests/tiling/test_tiling_branch1.cpp）直接调同一函数、以独立
// oracle 交叉验证。
//
// 覆盖口径：
//   - UB 切分（FindSplitAxis，perBufElems 按 f32 口径 = perBufBytes / 4）
//   - 多核切分（MultiCoreSplit：numCores = min(totalTiles, coreNum) 动态计算）
//   - 字段逐项填充 + RANK=8 前补右移（delta = 8 − rank = 3，split.axis += 3、
//     axis ∈ [3,7]，前 delta 维 shape=1 / stride=0）

// Branch-1 Tiling 输入：公共预处理结果（有效 rank = 5，effective 坐标系）+ 平台量
struct Branch1TilingInputs {
    int32_t rank;                     // 有效 rank
    int64_t maxBroShape[8];           // maximumBroShape（仅前 rank 维有效）
    int64_t normalInputShapes[7][8];  // 各输入归一 shape（仅前 rank 维有效）
    int64_t normalOutputShapes[1][8]; // 输出 y 归一 shape（仅前 rank 维有效）
    int64_t perBufBytes;              // per_buf_bytes = (ubSize/P) & ~31
    int64_t coreNum;                  // GetCoreNumAiv()
    float epsilon;                    // attr epsilon
    int64_t num;                      // num = N·D·H·W
};

// 分支 tiling 纯公式（无 TilingContext 依赖，无失败路径——输入已由公共校验
//   保证非空合法）：FindSplitAxis → MultiCoreSplit → stride 预计算（广播轴 stride=0）

void ComputeBranch1Tiling(const Branch1TilingInputs& in, BN3DTrainingReduceGradTilingData<8>& out);

} // namespace optiling

#endif
