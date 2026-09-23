/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// norm/l2_normalize/op_kernel/arch35/l2_normalize_empty.h
// =============================================================================
//
// ROLE: 空 tensor 模板 kernel 类（TPL_SEL_1，isGroup=0 / isEmptyTensor=1）——
//   EMPTY_A / EMPTY_R 合一全核早退。
// 判定结论：y.shape = x.shape，因此任一维为 0 时输出恒为同维空 tensor。
// EMPTY_A 与 EMPTY_R 共用同一早退：usedCoreNum = 0，SetBlockDim(1) 单核启动，
// 唯一 block 进入即 return。
//
// 零开销清单（范式 [5.5]/[5.8]/[5.10]）：零计算、零 GM IO（不绑 x/y）、
//   零 TBuf 分配（UB 占用 0）、零同步（不 FetchEventID、不 SyncAll）。
//   L2NormalizeEmptyTilingData 的 aTotal / aUbFactor / aBigCoreCnt /
//   aBigCoreLoopCnt / aSmallCoreLoopCnt / postBufSize 字段仅保留范式布局一致性，
//   早退路径不消费。
//
// 输出 tensor 由框架按 shape（含 0 维）预分配，kernel 无需写任何数据
//   （空 tensor 的内容集合为空）。
// =============================================================================

#ifndef OPS_NORM_L2_NORMALIZE_EMPTY_H_
#define OPS_NORM_L2_NORMALIZE_EMPTY_H_

#include "kernel_operator.h"            // Ascend C kernel framework
#include "l2_normalize_tiling_struct.h" // L2NormalizeEmptyTilingData

namespace NsL2Normalize {

using namespace AscendC;

// ════════════════════════════════════════════════════════════════════════════
// L2NormalizeEmptyKernel — 空 tensor 模板 kernel 类（TPL_SEL_1）
//   fp16 / fp32 各 1 binary（DTYPE_X 编译期实例化，空 tensor 语义与 dtype 无关）。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
class L2NormalizeEmptyKernel {
public:
    __aicore__ inline L2NormalizeEmptyKernel() {}

    // Empty 初始化：仅缓存 TilingData 与 TPipe（零 IO：不绑 x/y GM；零 buffer：
    // 不 InitBuffer；零同步：不 FetchEventID）——与范式 EMPTY_A 一致
    // 初始化阶段不绑定 GM，也不申请 TBuf 或事件。
    __aicore__ inline void Init(const L2NormalizeEmptyTilingData* td, TPipe* pipe)
    {
        td_ = td;
        pipe_ = pipe;
    }

    // EMPTY_A / EMPTY_R 合一早退：usedCoreNum=0 → 所有核进入即返回
    //   （零计算零 GM IO；无 EMPTY_R 后续路径——范式 Duplicate 固化值 /
    //   CopyIn_post / PostElewise / CopyOut 对本算子全部 N/A，输出恒为空 tensor）
    __aicore__ inline void Process()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
            return; // usedCoreNum=0 → 全核早退，零计算零 IO
        }
    }

private:
    // ─── tilingdata / pipe（早退路径仅缓存，不消费切分字段）───
    const L2NormalizeEmptyTilingData* td_ = nullptr;
    TPipe* pipe_ = nullptr;
};

} // namespace NsL2Normalize

#endif // OPS_NORM_L2_NORMALIZE_EMPTY_H_
