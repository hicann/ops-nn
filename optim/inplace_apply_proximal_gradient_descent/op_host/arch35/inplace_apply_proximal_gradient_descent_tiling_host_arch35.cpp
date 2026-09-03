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
 * inplace_apply_proximal_gradient_descent_tiling_host_arch35.cpp
 * =============================================================================
 * Role: DESIGN §9 Host 侧 Tiling 纯公式实现（§9.3 / §9.4 / §9.5 / §9.7）与
 *       DESIGN-BRANCH-0/1/27 §2 的 small-data SB 分支切分与
 *       DESIGN-BRANCH-256/257/283 §2 的 fp32/fp16/bf16-large-data DB 分支切分。
 *
 * 本文件同时被 host 编译链与 tests/ut（same-source）编译，UT 测的就是本实现。
 * 所有函数为纯公式/纯提交逻辑，不读 TilingContext，不读取任何输入 payload。
 *
 * 失败语义（§9.9 失败点表）：bool 返回 false 时，各 out 参数保持调用前取值
 * （CalcUbFactor 的合法输入下界分支除外：对齐后 ubFactor==0 时按 §9.4 行序
 * 先写入 0 再返回 false）；CommitTilingData 在 td/workspace 为空时不做任何
 * 写入也不调 SetBlockDim 回调。
 * =============================================================================
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include "inplace_apply_proximal_gradient_descent_tiling_host_arch35.h"

namespace optiling {

namespace {
// §9.5 多核切分常量（手抄自 DESIGN §9.5）
constexpr int64_t kMinElemsPerCore = 2048; // 候选核数阈值：CeilDiv(dim0, 2048)
// §9.5 common 256B alignment; the dim0=2049 balance tradeoff is UT-gated.
constexpr int64_t kElemAlignFactor = 512;
// DESIGN-BRANCH-0 §2 BUFFER_MODE 阈值：dim0<=1024 → 小数据阈值 SB（BUFFER_MODE=0）
constexpr int64_t kMinSplitThreshold = 1024;
} // namespace

// ======================== §9.3 输入预处理 / shape 校验 ========================

bool CalcDim0(const gert::Shape& shape, int64_t& dim0)
{
    bool hasZero = false;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            return false; // Tiling 时仍未知或非法：不写 dim0
        }
        hasZero = hasZero || (dim == 0);
    }
    if (hasZero) {
        dim0 = 0;
        return true;
    }
    dim0 = 1; // rank-0 空乘积也得到 1
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim0 > INT64_MAX / dim) {
            return false; // 乘法溢出：dim0 停在上一个已完成的部分积
        }
        dim0 *= dim;
    }
    return true;
}

bool ExactShapeEqual(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

bool IsSharedScalarShape(const gert::Shape& shape)
{
    return shape.GetDimNum() == 0 || (shape.GetDimNum() == 1 && shape.GetDim(0) == 1);
}

// ======================== §9.4 UB 切分 ========================

bool CalcUbFactor(uint64_t ubSize, uint8_t bufferMode, int64_t& ubFactor)
{
    constexpr int64_t kReserveBytes = 8192; // UB_RESERVE_BYTES
    constexpr int64_t kAlignElems = 128;    // ALIGN_ELEMS
    if (ubSize > static_cast<uint64_t>(INT64_MAX) || ubSize <= kReserveBytes || bufferMode > 1) {
        return false;
    }
    // §9.4 PERF-5 exception: B16 DB cannot reach 8192 elems within the current UB layout.
    const int64_t bytesPerElemWorst = (bufferMode == 0) ? 26 : 36;
    const int64_t usable = static_cast<int64_t>(ubSize) - kReserveBytes;
    ubFactor = (usable / bytesPerElemWorst / kAlignElems) * kAlignElems;
    return ubFactor > 0;
}

// ======================== §9.5 多核切分（正整数安全算术） ========================

bool CeilDivPositive(int64_t value, int64_t divisor, int64_t& result)
{
    if (value <= 0 || divisor <= 0) {
        return false;
    }
    // 无加法型 CeilDiv：(a+b-1)/b 在 a+b 溢出；本式仅除法与取模
    result = value / divisor + static_cast<int64_t>(value % divisor != 0);
    return true;
}

bool AlignUpPositive(int64_t value, int64_t align, int64_t& result)
{
    if (value <= 0 || align <= 0) {
        return false;
    }
    const int64_t rem = value % align;
    if (rem == 0) {
        result = value;
        return true;
    }
    const int64_t add = align - rem;
    if (value > INT64_MAX - add) {
        return false; // 下一 align 倍数超出 int64_t 表示范围
    }
    result = value + add;
    return true;
}

bool CalcLoopTail(int64_t length, int64_t factor, int64_t& loop, int64_t& tail)
{
    if (!CeilDivPositive(length, factor, loop)) {
        return false;
    }
    const int64_t formerLoops = loop - 1;
    // 不变量 (loop-1)*factor < length 的防溢出等价式
    if (formerLoops > (length - 1) / factor) {
        return false;
    }
    tail = length - formerLoops * factor;
    return tail > 0 && tail <= factor;
}

bool CalcMultiCore(int64_t dim0, uint32_t availableCoreNum, int32_t& usedCoreNum, int64_t& blockFactor,
                   int64_t& blockTail)
{
    if (dim0 <= 0 || availableCoreNum == 0) {
        return false;
    }
    int64_t requested = 0;
    if (!CeilDivPositive(dim0, kMinElemsPerCore, requested)) {
        return false;
    }
    // 候选核数：不超过可用核数，也不超过 int32_t 的核数上限
    const int64_t coreCap = std::min<int64_t>(static_cast<int64_t>(availableCoreNum), INT32_MAX);
    const int64_t candidate = std::min<int64_t>(requested, coreCap);
    int64_t rawFactor = 0;
    int64_t actualCoreNum = 0;
    // 候选核 → 512 对齐 blockFactor → 实际核数重算（缺陷反例修复，§9.5）
    if (!CeilDivPositive(dim0, candidate, rawFactor) || !AlignUpPositive(rawFactor, kElemAlignFactor, blockFactor) ||
        !CeilDivPositive(dim0, blockFactor, actualCoreNum) || actualCoreNum > candidate || actualCoreNum > INT32_MAX) {
        return false;
    }
    const int64_t formerCores = actualCoreNum - 1;
    // 不变量 (usedCoreNum-1)*blockFactor < dim0 的防溢出等价式
    if (formerCores > (dim0 - 1) / blockFactor) {
        return false;
    }
    blockTail = dim0 - formerCores * blockFactor;
    if (blockTail <= 0 || blockTail > blockFactor) {
        return false;
    }
    usedCoreNum = static_cast<int32_t>(actualCoreNum);
    return true;
}

// ======================== §9.7 填 TilingData + 设 TilingKey（typed 提交辅助） ========================

bool CommitTilingData(InplaceApplyProximalGradientDescentTilingData* td, size_t* workspace,
                      const SetBlockDimFn& setBlockDim, int64_t dim0, int32_t usedCoreNum, int64_t blockFactor,
                      int64_t blockTail, int64_t ubFactor, int64_t ubLoopOfFormerBlock, int64_t ubTailOfFormerBlock,
                      int64_t ubLoopOfTailBlock, int64_t ubTailOfTailBlock)
{
    // §9.9：typed TilingData / workspace slot 为空 → 无任何副作用即返回 false
    if (td == nullptr || workspace == nullptr) {
        return false;
    }
    td->dim0 = dim0;
    td->usedCoreNum = usedCoreNum;
    td->reserved = 0;
    td->blockFactor = blockFactor;
    td->blockTail = blockTail;
    td->ubFactor = ubFactor;
    td->ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    td->ubTailOfFormerBlock = ubTailOfFormerBlock;
    td->ubLoopOfTailBlock = ubLoopOfTailBlock;
    td->ubTailOfTailBlock = ubTailOfTailBlock;
    *workspace = static_cast<size_t>(0); // §9.6 workspace[0]=0
    if (!setBlockDim) {
        return false;
    }
    return setBlockDim(static_cast<uint32_t>(usedCoreNum));
}

// ======================== DESIGN-BRANCH-0 §2（历史 Branch ID；runtime key 0） ========================

bool ComputeBranch0Tiling(const Branch0TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td)
{
    // §0 进入条件：input 0 datatype=C_DT_FLOAT 且 dim0<=1024（BUFFER_MODE=0）。
    // 负 dim0 或超出本分支阈值范围返回 false 且不写 td。
    if (in.dim0 < 0 || in.dim0 > kMinSplitThreshold) {
        return false;
    }
    // 先全部算入局部量，任意一步失败则 td 保持调用前取值（§9.9 失败点表）
    int32_t usedCoreNum = 1;
    int64_t blockFactor = 0;
    int64_t blockTail = 0;
    int64_t ubFactor = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfTailBlock = 0;
    if (in.dim0 > 0) {
        // §2.2 非空小数据：SB 统一 26 B/elem 预算求 ubFactor（§9.4 同构），候选核→
        // 512 对齐 blockFactor→实际核数重算（§9.5 同构）。dim0<=1024<2048 时
        // candidateCoreNum=1，故恒单核：blockFactor=AlignUp(dim0,512)∈{512,1024}、
        // blockTail=dim0；ubFactor=9344>1024 使 ubLoopOfTailBlock=1、
        // ubTailOfTailBlock=dim0。UB loop/tail 一律读公共 CalcLoopTail。
        if (!CalcUbFactor(in.ubSize, 0, ubFactor) ||
            !CalcMultiCore(in.dim0, in.availableCoreNum, usedCoreNum, blockFactor, blockTail) ||
            !CalcLoopTail(blockFactor, ubFactor, ubLoopOfFormerBlock, ubTailOfFormerBlock) ||
            !CalcLoopTail(blockTail, ubFactor, ubLoopOfTailBlock, ubTailOfTailBlock)) {
            return false;
        }
    }
    // §2.1 empty（dim0==0）：usedCoreNum 保持 1，其余切分字段保持 0；非空逐字段提交
    td.dim0 = in.dim0;
    td.usedCoreNum = usedCoreNum;
    td.reserved = 0;
    td.blockFactor = blockFactor;
    td.blockTail = blockTail;
    td.ubFactor = ubFactor;
    td.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    td.ubTailOfFormerBlock = ubTailOfFormerBlock;
    td.ubLoopOfTailBlock = ubLoopOfTailBlock;
    td.ubTailOfTailBlock = ubTailOfTailBlock;
    return true;
}

// ======================== DESIGN-BRANCH-1 §2（历史 Branch ID；runtime key 0） ========================

bool ComputeBranch1Tiling(const Branch1TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td)
{
    // dtype 由外层 binary 承载；本函数只处理 runtime key 0 的类型无关切分。
    if (in.dim0 < 0 || in.dim0 > kMinSplitThreshold) {
        return false;
    }

    int32_t usedCoreNum = 1;
    int64_t blockFactor = 0;
    int64_t blockTail = 0;
    int64_t ubFactor = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfTailBlock = 0;
    if (in.dim0 > 0) {
        // §2.2：SB 统一按 26 B/elem 预算；dim0<=1024 使候选核恒为 1，
        // blockFactor=AlignUp(dim0,512)，而目标 UB 下 ubFactor=9344。
        if (!CalcUbFactor(in.ubSize, 0, ubFactor) ||
            !CalcMultiCore(in.dim0, in.availableCoreNum, usedCoreNum, blockFactor, blockTail) ||
            !CalcLoopTail(blockFactor, ubFactor, ubLoopOfFormerBlock, ubTailOfFormerBlock) ||
            !CalcLoopTail(blockTail, ubFactor, ubLoopOfTailBlock, ubTailOfTailBlock)) {
            return false;
        }
    }

    // §2.1 empty 保持 usedCoreNum=1、其余切分字段为 0；所有字段写入 §7 POD。
    td.dim0 = in.dim0;
    td.usedCoreNum = usedCoreNum;
    td.reserved = 0;
    td.blockFactor = blockFactor;
    td.blockTail = blockTail;
    td.ubFactor = ubFactor;
    td.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    td.ubTailOfFormerBlock = ubTailOfFormerBlock;
    td.ubLoopOfTailBlock = ubLoopOfTailBlock;
    td.ubTailOfTailBlock = ubTailOfTailBlock;
    return true;
}

// ======================== DESIGN-BRANCH-27 §2（历史 Branch ID；runtime key 0） ========================

bool ComputeBranch27Tiling(const Branch27TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td)
{
    // BF16 由外层 binary 承载；本函数只处理 runtime key 0，不读取 scalar payload。
    if (in.dim0 < 0 || in.dim0 > kMinSplitThreshold) {
        return false;
    }

    int32_t usedCoreNum = 1;
    int64_t blockFactor = 0;
    int64_t blockTail = 0;
    int64_t ubFactor = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfTailBlock = 0;
    if (in.dim0 > 0) {
        // §2.2：dim0<=1024 时候选核恒为 1，blockFactor 为 512/1024；
        // SB 预算得到 ubFactor，随后完整计算 former/tail 的 loop/tail 字段。
        if (!CalcUbFactor(in.ubSize, 0, ubFactor) ||
            !CalcMultiCore(in.dim0, in.availableCoreNum, usedCoreNum, blockFactor, blockTail) ||
            !CalcLoopTail(blockFactor, ubFactor, ubLoopOfFormerBlock, ubTailOfFormerBlock) ||
            !CalcLoopTail(blockTail, ubFactor, ubLoopOfTailBlock, ubTailOfTailBlock)) {
            return false;
        }
    }

    // §2.1 empty：usedCoreNum=1，reserved 与全部 factor/loop/tail 字段为 0。
    td.dim0 = in.dim0;
    td.usedCoreNum = usedCoreNum;
    td.reserved = 0;
    td.blockFactor = blockFactor;
    td.blockTail = blockTail;
    td.ubFactor = ubFactor;
    td.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    td.ubTailOfFormerBlock = ubTailOfFormerBlock;
    td.ubLoopOfTailBlock = ubLoopOfTailBlock;
    td.ubTailOfTailBlock = ubTailOfTailBlock;
    return true;
}

// ======================== DESIGN-BRANCH-256 §2（历史 Branch ID；runtime key 1） ========================

bool ComputeBranch256Tiling(const Branch256TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td)
{
    // FP32 由外层 binary 承载；本函数只处理 runtime key 1 的 dim0 范围。
    if (in.dim0 <= kMinSplitThreshold) {
        return false;
    }

    int32_t usedCoreNum = 0;
    int64_t blockFactor = 0;
    int64_t blockTail = 0;
    int64_t ubFactor = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfTailBlock = 0;

    // §2.1：候选核（每核至少 2048 元素）→512 元素对齐 blockFactor→
    // 重算 usedCoreNum→blockTail。§2.2：模式 1 统一按 36 B/elem、8192B
    // 预留和 128 元素对齐求 ubFactor，再分别计算前核与尾核 loop/tail。
    if (!CalcUbFactor(in.ubSize, 1, ubFactor) ||
        !CalcMultiCore(in.dim0, in.availableCoreNum, usedCoreNum, blockFactor, blockTail) ||
        !CalcLoopTail(blockFactor, ubFactor, ubLoopOfFormerBlock, ubTailOfFormerBlock) ||
        !CalcLoopTail(blockTail, ubFactor, ubLoopOfTailBlock, ubTailOfTailBlock)) {
        return false;
    }

    // 所有公式先落局部量；仅在全链成功后一次写全，失败路径保持 td 不变。
    td.dim0 = in.dim0;
    td.usedCoreNum = usedCoreNum;
    td.reserved = 0;
    td.blockFactor = blockFactor;
    td.blockTail = blockTail;
    td.ubFactor = ubFactor;
    td.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    td.ubTailOfFormerBlock = ubTailOfFormerBlock;
    td.ubLoopOfTailBlock = ubLoopOfTailBlock;
    td.ubTailOfTailBlock = ubTailOfTailBlock;
    return true;
}

// ======================== DESIGN-BRANCH-257 §2（历史 Branch ID；runtime key 1） ========================

bool ComputeBranch257Tiling(const Branch257TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td)
{
    // FP16 由外层 binary 承载；本函数只处理 runtime key 1 的 dim0 范围。
    if (in.dim0 <= kMinSplitThreshold) {
        return false;
    }

    int32_t usedCoreNum = 0;
    int64_t blockFactor = 0;
    int64_t blockTail = 0;
    int64_t ubFactor = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfTailBlock = 0;

    // §2.1：候选核（每核至少 2048 元素）→512 元素对齐 blockFactor→
    // 重算 usedCoreNum→blockTail。§2.2：模式 1 统一按 36 B/elem、8192B
    // 预留和 128 元素对齐求 ubFactor，再分别计算前核与尾核 loop/tail。
    if (!CalcUbFactor(in.ubSize, 1, ubFactor) ||
        !CalcMultiCore(in.dim0, in.availableCoreNum, usedCoreNum, blockFactor, blockTail) ||
        !CalcLoopTail(blockFactor, ubFactor, ubLoopOfFormerBlock, ubTailOfFormerBlock) ||
        !CalcLoopTail(blockTail, ubFactor, ubLoopOfTailBlock, ubTailOfTailBlock)) {
        return false;
    }

    // 所有公式先落局部量；仅在全链成功后一次写全，失败路径保持 td 不变。
    td.dim0 = in.dim0;
    td.usedCoreNum = usedCoreNum;
    td.reserved = 0;
    td.blockFactor = blockFactor;
    td.blockTail = blockTail;
    td.ubFactor = ubFactor;
    td.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    td.ubTailOfFormerBlock = ubTailOfFormerBlock;
    td.ubLoopOfTailBlock = ubLoopOfTailBlock;
    td.ubTailOfTailBlock = ubTailOfTailBlock;
    return true;
}

// ======================== DESIGN-BRANCH-283 §2（历史 Branch ID；runtime key 1） ========================

bool ComputeBranch283Tiling(const Branch283TilingInputs& in, InplaceApplyProximalGradientDescentTilingData& td)
{
    // BF16 由外层 binary 承载；本函数只处理 runtime key 1 的 dim0 范围。
    if (in.dim0 <= kMinSplitThreshold) {
        return false;
    }

    int32_t usedCoreNum = 0;
    int64_t blockFactor = 0;
    int64_t blockTail = 0;
    int64_t ubFactor = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfTailBlock = 0;

    // §2.1：候选核（每核至少 2048 元素）→512 元素对齐 blockFactor→
    // 重算 usedCoreNum→blockTail。§2.2：模式 1 统一按 36 B/elem、8192B
    // 预留和 128 元素对齐求 ubFactor，再分别计算前核与尾核 loop/tail。
    if (!CalcUbFactor(in.ubSize, 1, ubFactor) ||
        !CalcMultiCore(in.dim0, in.availableCoreNum, usedCoreNum, blockFactor, blockTail) ||
        !CalcLoopTail(blockFactor, ubFactor, ubLoopOfFormerBlock, ubTailOfFormerBlock) ||
        !CalcLoopTail(blockTail, ubFactor, ubLoopOfTailBlock, ubTailOfTailBlock)) {
        return false;
    }

    // 所有公式先落局部量；仅在全链成功后一次写全，失败路径保持 td 不变。
    td.dim0 = in.dim0;
    td.usedCoreNum = usedCoreNum;
    td.reserved = 0;
    td.blockFactor = blockFactor;
    td.blockTail = blockTail;
    td.ubFactor = ubFactor;
    td.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    td.ubTailOfFormerBlock = ubTailOfFormerBlock;
    td.ubLoopOfTailBlock = ubLoopOfTailBlock;
    td.ubTailOfTailBlock = ubTailOfTailBlock;
    return true;
}

} // namespace optiling
