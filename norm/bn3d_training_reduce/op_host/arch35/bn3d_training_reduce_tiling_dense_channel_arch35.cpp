/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file arch35/bn3d_training_reduce_tiling_dense_channel_arch35.cpp
 * \brief DENSE_CHANNEL 路线：按通道分核，通道独占，无跨核归并。
 */
#include <vector>
#include <algorithm>
#include <limits>
#include "bn3d_training_reduce_tiling.h"

using namespace ge;

namespace optiling {
// 每通道归约成 1 个标量（storage NCDHW / NCHW）。
constexpr uint64_t TILINGKEY_DENSE_CHANNEL = 100000;
// 每通道归约成 C0 个标量（storage NDC1HWC0）。搬运模型与上者同构，只是收尾方式不同。
constexpr uint64_t TILINGKEY_NDC1HWC0_CHANNEL = 200000;
// 低通道、超大归约轴：每个通道由多核分段归约，通过 workspace 确定性合并部分和。
constexpr uint64_t TILINGKEY_SPLIT_REDUCE = 300000;

constexpr int64_t RESERVE_FOR_ALIGN = 512;
constexpr int64_t FP32_BYTE = 4;
constexpr int64_t B16_BYTE = 2;
constexpr int64_t NUM_2 = 2;
// 每轮在 UB 暂存的通道结果数上限：512 * 4B = 2KB，双输出双 buffer 共 8KB，
// 既避免 C 很大时输出缓存吃满 UB，又能把逐通道的 4B 小写回合并成一次搬运。
constexpr int64_t C_ROUND_CAP = 512;
// Kernel accBuf_ 实际申请的槽数：每个累加槽占 2 个 VL 宽（Σx 与 Σx² 各一），
// 再多留一个 VL 宽的保护槽——C0 折叠要从 accUb + VL_FP32 + k * C0 处全宽载入，
// 最远读到 3 * VL_FP32 - C0。总槽数 = 2 * numAccSlots + 1。
// 必须与 Kernel 侧 AccBufSlotNum() 保持一致。
constexpr int64_t ACC_SLOTS_PER_ACCUM = 2;
constexpr int64_t ACC_GUARD_SLOT_NUM = 1;
// 每个累加槽还配一份等宽的 Kahan 补偿量（accBuf_ 里累加器区与补偿区各占一半）。
// 必须与 Kernel 侧 NUM_ACC_AND_COMP 一致。
constexpr int64_t NUM_ACC_AND_COMP = 2;
// 多累加槽上限。再大收益递减（链长降到 T/S 后 log2(S) 的归并项开始占比），
// 且 accBuf_ 会挤占输入缓存把 nTile 压小、反过来伤性能。
constexpr int64_t ACC_SLOTS_MAX = 32;
// accBuf_ 允许占用的 UB 上限（1/16）。超过就降 S——输入缓存优先，
// 因为 nTile 变小会直接增加 DataCopyPad 次数。
constexpr int64_t ACC_BUDGET_DIVISOR = 16;
// DataCopyExtParams 字段位宽约束。
constexpr int64_t BLOCK_COUNT_MAX = 65535;       // blockCount: uint16_t
constexpr int64_t UINT32_MAX_VALUE = 4294967295; // srcStride: uint32_t
constexpr int64_t SPLIT_REDUCE_MAX_CHANNELS = 4;
constexpr int64_t SPLIT_REDUCE_MIN_ELEMS_PER_CHANNEL = 1LL << 20;
constexpr int64_t SPLIT_REDUCE_CORE_ALIGN = 8; // 8 个 FP32 partial 恰为一个 32B 搬运块

bool BN3DTrainingReduceDenseChannelTiling::IsCapable()
{
    // C == 0 由 DoOpTiling 走 no-work 分支承接，此处也需放行。
    if (isEmptyChannel_) {
        return true;
    }
    return r0_ > 0 && r1_ > 0 && a_ > 0;
}

uint64_t BN3DTrainingReduceDenseChannelTiling::GetTilingKey() const
{
    if (splitReduce_) {
        return TILINGKEY_SPLIT_REDUCE;
    }
    return (c0_ > 0) ? TILINGKEY_NDC1HWC0_CHANNEL : TILINGKEY_DENSE_CHANNEL;
}

// 在 UB 预算内求解 nTile（一次载入的 n 行数）与 sub-R 分块参数。
namespace {
// 本文件的 shape 量均为正 int64。所有派生乘法/对齐也必须显式检查，不能依赖
// 有符号回绕：Tiling 只处理元数据，极端 shape 即使无法实际分配，也能走到这里。
inline bool CheckedMul(int64_t lhs, int64_t rhs, int64_t& out) { return !__builtin_mul_overflow(lhs, rhs, &out); }

inline bool CheckedCeilAlign(int64_t value, int64_t align, int64_t& out)
{
    if (value < 0 || align <= 0) {
        return false;
    }
    const int64_t remainder = value % align;
    if (remainder == 0) {
        out = value;
        return true;
    }
    return !__builtin_add_overflow(value, align - remainder, &out);
}

// 仅用于正数；用商和余数实现，避免常见的 (x + y - 1) / y 在 x 接近 INT64_MAX 时溢出。
inline int64_t CeilDivPositive(int64_t dividend, int64_t divisor)
{
    return dividend / divisor + static_cast<int64_t>(dividend % divisor != 0);
}

inline int64_t Log2Exact(int64_t p)
{
    int64_t n = 0;
    while (p > 1) {
        p /= NUM_2;
        ++n;
    }
    return n;
}
} // namespace

// 选累加槽数：把长度 totalChain 的线性依赖链拆成 numSteps 段轮转累加。
// 折叠后误差量级 ~ totalChain / S + log2(S)，对 S 求极小得 S ≈ sqrt(totalChain)；
// 再夹到 [1, min(numSteps, ACC_SLOTS_MAX)] 并向下取 2 的幂——
//   * 不能超过 numSteps：槽比轮转次数还多，多出来的槽恒为 0，白占 UB；
//   * 取 2 的幂是为了归并阶段能纯两两折叠，不用处理奇数尾。
int64_t BN3DTrainingReduceDenseChannelTiling::PickAccSlots(int64_t numSteps, int64_t totalChain, int64_t ubBudget) const
{
    if (numSteps <= 1 || totalChain <= 1) {
        return 1;
    }
    // 只需要不超过 ACC_SLOTS_MAX 的 2 次幂，无需真的求 sqrt(totalChain)。旧的线性 IntSqrt
    // 在 N 极大但总元素仍可由 int64 表示时会迭代数十亿次，且平方本身可能溢出。
    const int64_t slotLimit = std::min(numSteps, ACC_SLOTS_MAX);
    int64_t want = 1;
    while (want < slotLimit) {
        const int64_t next = want * NUM_2; // next <= 64，平方不会溢出
        if (next > slotLimit || next * next > totalChain) {
            break;
        }
        want = next;
    }
    // accBuf_ 不得超过 UB 的 1/16，否则宁可少切几段也要把输入缓存留给 nTile。
    const int64_t accBudget = ubBudget / ACC_BUDGET_DIVISOR;
    while (want > 1 && AccBytes(want) > accBudget) {
        want /= NUM_2;
    }
    return want;
}

int64_t BN3DTrainingReduceDenseChannelTiling::AccBytes(int64_t accSlots) const
{
    return (ACC_SLOTS_PER_ACCUM * accSlots * NUM_ACC_AND_COMP + ACC_GUARD_SLOT_NUM) * vlfp32_ * FP32_BYTE;
}

// 两趟求解：先按单槽解出 UB 切分，据此算出轮转次数与总链长、选定槽数，
// 再用放大后的 accBuf_ 复解一次。复解若把路径变差（全载退化成 sub-R，或 nTile 变小
// ——两者都会增加 DataCopyPad 次数），就把 S 减半重试，直到不变差为止。
// 精度是相对竞品的比值指标，性能是绝对指标，冲突时性能优先。
bool BN3DTrainingReduceDenseChannelTiling::SolveUbSplit(int64_t r0Align, int64_t elemSize, int64_t ubBudget)
{
    if (!SolveUbSplitWithSlots(r0Align, elemSize, ubBudget, 1)) {
        return false;
    }
    const uint64_t baseIsSubR = td_.isSubR;
    const uint64_t baseNTile = td_.nTile;

    const int64_t chunksPerCall = (td_.isSubR == 0) ? CeilDivPositive(r0_, vlfp32_) :
                                                      CeilDivPositive(static_cast<int64_t>(td_.r0Factor), vlfp32_);
    // numSteps：每通道调用累加 VF 的次数，也就是槽轮转能切开的段数上限。
    int64_t numSteps = 0;
    if (td_.isSubR == 0) {
        numSteps = CeilDivPositive(r1_, static_cast<int64_t>(td_.nTile));
    } else if (!CheckedMul(r1_, static_cast<int64_t>(td_.numChunks), numSteps)) {
        // 该值只参与槽数启发式；溢出时按最大链长处理即可，不得让 Host 回绕。
        numSteps = std::numeric_limits<int64_t>::max();
    }
    // 单次调用内部本身还有一条 rows * chunks 的链，轮转切不开它，故一并计入总链长。
    const int64_t rowsPerCall = (td_.isSubR == 0) ? static_cast<int64_t>(td_.nTile) : 1;
    int64_t totalChain = 0;
    if (!CheckedMul(numSteps, rowsPerCall, totalChain) || !CheckedMul(totalChain, chunksPerCall, totalChain)) {
        // PickAccSlots 在 totalChain >= ACC_SLOTS_MAX^2 后结果已饱和，故安全饱和不改变选择。
        totalChain = std::numeric_limits<int64_t>::max();
    }

    int64_t accSlots = PickAccSlots(numSteps, totalChain, ubBudget);
    while (accSlots > 1) {
        if (SolveUbSplitWithSlots(r0Align, elemSize, ubBudget, accSlots) && td_.isSubR == baseIsSubR &&
            td_.nTile == baseNTile) {
            break;
        }
        accSlots /= NUM_2;
    }
    if (accSlots <= 1) {
        // 退回单槽：必须重解一次，否则 td_ 里留着上一轮失败/变差的切分。
        accSlots = 1;
        if (!SolveUbSplitWithSlots(r0Align, elemSize, ubBudget, 1)) {
            return false;
        }
    }
    td_.numAccSlots = static_cast<uint64_t>(accSlots);
    td_.foldPasses = static_cast<uint64_t>(Log2Exact(accSlots));
    return true;
}

bool BN3DTrainingReduceDenseChannelTiling::SolveUbSplitWithSlots(int64_t r0Align, int64_t elemSize, int64_t ubBudget,
                                                                 int64_t accSlots)
{
    // 每通道产出的 fp32 个数：NCDHW/NCHW 为 1，NDC1HWC0 为 C0。
    const int64_t outPerChannel = (c0_ > 0) ? c0_ : 1;
    // C_ROUND_CAP 约束的是每轮输出缓存的字节数（512 * 4B = 2KB），不是通道数本身；
    // C0 打包时单通道就占 C0 个 fp32，故按 outPerChannel 折算通道数上限，
    // 避免 C0 打包下输出缓存被放大 C0 倍而吃满 UB。
    const int64_t cRoundCap = std::max<int64_t>(C_ROUND_CAP / outPerChannel, 1);
    const int64_t cRound = std::min(static_cast<int64_t>(td_.cPerCore), cRoundCap);
    // 输出：sum + square_sum，各双 buffer，按 32B 对齐。
    int64_t outBytesRaw = 0;
    int64_t outBytesAligned = 0;
    int64_t outBytes = 0;
    if (!CheckedMul(cRound, outPerChannel, outBytesRaw) || !CheckedMul(outBytesRaw, FP32_BYTE, outBytesRaw) ||
        !CheckedCeilAlign(outBytesRaw, ubBlockSize_, outBytesAligned) ||
        !CheckedMul(outBytesAligned, NUM_2 * NUM_2, outBytes)) {
        return false;
    }
    // Kernel 侧 accBuf_：numAccSlots 个累加槽（每槽 Σx / Σx² 各一个 VL 宽，跨 tile 常驻 UB）
    // + 等量的 Kahan 补偿量 + 一个保护槽。
    // 槽数在下面两趟求解里定：先按单槽解出轮转次数，再据此选 S 复解。
    const int64_t accBytes = AccBytes(accSlots);
    if (outBytes > ubBudget || accBytes > ubBudget - outBytes) {
        return false;
    }
    const int64_t inputBudget = ubBudget - outBytes - accBytes;

    td_.cRound = static_cast<uint64_t>(cRound);

    // 单行字节数（双 buffer 计入）。
    int64_t rowBytesDoubleBuf = 0;
    if (!CheckedMul(r0Align, elemSize, rowBytesDoubleBuf) || !CheckedMul(rowBytesDoubleBuf, NUM_2, rowBytesDoubleBuf) ||
        rowBytesDoubleBuf <= 0) {
        return false;
    }
    int64_t nTile = inputBudget / rowBytesDoubleBuf;

    if (nTile >= 1) {
        // 全载路径会把 r0Align/numR0 交给 uint32_t Kernel 参数，blockLen 也按 uint32_t 字节数下发。
        if (r0Align > UINT32_MAX_VALUE || r0_ > UINT32_MAX_VALUE / elemSize) {
            return false;
        }
        // R0 全载：一次 DataCopyPad 搬 nTile 个 n 行（跨 N 用 srcStride 跳过其他通道）。
        nTile = std::min(nTile, r1_);
        // DataCopyExtParams.blockCount 是 uint16_t。
        nTile = std::min(nTile, BLOCK_COUNT_MAX);
        // DataCopyExtParams.srcStride 是 uint32_t：同一通道跨 N 的字节间隔超出该范围时，
        // 无法用一次多 block 搬运表达，退回单行搬运（此时 srcStride 不参与寻址）。
        int64_t srcStrideElements = 0;
        int64_t srcStrideBytes = 0;
        if (!CheckedMul(a_ - 1, r0_, srcStrideElements) || !CheckedMul(srcStrideElements, elemSize, srcStrideBytes) ||
            srcStrideBytes > UINT32_MAX_VALUE) {
            nTile = 1;
        }
        td_.nTile = static_cast<uint64_t>(nTile);
        td_.isSubR = 0;
        td_.r0Factor = 0;
        td_.numChunks = 0;
        td_.tailLen = 0;
        return true;
    }

    // 单行 R0 都放不下 → sub-R 分块，nTile 恒为 1，按 r0Factor 逐块搬入并累加。
    int64_t r0Factor = inputBudget / (elemSize * NUM_2);
    r0Factor = r0Factor / vlfp32_ * vlfp32_; // 向下对齐到 VL_FP32
    if (r0Factor < vlfp32_) {
        return false; // 连一个 VL 都放不下，UB 预算不可行
    }
    if (r0Factor > r0Align) {
        r0Factor = r0Align;
    }
    // sub-R 的 r0Factor/tailLen 在 Kernel 内为 uint32_t，DataCopy blockLen 也是 uint32_t 字节数。
    if (r0Factor > UINT32_MAX_VALUE / elemSize) {
        return false;
    }
    const int64_t numChunks = CeilDivPositive(r0_, r0Factor);
    // Kernel 的循环计数成员是 uint32_t；TilingData 虽为 uint64_t，也不能静默窄化。
    if (numChunks > UINT32_MAX_VALUE) {
        return false;
    }
    int64_t processedBeforeTail = 0;
    if (!CheckedMul(numChunks - 1, r0Factor, processedBeforeTail)) {
        return false;
    }
    const int64_t tailLen = r0_ - processedBeforeTail;
    if (tailLen <= 0 || tailLen > UINT32_MAX_VALUE / elemSize) {
        return false;
    }
    td_.nTile = 1;
    td_.isSubR = 1;
    td_.r0Factor = static_cast<uint64_t>(r0Factor);
    td_.numChunks = static_cast<uint64_t>(numChunks);
    td_.tailLen = static_cast<uint64_t>(tailLen);
    return true;
}

ge::graphStatus BN3DTrainingReduceDenseChannelTiling::SetEmptyTilingData()
{
    // C == 0：两个输出为空，不做任何归约。沿用仓内 no-work 写法——blockDim 置 1，
    // 由 TilingData 的 usedCoreNum == 0 让 Kernel 立即返回，不使用未经验证的 blockDim = 0。
    OP_LOGW(context_->GetNodeName(),
            "BN3DTrainingReduce: the C-dimension is 0, empty outputs are returned without launching reduction.");
    td_.numN = static_cast<uint64_t>(r1_ < 0 ? 0 : r1_);
    td_.numC = 0;
    td_.usedCoreNum = 0;
    // Kernel 会用 numAccSlots 算槽步长，即使这里立即返回也不能留 0。
    td_.numAccSlots = 1;
    td_.foldPasses = 0;
    blockNum_ = 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BN3DTrainingReduceDenseChannelTiling::ValidateAndNormalizeShape()
{
    // C0 打包路线对 C0 有两条约束，缺一不可：
    //
    //   (1) C0 | VL_FP32 —— Kernel 用 VL 宽累加器，lane L 恒对应 c0 = L % C0，该映射成立的
    //       充要条件就是 C0 整除 VL_FP32（此时 r0Align 与 r0Factor 作为 VL_FP32 的整数倍
    //       也自动是 C0 的整数倍，跨行 / 跨分块的 lane 映射不会错位）。
    //
    //   (2) C0 * sizeof(float) 是 UB block（32B）的整数倍 —— 收尾的 C0 折叠里，
    //       载入偏移是 k * C0、写回偏移是 slot * C0（单位都是 fp32），二者都只按 C0 递进。
    //       C0 < ubBlockSize/sizeof(float)（即 8）时这些偏移落不到 32B 边界上，向量访存
    //       非对齐，实测在真机上直接 VEC_ERROR。只校验 (1) 会放行 C0 ∈ {1,2,4}：这些形状
    //       Host 侧算得出 tiling，Kernel 却必然崩——Host 契约不得放行 Kernel 执行不了的输入。
    //
    // 两条合起来把 C0 限定为 {8, 16, 32, 64}，是 GE 实际会产生的取值（canndev
    // GetCubeSizeByDataType：1 字节 dtype 为 32，其余为 kCubeSize=16）的超集，不裁剪真实支持面。
    // vlfp32_ / ubBlockSize_ 都来自平台信息，故校验放在此处——GetShapeAttrsInfo 早于 GetPlatformInfo 执行。
    // 整个校验必须裹在 c0_ > 0 里：c0_ == 0 表示 channel-first（NCDHW / NCHW），
    // 此时下面两个取模的除数就是 0。原写法靠 `c0_ > 0 &&` 的短路来保护，若把条件
    // 提成先行求值的局部变量，短路即失效 —— NCDHW 会在此处整数除零（SIGFPE）。
    if (c0_ > 0) {
        const bool c0DividesLane = (vlfp32_ > 0 && vlfp32_ % c0_ == 0);
        int64_t c0Bytes = 0;
        const bool c0BlockAligned = (c0DividesLane && CheckedMul(c0_, FP32_BYTE, c0Bytes) && ubBlockSize_ > 0 &&
                                     c0Bytes % ubBlockSize_ == 0);
        OP_CHECK_IF(!(c0DividesLane && c0BlockAligned),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "x", Ops::Base::ToString(xStorageShape_).c_str(),
                        "The C0-dimension of input x must divide the FLOAT32 lane count of one vector instruction "
                        "and occupy a whole number of UB blocks when the format of x is NDC1HWC0"),
                    return ge::GRAPH_FAILED);
    }

    // ── A == 1：把 R1 折进 R0，消除单 lane 串行累加 ──────────────────────────
    // GM 下标 idx(r1, a, r0) = r1 * (A * R0) + a * R0 + r0。A == 1 时退化为 r1 * R0 + r0，
    // 即整个张量就是该唯一通道的一整条连续数据，把 R1 折进 R0 与原布局逐元素等价，
    // 折叠后 R1' = 1、R0' = R1 * R0，归约集合一字不差。
    //
    // 这不是性能优化，是精度必需：Kernel 的 AccumulateRows 按"行"推进，每行发
    // ceil(R0 / VL_FP32) 次向量加，R0 < VL_FP32 时一次向量加只有 R0 个 lane 有效。
    // R0 == 1 就退化成 lane 0 上的串行标量累加，累加链长 = R1。fp32 全正序列串行累加的
    // 相对误差随链长线性增长（~ eps * R1 / 2），R1 到百万量级必然超过 stat_rel_err 的
    // 2^-13 阈值 —— 实测 (3900000, 1, 1) 的 Σx² 相对误差 2.7e-3，是阈值的 22 倍，
    // 而同一份数据按 VL 满宽累加只有 7.5e-6。折叠后走 sub-R 满宽分块，累加链缩短到
    // R1 * R0 / VL_FP32，误差回到阈值以下；顺带把向量指令数降到 1/VL_FP32。
    //
    // C0 打包路线同样安全：R0 = H * W * C0 是 C0 的整数倍，折叠后 R0' = R1 * R0 仍是
    // C0 的整数倍；又有 C0 | VL_FP32（本函数上方已校验），故 lane L ↔ c0 = L % C0 的
    // 映射在跨分块推进时保持不变。
    //
    // A > 1 时通道数据在 GM 上不连续，不能做该布局折叠；普通多通道路径由 Kernel 的
    // 补偿累加策略保证精度，不因此缩小支持范围。
    if (a_ == 1 && r1_ > 1) {
        OP_CHECK_IF(!CheckedMul(r1_, r0_, r0_),
                    OP_LOGE(context_->GetNodeName(), "Failed to fold R1 into R0 without int64 overflow"),
                    return ge::GRAPH_FAILED);
        r1_ = 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BN3DTrainingReduceDenseChannelTiling::BuildDenseTilingData()
{
    OP_CHECK_IF(vlfp32_ <= 0 || ubBlockSize_ <= 0 || aicoreParams_.blockDim == 0 ||
                    aicoreParams_.blockDim > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
                    aicoreParams_.ubSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
                    aicoreParams_.ubSize <= static_cast<uint64_t>(RESERVE_FOR_ALIGN),
                OP_LOGE(context_->GetNodeName(), "Invalid platform tiling parameters"), return ge::GRAPH_FAILED);

    const int64_t elemSize = (dataType_ == ge::DT_FLOAT) ? FP32_BYTE : B16_BYTE;
    // UB 内行步长对齐到 VL_FP32：保证 VF 每次取满一个向量寄存器时不跨行，
    // 尾段无效 lane 由 mask 清零，不会污染有效通道。
    int64_t r0Align = 0;
    OP_CHECK_IF(!CheckedCeilAlign(r0_, vlfp32_, r0Align),
                OP_LOGE(context_->GetNodeName(), "Failed to align R0 without int64 overflow, r0: %ld", r0_),
                return ge::GRAPH_FAILED);

    // 默认按通道分核：一个通道只由一个核完成，无跨核归并。
    const int64_t cPerCore = CeilDivPositive(a_, static_cast<int64_t>(aicoreParams_.blockDim));
    const int64_t usedCoreNum = CeilDivPositive(a_, cPerCore);

    td_.numN = static_cast<uint64_t>(r1_);
    td_.numC = static_cast<uint64_t>(a_);
    td_.numR0 = static_cast<uint64_t>(r0_);
    td_.r0Align = static_cast<uint64_t>(r0Align);
    td_.usedCoreNum = static_cast<uint64_t>(usedCoreNum);
    td_.cPerCore = static_cast<uint64_t>(cPerCore);
    td_.numC0 = static_cast<uint64_t>(c0_ > 0 ? c0_ : 0);

    const int64_t ubBudget = static_cast<int64_t>(aicoreParams_.ubSize) - RESERVE_FOR_ALIGN;
    OP_CHECK_IF(!SolveUbSplit(r0Align, elemSize, ubBudget),
                OP_LOGE(context_->GetNodeName(),
                        "Failed to solve UB split, r0: %ld, r0Align: %ld, elemSize: %ld, ubBudget: %ld", r0_, r0Align,
                        elemSize, ubBudget),
                return ge::GRAPH_FAILED);

    blockNum_ = usedCoreNum;
    int64_t elemsPerChannel = 0;
    const bool canSplitReduce = c0_ == 0 && a_ <= SPLIT_REDUCE_MAX_CHANNELS && td_.isSubR != 0 &&
                                CheckedMul(r1_, r0_, elemsPerChannel) &&
                                elemsPerChannel >= SPLIT_REDUCE_MIN_ELEMS_PER_CHANNEL &&
                                static_cast<int64_t>(aicoreParams_.blockDim) >= a_ * SPLIT_REDUCE_CORE_ALIGN;
    if (canSplitReduce) {
        // 每通道平分尽可能多的核，并向下对齐到 8 核。根核按连续 FP32 搬入 partial，
        // 32B 对齐可避免 DMA 尾块把下一通道 partial 带入横向归约。
        const int64_t rawCoresPerChannel = std::min(static_cast<int64_t>(aicoreParams_.blockDim) / a_, vlfp32_);
        const int64_t coresPerChannel = rawCoresPerChannel / SPLIT_REDUCE_CORE_ALIGN * SPLIT_REDUCE_CORE_ALIGN;
        OP_CHECK_IF(coresPerChannel == 0,
                    OP_LOGE(context_->GetNodeName(), "Insufficient cores for aligned split reduce, raw cores: %ld",
                            rawCoresPerChannel),
                    return ge::GRAPH_FAILED);
        blockNum_ = coresPerChannel * a_;
        td_.usedCoreNum = static_cast<uint64_t>(blockNum_);
        td_.cPerCore = static_cast<uint64_t>(coresPerChannel); // split key 下表示每通道核数
        td_.cRound = static_cast<uint64_t>(a_);                // core 0 一次清零全部输出
        // 每核只处理原链的 1/coresPerChannel，再在根核做一层固定顺序树形合并；
        // 保留多槽只会增加清零/折叠开销，故 split key 固定单槽。
        td_.numAccSlots = 1;
        td_.foldPasses = 0;
        splitReduce_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BN3DTrainingReduceDenseChannelTiling::DoOpTiling()
{
    td_ = BN3DTrainingReduceDenseChannelTilingData{};
    if (isEmptyChannel_) {
        return SetEmptyTilingData();
    }
    if (ValidateAndNormalizeShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return BuildDenseTilingData();
}

ge::graphStatus BN3DTrainingReduceDenseChannelTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    if (splitReduce_) {
        OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Failed to set batch schedule mode for split reduce"),
                    return ge::GRAPH_FAILED);
    }
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    if (splitReduce_) {
        currentWorkspace[0] += static_cast<size_t>(blockNum_) * ACC_SLOTS_PER_ACCUM * sizeof(float);
    }

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(td_), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&td_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(td_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BN3DTrainingReduce, BN3DTrainingReduceDenseChannelTiling,
                             BN3D_TRAINING_REDUCE_DENSE_CHANNEL_PRIORITY);
} // namespace optiling
