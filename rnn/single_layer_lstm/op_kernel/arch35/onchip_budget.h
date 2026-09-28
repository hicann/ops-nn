/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file onchip_budget.h
 * \brief dav_3510 on-chip capacities and the pure-scalar layout arithmetic built on them.
 *
 * Separate from cube_helper.h because op_host must compute the same buffer layout the kernel does --
 * an on-chip overflow is the one failure mode with no error message -- and op_host is compiled by
 * g++ while cube_helper.h includes kernel_operator.h. Two copies of a capacity formula drifting
 * apart is that silent overflow, so the arithmetic lives here: plain C++, includable by both.
 * Nothing here may acquire an AscendC dependency.
 *
 * Shared by rnn/single_layer_lstm and rnn/single_layer_lstm_grad through a cross-operator include.
 */
#ifndef OPS_RNN_SINGLE_LAYER_LSTM_ONCHIP_BUDGET_H
#define OPS_RNN_SINGLE_LAYER_LSTM_ONCHIP_BUDGET_H

#include <cstdint>

/* 命名空间用算子全名。它们原先叫 ch / vh / sh (cube / vector / sync helper)、fwd、proj --
 * 按范式取的短名, 读起来顺, 但摆在顶层是抢地盘: 同名的 fwd::Layout 在另一份 RNN 实现里也
 * 存在过, 两者一旦被同一次编译收进来就是 struct 重定义, 而报错只会说 "redefinition of
 * struct fwd::Layout", 不会告诉你是哪两个算子撞了。仓内的同类都用全名 (LstmGradRegbase、
 * ThnnFusedLstmCellNS), 这里照办。 */
namespace SingleLayerLstmCube {

/* HOW ONE PIECE OF ARITHMETIC SERVES BOTH THE KERNEL AND THE HOST.
 *
 * A plain function is implicitly __host__ and cannot be called from a __global__ __aicore__ body
 * ("call to __host__ function from __global__ __aicore__ function"), while an __aicore__ function
 * is harmless to merely define during a host compile. The asymmetry is one-directional, so this
 * macro gives each compile the decoration it needs from a SINGLE definition. */
#ifdef __CCE_AICORE__
#define CH_BOTH __aicore__ inline
#else
#define CH_BOTH inline
#endif

constexpr uint32_t CUBE_BLOCK = 16; // the M-side fractal edge -- NOT type dependent
constexpr uint32_t C0_BYTES = 32;   // BYTES. AscendC also has a C0_SIZE = 16 in
                                    // matmul_tiling_base.h which is in ELEMENTS and assumes a
                                    // 2-byte type. Always annotate the unit.
constexpr uint32_t BT_ALIGN = 64;   // bias-table burst granularity

/* On-chip capacities for dav_3510, taken from the toolkit's own header rather than from memory:
 * $ASCEND_HOME_PATH/<arch>/include/pto/common/buffer_limits.hpp, PTO_NPU_ARCH_A5 branch.
 * L0C is 256KB on Ascend 950PR but only 128KB on Atlas A2/A3 -- this does not port backwards. */
constexpr uint32_t CAP_L1 = 512 * 1024;
constexpr uint32_t CAP_L0A = 64 * 1024;
constexpr uint32_t CAP_L0B = 64 * 1024;
constexpr uint32_t CAP_L0C = 256 * 1024;
constexpr uint32_t CAP_BT = 4 * 1024;
/* Per AIV, and there are two of them. UB has two figures in circulation and both are correct for
 * different purposes: the physical array is 256KB, the programmable budget is 248KB. Every budget
 * formula here uses 248 -- budgeting at 248 on 256KB hardware is safe, the reverse is not. */
constexpr uint32_t CAP_UB = 248 * 1024;

/* Elements of fp32 in one vector repeat. `Compare` / `Compares` / `CompareScalar` are the only
 * level-2 vector APIs whose contract requires `count * sizeof(T)` to be a whole number of repeats
 * (kernel_operator_vec_cmpsel_intf_impl.h); every other op used here takes an arbitrary count. It
 * lives here, next to the other capacities, because the HOST needs it too: a UB plan that holds a
 * compare's operand must reserve the rounded-up length, not the logical one. */
constexpr uint32_t CMP_REPEAT_ELEMS = 256U / sizeof(float);

CH_BOTH uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

CH_BOTH uint32_t CeilAlign(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

/* Bias-table footprint, in fp32 ELEMENTS. Its instruction's burst is 64-byte granular, so the L1
 * tile and the BT slot are both allocated at this rounded-up count while the SOURCE read stays
 * exactly n -- see the note at the CopyInFlat call in single_layer_lstm_proj_impl.h.
 *
 * IT LIVES HERE, NOT BESIDE THE BT PLUMBING IN cube_helper.h, for the reason at the top of this
 * file: SingleLayerLstmBudget::PickProjTiling has to count the same L1 bytes the kernel takes, and
 * op_host cannot include cube_helper.h. */
CH_BOTH uint32_t BtElems(uint32_t n)
{
    return CeilAlign(n * static_cast<uint32_t>(sizeof(float)), BT_ALIGN) / static_cast<uint32_t>(sizeof(float));
}

/* C0 in ELEMENTS for T. This is the one place the fractal's type dependence lives. */
template <typename T>
CH_BOTH constexpr uint32_t C0()
{
    return C0_BYTES / sizeof(T);
}

/* Bump -- runtime layout of one on-chip buffer. With runtime shapes the offsets must be computed,
 * and an offset one plane short aliases two buffers silently; a bump allocator makes that
 * unrepresentable, and its high-water mark `cur` is what the capacity check reads, so the check sees
 * the real layout rather than a re-derivation. Every allocation is rounded up to 32 bytes, because
 * the DataCopy family expresses lengths and strides in 32-byte blocks. */
struct Bump {
    uint32_t cur;

    CH_BOTH explicit Bump(uint32_t base = 0) : cur(base) {}

    CH_BOTH uint32_t Take(uint32_t bytes)
    {
        uint32_t o = cur;
        cur += CeilAlign(bytes, C0_BYTES);
        return o;
    }

    template <typename T>
    CH_BOTH uint32_t TakeT(uint32_t elems)
    {
        return Take(elems * static_cast<uint32_t>(sizeof(T)));
    }
};

} // namespace SingleLayerLstmCube

#endif // OPS_RNN_SINGLE_LAYER_LSTM_ONCHIP_BUDGET_H
