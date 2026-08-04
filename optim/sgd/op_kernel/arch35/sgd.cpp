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
 * \file sgd.cpp
 * \brief sgd
 */

#include <type_traits>
#include "kernel_operator.h"
#include "sgd_dag.h"
#include "sgd_tiling_key.h"
#include "sgd_tiling_data.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;
using namespace SgdOp;

namespace {
/**
 * 读 momentum GM 的第 0 个元素并升到 float32。
 *
 * fp32 / fp16：直接 GetValue(0) 后 static_cast<float>。
 * bf16      ：按位解码。理由是编译器限制，不是风格选择 ——
 *             optim/apply_centered_rms_prop/op_kernel/arch35/apply_centered_rms_prop.h:184-185
 *             明确记录「BF16 fallback: bitwise conversion to avoid unsupported LLVM
 *             bf16→fp32 scalar cast（"not support bf16 type cast"）」，而其 fp32 / fp16
 *             分支照常直接 cast。本函数与该先例逐行同形态。
 *             注意受限的只有【标量 cast】：GlobalTensor<bfloat16_t>::GetValue(0) 本身可用，
 *             不需要把 GM 指针 reinterpret_cast 成 __gm__ uint16_t*。
 *
 * 红线：位操作【仅用于解码】，判定发生在解码后的 float 上用 IEEE !=。
 *       -0.0 的 bf16 位模式 0x8000 解码为 -0.0f，而 -0.0f != 0.0f 为 false
 *       → 正确判为 0（走"不回写"分支），与 spec `m32 != np.float32(0.0)` 同口径。
 *       【不做】位模式比较、【不做】floor/ceil、【不降】fp16 —— canndev sgd.py:142-145
 *       那套会让 1e-8 这类极小值在不同芯片上行为相反。
 */
template <typename T>
__aicore__ inline float LoadMomentumScalarF32(GM_ADDR momentumGm)
{
    GlobalTensor<T> momentumGlobal;
    momentumGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(momentumGm));
    T raw = momentumGlobal.GetValue(0);

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        union {
            uint16_t u;
            bfloat16_t b;
        } src;
        union {
            uint32_t u;
            float f;
        } dst;
        src.b = raw;
        dst.u = static_cast<uint32_t>(src.u) << 16; // bf16 → fp32 位扩展，精确无损
        return dst.f;
    } else {
        return static_cast<float>(raw);
    }
}
} // namespace

/**
 * SGD kernel 入口。
 *
 * 7 个业务 GM_ADDR，顺序与图原型一致。图上只声明 1 个 output（parameters）；
 * accum / stat 通过【覆写输入 GM】返回 —— sch.Init 的输出位直接填入 accum / stat
 * 这两个输入地址（先例 optim/apply_ftrl/op_kernel/arch35/apply_ftrl.cpp:33）。
 *
 * 运行期掩码分支：momentum == 0（含 -0.0）时选掩码 DAG，accum / stat 的 GM
 * 【不出现在输出位】→ 零写事务 → 逐位保持原值。两个分支共用同一个 TPipe。
 */
template <uint64_t schMode, uint64_t useNesterov, uint64_t hasWeightDecay, uint64_t hasDampening>
__global__ __aicore__ void sgd(GM_ADDR parameters, GM_ADDR gradient, GM_ADDR learning_rate, GM_ADDR accum,
                               GM_ADDR momentum, GM_ADDR stat, GM_ADDR parameters_out, GM_ADDR workspace,
                               GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SgdRegbaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(SgdRegbaseTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    constexpr bool kUseNesterov = (static_cast<int>(useNesterov) == 1);
    constexpr bool kHasWeightDecay = (static_cast<int>(hasWeightDecay) == 1);
    constexpr bool kHasDampening = (static_cast<int>(hasDampening) == 1);

    // Var 索引必须从 0 起连续无空洞（sch.SetVar 带 static_assert(index < Vars::Size)）：
    // weight_decay 恒占 0；dampening 在无 weight_decay 时让位到 0。
    constexpr int kVarIdxWeightDecay = SGD_VAR_IDX_WEIGHT_DECAY;
    constexpr int kVarIdxDampening = kHasWeightDecay ? 1 : 0;

    using WritebackDag = SgdOp::SgdDag<DTYPE_PARAMETERS, kUseNesterov, kHasWeightDecay, kHasDampening, true>;
    using MaskedDag = SgdOp::SgdDag<DTYPE_PARAMETERS, kUseNesterov, kHasWeightDecay, kHasDampening, false>;

    const float momentumScalar = LoadMomentumScalarF32<DTYPE_PARAMETERS>(momentum);

    TPipe pipe; // 单一 TPipe，两个分支共用

    if (momentumScalar != 0.0f) {
        // 回写分支：三路输出。输出位 2、3 填的是【输入】accum / stat 的 GM 地址。
        ElementwiseSch<schMode, typename WritebackDag::OpDag> sch(&(tilingData.elewiseTiling), &pipe);
        if constexpr (kHasWeightDecay) {
            sch.template SetVar<float, kVarIdxWeightDecay>(tilingData.weightDecay);
        }
        if constexpr (kHasDampening) {
            sch.template SetVar<float, kVarIdxDampening>(tilingData.dampening);
        }
        if constexpr (kHasDampening) {
            sch.Init(parameters, gradient, learning_rate, accum, momentum, stat, parameters_out, accum, stat);
        } else {
            // hasDampening == 0：不读 stat，输入退化为 5 路（In5 不在 DAG 闭包内）。
            // 但 stat 仍要被回写为 0，故输出位保留 3 个。
            sch.Init(parameters, gradient, learning_rate, accum, momentum, parameters_out, accum, stat);
        }
        sch.Process();
    } else {
        // 掩码分支：只回写 parameters。accum / stat 的 GM【从不出现在输出位】。
        // ⛔ 输入侧完全不变 —— accum_t 是 parameters_out 的上游必须照常算，
        //    且 accum 含 ±inf 时 0 * inf = NaN 须按 IEEE 传播进 parameters_out。
        ElementwiseSch<schMode, typename MaskedDag::OpDag> sch(&(tilingData.elewiseTiling), &pipe);
        if constexpr (kHasWeightDecay) {
            sch.template SetVar<float, kVarIdxWeightDecay>(tilingData.weightDecay);
        }
        if constexpr (kHasDampening) {
            sch.template SetVar<float, kVarIdxDampening>(tilingData.dampening);
        }
        if constexpr (kHasDampening) {
            sch.Init(parameters, gradient, learning_rate, accum, momentum, stat, parameters_out);
        } else {
            sch.Init(parameters, gradient, learning_rate, accum, momentum, parameters_out);
        }
        sch.Process();
    }
    return;
}
