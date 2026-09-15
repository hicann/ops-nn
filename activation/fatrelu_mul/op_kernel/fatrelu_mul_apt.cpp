/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// fatrelu_mul_package/op_kernel/fatrelu_mul_apt.cpp
// =============================================================================
//
// ROLE: Ascend C kernel entry point for FatreluMul (implemented stage).
//   This file contains the __global__ kernel function that runs on the NPU.
//   Per docs/fatrelu_mul/develop/proto.md §3, the kernel entry signature is:
//
//     template <typename D_T_X, int64_t kPath>
//     __global__ __aicore__ void FatreluMul(
//         GM_ADDR input, GM_ADDR threshold, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
//
//   Entry chain per docs/fatrelu_mul/design/Kernel.md「Kernel 入口」: TPL 分发
//   （D_T_X / kPath 由 ASCENDC_TPL_SEL 实例化，host 侧 ASCENDC_TPL_SEL_PARAM
//   编码 tilingKey 选择 sub-kernel）+
//   TilingData 反序列化（GET_TILING_DATA_WITH_STRUCT，非模板化共用结构体）+
//   核守卫（blockIdx ≥ need_core_num 直接返回，含空 Tensor 短路）+
//   FatreluMulKernel<D_T_X, kPath> 实例化（Init / Process）。
//   The compute chain lives in arch35/FatreluMul_kernel.h（small-tail 行组路径
//   per branches/DESIGN-BRANCH-0.md §3–§5；big-tail per DESIGN-BRANCH-1.md）.
//   The original sample compute logic (ScaleCustom-style broadcast mul-add)
//   is preserved below as a `// [REF_SAMPLE]` comment block.
//
//   NOTE: the OpDef's ExtendCfgInfo("opFile.value", "fatrelu_mul_apt") 按仓内
//   约定（elu / fast_gelu_v2 等）使用 snake_case 算子目录名 + "_apt" 后缀，
//   框架据此定位 op_kernel/fatrelu_mul_apt.cpp 并生成 dynamic/fatrelu_mul_apt.py.
//
// CONTENTS:
//   - GET_TILING_DATA_WITH_STRUCT(FatreluMulTilingData, td, tiling)
//                                        (global non-templated TilingData)
//   - template<typename D_T_X, int64_t kPath> __global__ void FatreluMul(...)  (proto.md kernel entry)
//   - template<typename D_T_X, int64_t kPath> __global__ void fatrelu_mul(...) (build-system adapter entry)
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FatreluMul   — proto.md kernel entry name, kernel class prefix
//   snake_case   : fatrelu_mul  — opInterface name used by the build system
//   UPPER        : FATRELUMUL   — TPL path value macros (FatreluMul_struct.h)
//
// BUILD-SYSTEM NAMING CONTRACT:
//   op_build derives `opInterface.value` from the OpType via ConvertToSnakeCase,
//   i.e. "fatrelu_mul". The auto-generated wrapper (auto_gen_fatrelu_mul_kernel,
//   renamed per tiling key via -Dfatrelu_mul=fatrelu_mul_<tilingKey>_tilingkey)
//   invokes the kernel entry through the identifier `fatrelu_mul<TEMPLATE_PARAMS>`.
//   Therefore the snake_case adapter entry `fatrelu_mul` below forwards to the
//   proto.md-declared entry `FatreluMul`. Both are compiled inline into the
//   generated wrapper TU (__global__ is #undef'd to inline before inclusion).
//
// KEY MACROS AND CONVENTIONS:
//   __global__ __aicore__  — Ascend C attribute for NPU kernel functions
//   GM_ADDR                — global memory address type (pointer to device DRAM)
//   REGISTER_NONE_TILING   — registers with the AICore runtime (tiling data read
//                            manually via GET_TILING_DATA_WITH_STRUCT)
//   KERNEL_TASK_TYPE_DEFAULT — sets the task type (AIV only, pure Vector op)
//   GET_TILING_DATA_WITH_STRUCT(StructType, varName, tilingAddr) — deserializes
//     tiling data from the tiling buffer into the struct (TilingData registration)
//
// =============================================================================

#include "kernel_operator.h"               // Ascend C kernel framework (AscendC:: namespace)
#include "arch35/FatreluMul_struct.h"      // ASCENDC_TPL_* 模板参数（= TilingKey.md）
#include "arch35/FatreluMul_tiling_data.h" // FatreluMulTilingData / kMaxInputSlots / kMaxOutputSlots（= TilingData.md）
#include "arch35/FatreluMul_kernel.h"      // FatreluMulKernel<T, kPath>（= Kernel.md + branches/ 各分支设计）

// FatreluMulTilingData is a single non-templated struct shared by both TPL
// instantiations (kPath): the row model (x as (batch_size, 2*half_dim))
// carries rank 2-8 through the batch_size / half_dim scalars, so no per-rank
// tiling data variants exist (docs/fatrelu_mul/design/TilingData.md).

// ===========================================================================
// template<typename D_T_X, int64_t kPath> __global__ __aicore__ void FatreluMul(...)
//
// The proto.md §3 kernel entry (PascalCase). Each AIV core reaches this
// function through the snake_case adapter entry below.
//
// Template parameters:
//   D_T_X — TPL dtype 参数（ASCENDC_TPL_DATATYPE_DECL，FatreluMul_struct.h），
//           框架按 tilingKey 的 dtype 位选择实例（float / half / bfloat16_t）
//   kPath — TPL-instantiated path selector (FatreluMul_struct.h:
//           FATRELUMUL_PATH_SMALL_TAIL = 0 / FATRELUMUL_PATH_BIG_TAIL = 1，per
//           docs/fatrelu_mul/design/TilingKey.md；host 侧 TilingFunc 按 BranchRoute.md
//           判界 SetTilingKey 选择 sub-kernel).
//
// Parameters (all GM_ADDR = global memory pointers on device):
//   input     — x: gate_up concatenated tensor, last dim 2d (d = half_dim)
//   threshold — FatReLU threshold scalar Tensor (single element)
//   output    — y: y.shape = x.shape[:-1] + (d,)
//   workspace — workspace buffer (framework entry-signature convention only:
//               本算子无 device workspace，不绑定，Kernel.md「Kernel 入口」)
//   tiling    — tiling data buffer (contains FatreluMulTilingData)
//
// Entry chain (docs/fatrelu_mul/design/Kernel.md「Kernel 入口」):
//   REGISTER_NONE_TILING → KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY) →
//   GET_TILING_DATA_WITH_STRUCT(FatreluMulTilingData) → 核守卫（blockIdx ≥
//   need_core_num 直接返回，空 Tensor 短路时 need_core_num=1 / rows_former=0，
//   block 0 进入 Process 后行数 0 空转返回）→ FatreluMulKernel<D_T_X, kPath>
//   Init / Process（D_T_X 来自 TPL DATATYPE 模板参数，由 tilingKey dtype 位选定）.
// ===========================================================================
template <typename D_T_X, int64_t kPath>
__global__ __aicore__ void FatreluMul(GM_ADDR input, GM_ADDR threshold, GM_ADDR output, GM_ADDR workspace,
                                      GM_ADDR tiling)
{
    // 槽位绑定：输入 x / threshold 两路、输出 y 一路（kMaxInputSlots /
    // kMaxOutputSlots，TilingData.md §1；threshold 为单元素标量 Tensor，GM 直读、
    // 不占 UB 槽位）
    GM_ADDR ins[kMaxInputSlots] = {input, threshold};
    GM_ADDR outs[kMaxOutputSlots] = {output};

    // REGISTER_NONE_TILING: registers the kernel with the AICore runtime.
    // "NONE_TILING" means the tiling data is read manually via GET_TILING_DATA_WITH_STRUCT
    // rather than being auto-tiled by the framework. This gives full control over tiling.
    REGISTER_NONE_TILING;

    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY):
    //   FatreluMul is a pure Vector operator — runs on AIV (AI Vector) cores only.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // TilingData 非模板化：双路径共用同一结构体（无 if constexpr 分档）；
    // GET_TILING_DATA_WITH_STRUCT 反序列化 tiling buffer → FatreluMulTilingData td
    GET_TILING_DATA_WITH_STRUCT(FatreluMulTilingData, td, tiling);

    // 核守卫：blockIdx 超出参与计算核数直接返回（need_core_num = SetBlockDim 值，
    // 对齐内置旧代际 kernel 口径，REQUIREMENTS §5.7）
    if (AscendC::GetBlockIdx() >= td.need_core_num) {
        return;
    }

    // D_T_X：TPL DATATYPE 模板参数（FatreluMul_struct.h ASCENDC_TPL_DATATYPE_DECL），
    // 与 def.cpp DataType({DT_BF16, DT_FLOAT16, DT_FLOAT}) 对应，由 host 侧
    // ASCENDC_TPL_SEL_PARAM 编码进 tilingKey 的 dtype 位选定实例
    FatreluMulKernel<D_T_X, kPath> kernel;
    kernel.Init(ins, outs, &td);
    kernel.Process();

    // -----------------------------------------------------------------------
    // [REF_SAMPLE] Original sample entry logic (ScaleCustom-style broadcast
    // mul-add, kept verbatim for reference; superseded by the FatreluMul
    // entry chain above per docs/fatrelu_mul/design/Kernel.md「Kernel 入口」):
    //
    // // Bundle input pointers into an array for the kernel.
    // // Index: 0=x, 1=scale, 2=bias (may be null/unused if has_bias==0)
    // GM_ADDR ins[3]   = {x, scale_in, bias};
    // // Bundle output pointers: index 0=y
    // GM_ADDR outs[1]  = {y};
    //
    // if constexpr (RANK == 4) {
    //     GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
    //     // DTYPE_X expands to the actual data type (float, half, bfloat16)
    //     FatreluMulKernel<DTYPE_X, 4> kernel;
    //     kernel.Init(ins, outs, &td);
    //     kernel.Process();
    // } else {
    //     // Rank 5-8: use FatreluMulTilingData<8>
    //     GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
    //     FatreluMulKernel<DTYPE_X, 8> kernel;
    //     kernel.Init(ins, outs, &td);
    //     kernel.Process();
    // }
    // -----------------------------------------------------------------------
}

// ===========================================================================
// template<typename D_T_X, int64_t kPath> __global__ __aicore__ void fatrelu_mul(...)
//
// Build-system adapter entry (snake_case = opInterface.value). The auto-gen
// wrapper calls `fatrelu_mul<TEMPLATE_PARAMS>(...)` (macro-renamed per tiling
// key), so this entry must exist under the snake_case name; it forwards to the
// proto.md §3 kernel entry FatreluMul<D_T_X, kPath> one-to-one.
// ===========================================================================
template <typename D_T_X, int64_t kPath>
__global__ __aicore__ void fatrelu_mul(GM_ADDR input, GM_ADDR threshold, GM_ADDR output, GM_ADDR workspace,
                                       GM_ADDR tiling)
{
    FatreluMul<D_T_X, kPath>(input, threshold, output, workspace, tiling);
}
