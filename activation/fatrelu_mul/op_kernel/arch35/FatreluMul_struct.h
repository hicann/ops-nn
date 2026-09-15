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
// fatrelu_mul_package/op_kernel/arch35/FatreluMul_struct.h
// =============================================================================
//
// ROLE: FatreluMul TilingKey 模板参数（ASCENDC_TPL_*）声明与实例化选择。
//   本文件是 tilingKey 位编码的唯一事实源（参照仓内 fast_gelu_v2_tiling_key.h 模式）：
//     - host 侧（fatrelu_mul_tiling_arch35.cpp TilingFunc）经
//       ASCENDC_TPL_SEL_PARAM(context, dtype, FATRELUMUL_PATH_SMALL_TAIL / BIG_TAIL)
//       设置 tilingKey（BranchRoute.md 判定顺序：空 Tensor 短路 → 判界 halfDim vs tileElems）；
//     - kernel 侧（fatrelu_mul_apt.cpp）按 ASCENDC_TPL_SEL 声明实例化 sub-kernel：
//       D_T_X（dtype 模板参数）× PATH → 3 dtype × 2 path = 6 个实例，
//       框架为每个实例生成独立 .o 并按 tilingKey 分发。
//
// CONTENTS:
//   - FATRELUMUL_PATH_SMALL_TAIL = 0（行打包路径：空 Tensor 短路 或 0 < d ≤ tileElems）
//   - FATRELUMUL_PATH_BIG_TAIL   = 1（行分段路径：d > tileElems）
//   - ASCENDC_TPL_ARGS_DECL(FatreluMul, DATATYPE(D_T_X) + UINT(PATH))
//   - ASCENDC_TPL_SEL(...)  — 每 dtype 一行 ARGS_SEL（无空位，6 组合均实例化）
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FatreluMul   — ASCENDC_TPL_ARGS_DECL 第一参数（生成代码前缀）
//   文件名前缀    : FatreluMul_  — DevView.md「命名规范」
//   UPPER        : FATRELUMUL   — 宏名 / 头文件守卫 / TPL 路径常量前缀
//
// =============================================================================

// FatreluMul TilingKey 模板参数 —— dtype + 行模型双路径（small-tail 行打包 / big-tail 行分段）
// 落位：op_kernel/arch35/FatreluMul_struct.h（与 DevView.md 工程结构一致）
#ifndef FATRELUMUL_STRUCT_H_
#define FATRELUMUL_STRUCT_H_
#include "ascendc/host_api/tiling/template_argument.h"

// 路径常量：取值仅作 TPL 枚举标识。
// tilingKey 位编码取「值在 DECL 枚举列表中的下标」，非常量值本身
// （GET_TPL_TILING_KEY 位编码规则：UINT 参数按下标编码占 bitWidth 位，见 ascendc-build-system skill
//   references/tiling-key-encoding.md）：
//   GET_TPL_TILING_KEY(dtype, FATRELUMUL_PATH_SMALL_TAIL) → dtype 值占低 8 bit，PATH 下标 0 → tilingKey = dtype 值
//   GET_TPL_TILING_KEY(dtype, FATRELUMUL_PATH_BIG_TAIL)   → dtype 值占低 8 bit，PATH 下标 1 占 bit 8
#define FATRELUMUL_PATH_SMALL_TAIL 0 // 行打包路径：空 Tensor 短路 或 0 < d ≤ tileElems
#define FATRELUMUL_PATH_BIG_TAIL 1   // 行分段路径：d > tileElems

// 模板参数声明：
//   - D_T_X：首输入 x 的 dtype（ASCENDC_TPL_INPUT(0)），框架按 DECL 列举的 dtype 逐一实例化
//     sub-kernel（float / fp16 / bf16），与 def.cpp 的 DataType({DT_BF16, DT_FLOAT16, DT_FLOAT}) 对应
//   - PATH：单 UINT 参数，位宽 1 bit（2^1 ≥ 2 个枚举值），可选 {0, 1}
//     （bitWidth 须 ≥ 1：asc_op_compile_base template_tiling.py check_bit_width_valid 拒绝 0；
//       下标编码下 1 bit 即覆盖两枚举值）
ASCENDC_TPL_ARGS_DECL(FatreluMul,
                      ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
                      ASCENDC_TPL_UINT_DECL(PATH, 1, ASCENDC_TPL_UI_LIST, FATRELUMUL_PATH_SMALL_TAIL,
                                            FATRELUMUL_PATH_BIG_TAIL));

// 实例化选择：每 dtype 一行 ARGS_SEL，行内 PATH 按 UI_LIST 展开 → 3 dtype × 2 path = 6 个 sub-kernel
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, FATRELUMUL_PATH_SMALL_TAIL,
                                                          FATRELUMUL_PATH_BIG_TAIL)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, FATRELUMUL_PATH_SMALL_TAIL,
                                                          FATRELUMUL_PATH_BIG_TAIL)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, FATRELUMUL_PATH_SMALL_TAIL,
                                                          FATRELUMUL_PATH_BIG_TAIL)));
#endif
