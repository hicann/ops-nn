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
// gn_training_update_package/op_kernel/arch35/gn_training_update_struct.h
// =============================================================================
//
// ROLE: Ascend C template parameter (TPL) declarations for GnTrainingUpdate.
//   This file uses ASCENDC_TPL macros to declare compile-time template
//   parameters for the Ascend C kernel. The template parameter RANK controls
//   whether the kernel is compiled for rank-4 or rank-8 tensors.
//
//   The ASCENDC_TPL system allows the kernel to be compiled with different
//   compile-time parameters (rank, dtype, etc.) and generates separate
//   kernel binaries for each combination. The runtime dispatches to the
//   correct binary based on the tiling key.
//
//   The macros work as follows:
//   1. ASCENDC_TPL_ARGS_DECL — declares the template argument list
//      Defines a template parameter named RANK with default value 8,
//      selectable from {GN_TRAINING_UPDATE_RANK_4, GN_TRAINING_UPDATE_RANK_8}
//   2. ASCENDC_TPL_SEL — generates the template specialization dispatch
//      Maps each rank value to a specific kernel compilation
//   3. GET_TPL_TILING_KEY — used by host-side tiling to return a key
//      that tells the runtime which kernel binary to select
//
// CONTENTS:
//   - GN_TRAINING_UPDATE_RANK_4 = 4  (rank value for ≤4D tensors)
//   - GN_TRAINING_UPDATE_RANK_8 = 8  (rank value for 5-8D tensors)
//   - ASCENDC_TPL_ARGS_DECL(GnTrainingUpdate, ...) — template argument declaration
//   - ASCENDC_TPL_SEL(...) — template specialization selector
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : GnTrainingUpdate   — first argument to ASCENDC_TPL_ARGS_DECL
//   snake_case   : gn_training_update  — filename
//   UPPER_SNAKE  : GN_TRAINING_UPDATE  — rank value macro names, header guard
//
// NAME REPLACEMENT RULES (to create FooBar operator):
//   GnTrainingUpdate → FooBar                (ASCENDC_TPL_ARGS_DECL argument)
//   GN_TRAINING_UPDATE → FOO_BAR             (rank value macros)
//   GN_TRAINING_UPDATE_RANK_4/8 → FOO_BAR_RANK_4/8
//   gn_training_update_struct → foo_bar_struct (filename)
//   RANK: keep as-is (standard TPL parameter name)
//   ASCENDC_TPL macros: keep as-is (CANN framework macros)
//
// =============================================================================

#ifndef GN_TRAINING_UPDATE_STRUCT_H_
#define GN_TRAINING_UPDATE_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h" // ASCENDC_TPL macros

// ---------------------------------------------------------------------------
// Rank value constants for TPL template dispatch
//   These values correspond to the RANK template parameter.
//   RANK=4: tensors with 1-4 dimensions (uses TilingData4)
//   RANK=8: tensors with 5-8 dimensions (uses TilingData8)
//
// To create FooBar with different rank splits:
//   #define FOO_BAR_RANK_4 4
//   #define FOO_BAR_RANK_8 8
//   (or add custom rank values for different dim-split strategies)
// ---------------------------------------------------------------------------
#define GN_TRAINING_UPDATE_RANK_4 4
#define GN_TRAINING_UPDATE_RANK_8 8

// ---------------------------------------------------------------------------
// ASCENDC_TPL_ARGS_DECL — declares compile-time template arguments
//
// Arguments:
//   GnTrainingUpdate  → operator type name (used as namespace/prefix for generated code)
//   ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST, GN_TRAINING_UPDATE_RANK_4, GN_TRAINING_UPDATE_RANK_8)
//     → declares a template parameter "RANK" of type uint
//     → default value: 8
//     → list type: ASCENDC_TPL_UI_LIST (uint-list)
//     → allowed values: GN_TRAINING_UPDATE_RANK_4 (4) and GN_TRAINING_UPDATE_RANK_8 (8)
//
// To create FooBar:
//   ASCENDC_TPL_ARGS_DECL(FooBar,
//       ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST,
//           FOO_BAR_RANK_4, FOO_BAR_RANK_8)
//   );
// ---------------------------------------------------------------------------
ASCENDC_TPL_ARGS_DECL(GnTrainingUpdate, ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST, GN_TRAINING_UPDATE_RANK_4,
                                                              GN_TRAINING_UPDATE_RANK_8));

// ---------------------------------------------------------------------------
// ASCENDC_TPL_SEL — generates template specialization selectors
//
// Each ASCENDC_TPL_ARGS_SEL line creates a specialization for one combination
// of template parameter values. The compiler will generate separate kernel
// binaries for each specialization.
//
// Line 1: ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ..., GN_TRAINING_UPDATE_RANK_4))
//   → specialization for RANK=4
// Line 2: ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ..., GN_TRAINING_UPDATE_RANK_8))
//   → specialization for RANK=8
//
// To create FooBar:
//   ASCENDC_TPL_SEL(
//       ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, FOO_BAR_RANK_4)),
//       ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, FOO_BAR_RANK_8))
//   );
// ---------------------------------------------------------------------------
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, GN_TRAINING_UPDATE_RANK_4)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, GN_TRAINING_UPDATE_RANK_8)));

#endif
