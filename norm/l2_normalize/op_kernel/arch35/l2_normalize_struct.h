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
// norm/l2_normalize/op_kernel/arch35/l2_normalize_struct.h
// =============================================================================
//
// ROLE: Ascend C template parameter (TPL) declarations for L2Normalize.
//   TilingKey 模板参数按双 bool 轴声明（isGroup / isEmptyTensor），与 kernel entry
//   的函数模板参数一一对应：
//   - TPL_SEL_0 (0,0) base  模板：非空 tensor 默认路径
//   - TPL_SEL_1 (0,1) empty 模板：空 tensor（某维为 0）全核早退
//   - TPL_SEL_2 (1,0) group 模板：A 用不满核且 R 有并行度
//   dtype（fp16/fp32）由框架按 REG_OP 输入名经 DTYPE_X 编译期实例化，不进 TilingKey。
//   tilingKey 数值由框架宏 GET_TPL_TILING_KEY(isGroup, isEmptyTensor) 生成，不自编序号。
// =============================================================================

#ifndef OPS_NORM_L2_NORMALIZE_TILING_KEY_H_
#define OPS_NORM_L2_NORMALIZE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h" // ASCENDC_TPL macros

ASCENDC_TPL_ARGS_DECL(L2Normalize, ASCENDC_TPL_BOOL_DECL(isGroup, 0, 1), ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1));

ASCENDC_TPL_SEL(
    // base 模板
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)),
    // 空 tensor 模板
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1)),
    // group 模板（isEmptyTensor 固定 0，与 empty 互斥）
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 1), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)));

#endif // OPS_NORM_L2_NORMALIZE_TILING_KEY_H_
