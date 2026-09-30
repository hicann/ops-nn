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
// bn3d_training_update_grad_package/op_kernel/arch35/bn3d_training_update_grad_struct.h
// =============================================================================
//
// ROLE: Ascend C template-parameter (TPL) declarations for BN3DTrainingUpdateGrad.
//   2 bool selection bits (isGroup, isEmptyTensor),
//   bit0=isGroup, bit1=isEmptyTensor. Three legal combinations:
//     (0,0) -> key 0 = base
//     (1,0) -> key 1 = group
//     (0,1) -> key 2 = empty
//   (1,1) is illegal (group and empty are mutually exclusive) and is neither
//   declared nor compiled. The host TilingFunc calls SetTilingKey(0/1/2) directly.
// =============================================================================

#ifndef BN3D_TRAINING_UPDATE_GRAD_STRUCT_H_
#define BN3D_TRAINING_UPDATE_GRAD_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h" // ASCENDC_TPL macros

// 字段顺序 (isGroup, isEmptyTensor)，2 个 bool；合法组合 3 个（(1,1) 互斥非法）
ASCENDC_TPL_ARGS_DECL(BN3DTrainingUpdateGrad,
                      ASCENDC_TPL_BOOL_DECL(isGroup, 0, 1),      // 0=base/empty, 1=group（A×R 2D 分核）
                      ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1) // 0=非空, 1=空 tensor（EMPTY_A/EMPTY_R 共用）
);

// 合法组合（3 个；(1,1) 非法不声明、不生成 binary）
ASCENDC_TPL_SEL(
    // base (0,0)
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)),
    // empty (0,1)（EMPTY_A/EMPTY_R 共用）
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1)),
    // group (1,0)（isEmptyTensor 固定 0，与 empty 互斥）
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 1), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)));

#endif
