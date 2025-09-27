/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_transpose_v2_tiling_key.h
 * \brief
 */
#ifndef CONV3D_TRANSPOSE_V2_TILING_KEY_H
#define CONV3D_TRANSPOSE_V2_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_TRS_BASEK_LT_HKWK 0 // BaseK / blockSize < Hk*Wk;
#define TPL_TRS_BASEK_GE_HKWK 1 // BaseK / blockSize >= Hk*Wk;
#define TPL_TRS_HKWK_EQ_ONE 2 // Hk*Wk = 1;

// 模板参数
ASCENDC_TPL_ARGS_DECL(Conv3dTransposeV2,
    ASCENDC_TPL_UINT_DECL(loadB2Condition, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, \
                          TPL_TRS_BASEK_LT_HKWK, TPL_TRS_BASEK_GE_HKWK, TPL_TRS_HKWK_EQ_ONE), // LIST模式, 穷举
    ASCENDC_TPL_BOOL_DECL(enableKernelSplit, 0, 1),
    ASCENDC_TPL_BOOL_DECL(useBasicBlock, 0, 1)
);

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_RANGE, 1, TPL_TRS_BASEK_LT_HKWK, TPL_TRS_HKWK_EQ_ONE), // RANGE模式, 第一个值1表示范围个数1，后两个值表示范围起、终位置
        ASCENDC_TPL_BOOL_SEL(enableKernelSplit, 0, 1),
        ASCENDC_TPL_BOOL_SEL(useBasicBlock, 0, 1)
    ),
);

#endif // CONV3D_TRANSPOSE_V2_TILING_KEY_H