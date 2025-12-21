/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_backprop_input_v2_arch35_tiling_key.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_ARCH35_TILING_KEY_H
#define CONV3D_BACKPROP_INPUT_V2_ARCH35_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_NO_TRANSPOSE_NO_REVERSE 0
#define TPL_TRANSPOSE_AND_REVERSE 1
#define TPL_REVERSE_ONLY 2
#define TPL_TRANSPOSE_ONLY 3

#define TPL_NO_SPLIT_KERNEL 0
#define TPL_SPLIT_KERNEL_HW 1
#define TPL_SPLIT_KERNEL_H 2

#define TPL_GROUP_MODE_ORIGIN 0
#define TPL_GROUP_MODE_ENLARGE 1

#define TPL_GM_TO_L1 0
#define TPL_VEC_TO_L1_C04 1
#define TPL_GM_TO_L1_NO_HK 2
#define TPL_GM_TO_L1_NO_HK_WK 3

// 模板参数
ASCENDC_TPL_ARGS_DECL(Conv3dBackPropInputV2Arch35,
    ASCENDC_TPL_UINT_DECL(loadB2Condition, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, \
                          TPL_NO_TRANSPOSE_NO_REVERSE, TPL_TRANSPOSE_AND_REVERSE, TPL_REVERSE_ONLY, TPL_TRANSPOSE_ONLY), // LIST模式, 穷举
    ASCENDC_TPL_UINT_DECL(kernelSplitMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, \
                          TPL_NO_SPLIT_KERNEL, TPL_SPLIT_KERNEL_HW, TPL_SPLIT_KERNEL_H),
    ASCENDC_TPL_UINT_DECL(groupConvMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, \
                          TPL_GROUP_MODE_ORIGIN, TPL_GROUP_MODE_ENLARGE),
    ASCENDC_TPL_BOOL_DECL(isBasicBlockTiling, 0, 1),
    ASCENDC_TPL_UINT_DECL(loadB1Condition, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, \
                          TPL_GM_TO_L1, TPL_VEC_TO_L1_C04, TPL_GM_TO_L1_NO_HK, TPL_GM_TO_L1_NO_HK_WK)
);

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_RANGE, 1, TPL_NO_TRANSPOSE_NO_REVERSE, TPL_TRANSPOSE_ONLY), // RANGE模式, 第一个值1表示范围个数1，后两个值表示范围起、终位置
        ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_RANGE, 1, TPL_NO_SPLIT_KERNEL, TPL_SPLIT_KERNEL_H),
        ASCENDC_TPL_UINT_SEL(groupConvMode, ASCENDC_TPL_UI_RANGE, 1, TPL_GROUP_MODE_ORIGIN, TPL_GROUP_MODE_ENLARGE),
        ASCENDC_TPL_BOOL_SEL(isBasicBlockTiling, 0, 1),
        ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_RANGE, 1, TPL_GM_TO_L1, TPL_GM_TO_L1_NO_HK_WK)
    ),
);

#endif // CONV3D_BACKPROP_INPUT_V2_ARCH35_TILING_KEY_H