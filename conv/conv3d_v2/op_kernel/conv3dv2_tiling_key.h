/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file conv3dv2_tiling_key.h
 * \brief
 */

#ifndef CONV3DV2_TILING_KEY_H
#define CONV3DV2_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace Conv3dTilingKey {

#define L0_PINGPONG_ALL_CLOSE 0
#define L0_PINGPONG_L0A_OPEN 1
#define L0_PINGPONG_L0B_OPEN 2
#define L0_PINGPONG_ALL_OPEN 3

#define L1_BYPASS_OFF 0
#define L1_BYPASS_ON 1

#define NOGROUP_CONV 0
#define GROUPCONV_WEIGHT_GFZ 1

#define M_MODE 0
#define HW_MODE 1

ASCENDC_TPL_ARGS_DECL(
    Conv3DV2,
    ASCENDC_TPL_UINT_DECL(ConvL0PingPong,
        ASCENDC_TPL_2_BW,
        ASCENDC_TPL_UI_LIST,
        L0_PINGPONG_ALL_CLOSE,
        L0_PINGPONG_L0A_OPEN,
        L0_PINGPONG_L0B_OPEN,
        L0_PINGPONG_ALL_OPEN),
    ASCENDC_TPL_UINT_DECL(ConvBL1ByPass,
        ASCENDC_TPL_1_BW,
        ASCENDC_TPL_UI_LIST,
        L1_BYPASS_OFF,
        L1_BYPASS_ON),
    ASCENDC_TPL_UINT_DECL(GroupConvType,
        ASCENDC_TPL_1_BW,
        ASCENDC_TPL_UI_LIST,
        NOGROUP_CONV,
        GROUPCONV_WEIGHT_GFZ),
    ASCENDC_TPL_UINT_DECL(OutputOrder,
        ASCENDC_TPL_1_BW,
        ASCENDC_TPL_UI_LIST,
        M_MODE,
        HW_MODE)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(ConvL0PingPong,
            ASCENDC_TPL_UI_LIST,
            L0_PINGPONG_ALL_CLOSE,
            L0_PINGPONG_L0A_OPEN,
            L0_PINGPONG_L0B_OPEN,
            L0_PINGPONG_ALL_OPEN),
        ASCENDC_TPL_UINT_SEL(ConvBL1ByPass,
            ASCENDC_TPL_UI_LIST,
            L1_BYPASS_OFF,
            L1_BYPASS_ON),
        ASCENDC_TPL_UINT_SEL(GroupConvType,
            ASCENDC_TPL_UI_LIST,
            NOGROUP_CONV,
            GROUPCONV_WEIGHT_GFZ),
        ASCENDC_TPL_UINT_SEL(OutputOrder,
            ASCENDC_TPL_UI_LIST,
            M_MODE)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(ConvL0PingPong,
            ASCENDC_TPL_UI_LIST,
            L0_PINGPONG_ALL_CLOSE,
            L0_PINGPONG_L0A_OPEN,
            L0_PINGPONG_L0B_OPEN,
            L0_PINGPONG_ALL_OPEN),
        ASCENDC_TPL_UINT_SEL(ConvBL1ByPass,
            ASCENDC_TPL_UI_LIST,
            L1_BYPASS_OFF,
            L1_BYPASS_ON),
        ASCENDC_TPL_UINT_SEL(GroupConvType,
            ASCENDC_TPL_UI_LIST,
            NOGROUP_CONV),
        ASCENDC_TPL_UINT_SEL(OutputOrder,
            ASCENDC_TPL_UI_LIST,
            HW_MODE)
    )
);

} // namespace Conv3dTilingKey

#endif // CONV3DV2_TILING_KEY_H