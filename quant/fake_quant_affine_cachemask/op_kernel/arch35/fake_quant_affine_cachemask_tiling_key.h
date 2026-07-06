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
 * \file fake_quant_affine_cachemask_tiling_key.h
 * \brief TilingKey 模板参数：D_T_X × ZP_DT × MODE × HAS_ZP
 *
 *   D_T_X : x/scale/y dtype（C_DT_FLOAT / C_DT_FLOAT16 / C_DT_BF16）
 *   ZP_DT : zero_point dtype（C_DT_INT32 / C_DT_FLOAT / C_DT_FLOAT16 / C_DT_BF16）
 *           对每个 D_T_X，只展开 INT32 + 同 D_T_X dtype 两个选项（共 6 SEL）
 *   MODE  : 0=PT, 1=PC, 2=PC_NDDMA, 3=PH （PC_NDDMA 占位，kernel 走 PC 同实现）
 *   HAS_ZP: 0=symmetric (zp ignored), 1=asymmetric
 */
#ifndef FAKE_QUANT_AFFINE_CACHEMASK_TILING_KEY_H
#define FAKE_QUANT_AFFINE_CACHEMASK_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(FakeQuantAffineCachemask,
                      ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
                      ASCENDC_TPL_DATATYPE_DECL(ZP_DT, C_DT_INT32, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16,
                                                ASCENDC_TPL_INPUT(2)),
                      ASCENDC_TPL_UINT_DECL(MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
                      ASCENDC_TPL_UINT_DECL(HAS_ZP, 8, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    // x = FLOAT, zp = INT32
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT), ASCENDC_TPL_DATATYPE_SEL(ZP_DT, C_DT_INT32),
                         ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(HAS_ZP, ASCENDC_TPL_UI_LIST, 0, 1)),
    // x = FLOAT, zp = FLOAT
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT), ASCENDC_TPL_DATATYPE_SEL(ZP_DT, C_DT_FLOAT),
                         ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(HAS_ZP, ASCENDC_TPL_UI_LIST, 0, 1)),
    // x = FLOAT16, zp = INT32
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16), ASCENDC_TPL_DATATYPE_SEL(ZP_DT, C_DT_INT32),
                         ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(HAS_ZP, ASCENDC_TPL_UI_LIST, 0, 1)),
    // x = FLOAT16, zp = FLOAT16
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16), ASCENDC_TPL_DATATYPE_SEL(ZP_DT, C_DT_FLOAT16),
                         ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(HAS_ZP, ASCENDC_TPL_UI_LIST, 0, 1)),
    // x = BF16, zp = INT32
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16), ASCENDC_TPL_DATATYPE_SEL(ZP_DT, C_DT_INT32),
                         ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(HAS_ZP, ASCENDC_TPL_UI_LIST, 0, 1)),
    // x = BF16, zp = BF16
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16), ASCENDC_TPL_DATATYPE_SEL(ZP_DT, C_DT_BF16),
                         ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
                         ASCENDC_TPL_UINT_SEL(HAS_ZP, ASCENDC_TPL_UI_LIST, 0, 1)), );

#endif // FAKE_QUANT_AFFINE_CACHEMASK_TILING_KEY_H
