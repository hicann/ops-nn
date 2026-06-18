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
 * \file dynamic_mx_quant_struct.h
 * \brief
 */

#ifndef _DYNAMIC_MX_QUANT_STRUCT_H_
#define _DYNAMIC_MX_QUANT_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_TAIL_AXIS_QUANT_NORMAL 0
#define TPL_TAIL_AXIS_QUANT_OPTI 1
#define TPL_NOT_TAIL_AXIS_QUANT_NORMAL 2
#define TPL_NOT_TAIL_AXIS_QUANT_OPTI 3
#define TPL_NOT_TAIL_AXIS_QUANT_SMALL_OPTI 10
#define TPL_NOT_TAIL_AXIS_QUANT_LARGE_OPTI 20

#define TPL_NOT_CARE_SCALE_ALG 10
#define TPL_SCALE_ALG_0 0
#define TPL_SCALE_ALG_1 1
#define TPL_SCALE_ALG_2_SPEC_DST_MAX_VALUE 2
#define TPL_SCALE_ALG_2_OTHERS 3

#define TPL_NOT_CARE_ROUND_MODE 10
#define TPL_RINT 4
#define TPL_ROUND 0
#define TPL_FLOOR 1

#define TPL_NOT_CARE_SCALE 10
#define TPL_ODD_SCALE 1
#define TPL_EVEN_SCALE 2

/**
 * ============================================================
 * Tiling Key Encoding
 * ============================================================
 *
 * GET_TPL_TILING_KEY encodes each ASCENDC_TPL_UINT_DECL parameter
 * by its 0-based INDEX in the DECL values list (not the macro
 * value), then shifts by accumulated bit widths:
 *
 *   key = (optiMode_idx    << 0)   // bits 0-4,  bitWidth=5
 *       + (scaleAlg_idx    << 5)   // bits 5-7,  bitWidth=3
 *       + (roundMode_idx   << 8)   // bits 8-11, bitWidth=4
 *       + (isOddScale_idx  << 12)  // bits 12-13,bitWidth=2
 *
 * Index mapping (macro value → 0-based index in DECL vals list):
 *   optiMode:   TAIL_AXIS_QUANT_NORMAL(0)→0,  TAIL_AXIS_QUANT_OPTI(1)→1,
 *               NOT_TAIL_AXIS_QUANT_NORMAL(2)→2,  NOT_TAIL_AXIS_QUANT_OPTI(3)→3,
 *               NOT_TAIL_AXIS_QUANT_SMALL_OPTI(10)→4,  NOT_TAIL_AXIS_QUANT_LARGE_OPTI(20)→5
 *   scaleAlg:   NOT_CARE_SCALE_ALG(10)→0,  SCALE_ALG_0(0)→1,
 *               SCALE_ALG_1(1)→2,  SCALE_ALG_2_SPEC(2)→3,  SCALE_ALG_2_OTHERS(3)→4
 *   roundMode:  NOT_CARE_ROUND_MODE(10)→0,  RINT(4)→1,
 *               ROUND(0)→2,  FLOOR(1)→3
 *   isOddScale: NOT_CARE_SCALE(10)→0,  ODD_SCALE(1)→1,  EVEN_SCALE(2)→2
 *
 * ============================================================
 * Tiling Path Dispatch
 * ============================================================
 *
 *   1) isTailAxis && blockSize==32 → CalcTilingKeyForTail (Group 1)
 *   2) IsOptForNotLastQuantAxis rejects isTailAxis early →
 *      only non-tail axis enters optimize/main tiling
 *   3) isOptimizable (non-tail)    → optimize tiling     (Group 3)
 *   4) fallback (non-tail)         → main CalcTilingKey   (Group 2 or 4)
 *
 * ============================================================
 * Group 1 — 尾轴 + blockSize=32  (3 keys)
 * ============================================================
 *   From: CalcTilingKeyForTail in tail_axis_tiling
 *   Params: optiMode=TAIL_AXIS_QUANT_OPTI(idx=1), scaleAlg varies,
 *           roundMode=NOT_CARE(idx=0), isOddScale=NOT_CARE(idx=0)
 *   Key = 1 + scaleAlg_idx*32
 *
 *   tilingKey  hex     scaleAlg(user)  scaleAlg_idx
 *   33         0x21    0               1 (SCALE_ALG_0)
 *   65         0x41    1               2 (SCALE_ALG_1)
 *   97         0x61    2               3 (SCALE_ALG_2_SPEC_DST_MAX_VALUE)
 *
 * ============================================================
 * Group 2 — 尾轴 + blockSize!=32  (2 keys)
 * ============================================================
 *   From: main CalcTilingKey
 *   Params: optiMode=TAIL_AXIS_QUANT_NORMAL(idx=0),
 *           scaleAlg=NOT_CARE(idx=0), roundMode=NOT_CARE(idx=0)
 *   Key = 0 + isOddScale_idx*4096
 *
 *   tilingKey  hex       isOddScale         note
 *   4096       0x1000    ODD_SCALE(1)       block数奇 or 非尾轴
 *   8192       0x2000    EVEN_SCALE(2)      尾轴+block数偶
 *
 * ============================================================
 * Group 3 — 非尾轴 + optimize tiling  (24 keys)
 * ============================================================
 *   From: optimize tiling SetTilingKey
 *   Params: optiMode=SMALL_OPTI(idx=4) or LARGE_OPTI(idx=5),
 *           isOddScale=NOT_CARE(idx=0)
 *   Key = optiMode_idx + scaleAlg_idx*32 + roundMode_idx*256
 *
 *   SMALL_OPTI (postAxisSize<=nAlignNum): nAlignNum=vfLen/inputDtypeSize
 *     bf16/fp16: nAlignNum=128;  fp32: nAlignNum=64
 *     optiMode_idx=4:
 *       scaleAlg_idx=1(ALG_0):   rint(1)=292  round(2)=548   floor(3)=804
 *       scaleAlg_idx=2(ALG_1):   rint(1)=324  round(2)=580   floor(3)=836
 *       scaleAlg_idx=3(SPEC):    rint(1)=356  round(2)=612   floor(3)=868
 *       scaleAlg_idx=4(OTHERS):  rint(1)=388  round(2)=644   floor(3)=900
 *
 *   LARGE_OPTI (postAxisSize>nAlignNum):
 *     optiMode_idx=5:
 *       scaleAlg_idx=1(ALG_0):   rint(1)=293  round(2)=549   floor(3)=805
 *       scaleAlg_idx=2(ALG_1):   rint(1)=325  round(2)=581   floor(3)=837
 *       scaleAlg_idx=3(SPEC):    rint(1)=357  round(2)=613   floor(3)=869
 *       scaleAlg_idx=4(OTHERS):  rint(1)=389  round(2)=645   floor(3)=901
 *
 * ============================================================
 * Group 4 — 非尾轴 + blockSize!=32 + 普通模板  (2 reachable keys)
 * ============================================================
 *   From: main CalcTilingKey
 *   Params: optiMode=NOT_TAIL_AXIS_QUANT_NORMAL(idx=2)
 *           or NOT_TAIL_AXIS_QUANT_OPTI(idx=3),
 *           scaleAlg=NOT_CARE(idx=0), roundMode=NOT_CARE(idx=0)
 *   Key = optiMode_idx + isOddScale_idx*4096
 *   Note: non-tail axis always sets isOddScale=ODD(1) in main path
 *
 *   tilingKey  hex       optiMode_idx      isOddScale_idx
 *   4098       0x1002    2 (NOT_TAIL_NORMAL)  1 (ODD_SCALE)
 *   4099       0x1003    3 (NOT_TAIL_OPTI)    1 (ODD_SCALE)
 *
 * ============================================================
 * Total valid tiling keys: 3 + 2 + 24 + 2 = 31
 * ============================================================
 */
namespace DynamicMxQuant {
ASCENDC_TPL_ARGS_DECL(
    DynamicMxQuant,
    ASCENDC_TPL_UINT_DECL(
        optiMode, 5, ASCENDC_TPL_UI_LIST, TPL_TAIL_AXIS_QUANT_NORMAL, TPL_TAIL_AXIS_QUANT_OPTI,
        TPL_NOT_TAIL_AXIS_QUANT_NORMAL, TPL_NOT_TAIL_AXIS_QUANT_OPTI, TPL_NOT_TAIL_AXIS_QUANT_SMALL_OPTI,
        TPL_NOT_TAIL_AXIS_QUANT_LARGE_OPTI),
    ASCENDC_TPL_UINT_DECL(
        scaleAlg, 3, ASCENDC_TPL_UI_LIST, TPL_NOT_CARE_SCALE_ALG, TPL_SCALE_ALG_0, TPL_SCALE_ALG_1,
        TPL_SCALE_ALG_2_SPEC_DST_MAX_VALUE, TPL_SCALE_ALG_2_OTHERS),
    ASCENDC_TPL_UINT_DECL(roundMode, 4, ASCENDC_TPL_UI_LIST, TPL_NOT_CARE_ROUND_MODE, TPL_RINT, TPL_ROUND, TPL_FLOOR),
    ASCENDC_TPL_UINT_DECL(isOddScale, 2, ASCENDC_TPL_UI_LIST, TPL_NOT_CARE_SCALE, TPL_ODD_SCALE, TPL_EVEN_SCALE));

ASCENDC_TPL_SEL(
    // 尾轴量化，blocksize=32量化
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(optiMode, ASCENDC_TPL_UI_LIST, TPL_TAIL_AXIS_QUANT_OPTI),
        ASCENDC_TPL_UINT_SEL(
            scaleAlg, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_1, TPL_SCALE_ALG_2_SPEC_DST_MAX_VALUE),
        ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_NOT_CARE_ROUND_MODE),
        ASCENDC_TPL_UINT_SEL(isOddScale, ASCENDC_TPL_UI_LIST, TPL_NOT_CARE_SCALE)),
    // 尾轴量化，blocksize!=32量化
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(optiMode, ASCENDC_TPL_UI_LIST, TPL_TAIL_AXIS_QUANT_NORMAL),
        ASCENDC_TPL_UINT_SEL(scaleAlg, ASCENDC_TPL_UI_LIST, TPL_NOT_CARE_SCALE_ALG),
        ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_NOT_CARE_ROUND_MODE),
        ASCENDC_TPL_UINT_SEL(isOddScale, ASCENDC_TPL_UI_LIST, TPL_ODD_SCALE, TPL_EVEN_SCALE)),
    // 非尾轴量化，blocksize=32，小尾轴优化模板
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(optiMode, ASCENDC_TPL_UI_LIST, TPL_NOT_TAIL_AXIS_QUANT_SMALL_OPTI),
        ASCENDC_TPL_UINT_SEL(
            scaleAlg, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_1, TPL_SCALE_ALG_2_SPEC_DST_MAX_VALUE,
            TPL_SCALE_ALG_2_OTHERS),
        ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_ROUND, TPL_FLOOR),
        ASCENDC_TPL_UINT_SEL(isOddScale, ASCENDC_TPL_UI_LIST, TPL_NOT_CARE_SCALE)),
    // 非尾轴量化，大尾轴优化模板
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(optiMode, ASCENDC_TPL_UI_LIST, TPL_NOT_TAIL_AXIS_QUANT_LARGE_OPTI),
        ASCENDC_TPL_UINT_SEL(
            scaleAlg, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_1, TPL_SCALE_ALG_2_SPEC_DST_MAX_VALUE,
            TPL_SCALE_ALG_2_OTHERS),
        ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_ROUND, TPL_FLOOR),
        ASCENDC_TPL_UINT_SEL(isOddScale, ASCENDC_TPL_UI_LIST, TPL_NOT_CARE_SCALE)),
    // 非尾轴，普通模板和多行处理模板
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(
            optiMode, ASCENDC_TPL_UI_LIST, TPL_NOT_TAIL_AXIS_QUANT_NORMAL, TPL_NOT_TAIL_AXIS_QUANT_OPTI),
        ASCENDC_TPL_UINT_SEL(scaleAlg, ASCENDC_TPL_UI_LIST, TPL_NOT_CARE_SCALE_ALG),
        ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_NOT_CARE_ROUND_MODE),
        ASCENDC_TPL_UINT_SEL(isOddScale, ASCENDC_TPL_UI_LIST, TPL_ODD_SCALE, TPL_EVEN_SCALE)), );

} // namespace DynamicMxQuant

#endif // _DYNAMIC_MX_QUANT_STRUCT_H_