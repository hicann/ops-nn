/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_grad_tiling_key.h
 * \brief SwigluGroupGrad TilingKey definition (arch35, Ascend950)
 */

#ifndef __SWIGLU_GROUP_GRAD_TILING_KEY_H__
#define __SWIGLU_GROUP_GRAD_TILING_KEY_H__

#include <cstdint>
#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_REGBASE_KERNEL 0
#define TPL_SIMT_KERNEL 1

#pragma pack(push, 8)
struct alignas(8) SwigluGroupGradTilingData {
    int64_t coreNumAll;
    int64_t totalRows;
    int64_t hiddenSize;
    int64_t rowsPerTile;
    int64_t splitHiddenMode;
    int64_t launchedCoreNum;
    int64_t groupIndexG;
    int64_t hiddenChunkSize;
    int64_t chunksPerRow;
    float clampLimit;
    float clampLimitRecp;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) SwigluGroupGradSimtTilingData {
    int64_t totalRows;
    int64_t hiddenSize;
    int64_t groupIndexG;
    float clampLimit;
    float clampLimitRecp;
};
#pragma pack(pop)

ASCENDC_TPL_ARGS_DECL(SwigluGroupGrad,
                      ASCENDC_TPL_UINT_DECL(SCHMODE, 1, ASCENDC_TPL_UI_LIST, TPL_REGBASE_KERNEL, TPL_SIMT_KERNEL),
                      ASCENDC_TPL_UINT_DECL(HAS_CLAMP, 1, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(IS_WEIGHT, 1, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(IS_Y_ORIGIN, 1, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(IS_GROUP_INDEX, 1, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(SCHMODE, ASCENDC_TPL_UI_LIST, TPL_REGBASE_KERNEL),
                                     ASCENDC_TPL_UINT_SEL(HAS_CLAMP, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(IS_WEIGHT, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(IS_Y_ORIGIN, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(IS_GROUP_INDEX, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(SwigluGroupGradTilingData)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(SCHMODE, ASCENDC_TPL_UI_LIST, TPL_SIMT_KERNEL),
                                     ASCENDC_TPL_UINT_SEL(HAS_CLAMP, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(IS_WEIGHT, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(IS_Y_ORIGIN, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(IS_GROUP_INDEX, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(SwigluGroupGradSimtTilingData)));

#endif // __SWIGLU_GROUP_GRAD_TILING_KEY_H__
