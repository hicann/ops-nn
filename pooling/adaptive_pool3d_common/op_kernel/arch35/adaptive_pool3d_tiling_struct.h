/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /* !
 * \file adaptive_pool3d_tiling_struct.h
 * \brief adaptive_pool3d tiling data
 */

#ifndef ADAPTIVE_POOL_3D_WITH_TILING_DATA_H
#define ADAPTIVE_POOL_3D_WITH_TILING_DATA_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace AdaptivePool3DTiling{

#define TPL_DYTPE_0 0
#define TPL_INT32_UINT32 1
#define TPL_INT64_UINT32 2
#define TPL_INT32_UINT64 3
#define TPL_INT64_UINT64 4
#define TPL_MODE_0 0
#define TPL_MODE_1 1
#define TPL_MODE_2 2
#define TPL_MULTI_MODE_0 0
#define TPL_DATA_FORMAT_MODE_0 0
#define TPL_DATA_FORMAT_MODE_1 1

ASCENDC_TPL_ARGS_DECL(AdaptiveMaxPool3d,
    ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, 2, ASCENDC_TPL_UI_LIST, TPL_MODE_0, TPL_MODE_1, TPL_MODE_2),
    ASCENDC_TPL_UINT_DECL(DYTPE_MODE, 3, ASCENDC_TPL_UI_LIST, TPL_DYTPE_0, TPL_INT32_UINT32, TPL_INT64_UINT32, TPL_INT32_UINT64, TPL_INT64_UINT64),
    ASCENDC_TPL_UINT_DECL(MULTI_MODE, 1, ASCENDC_TPL_UI_LIST, TPL_MULTI_MODE_0),
    ASCENDC_TPL_UINT_DECL(FORMAT_MODE, 1, ASCENDC_TPL_UI_LIST, TPL_DATA_FORMAT_MODE_0, TPL_DATA_FORMAT_MODE_1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_2),
        ASCENDC_TPL_UINT_SEL(DYTPE_MODE, ASCENDC_TPL_UI_LIST, TPL_INT32_UINT32, TPL_INT64_UINT32, TPL_INT32_UINT64, TPL_INT64_UINT64),
        ASCENDC_TPL_UINT_SEL(MULTI_MODE, ASCENDC_TPL_UI_LIST, TPL_MULTI_MODE_0),
        ASCENDC_TPL_UINT_SEL(FORMAT_MODE, ASCENDC_TPL_UI_LIST, TPL_DATA_FORMAT_MODE_0, TPL_DATA_FORMAT_MODE_1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_1),
        ASCENDC_TPL_UINT_SEL(DYTPE_MODE, ASCENDC_TPL_UI_LIST, TPL_DYTPE_0),
        ASCENDC_TPL_UINT_SEL(MULTI_MODE, ASCENDC_TPL_UI_LIST, TPL_MULTI_MODE_0),
        ASCENDC_TPL_UINT_SEL(FORMAT_MODE, ASCENDC_TPL_UI_LIST, TPL_DATA_FORMAT_MODE_0)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_0),
        ASCENDC_TPL_UINT_SEL(DYTPE_MODE, ASCENDC_TPL_UI_LIST, TPL_DYTPE_0),
        ASCENDC_TPL_UINT_SEL(MULTI_MODE, ASCENDC_TPL_UI_LIST, TPL_MULTI_MODE_0),
        ASCENDC_TPL_UINT_SEL(FORMAT_MODE, ASCENDC_TPL_UI_LIST, TPL_DATA_FORMAT_MODE_0)
    ),
);


class AdaptivePool3DSimtTilingData {
public:
    int64_t nDim = 0;
    int64_t cDim = 0;
    int64_t dInDim = 0;
    int64_t hInDim = 0;
    int64_t wInDim = 0;
    int64_t dOutDim = 0;
    int64_t hOutDim = 0;
    int64_t wOutDim = 0;
};

class AdaptivePool3dBigKernelTilingData {
public:
    int64_t nc = 1;
    int64_t dInDim = 1;
    int64_t hInDim = 1;
    int64_t wInDim = 1;
    int64_t dOutDim = 1;
    int64_t hOutDim = 1;
    int64_t wOutDim = 1;
    int64_t blockFactor = 1;
    int64_t blockTail = 1;
    int64_t totalIdx = 1;
    int64_t coreNums = 1;
    int64_t maxCount = 1;
    int64_t batchCount = 1;
};

class AdaptivePool3dParaKernelTilingData {
public:
    int64_t dIn = 1;
    int64_t hIn = 1;
    int64_t wIn = 1;
    int64_t dOut = 1;
    int64_t hOut = 1;
    int64_t wOut = 1;
    int64_t useCoreNum = 1;
    int64_t blockFactor = 1;
    int64_t blockTail = 1;
    int64_t ncFactor = 1;
    int64_t doFactor = 1;
    int64_t hoFactor = 1;
    int64_t woFactor = 1;
    int64_t ncOuter = 1;
    int64_t doOuter = 1;
    int64_t hoOuter = 1;
    int64_t woOuter = 1;
    int64_t ncTail = 1;
    int64_t doTail = 1;
    int64_t hoTail = 1;
    int64_t woTail = 1;
    int64_t maxInputSize = 1;
};

}

#endif