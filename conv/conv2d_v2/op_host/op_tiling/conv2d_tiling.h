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
 * \file conv2d_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_TILING_H

#include "register/tilingdata_base.h"
#include "arch35/conv2d_api_tilingdata.h"
#include "arch35/conv2d_api_tiling.h"
#include "tiling/tiling_api.h"

/**
 * Conv2DRunInfo define here
 * BEGIN_TILING_DATA_DEF and TILING_DATA_FIELD_DEF should same with CANN
*/

namespace optiling {
    BEGIN_TILING_DATA_DEF(Conv2DRunInfo)
    TILING_DATA_FIELD_DEF(uint64_t, hin);
    TILING_DATA_FIELD_DEF(uint64_t, win);
    TILING_DATA_FIELD_DEF(uint64_t, hout);
    TILING_DATA_FIELD_DEF(uint64_t, wout);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);
    TILING_DATA_FIELD_DEF(uint32_t, batchDim);
    TILING_DATA_FIELD_DEF(uint32_t, groupDim);
    TILING_DATA_FIELD_DEF(uint32_t, nDim);
    TILING_DATA_FIELD_DEF(uint32_t, hoDim);
    TILING_DATA_FIELD_DEF(uint32_t, woDim);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
    TILING_DATA_FIELD_DEF(uint32_t, padTop);
    TILING_DATA_FIELD_DEF(uint32_t, padLeft);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
    TILING_DATA_FIELD_DEF(uint32_t, enlarge);
    TILING_DATA_FIELD_DEF(uint32_t, cinOpt);
    TILING_DATA_FIELD_DEF(uint32_t, coutOpt);
    TILING_DATA_FIELD_DEF(uint32_t, groupOpt);
    TILING_DATA_FIELD_DEF(uint8_t, hasBias);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Conv2DRunInfoOp, Conv2DRunInfo)

    BEGIN_TILING_DATA_DEF(Conv2DTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TConv2DTiling, conv2dApiTiling);
    TILING_DATA_FIELD_DEF_STRUCT(Conv2DRunInfo, conv2dRunInfo);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Conv2DV2, Conv2DTilingData)
    REGISTER_TILING_DATA_CLASS(QuantConv2D, Conv2DTilingData) 
    REGISTER_TILING_DATA_CLASS(ExtendConv2D, Conv2DTilingData)
}
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_TILING_H