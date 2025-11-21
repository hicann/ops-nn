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
 * \file conv3d_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "conv3d_api_tiling_algorithm.h"
#include "conv3d_api_tiling_algorithm_pointwise.h"
#include "conv3d_api_tiling_algorithm_hw_mode.h"
#include "conv3d_api_tiling_base.h"
#include "conv3d_api_tiling.h"

namespace Conv3dApiTiling {

int64_t Conv3dTiling::GetTiling(optiling::TConv3DTiling &tiling)
{
    int64_t ret = Compute();
    if (ret == INVALID_VALUE) {
        TILING_ERROR_LOG("can not gen conv3d api tiling");
        return INVALID_VALUE;
    }

    SetFinalTiling(tiling);
    PrintTilingData();
    return ret;
}

int64_t Conv3dTiling::Compute()
{
    bool isPointWise = (this->descInfo.fMapType.format == ConvFormat::NCDHW);
    if (!isPointWise && !CheckInputParam()) {
        return INVALID_VALUE;
    }

    if (isPointWise && !CheckInputParamPointWise()) {
        return INVALID_VALUE;
    }
    // get cube info
    GetCubeInfo();
    // cal output and check valid
    if (!ShapeInitCalc()) {
        return INVALID_VALUE;
    }
    if (!CheckParamsOverflow()) {
        return INVALID_VALUE;
    }
    if (isPointWise) {
        Conv3dTilingAlgorithmPointWise tilingAlgo(this);
        int64_t ret = tilingAlgo.Process();
        return ret;
    } else if (outputOrder_ == M_Mode) {
        Conv3dTilingAlgorithm tilingAlgo(this);
        int64_t ret = tilingAlgo.Process();
        return ret;
    } else {
        Conv3dTilingAlgorithmHwMode tilingAlgo(this);
        int64_t ret = tilingAlgo.Process();
        return ret;
    }
}

} // namespace Conv3dApiTiling