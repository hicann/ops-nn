/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file dynamic_quant_update_scatter_tiling_data.h
 * \brief DynamicQuantUpdateScatter arch35 (Ascend950) plain tiling-data struct.
 *
 * The field order remains compatible with the operator's existing tiling schema.
 * Arch35 uses eachCoreBsNum/lastCoreBsNum for complete scatter row groups and
 * innerLoopEle for the UB tile length used by the two-pass RegBase kernel.
 */

#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_ARCH35_TILING_DATA_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_ARCH35_TILING_DATA_H

#include <cstdint>

struct DynamicQuantUpdateScatterRegbaseTilingData {
    int64_t coreNum = 0;
    int64_t eachCoreBsNum = 0;
    int64_t lastCoreBsNum = 0;
    int64_t updateAxisShape = 0;
    int64_t srcBsStride = 0;
    int64_t dstBsStride = 0;
    int64_t indexElements = 0;
    int64_t numHead = 0;
    int64_t sizePerHead = 0;
    int64_t dataAxisShape = 0;
    int64_t numOneBlock = 0;
    int64_t innerLoopEle = 0;
    int64_t indicesShapeRank = 0;
    int64_t srcFirBsStride = 0;
    int64_t dstFirSecBsStride = 0;
    int64_t updateDim0 = 0;
    int64_t updateDim1 = 0;
    int64_t varElements = 0;
    int64_t varScalesElements = 0;
    int64_t updatesElements = 0;
    int64_t quantReptNum = 0;
    int64_t varOrigLastDimSize = 0;
    int64_t sizeSrcPerHead = 0;
    int64_t innerLoopFullRpt = 0;
    int64_t innerLoopTimes = 0;
    int64_t innerLoopTail = 0;
    int64_t innerLoopTimesLastCore = 0;
    int64_t innerLoopTailLastCore = 0;
};

#endif // DYNAMIC_QUANT_UPDATE_SCATTER_ARCH35_TILING_DATA_H
