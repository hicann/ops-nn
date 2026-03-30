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
 * \file sorted_sparse_segment_mean_grad_struct.h
 * \brief tiling base data
 */

#ifndef SORTED_SPARSE_SEGMENT_MEAN_GRAD_STRUCT_H
#define SORTED_SPARSE_SEGMENT_MEAN_GRAD_STRUCT_H


class SortedSparseSegmentMeanGradSimtTilingData {
public:
    int64_t needCoreNum;
    int64_t innerSize;
    int64_t segmentNum;
    int64_t outterSize;
    int64_t threadNumX;
    int64_t threadNumY;
    int32_t segThreadNumX;
    int32_t segThreadNumY;
    int32_t indexThreadNumX;
    int32_t indexThreadNumY;
    int64_t perCoreIndicesNum;
    int64_t resIndicesNum;
    int32_t innerCore;
    int64_t perCoreInnerSize;
    int64_t resCoreInnerSize;
    int32_t outputDim0;
};

#endif