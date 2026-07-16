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
 * \file log_softmax_grad_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __LOG_SOFTMAX_GRAD_TILING_DATA_H__
#define __LOG_SOFTMAX_GRAD_TILING_DATA_H__

struct LogSoftmaxGradTilingData {
    uint64_t singleBufElems;
    uint64_t mergedDim0;
    uint64_t mergedDim1;
    uint64_t mergedDim2;
    uint64_t dim0Tile;
    uint64_t dim1Tile;
    uint64_t dim2Tile;
    uint64_t totalElems;
    uint64_t dim0LoopTime;
    uint64_t dim0Remained;
    uint64_t dim1LoopTime;
    uint64_t dim1Remained;
    uint64_t dim2LoopTime;
    uint64_t dim2Remained;
};

#endif
