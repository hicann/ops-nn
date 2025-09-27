/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad_v2_struct.h
 * \brief tiling base data
 */

#ifndef EMBEDDING_DENSE_GRAD_V2_STRUCT_H
#define EMBEDDING_DENSE_GRAD_V2_STRUCT_H

class EmbeddingDenseGradV2TilingData4RegBase {
public:
    int64_t scatterDimSize;
    int64_t elewiseDimSize;
    int64_t scatterDimLoopNum;
    int64_t elewiseDimLoopNum;
    int64_t elewiseDimOuterLoopNum;
    int64_t normalBlockLoopNum;
    int64_t tailBlockLoopNum;
    int64_t numWeights;
    int64_t paddingIdx;
    uint32_t scatterFactor;
    uint32_t elewiseFactor;
    uint32_t usedCoreNum;
    uint32_t blockNumForLastProcess;
    uint32_t elewiseDimLastPerUbProcessNum;
    uint32_t elewiseDimLastProcessTailNum;
    uint64_t elewiseDimLastNormalLoopNum;
    uint64_t elewiseDimLastTailLoopNum;
};

#endif