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
 * \file sparse_to_dense_struct.h
 * \brief tiling base data
 */

#ifndef SPARSE_TO_DENSE_STRUCT_H
#define SPARSE_TO_DENSE_STRUCT_H

class SparseToDenseTilingData {
public:
    int64_t numDims;
    int64_t normCoreHandleDefaultValues;
    int64_t defaultUbFactor;
    int64_t normBlockTailLoopSize;
    int64_t tailBlockTailLoopSize;
    int64_t normBlockLoop;
    int64_t tailBlockLoop;
    int64_t normCoreHandleSparses;
    int64_t tailCoreHandleSparses;
    int64_t defaultValueUsedCoreNum;
    int64_t sparseUsedCoreNum;
    bool isNeedSyncall;
    bool validateIndices;
};

#endif