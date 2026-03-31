/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_fill_struct.h
 * \brief
 */

#ifndef INPLACE_INDEX_FILL_STRUCT_H_
#define INPLACE_INDEX_FILL_STRUCT_H_

namespace InplaceIndexFill {
class InplaceIndexFillSimtTilingData {
public:
    int64_t tilingKeySimt = 0;
    int64_t preDimProduct = 0;  // x在dim轴前面的轴维度乘积
    int64_t dimSize = 0;        // dim轴的维度
    int64_t postDimProduct = 0; // x在dim轴后面的轴维度乘积
    int64_t indicesNum = 0; // indices的长度
    int64_t totalDataSize = 0;  // 需处理的总数据量
    int64_t usedCoreNum = 0;    // 使用核个数
    int64_t perBlockData = 0;   // 核处理的数据量
    int64_t tailBlockData = 0;  // 尾核处理的数据量
    int64_t threadNum = 0;      // 核内处理线程数
};

class InplaceIndexFillSimdTilingData {
public:
    int64_t tilingKey = 0;
    int64_t preDimProduct = 0;   // x在dim轴前面的维度乘积
    int64_t dimSize = 0;         // dim轴的维度
    int64_t postDimProduct = 0;  // x在dim轴后面的轴维度乘积
    int64_t indicesNum = 0;      // indicesNum

    int64_t perBlockData = 0;
    int64_t tailBlockData = 0;
    int64_t tailBlockNum = 0;
    int64_t qBlockFactor = 0;
    int64_t qUsedCoreNum = 0;
    int64_t usedCoreNum = 0;

    //UB参数
    int64_t qBufferSize = 0;
    int64_t indicesBufferSize = 0;
    int64_t indicesUbFactor = 0;
    int64_t qUbFactor = 0;
    int64_t qLoopSize = 0;
    int64_t qUbTailFactor = 0;
};

}   // namspace InplaceIndexFill
#endif //INPLACE_INDEX_FILL_STRUCT_H_