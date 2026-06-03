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
 * \file foreach_copy_tiling_data.h
 * \brief tiling data struct for foreach_copy operator (kernel side)
 */

#ifndef FOREACH_COPY_TILING_DATA_H
#define FOREACH_COPY_TILING_DATA_H

constexpr int32_t MAX_TENSOR_NUM_FOREACH_COPY = 128;

class ForeachCopyTilingData {
public:
    int32_t needCoreNum;       // 需要的核数（≤ maxCoreNum）
    int32_t tensorNum;         // 张量列表中的张量总数
    int32_t tensorStart;       // 本核处理的起始张量索引（0-based）
    int64_t tensorOffset;      // 本核在起始张量内的元素偏移
    int64_t totalElements;     // 所有张量的元素总数（用于核内 stride 循环终止判断）
    uint64_t dtypeKey;         // 数据类型枚举值，用于 kernel 分发到对应模板实例
    int64_t numels[MAX_TENSOR_NUM_FOREACH_COPY];  // 每个张量的元素数量
};

#endif // FOREACH_COPY_TILING_DATA_H
