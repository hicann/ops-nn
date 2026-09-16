/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file update_tensor_desc_tiling_data.h
 * \brief UpdateTensorDesc TilingData（host / kernel 共用同构布局）。
 *   非模板结构体，无 TPL 模板参数（tilingKey 恒 0）：
 *   host 侧 TilingFunc 填充，kernel 侧经 GET_TILING_DATA_WITH_STRUCT 直取。
 *   仅承载 kernel 无法自取的 attr shape 数据。
 */

#pragma once
#include <cstdint>

constexpr int64_t kDescSize = 128; // 描述缓冲元素数：128 × int64 = 1 KB（kernel RMW 粒度）
constexpr int64_t kDimBaseIdx = 3; // rank 写入下标基址：y[3] = N
constexpr int64_t kMaxRank = 124;  // rank(attr shape) 上限 = kDescSize - kDimBaseIdx - 1

struct UpdateTensorDescTilingData {
    int64_t rank;            // N = len(attr shape) ∈ [1, kMaxRank]
    int64_t shape[kMaxRank]; // attr shape 各维大小，仅前 rank 个槽位有效
};
