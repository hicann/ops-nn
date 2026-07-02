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
 * \file apply_adam_with_amsgrad_v2_tiling_data.h
 * \brief ApplyAdamWithAmsgradV2 TilingData 结构体（arch35）
 *
 * POD struct，host/kernel 共用。
 * fp32-only 单变体；blockFactor=每核基础元素数（含尾），ubFactor=单块元素数。
 * 6 标量缓存（host Tiling 阶段一次读出，kernel 直取）。
 */
#ifndef _APPLY_ADAM_WITH_AMSGRAD_V2_TILING_DATA_H_
#define _APPLY_ADAM_WITH_AMSGRAD_V2_TILING_DATA_H_

#include <cstdint>

struct ApplyAdamWithAmsgradV2TilingData {
    int64_t totalNum = 0;     // var 总元素数
    int64_t blockFactor = 0;  // 每核负责的标准元素数（32B 对齐向上取整）
    int64_t ubFactor = 0;     // UB 内一次循环处理的元素数（tile_numel）

    // 缓存标量参数（统一 fp32 升精度存储）
    float beta1Power = 0.0f;
    float beta2Power = 0.0f;
    float lr = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float epsilon = 0.0f;
};

#endif // _APPLY_ADAM_WITH_AMSGRAD_V2_TILING_DATA_H_
