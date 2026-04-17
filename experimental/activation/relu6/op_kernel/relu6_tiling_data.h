/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file relu6_tiling_data.h
 * \brief Relu6 TilingData 结构体定义（arch35 = Ascend950）
 *
 * 迭代一：仅 float16 单 dtype 骨架
 * 迭代二：新增 dataType 字段，支持多 dtype 分发
 * 字段说明：
 *   - totalNum:   输入张量展平后的总元素数量
 *   - blockFactor: 每个 AI Core 处理的元素数量
 *   - ubFactor:    UB 单次循环处理的元素数量
 *   - dataType:    数据类型标识（0=float16, 1=float, 2=int32, 3=bfloat16）
 */

#ifndef _RELU6_TILING_DATA_H_
#define _RELU6_TILING_DATA_H_

#include <cstdint>

struct Relu6TilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 0;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
    int32_t dataType = 0;     // 数据类型：0=float16, 1=float, 2=int32, 3=bfloat16
};

#endif // _RELU6_TILING_DATA_H_
