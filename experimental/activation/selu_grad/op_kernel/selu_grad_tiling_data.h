/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file selu_grad_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SELUGRAD_TILING_DATA_H_
#define _SELUGRAD_TILING_DATA_H_

struct SeluGradTilingData {
    uint64_t totalNum = 0;    // 总元素数量
    uint32_t blockFactor = 1; // 每个核处理的元素数量
    uint32_t ubFactor = 0;    // 每次 UB 循环处理的元素数量
};
#endif
