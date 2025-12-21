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
 * \file linear_index_v2_common.h
 * \brief
 */
#ifndef LINEAR_INDEX_V2_COMMON_H
#define LINEAR_INDEX_V2_COMMON_H
#include "kernel_operator.h"
#define IS_CAST_INT (isSame<T, int64_t>::value)

template <typename Tp, Tp v>
struct integralConstant {
    static constexpr Tp value = v;
};
using true_type = integralConstant<bool, true>;
using false_type = integralConstant<bool, false>;
template <typename, typename>
struct isSame : public false_type {
};
template <typename Tp>
struct isSame<Tp, Tp> : public true_type {
};

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_SIZE = 32;
#endif // LINEAR_INDEX_V2_COMMON_H