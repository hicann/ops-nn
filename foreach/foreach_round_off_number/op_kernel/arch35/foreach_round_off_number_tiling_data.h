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
 * \file foreach_round_off_number_tiling_data.h
 * \brief TilingData struct for foreach_round_off_number operator.
 */

#ifndef FOREACH_ROUND_OFF_NUMBER_TILING_DATA_H
#define FOREACH_ROUND_OFF_NUMBER_TILING_DATA_H

constexpr int64_t MAX_TENSOR_NUM_FOREACH_ROUND = 256;

struct ForeachRoundOffNumberTilingData {
    int32_t needCoreNum;
    int32_t tensorCount;
    int64_t totalElements;
    int64_t tensorElements[MAX_TENSOR_NUM_FOREACH_ROUND];
};

#endif // FOREACH_ROUND_OFF_NUMBER_TILING_DATA_H
