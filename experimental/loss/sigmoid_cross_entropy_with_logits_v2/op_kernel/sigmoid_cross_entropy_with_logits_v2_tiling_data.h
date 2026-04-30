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
 * \file sigmoid_cross_entropy_with_logits_v2_tiling_data.h
 * \brief tiling data struct
 */

#ifndef SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_TILING_DATA_H
#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct SigmoidCrossEntropyWithLogitsV2TilingData {
	uint32_t totalLength;
	uint32_t tileNum;
	uint32_t tileLength;
	uint8_t has_weight;
	uint8_t has_pos_weight;
	uint32_t dtype_enum;
};

#endif // SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_TILING_DATA_H
