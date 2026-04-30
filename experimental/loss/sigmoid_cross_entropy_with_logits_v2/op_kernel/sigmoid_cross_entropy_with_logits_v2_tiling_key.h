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
 * \file sigmoid_cross_entropy_with_logits_v2_tiling_key.h
 * \brief sigmoid_cross_entropy_with_logits_v2 tiling key declare
 */

#ifndef __SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_TILING_KEY_H__
#define __SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_TILING_KEY_H__

// 对应 float16 (half)
#define TILING_KEY_HALF 0

// 对应 float (float32)
#define TILING_KEY_FLOAT 1

// 对应 bfloat16
#define TILING_KEY_BFLOAT16 2

#endif // __SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_TILING_KEY_H__
