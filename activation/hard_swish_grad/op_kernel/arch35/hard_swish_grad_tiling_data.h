/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file hard_swish_grad_tiling_data.h
 * \brief HardSwishGrad tiling data struct
 */

#ifndef _HARD_SWISH_GRAD_TILING_DATA_H_
#define _HARD_SWISH_GRAD_TILING_DATA_H_

struct HardSwishGradTilingData {
    int64_t totalNum = 0;      // total number of elements
    int64_t blockFactor = 0;   // number of elements per AI Core
    int64_t ubFactor = 0;      // number of elements per UB iteration
};

#endif
