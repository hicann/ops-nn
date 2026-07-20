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
 * \file masked_scatter_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __MASKED_SCATTER_TILING_DATA_H__
#define __MASKED_SCATTER_TILING_DATA_H__

struct MaskedScatterTilingData {
    int64_t numElemX;
    int64_t numElemMask;
    int64_t numElemUpdates;
    int64_t tilingCoreNum;
};

#endif // __MASKED_SCATTER_TILING_DATA_H__
