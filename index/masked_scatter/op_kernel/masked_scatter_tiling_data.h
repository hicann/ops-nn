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

 #ifndef _MASKED_SCATTER_TILING_DATA_H_
 #define _MASKED_SCATTER_TILING_DATA_H_

struct MaskedScatterV1TilingData {
    uint32_t loopNum = 0;
    uint32_t remainNum = 0;
    uint32_t updatesLineNum = 0;
    uint32_t totalUpdatesNum = 0;
    uint32_t maskTileLength = 0;
};
#endif
