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
/**
 * \file threshold_v2.cpp
 * \brief ThresholdV2 Kernel entry
 *
 * Formula: y = (x > threshold) ? x : value
 */
#include "threshold_v2.h"

template <int BUFFER_MODE, int DTYPE_SELF_ID>
__global__ __aicore__ void threshold_v2(GM_ADDR self, GM_ADDR out,
                                         GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ThresholdV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(ThresholdV2TilingData, tilingData, tiling);
    NsThresholdV2::ThresholdV2<DTYPE_SELF, BUFFER_MODE> op;
    op.Init(self, out, &tilingData);
    op.Process();
}
