/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file smooth_l1_loss_v2_tiling_struct.h
 * \brief
 */
#ifndef SMOOTH_L1_LOSS_V2_TILINGDATA_H
#define SMOOTH_L1_LOSS_V2_TILINGDATA_H

#include "atvoss/elewise/elewise_base_struct.h"
#include "atvoss/reduce/reduce_tiling_data.h" 

using namespace Ops::Base;

namespace SmoothL1LossV2 {

struct SmoothL1LossV2TilingData {
    EleBaseTilingData baseTiling;
    ReduceOpTilingData reduceTiling;
    float Sigma;
    float MultiplyValue;
    float AddsValue;
};

}
#endif //SMOOTH_L1_LOSS_V2_TILING_STRUCT_H