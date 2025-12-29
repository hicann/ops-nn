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
 * \file mse_loss_tiling_def.h
 * \brief tiling data struct
 */

#ifndef __MSE_LOSS_TILING_DATA_H__
#define __MSE_LOSS_TILING_DATA_H__

#include "atvoss/elewise/elewise_base_struct.h"

using namespace Ops::Base;
struct ReduceOpTilingDataV2 {
    uint64_t factorACntPerCore;
    uint64_t factorATotalCnt;
    uint64_t ubFactorA;
    uint64_t factorRCntPerCore;
    uint64_t factorRTotalCnt;
    uint64_t ubFactorR;
    uint64_t groupR;
    uint64_t outSize;
    uint64_t basicBlock;
    uint64_t resultBlock;
    int32_t coreNum;
    int32_t useNddma;
    float meanVar;
    uint64_t shape[Ops::Base::ReduceOpTmpl::MAX_DIM];
    uint64_t stride[Ops::Base::ReduceOpTmpl::MAX_DIM];
    uint64_t dstStride[Ops::Base::ReduceOpTmpl::MAX_DIM];
};

struct MseLossTilingData {
    EleBaseTilingData baseTiling;
    ReduceOpTilingDataV2 reduceTiling;
};

#endif