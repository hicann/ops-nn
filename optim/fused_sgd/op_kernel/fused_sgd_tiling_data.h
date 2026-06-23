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
 * \file fused_sgd_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _FUSED_SGD_TILING_DATA_H_
#define _FUSED_SGD_TILING_DATA_H_

struct FusedSgdTilingData {
    float weightDecay;
    float momentum;
    float lr;
    float dampening;
    uint64_t nesterov;
    uint64_t maximize;
    uint64_t isFirstStep;
    uint64_t useGradScale;
    uint64_t useMomentum;
    uint64_t tensorNum;
    uint64_t tensorsPerCore;
    uint64_t usedCoreNum;
    uint64_t coreCalcMax;
};
#endif
