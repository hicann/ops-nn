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
 * \file optimized_transducer_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __OPTIMIZED_TRANSDUCER_TILLING_DATA_H__
#define __OPTIMIZED_TRANSDUCER_TILLING_DATA_H__

struct OptimizedTransducerTilingData {
    uint64_t maxT;
    uint64_t maxU;
    uint64_t V;
    uint64_t batch_size;
    uint64_t bigCoreNum;
    uint64_t bigCoreProcessNum;
    uint64_t smallCoreProcessNum;
    int blank;
    float clamp;
    bool fused_log_softmax;
    bool requires_grad;
};
#endif
