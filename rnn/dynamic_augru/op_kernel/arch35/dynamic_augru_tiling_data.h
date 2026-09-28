/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_RNN_DYNAMIC_AUGRU_OP_KERNEL_ARCH35_DYNAMIC_AUGRU_TILING_DATA_H_
#define OPS_RNN_DYNAMIC_AUGRU_OP_KERNEL_ARCH35_DYNAMIC_AUGRU_TILING_DATA_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// Serialized layout of the host DynamicAUGRUTilingData; field order is ABI.
struct DynamicAUGRUTilingData {
    int64_t timeSize;
    int64_t batchSize;
    int64_t inputSize;
    int64_t hiddenSize;
    int64_t tileHidden;
    uint32_t hasBiasInput;
    uint32_t hasBiasHidden;
    uint32_t hasInitH;
    uint32_t sequenceMode;
    uint32_t gateOrder;
    uint32_t stateType;
    uint32_t usedAicCoreNum;
    uint32_t usedAivCoreNum;
    uint32_t blockSize;
    uint32_t vectorLength;
    uint64_t inputProjectionOffset;
    uint64_t hiddenProjectionOffset;
    uint64_t stateFp32Offset;
    uint64_t weightHiddenFp32Offset;
    uint64_t userWorkspaceSize;
    TCubeTiling inputMMTiling;
    TCubeTiling hiddenMMTiling;
};

#endif
