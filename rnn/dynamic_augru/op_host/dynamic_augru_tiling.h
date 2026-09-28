/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_RNN_DYNAMIC_AUGRU_OP_HOST_DYNAMIC_AUGRU_TILING_H_
#define OPS_RNN_DYNAMIC_AUGRU_OP_HOST_DYNAMIC_AUGRU_TILING_H_

#include <cstdint>
#include <string>
#include "register/op_impl_registry.h"

namespace optiling {
enum class DynamicAUGRUSequenceMode : uint32_t {
    NONE = 0,
    LENGTH = 1,
    MASK = 2,
};

enum class DynamicAUGRUGateOrder : uint32_t {
    ZRH = 0,
    RZH = 1,
};

struct DynamicAUGRUCompileInfo {
    uint32_t aicCoreNum = 0;
    uint32_t aivCoreNum = 0;
    uint64_t ubSize = 0;
    uint64_t blockSize = 0;
    uint32_t vectorLength = 0;
    bool isArch35 = false;
};

struct DynamicAUGRUAttributes {
    std::string direction = "UNIDIRECTIONAL";
    int64_t cellDepth = 1;
    float keepProb = 1.0F;
    float cellClip = -1.0F;
    int64_t numProj = 0;
    bool timeMajor = true;
    std::string activation = "tanh";
    std::string gateOrder = "zrh";
    bool resetAfter = true;
    bool isTraining = true;
};

const char* InvalidDynamicAUGRUAttribute(const DynamicAUGRUAttributes& attrs);
} // namespace optiling
#endif // OPS_RNN_DYNAMIC_AUGRU_OP_HOST_DYNAMIC_AUGRU_TILING_H_
