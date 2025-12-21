/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_ADA_LAYER_NORM_V2_H
#define OP_API_INC_LEVEL0_ADA_LAYER_NORM_V2_H

#include "norm/ada_layer_norm/op_host/op_api/ada_layer_norm.h"
#include "opdev/op_executor.h"

namespace l0op {
struct AdaLayerNormV2InputTensor {
    const aclTensor* x;
    const aclTensor* scale;
    const aclTensor* shift;
    const aclTensor* weightOptional;
    const aclTensor* biasOptional;
};

struct AdaLayerNormV2OutputTensor {
    aclTensor* out;
    aclTensor* meanOutOptional;
    aclTensor* rstdOutOptional;
    bool useV2;
};

const std::tuple<aclTensor*, aclTensor*, aclTensor*> AdaLayerNormV2(AdaLayerNormV2InputTensor inputTensor, 
                                                                    float epsilon, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_ADA_LAYER_NORM_V2_H