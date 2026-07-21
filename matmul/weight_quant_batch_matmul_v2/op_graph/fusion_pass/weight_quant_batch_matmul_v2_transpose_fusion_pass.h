/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_WEIGHT_QUANT_BATCH_MATMUL_V2_TRANSPOSE_FUSION_PASS_H
#define NN_WEIGHT_QUANT_BATCH_MATMUL_V2_TRANSPOSE_FUSION_PASS_H

#include "ge/fusion/pass/fusion_base_pass.h"

namespace ops {

class __attribute__((visibility("default"))) WeightQuantBatchMatmulV2TransposeFusionPass
    : public ge::fusion::FusionBasePass {
public:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;

protected:
    virtual bool CheckPlatForm();
};

class __attribute__((visibility("default"))) WeightQuantBatchMatmulV2TransposeNZFusionPass
    : public WeightQuantBatchMatmulV2TransposeFusionPass {
protected:
    bool CheckPlatForm() override;
};

} // namespace ops

#endif // NN_WEIGHT_QUANT_BATCH_MATMUL_V2_TRANSPOSE_FUSION_PASS_H
