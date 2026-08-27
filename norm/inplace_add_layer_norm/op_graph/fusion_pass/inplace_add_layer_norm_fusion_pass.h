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
 * \file inplace_add_layer_norm_fusion_pass.h
 * \brief AddLayerNorm --> InplaceAddLayerNorm (graph base route)
 *
 *   x1 x2 gamma beta [bias]              x1 x2 gamma beta [bias]
 *     \  \   |    /    /                   \  \   |    /    /
 *      AddLayerNorm            -->        InplaceAddLayerNorm
 *     /   |    |    \                     /    |    |    \
 *    y  mean  rstd   x                  x1'  mean  rstd   x2'
 *
 */
#ifndef OPS_NORM_INPLACE_ADD_LAYER_NORM_OP_GRAPH_FUSION_PASS_INPLACE_ADD_LAYER_NORM_FUSION_PASS_H_
#define OPS_NORM_INPLACE_ADD_LAYER_NORM_OP_GRAPH_FUSION_PASS_INPLACE_ADD_LAYER_NORM_FUSION_PASS_H_

#include "ge/fusion/pass/fusion_base_pass.h"

namespace ops {
using namespace ge;
using namespace ge::fusion;

class __attribute__((visibility("default"))) ZInplaceAddLayerNormFusionPass : public FusionBasePass {
public:
    Status Run(GraphPtr& graph, CustomPassContext& passContext) override;
};
} // namespace ops

#endif // OPS_NORM_INPLACE_ADD_LAYER_NORM_OP_GRAPH_FUSION_PASS_INPLACE_ADD_LAYER_NORM_FUSION_PASS_H_
