/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_ASCEND_QUANT_V2_SCATTER_FUSION_PASS_H
#define NN_ASCEND_QUANT_V2_SCATTER_FUSION_PASS_H

#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
using namespace ge;
using namespace fusion;

/**
 * @brief AscendQuantV2 + Scatter -> QuantUpdateScatter fusion pass
 * @details Fusion pattern:
 *          x, scale, offset      var, indices, updates, scale, offset
 *                 |                          |
 *                 |                          |
 *           ascend_quant_v2          quant_update_scatter
 *                 |                          |
 *   var indices   |                          |
 *        \     \   |                          |
 *          \    \  |                          |
 *           \   scatter   ======>>>>          var
 *                 |
 *                 |
 *               var
 *
 * The key transformation:
 * - AscendQuantV2 has 2-3 inputs: x, scale, offset(optional)
 * - Scatter has 3 inputs: var, indices, updates(from AscendQuantV2 output)
 * - QuantUpdateScatter has 4-5 inputs: var, indices, x, scale, offset(optional)
 * - Attributes: reduce="update", axis, quant_axis=-1, reciprocal_scale=true
 */
class __attribute__((visibility("default"))) AscendQuantV2ScatterFusionPass : public PatternFusionPass {
protected:
    std::vector<PatternUniqPtr> Patterns() override;

    bool MeetRequirements(const std::unique_ptr<MatchResult>& match_result) override;

    std::unique_ptr<Graph> Replacement(const std::unique_ptr<MatchResult>& match_result) override;
};

} // namespace ops

#endif // NN_ASCEND_QUANT_V2_SCATTER_FUSION_PASS_H