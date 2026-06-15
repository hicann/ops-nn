/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file multi_add_rms_norm_dynamic_quant_fusion_pass.h
 * \brief quant fusion for multi-add, addrmsnorm
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_FUSION_PASS_H_

#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
using namespace ge;
using namespace fusion;

class __attribute__((visibility("default"))) MultiAddRmsNormDynamicQuantFusionPass : public PatternFusionPass {
protected:
    std::vector<PatternUniqPtr> Patterns() override;
    bool MeetRequirements(const std::unique_ptr<MatchResult>& match_result) override;
    std::unique_ptr<Graph> Replacement(const std::unique_ptr<MatchResult>& match_result) override;

private:
    static constexpr int64_t MAX_ADD_NUM = 4;
    static constexpr int64_t MAX_SMOOTH_NUM = 2;
    int64_t add_num_ = 0;
    int64_t smooth_num_ = 0;
    bool has_reshape_ = false;
};

} // namespace ops

#endif // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_FUSION_PASS_H_
