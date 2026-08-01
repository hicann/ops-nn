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
 * \file add_rms_norm_dynamic_quant_v2_fusion_pass.h
 * \brief Fusion pass for AddRmsNorm + Cast + DynamicQuant -> AddRmsNormDynamicQuantV2.
 */
#ifndef OPS_NORM_ADD_RMS_NORM_DYNAMIC_QUANT_V2_FUSION_PASS_H_
#define OPS_NORM_ADD_RMS_NORM_DYNAMIC_QUANT_V2_FUSION_PASS_H_

#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
using namespace ge;
using namespace fusion;

class __attribute__((visibility("default"))) AddRmsNormDynamicQuantV2FusionPass : public PatternFusionPass {
protected:
    std::vector<PatternUniqPtr> Patterns() override;

    bool MeetRequirements(const std::unique_ptr<MatchResult>& matchResult) override;

    GraphUniqPtr Replacement(const std::unique_ptr<MatchResult>& matchResult) override;
};
} // namespace ops

#endif // OPS_NORM_ADD_RMS_NORM_DYNAMIC_QUANT_V2_FUSION_PASS_H_
