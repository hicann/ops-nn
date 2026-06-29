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
 * \file inplace_update_fusion_pass.h
 * \brief InplaceUpdate fusion pass (InplaceUpdate --> TensorMove & ScatterUpdate)
 *
 *   x        indices    updates              x    indices  updates
 *    \         |          /                  |       /       /
 *     \        |         /                   |      /       /
 *      InplaceUpdate            -->     TensorMove /       /
 *          |                                |     /       /
 *          y                                |    /       /
 *                                       ScatterUpdate
 *                                            |
 *                                            y
 */
#ifndef NN_INPLACE_UPDATE_FUSION_PASS_H
#define NN_INPLACE_UPDATE_FUSION_PASS_H

#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
using namespace ge;
using namespace fusion;

class __attribute__((visibility("default"))) AInplaceUpdateFusionPass : public PatternFusionPass {
protected:
    std::vector<PatternUniqPtr> Patterns() override;
    bool MeetRequirements(const std::unique_ptr<MatchResult>& match_result) override;
    std::unique_ptr<Graph> Replacement(const std::unique_ptr<MatchResult>& match_result) override;
};
} // namespace ops

#endif // NN_INPLACE_UPDATE_FUSION_PASS_H