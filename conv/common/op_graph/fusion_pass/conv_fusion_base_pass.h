/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_CONV_FUSION_BASE_PASS_H
#define NN_CONV_FUSION_BASE_PASS_H

#include <memory>

#include "conv_fusion_utils_pass.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/subgraph_boundary.h"
#include "ge/ge_api_error_codes.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {

// Type-based node match in Run(); optional structure check via CheckMatchStructure().
class __attribute__((visibility("default"))) ConvFusionBasePass : public ge::fusion::FusionBasePass {
public:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& pass_context) override;

protected:
    virtual void InitMember() = 0;
    // Further structure match beyond single-node type match. Default: true.
    // Override for multi-node cascade topologies; return false to skip current node.
    virtual bool CheckMatchStructure(const ge::GNode& matchNode) { return true; }
    virtual bool MeetRequirements(const ge::GNode& convNode) = 0;
    virtual std::set<ge::AscendString> GetNodeTypes() const = 0;
    virtual void PrintGraphStructure() const = 0;

    // Stage 1: pre-process graph. Default: no-op, return SUCCESS.
    // CONV_NOT_CHANGED skips current node; FAILED aborts the pass.
    virtual ge::Status ConvFusionPreImpl(ge::GraphPtr& graph, ge::GNode& convNode, ge::CustomPassContext& passContext)
    {
        return ge::SUCCESS;
    }

    // Stage 2: subgraph replace entry. Default: skip replace, return true.
    // Override and call DefaultConvFusionReplaceImpl() for standard replace.
    virtual bool ConvFusionReplaceImpl(ge::GraphPtr& graph, ge::GNode& convNode, ge::CustomPassContext& passContext)
    {
        return true;
    }

    // Override with DefaultConvFusionReplaceImpl(). nullptr means replace failed.
    virtual std::unique_ptr<ge::fusion::SubgraphBoundary> ConstructBoundary(const ge::GNode& convNode)
    {
        return nullptr;
    }
    virtual ge::fusion::GraphUniqPtr Replacement(const ge::GNode& convNode) { return nullptr; }

    // ConstructBoundary -> Replacement -> SubgraphRewriter::Replace.
    bool DefaultConvFusionReplaceImpl(const ge::GNode& convNode, ge::CustomPassContext& passContext);

protected:
    NpuArch npuArch = NpuArch::DAV_RESV;
    ConvFusionUtils::ConvDescInfo convDescInfo = {};
};

constexpr uint32_t CONV_NOT_CHANGED = 53608649;
} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // NN_CONV_FUSION_BASE_PASS_H
