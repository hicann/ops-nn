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

#include <map>
#include <memory>
#include <set>

#include "conv_fusion_utils_pass.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/subgraph_boundary.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {

// Conv fusion base pass is used for single node matching, multi nodes replacing
class __attribute__((visibility("default"))) ConvFusionBasePass : public ge::fusion::FusionBasePass {
public:
    ge::Status Run(ge::GraphPtr &graph, ge::CustomPassContext &pass_context) override;

protected:
    virtual std::unique_ptr<ge::fusion::SubgraphBoundary> ConstructBoundary(const ge::GNode &convNode) = 0;
    // Required in quantization scene for creating FixPipe node.
    virtual bool FixpipeFusionImpl(
        ge::GraphPtr &graph, ge::GNode &convNode, const ge::CustomPassContext &pass_context) = 0;
    virtual void InitMember() = 0;
    virtual bool MeetRequirements(const ge::GNode &convNode) = 0;
    virtual ge::AscendString GetNodeType() const = 0;
    virtual std::map<std::string, NpuArch> GetSocSupportList() const = 0;
    virtual void PrintGraphStructure() const = 0;
    virtual ge::fusion::GraphUniqPtr Replacement(const ge::GNode &convNode) = 0;

protected:
    NpuArch npuArch;
    ConvFusionUtils::ConvDescInfo convDescInfo = {};
};

constexpr uint32_t CONV_NOT_CHANGED = 53608649;
} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // NN_CONV_FUSION_BASE_PASS_H