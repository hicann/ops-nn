/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef A_CONV2D_MUL_FUSION_PASS_H
#define A_CONV2D_MUL_FUSION_PASS_H

#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "graph/gnode.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace AConv2dMulFusionConsts {

const ge::AscendString MUL = "Mul";
const std::string FUSION_NAME = "AConv2dMulFusion";

constexpr int32_t MUL_INPUT_NUMS = 2;
constexpr int32_t CONV2D_DIM_SIZE = 4;
constexpr int32_t CONV3D_DIM_SIZE = 5;
constexpr int32_t NCHW_C_POSITION = 1;
constexpr int32_t NHWC_C_POSITION = 3;
constexpr int32_t NDHWC_C_POSITION = 4;
constexpr int32_t NCDHW_C_POSITION = 1;
constexpr size_t SINGLE_CONSUMER_CNT = 1;
const std::string FILTER_MUL_NAME_SUFFIX = "_filter";
const std::string BIAS_MUL_NAME_SUFFIX = "_bias";

const std::map<std::string, NpuArch> ND_SOC_LIST = {{"Ascend950", NpuArch::DAV_3510}};

} // namespace AConv2dMulFusionConsts

class __attribute__((visibility("default"))) AConv2dMulFusion : public ConvFusionBasePass {
protected:
    void InitMember() override;
    bool CheckMatchStructure(const ge::GNode& matchNode) override;
    bool MeetRequirements(const ge::GNode& convNode) override;
    std::set<ge::AscendString> GetNodeTypes() const override;
    void PrintGraphStructure() const override;
    bool ConvFusionReplaceImpl(ge::GraphPtr& graph, ge::GNode& convNode, ge::CustomPassContext& passContext) override;

private:
    bool DetermineMulInputIndices();
    bool CheckWeightConst(const ge::GNode& convNode);
    bool CheckInputConst(const ge::GNode& convNode, int32_t inputIdx, const char* weightName, ge::GNodePtr& weightNode);
    bool CheckScaleShape(const ge::GNode& convNode) const;
    bool CheckScaleShapeConv2d() const;
    bool CheckScaleShapeConv3d() const;
    bool CreateScaleMulNode(ge::Graph& graph, const ge::AscendString& name, const ge::TensorDesc& dataDesc,
                            const ge::TensorDesc& outDesc, ge::GNode& scaleMulNode) const;
    bool RelinkConvOutputToMulConsumers(ge::Graph& graph, ge::GNode& convNode);
    bool InsertWeightMul(ge::Graph& graph, ge::GNode& convNode, const ge::GNodePtr& weightNode, int32_t weightInputIdx,
                         const std::string& nameSuffix);

    ge::GNodePtr mulNode = nullptr;
    ge::GNodePtr scaleNode = nullptr;
    ge::GNodePtr filterNode = nullptr;
    ge::GNodePtr biasNode = nullptr;
    int32_t mulNonConstInputIdx = -1;
    int32_t mulConstInputIdx = -1;
    bool isDav3510 = false;
    NpuArch npuArch = NpuArch::DAV_RESV;
    std::vector<ge::GNode> insertedMulNodes;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // A_CONV2D_MUL_FUSION_PASS_H
