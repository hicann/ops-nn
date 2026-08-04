/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_CONV2D_CONCAT_FUSION_PASS_H
#define SPLIT_CONV2D_CONCAT_FUSION_PASS_H

#include <set>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "ge/es_graph_builder.h"
#include "ge/es_tensor_holder.h"
#include "ge/fusion/subgraph_boundary.h"
#include "graph/gnode.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace SplitConv2dConcatFusion {

const ge::AscendString ATTR_N = "N";
const ge::AscendString CONCAT = "Concat";
const ge::AscendString CONCAT_HOST_OP = "Concatv2HostCpuOp";
const ge::AscendString CONCAT_V2 = "ConcatV2";
const ge::AscendString SPLIT = "Split";
const ge::AscendString SPLIT_V = "SplitV";

const std::set<ge::AscendString> ALLOWED_CONST_LIST = {
    "AscendWeightQuant", "Const", "Constant", "QuantBiasOptimization", "QuantBiasRollBack", "QuantWeightRollBack"};
const std::set<ge::DataType> DATA_TYPE_IN = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT32};

const std::string FUSION_NAME = "ASplitConv2dConcatPass";

constexpr int64_t BOUNDARY_FILTER_INPUT_BASE = 1;
constexpr int64_t BOUNDARY_FMAP_INPUT_IDX = 0;
constexpr int64_t BOUNDARY_OUTPUT_IDX = 0;
constexpr int32_t AXIS_FROM_END = -1;
constexpr int32_t BIAS_AXIS = 0;
constexpr int32_t CONCAT_DIM_INPUT_IDX = 0;
constexpr int32_t NHWC_C_POSITION = 3;
constexpr int32_t SPLIT_DATA_INPUT_IDX = 1;
constexpr int32_t SPLIT_DIM_INPUT_IDX = 0;
constexpr int32_t SPLITV_DATA_INPUT_IDX = 0;
constexpr int32_t SPLITV_DIM_INPUT_IDX = 2;
constexpr int32_t SPLITV_SIZE_SPLITS_INPUT_IDX = 1;
constexpr size_t CONCAT_DIM_EXTRA_INPUT_CNT = 1;
constexpr int64_t CONCAT_DIM_SHAPE_SIZE = 1;
constexpr size_t CONCAT_SHAPE_SIZE = 4;
constexpr size_t FIRST_BRANCH_IDX = 0;
constexpr size_t SINGLE_REF_CNT = 1;

struct ConvBranchBaseline {
    size_t inputCnt = 0;
    std::vector<int64_t> weightShape;
    ge::Format weightFormat = ge::FORMAT_RESERVED;
    ge::AscendString concatName = "";
};
} // namespace SplitConv2dConcatFusion

class __attribute__((visibility("default"))) ASplitConv2dConcatPass : public ConvFusionBasePass {
protected:
    void InitMember() override;
    bool CheckMatchStructure(const ge::GNode& matchNode) override;
    bool MeetRequirements(const ge::GNode& splitNode) override;
    std::set<ge::AscendString> GetNodeTypes() const override;
    void PrintGraphStructure() const override;
    ge::Status ConvFusionPreImpl(ge::GraphPtr& graph, ge::GNode& splitNode,
                                 ge::CustomPassContext& passContext) override;
    bool ConvFusionReplaceImpl(ge::GraphPtr& graph, ge::GNode& splitNode, ge::CustomPassContext& passContext) override;
    std::unique_ptr<ge::fusion::SubgraphBoundary> ConstructBoundary(const ge::GNode& splitNode) override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& splitNode) override;

private:
    bool AnalyzeMidLayer(const ge::GNode& splitNode);
    bool CheckOneConvBranch(const ge::GNode& convNode, SplitConv2dConcatFusion::ConvBranchBaseline& baseline,
                            bool isFirst);
    bool CheckConvWeight(const ge::GNode& convNode, SplitConv2dConcatFusion::ConvBranchBaseline& baseline,
                         bool isFirst);
    bool CheckConvOutputToConcat(const ge::GNode& convNode, SplitConv2dConcatFusion::ConvBranchBaseline& baseline,
                                 bool isFirst);
    bool CheckConvNonFmapInputs(const ge::GNode& convNode, bool isFirst) const;
    bool CheckConcatDataInputs() const;
    bool VerifySptCcatAxis(const ge::GNode& splitNode);
    bool CheckFormatsConsistent(const ge::GNode& splitNode);
    bool GetAxisValue(const ge::GNode& ownerNode, int32_t inputIdx, int32_t& axisVal) const;
    int32_t GetSplitDataInputIdx() const;
    int32_t GetSplitDimInputIdx() const;
    int32_t GetConcatDimInputIdx() const;
    int32_t GetFormatAxisPos(ge::Format format, char axisChar) const;
    ge::Status SafeRemoveConstNode(ge::GraphPtr& graph, ge::GNode& constNode) const;
    bool BuildHostConcatNode(ge::es::EsGraphBuilder& graphBuilder, const std::vector<ge::es::EsTensorHolder>& inputs,
                             int32_t axis, const ge::AscendString& name, ge::GNode& hostConcatNode) const;
    bool ExpandNDimInDesc(ge::TensorDesc& desc) const;
    bool UpdateHostConcatDescs(ge::GNode& hostConcatNode, const ge::TensorDesc& sampleDesc) const;
    bool BuildGroupConvNode(ge::es::EsGraphBuilder& graphBuilder, const std::vector<ge::es::EsTensorHolder>& inputs,
                            ge::GNode& groupConv);
    bool SetGroupConvAttrs(ge::GNode& groupConv);
    bool UpdateGroupConvDescs(ge::GNode& groupConv);

    ge::AscendString splitNodeName = "";
    std::vector<ge::GNode> convNodes = {};
    ge::GNodePtr concatNode = nullptr;
    ge::TensorDesc fmapDesc = {};
    ge::TensorDesc outputDesc = {};
    ge::Format splitDimOriginFormat = ge::FORMAT_ND;
    int64_t groups = 0;
    bool hasBias = false;
    bool isConcatV2 = false;
    bool isSplitV = false;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // SPLIT_CONV2D_CONCAT_FUSION_PASS_H
