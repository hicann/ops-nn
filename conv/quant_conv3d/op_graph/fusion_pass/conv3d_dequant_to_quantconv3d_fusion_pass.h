/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3D_DEQUANT_TO_QUANTCONV3D_FUSION_PASS_H
#define CONV3D_DEQUANT_TO_QUANTCONV3D_FUSION_PASS_H

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "../../common/graph_fusion/cube_utils/cube_utils.h"
#include "ge/fusion/subgraph_boundary.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace Conv3DDequantToQuantconv3DFusion {
using ge::AscendString;
using ge::DataType;
using ge::GNodePtr;
using ge::GNode;
using ge::Format;

const AscendString RELU_FLAG = "relu_flag";
const AscendString SQRT_MODE = "sqrt_mode";
const AscendString RINT = "rint";

const std::map<std::string, NpuArch> SUPPORT_SOC_LIST = {
    {"Ascend950", NpuArch::DAV_3510},
};

constexpr int32_t FUSION_LIST_LENGTH = 2;
constexpr int32_t CONV3D_INDEX = 0;
constexpr int32_t ASCEND_DEQUANT_INDEX = 1;
constexpr int32_t POST_CUBE_INPUT_QUANT_SCALE_0_INDEX = 2;
constexpr int32_t QUANTCONV3D_CONV3D_INPUT_SCALE_INDEX = 2;
constexpr int32_t QUANTCONV3D_CONV3D_INPUT_BIAS_INDEX = 3;
constexpr size_t SINGLE_OUTPUTNUM = 1;

// Fmap Filter Output Bias
const std::vector<std::vector<DataType>> CONV_SUPPORT_DTYPES = {
    {DataType::DT_INT8, DataType::DT_INT8, DataType::DT_INT32, DataType::DT_INT32},
};

// Fmap Filter Output
const std::vector<std::vector<Format>> CONV_SUPPORT_FORMATS_DAV_3510 = {
    {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW},
    {ge::FORMAT_NDHWC, ge::FORMAT_DHWCN, ge::FORMAT_NDHWC}
};

} // namespace Conv3DDequantToQuantconv3DFusion

class __attribute__((visibility("default"))) Conv3DDequantToQuantConv3DFusionPass : public ConvFusionBasePass {
protected:
    std::unique_ptr<ge::fusion::SubgraphBoundary> ConstructBoundary(const ge::GNode &convNode) override;
    bool PostCubeFusionImpl(
        ge::GraphPtr &graph, ge::GNode &convNode, const ge::CustomPassContext &pass_context) override;
    void InitMember() override;
    bool MeetRequirements(const ge::GNode &convNode) override;
    ge::AscendString GetNodeType() const override;
    std::map<std::string, NpuArch> GetSocSupportList() const override;
    void PrintGraphStructure() const override {};
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode &convNode) override;
private:
    bool GetPostCubeNodes(const ge::GNode &convNode);
    void SelectPostCubePassByWhiteList(std::vector<ops::PostCubePassInfo> &matchVec);
    bool UpdateQuantConv3DDesc(ge::GNode *quantConv3D, ge::TensorDesc &postCubeOutDesc);

    ge::GNodePtr postCubeNode = nullptr;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // CONV3D_DEQUANT_TO_QUANTCONV3D_FUSION_PASS_H