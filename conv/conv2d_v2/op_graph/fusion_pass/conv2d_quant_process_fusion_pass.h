/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_QUANT_PROCESS_FUSION_PASS_H
#define CONV2D_QUANT_PROCESS_FUSION_PASS_H

#include <set>
#include <string>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_fusion_base_pass.h"
#include "graph/gnode.h"
#include "op_graph/fusion_pass/bias_optimize_quant_rollback_base.h"
#include "register/register_custom_pass.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace Conv2dQuantProcessFusion {

const ge::AscendString PAD = "Pad";
const ge::AscendString PADV3 = "PadV3";
const ge::AscendString AIPP = "Aipp";
const std::string FUSION_NAME = "Conv2DQuantProcessFusionPass";
const std::string ATTR_SCALE = "scale";
const std::string ATTR_OFFSET = "offset";
const std::string ATTR_NAME_AIPP = "aipp";
const std::string K_AIPP_QUANT_NEED_ADJUST = "_aipp_quant_need_adjust";
const std::string K_AIPP_QUANT_MEAN_VEC = "_aipp_quant_mean_vec";
const std::string K_QUANT_SCALE_RATE = "_quant_scale_rate";
const std::string OPTION_FIRST_LAYER_QUANT = "enable_first_layer_quantization";
constexpr int64_t FP16_C0 = 16;
constexpr int32_t K_BASE_OFFSET = 128;
constexpr int32_t K_MAX_CHN_NUM = 4;

// 老 quant_host_cpu_op_common.h 的 kAippKey: {mean, min, var_reci} 通道键
const std::vector<std::vector<std::string>> K_AIPP_KEY = {{"mean_chn_0", "min_chn_0", "var_reci_chn_0"},
                                                          {"mean_chn_1", "min_chn_1", "var_reci_chn_1"},
                                                          {"mean_chn_2", "min_chn_2", "var_reci_chn_2"},
                                                          {"mean_chn_3", "min_chn_3", "var_reci_chn_3"}};

struct AippChnInfo {
    std::vector<int64_t> meanChnVec;
    std::vector<ge::float32_t> minChnVec;
    std::vector<ge::float32_t> varReciChnVec;
};

enum class AIPPFirstConvType { NONE_TYPE, NEED_ADJUST_TYPE, NO_NEED_ADJUST_TYPE };

} // namespace Conv2dQuantProcessFusion

class __attribute__((visibility("default"))) Conv2DQuantProcessFusionPass : public ConvFusionBasePass,
                                                                            public ops::BiasOptimizeQuantRollbackBase {
protected:
    void InitMember() override;
    bool CheckMatchStructure(const ge::GNode& matchNode) override;
    bool MeetRequirements(const ge::GNode& convNode) override;
    std::set<ge::AscendString> GetNodeTypes() const override;
    void PrintGraphStructure() const override;
    ge::Status ConvFusionPreImpl(ge::GraphPtr& graph, ge::GNode& convNode, ge::CustomPassContext& passContext) override;
    bool ConvFusionReplaceImpl(ge::GraphPtr& graph, ge::GNode& convNode, ge::CustomPassContext& passContext) override;

    // BiasOptimizeQuantRollbackBase 覆写钩子（老 Conv2DQuantProcessFusionPass 的 conv2d 特化）
    ge::Status GetQuantProcessMode(const ge::GNodePtr& quantNode, const ge::GNodePtr& cubeNode,
                                   ops::QuantProcessMode& quantProcessMode) override;
    ge::GNodePtr GetCubeNodeInputNode(const ge::GNodePtr& cubeNode) const override;
    ge::Status SetDataTypeOfNodes(const ge::GNodePtr& cubeNode) override;
    ge::Status GetCoValue(const ge::GNodePtr& cubeNode, int64_t& co) override;

private:
    // 老 SetPadNodeDataType：rollback 时 Pad 双向 desc 改 fp16
    ge::Status SetPadNodeDataType(ge::GNode& padNode);
    // AIPP first conv（老 IsAippWithFirstConv 链路，开关经 compression_optimize_conf 配置文件读取）
    bool IsFirstLayerQuantEnabled(const ge::CustomPassContext& passContext) const;
    ge::Status GetAippCfgInfo(const ge::GNode& aippNodeRef);
    bool CheckAippVarChn() const;
    ge::Status QuantParamRevise();
    void PostProcAfterOptimize(const std::vector<ge::GNode>& nodesAfterFuse);

    ge::GNodePtr dequantNode = nullptr;
    ge::GNodePtr padNode = nullptr;
    ge::GNodePtr quantNode = nullptr;
    ge::GNodePtr aippNode = nullptr;
    Conv2dQuantProcessFusion::AIPPFirstConvType
        aippFirstConvType = Conv2dQuantProcessFusion::AIPPFirstConvType::NONE_TYPE;
    ge::float32_t meanAipp = 0.0F;
    ge::float32_t scaleRate = 0.0F;
    Conv2dQuantProcessFusion::AippChnInfo aippChnInfo;
    ops::QuantProcessMode quantProcessMode = ops::QuantProcessMode::QUANT_UNDEFINED;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV2D_QUANT_PROCESS_FUSION_PASS_H
