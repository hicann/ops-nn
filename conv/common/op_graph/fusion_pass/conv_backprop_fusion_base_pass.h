/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV_BACKPROP_FUSION_BASE_PASS_H
#define CONV_BACKPROP_FUSION_BASE_PASS_H

#include "conv_backprop_fusion_utils_pass.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/es_tensor_holder.h"
#include "ge/fusion/pass/decompose_pass.h"
#include "log/log.h"
#include "register/register_custom_pass.h"

namespace ops {
using namespace ge;
using namespace ge::es;
using namespace fusion;
using namespace ConvBackpropFusionUtils;

// ConvBackprop属性结构体
struct ConvBackpropAttrs {
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    int64_t groups = 0;
    std::string dataFormat;
    int64_t opImplModeEnum = 0;
    bool hf32 = false;
    bool from_depthwise = false;

    void Reset()
    {
        strides.clear();
        pads.clear();
        dilations.clear();
        groups = 0;
        dataFormat = "";
        opImplModeEnum = 0;
        hf32 = false;
        from_depthwise = false;
    }
};

/**
 * ConvBackprop融合规则基类
 * 用于适配InputV2和FilterV2的公共成员和方法
 */
class __attribute__((visibility("default"))) ConvBackpropFusionBasePass : public ge::fusion::DecomposePass {
public:
    explicit ConvBackpropFusionBasePass(const std::vector<ge::AscendString>& opTypes) : DecomposePass(opTypes) {}

protected:
    virtual void InitMember();

    virtual bool CheckSocAndIntrinsic();

    bool MeetRequirements(const ge::GNode& convBpInputNode) override;

    virtual bool GetNodeDesc(const ge::GNode& node);

    virtual bool GetNodeAttrs(const ge::GNode& node);

    virtual bool UpdateNodeInputDescInfo(ge::GNode* node);

    void SetNodeAttrs(ge::GNode& outNode);

    virtual ge::AscendString GetNodeType() const = 0;

    virtual bool CheckTransposeNeeded() { return false; }

protected:
    NpuArch npuArch = NpuArch::DAV_RESV;

    ge::TensorDesc input0Desc;
    ge::TensorDesc input1Desc;
    ge::TensorDesc input2Desc;
    ge::TensorDesc outputDesc;

    ConvBackpropAttrs convBpAttr;
};

} // namespace ops

#endif // CONV_BACKPROP_FUSION_BASE_PASS_H
