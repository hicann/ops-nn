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
 * \file mean_variance_normalization_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {
static const int OUTPUT_SIZE = 1;
struct MeanVarianceNormalizationAttr {
    std::vector<int> v_axes = {};
    std::vector<int> DefaultAxes = {0, 2, 3};

    bool set_axes_flag = false;
};

static Status UpdateAttrFromOnnx(const ge::onnx::NodeProto* node, MeanVarianceNormalizationAttr& node_attr)
{
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            node_attr.set_axes_flag = true;
            for (int i = 0; i < attr.ints_size(); i++) {
                node_attr.v_axes.push_back(attr.ints(i));
            }
        }
    }
    return SUCCESS;
}

static Status SetAttrToDesc(ge::Operator& op_dest, MeanVarianceNormalizationAttr& node_attr)
{
    if (node_attr.set_axes_flag) {
        op_dest.SetAttr("axes", node_attr.v_axes);
    } else {
        OP_LOGD(GetOpName(op_dest).c_str(), "onnx MeanVarianceNormalization op has no axes attr, set it to [0, 2, 3].");
        op_dest.SetAttr("axes", node_attr.DefaultAxes);
    }

    return SUCCESS;
}

static Status ParseParamsMeanVarianceNormalization(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    int op_output_size = node->output_size();
    if (op_output_size != OUTPUT_SIZE) {
        OP_LOGE(GetOpName(op_dest).c_str(), "The output of Indices is not support, transforming failed.");
        return FAILED;
    }
    MeanVarianceNormalizationAttr node_attr;
    if (UpdateAttrFromOnnx(node, node_attr) != SUCCESS) {
        return FAILED;
    }

    if (SetAttrToDesc(op_dest, node_attr) != SUCCESS) {
        return FAILED;
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("MVNV2")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::9::MeanVarianceNormalization"),
         ge::AscendString("ai.onnx::10::MeanVarianceNormalization"),
         ge::AscendString("ai.onnx::11::MeanVarianceNormalization"),
         ge::AscendString("ai.onnx::12::MeanVarianceNormalization"),
         ge::AscendString("ai.onnx::13::MeanVarianceNormalization"),
         ge::AscendString("ai.onnx::14::MeanVarianceNormalization"),
         ge::AscendString("ai.onnx::15::MeanVarianceNormalization"),
         ge::AscendString("ai.onnx::16::MeanVarianceNormalization"),
         ge::AscendString("ai.onnx::17::MeanVarianceNormalization"),
         ge::AscendString("ai.onnx::18::MeanVarianceNormalization")})
    .ParseParamsFn(ParseParamsMeanVarianceNormalization)
    .ImplyType(ImplyType::TVM);
} // namespace domi
