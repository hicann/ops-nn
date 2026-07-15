/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "op_nn_proto_extend.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

static Status ParseParamsInt8Fc(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE("ParseParamsInt8Fc", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::map<string, float> scale_map;
    std::map<string, int> offset_map;
    for (auto& attr : node->attribute()) {
        if (attr.name().find("scale") != std::string::npos && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            scale_map[attr.name()] = attr.f();
        } else if (attr.name().find("zero_point") != std::string::npos &&
                   attr.type() == ge::onnx::AttributeProto::INT) {
            offset_map[attr.name()] = attr.i();
        }
    }

    int op_input_size = node->input_size();
    int op_output_size = node->output_size();
    op_dest.DynamicInputRegister("x", op_input_size);
    op_dest.DynamicOutputRegister("y", op_output_size);

    op_dest.SetAttr("original_type", "ai.onnx::11::Int8FC");

    op_dest.SetAttr("num_input", op_input_size);
    op_dest.SetAttr("num_output", op_output_size);

    op_dest.SetAttr("name", node->name());

    unsigned int s_conter = 0;
    for (auto iter = scale_map.begin(); iter != scale_map.end(); ++iter) {
        if (s_conter == 3) {
            op_dest.SetAttr("ascend_dequant_scale", iter->second);
        } else if (s_conter == 0) {
            op_dest.SetAttr("ascend_quant_scale", iter->second);
        }
        ++s_conter;
    }

    unsigned int o_counter = 0;
    for (auto iter = offset_map.begin(); iter != offset_map.end(); ++iter) {
        if (o_counter == 3) {
            if (iter->second != 0) {
                OP_LOGW("Int8Fc", "The offset of operator AscendDequant in NPU must 0.");
            }
            op_dest.SetAttr("ascend_dequant_offset", 0);
        } else if (o_counter == 0) {
            op_dest.SetAttr("ascend_quant_offset", static_cast<float>(iter->second));
        }
        ++o_counter;
    }

    return SUCCESS;
}

static Status ParseOpToGraphInt8Fc(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    float quant_scale = 1.0;
    float quant_offset = 0.0;
    float deq_scale = 1.0;
    int input_size = 2;
    int output_size = 1;
    op.GetAttr("num_input", input_size);
    op.GetAttr("num_output", output_size);
    op.GetAttr("ascend_quant_scale", quant_scale);
    op.GetAttr("ascend_quant_offset", quant_offset);
    op.GetAttr("ascend_dequant_scale", deq_scale);

    std::vector<ge::Operator> inputs;
    ge::Operator fc_op;
    if (input_size == 2) {
        auto data1 = op::Data(ori_name + "_data1").set_attr_index(0);
        auto data2 = op::Data(ori_name + "_data2").set_attr_index(1);
        fc_op = op::FullyConnection(ori_name + "_Int8FcFullyconnection")
                    .set_input_x(data1)
                    .set_input_w(data2)
                    .set_attr_num_output(output_size)
                    .set_attr_transpose(false);
        inputs = {data1, data2};
    } else if (input_size == 3) {
        auto data1 = op::Data(ori_name + "_data1").set_attr_index(0);
        auto data2 = op::Data(ori_name + "_data2").set_attr_index(1);
        auto data3 = op::Data(ori_name + "_data3").set_attr_index(2);
        fc_op = op::FullyConnection(ori_name + "_Int8FcFullyconnection")
                    .set_input_x(data1)
                    .set_input_w(data2)
                    .set_input_b(data3)
                    .set_attr_num_output(output_size)
                    .set_attr_transpose(false);
        inputs = {data1, data2, data3};
    } else {
        OP_LOGE("ParseParamsInt8Fc", "Numbers of input cannot be parsered.");
        return FAILED;
    }

    ge::Tensor scalar_const_deq_scale = CreateScalar(deq_scale, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0);
    auto const_op = op::Const(ori_name + "_deq_scale").set_attr_value(scalar_const_deq_scale);
    auto ascend_deq = op::AscendDequant("Int8FcAscendDequant")
                          .set_input_x(fc_op)
                          .set_input_deq_scale(const_op)
                          .set_attr_sqrt_mode(false)
                          .set_attr_relu_flag(false)
                          .set_attr_dtype(DT_FLOAT16);

    auto ascend_quant = op::AscendQuant(ori_name + "_Int8FcAscendQuant")
                            .set_input_x(ascend_deq)
                            .set_attr_scale(quant_scale)
                            .set_attr_offset(quant_offset)
                            .set_attr_sqrt_mode(false)
                            .set_attr_round_mode("Round");

    ge::TensorDesc tensor = ascend_quant.GetOutputDesc(0);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    uint32_t output_idx = 0U;
    auto ret_y = ascend_quant.UpdateOutputDesc(output_idx, tensor);
    if (ret_y != ge::GRAPH_SUCCESS) {
        OP_LOGE("Int8FC", "update quant output format failed.");
        return FAILED;
    }
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;

    output_indexs.emplace_back(ascend_quant, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Int8FC", "ai.onnx::9::Int8FC", "ai.onnx::10::Int8FC", "ai.onnx::11::Int8FC",
                   "ai.onnx::12::Int8FC", "ai.onnx::13::Int8FC", "ai.onnx::14::Int8FC", "ai.onnx::15::Int8FC",
                   "ai.onnx::16::Int8FC"})
    .ParseParamsFn(ParseParamsInt8Fc)
    .ParseOpToGraphFn(ParseOpToGraphInt8Fc)
    .ImplyType(ImplyType::TVM);
} // namespace domi
