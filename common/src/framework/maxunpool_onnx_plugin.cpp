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
#include "index/scatter_elements/op_graph/scatter_elements_proto.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace {
constexpr int X_INDEX = 0;
constexpr int I_INDEX = 1;
constexpr int OUTPUT_SHAPE_INDEX = 2;
} // namespace
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

static Status ParseParamsMaxUnpool(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::11::MaxUnpool");
    int n = node->input_size();
    std::vector<int> kernel_shape = {};
    std::vector<int> pads = {};
    std::vector<int> strides = {};

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "kernel_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                kernel_shape.push_back(attr.ints(i));
            }
        } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                pads.push_back(attr.ints(i));
            }
        } else if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                strides.push_back(attr.ints(i));
            }
        }
    }

    if (kernel_shape.empty()) {
        OP_LOGE(op_dest.GetName().c_str(), "node must have attr kernel_shape");
        return FAILED;
    }

    if (pads.size() == 0) {
        pads.resize(kernel_shape.size() * 2, 0);
    }
    if (strides.size() == 0) {
        strides.resize(kernel_shape.size(), 1);
    }
    op_dest.SetAttr("num", n);
    op_dest.SetAttr("kernel_shape", kernel_shape);
    op_dest.SetAttr("pads", pads);
    op_dest.SetAttr("strides", strides);
    op_dest.SetAttr("name", node->name());
    op_dest.DynamicInputRegister("x", n);
    op_dest.DynamicOutputRegister("y", 1);
    return SUCCESS;
}

static Status ParseOpToGraphMaxUnpool(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data(ori_name + "_data0").set_attr_index(X_INDEX);
    auto data1 = op::Data(ori_name + "_data1").set_attr_index(I_INDEX);
    std::vector<ge::Operator> inputs = {data0, data1};

    auto identity_op1 = op::Identity(ori_name + "_identity1").set_input_x(data0);
    std::vector<int> kernel_shape_value;
    if (op.GetAttr("kernel_shape", kernel_shape_value) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get kernel_shape from op failed");
        return FAILED;
    }

    size_t k_size = kernel_shape_value.size();
    std::vector<int> pads_value;
    if (op.GetAttr("pads", pads_value) != SUCCESS && pads_value.size() != (k_size * 2)) {
        OP_LOGE(GetOpName(op).c_str(), "get pads from op failed");
        return FAILED;
    }

    std::vector<int> strides_value;
    if (op.GetAttr("strides", strides_value) != SUCCESS && strides_value.size() != k_size) {
        OP_LOGE(GetOpName(op).c_str(), "get strides from op failed");
        return FAILED;
    }

    int64_t num1;
    if (op.GetAttr("num", num1) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get numbers of inputs from op failed");
        return FAILED;
    }

    ge::Operator data2;
    if (num1 == 2) {
        auto input_shape = op::Shape(ori_name + "_shape1").set_input_x(identity_op1).set_attr_dtype(ge::DT_INT32);

        int64_t d_num = kernel_shape_value.size();
        std::vector<int> dim_other;
        for (int i = 0; i < d_num; i++) {
            dim_other.push_back(pads_value[i] + pads_value[i + d_num] - kernel_shape_value[i]);
        }

        std::vector<int32_t> vector1(2, 0);
        std::vector<int32_t> vector2(d_num, 1);
        vector1.insert(vector1.end(), vector2.begin(), vector2.end());
        int64_t len1 = vector1.size();
        std::vector<int64_t> dims1 = {len1};
        ge::Tensor tensor1 = Vec2Tensor(vector1, dims1, ge::DT_INT32, ge::FORMAT_ND);
        auto const_value1 = op::Const(ori_name + "_const1").set_attr_value(tensor1);
        auto d_dim_out = op::Sub(ori_name + "_sub1").set_input_x1(input_shape).set_input_x2(const_value1);

        std::vector<int32_t> vector3(2, 1);
        vector3.insert(vector3.end(), strides_value.begin(), strides_value.end());
        int64_t len2 = vector3.size();
        std::vector<int64_t> dims2 = {len2};
        ge::Tensor tensor2 = Vec2Tensor(vector3, dims2, ge::DT_INT32, ge::FORMAT_ND);
        auto const_value2 = op::Const(ori_name + "_const2").set_attr_value(tensor2);
        auto d_dim_out_1 = op::Mul(ori_name + "_mul1").set_input_x1(d_dim_out).set_input_x2(const_value2);

        std::vector<int32_t> vector4(2, 0);
        vector4.insert(vector4.end(), dim_other.begin(), dim_other.end());
        int64_t len3 = vector4.size();
        std::vector<int64_t> dims3 = {len3};
        ge::Tensor tensor3 = Vec2Tensor(vector4, dims3, ge::DT_INT32, ge::FORMAT_ND);
        auto const_value3 = op::Const(ori_name + "_const3").set_attr_value(tensor3);
        data2 = op::Sub(ori_name + "_sub2").set_input_x1(d_dim_out_1).set_input_x2(const_value3);
    } else {
        data2 = op::Data(ori_name + "_data2").set_attr_index(OUTPUT_SHAPE_INDEX);
        inputs.push_back(data2);
    }

    auto identity_op2 = op::Identity(ori_name + "_identity2").set_input_x(data2);
    std::vector<int32_t> vector5(1, -1);
    int64_t len4 = vector5.size();
    const vector<int64_t> dims4 = {len4};
    ge::Tensor tensor4 = Vec2Tensor(vector5, dims4, ge::DT_INT32, ge::FORMAT_ND);
    auto const_value4 = op::Const(ori_name + "_const_data4").set_attr_value(tensor4);

    auto x_1 = op::Reshape(ori_name + "_reshape1")
                   .set_input_x(identity_op1)
                   .set_input_shape(const_value4)
                   .set_attr_axis(0)
                   .set_attr_num_axes(-1);

    auto i_1 = op::Reshape(ori_name + "_reshape2")
                   .set_input_x(data1)
                   .set_input_shape(const_value4)
                   .set_attr_axis(0)
                   .set_attr_num_axes(-1);
    std::vector<int32_t> vector6(1, 0);
    int64_t len5 = vector6.size();
    const vector<int64_t> dims5 = {len5};
    ge::Tensor tensor5 = Vec2Tensor(vector6, dims5, ge::DT_INT32, ge::FORMAT_ND);
    auto const_value5 = op::Const(ori_name + "_const_data6").set_attr_value(tensor5);

    std::vector<int32_t> vector7(1, 1);
    int64_t len6 = vector7.size();
    const vector<int64_t> dims6 = {len6};
    ge::Tensor tensor6 = Vec2Tensor(vector7, dims6, ge::DT_INT32, ge::FORMAT_ND);
    auto const_value6 = op::Const(ori_name + "_const_data7").set_attr_value(tensor6);

    auto x_2 = op::StridedSlice(ori_name + "_StridedSlice1")
                   .set_input_x(x_1)
                   .set_input_begin(const_value5)
                   .set_input_end(const_value6)
                   .set_input_strides(const_value6)
                   .set_attr_begin_mask(0)
                   .set_attr_end_mask(0)
                   .set_attr_ellipsis_mask(0)
                   .set_attr_new_axis_mask(0)
                   .set_attr_shrink_axis_mask(0);
    auto x_3 = op::ZerosLike(ori_name + "_zeroslike1").set_input_x(x_2);
    auto result_zero = op::BroadcastTo(ori_name + "_BroadcastTo1").set_input_x(x_3).set_input_shape(identity_op2);
    auto result_zero_1 = op::Reshape(ori_name + "_reshape3")
                             .set_input_x(result_zero)
                             .set_input_shape(const_value4)
                             .set_attr_axis(0)
                             .set_attr_num_axes(-1);
    auto result_zero_2 = op::ScatterElements(ori_name + "_ScatterElements1")
                             .set_input_data(result_zero_1)
                             .set_input_indices(i_1)
                             .set_input_updates(x_1)
                             .set_attr_axis(0);
    auto result_zero_3 = op::Reshape(ori_name + "_reshape4")
                             .set_input_x(result_zero_2)
                             .set_input_shape(identity_op2)
                             .set_attr_axis(0)
                             .set_attr_num_axes(-1);

    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(result_zero_3, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::MaxUnpool", "ai.onnx::11::MaxUnpool", "ai.onnx::12::MaxUnpool",
                   "ai.onnx::13::MaxUnpool", "ai.onnx::14::MaxUnpool", "ai.onnx::15::MaxUnpool",
                   "ai.onnx::16::MaxUnpool", "ai.onnx::17::MaxUnpool", "ai.onnx::18::MaxUnpool"})
    .ParseParamsFn(ParseParamsMaxUnpool)
    .ParseOpToGraphFn(ParseOpToGraphMaxUnpool)
    .ImplyType(ImplyType::TVM);
} // namespace domi
