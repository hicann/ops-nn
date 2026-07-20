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

namespace domi {

using OpDesc = std::shared_ptr<ge::OpDesc>;
using namespace ge;
static const int INPUT_4D = 4;
static const int INPUT_5D = 5;
static const int INPUT_NUM_2 = 2;
static const int INPUT_NUM_3 = 3;
static const int kIndex = 2;
static const int kLen2 = 2;
static const int kLen3 = 3;
static const int ONNX_1D_ATTR_PAD_LEN = 2;
bool is_set_output_shape = false;
bool is_set_auto_pad = false;
struct ConvTransposeAttr {
    std::vector<int64_t> dilations = {1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    int64_t groups = 1;
    std::string data_format = "NCHW";
    std::vector<int64_t> input_size = {0, 0, 0, 0};
    std::string auto_pad = "NOTSET";
    std::vector<int64_t> output_padding = {0, 0, 0, 0};
    int dim_size = 4;
    int input_num = 2;
    bool trans_2d = false;
    std::vector<int64_t> output_shape = {0, 0, 0, 0};
    int64_t fixed_shift_value = 0;
};

static Status AttrUpdate(std::vector<int32_t>& dst, std::vector<int32_t>& src, int offset, int count,
                         const ge::AscendString& op_name)
{
    if ((int)src.size() < count) {
        OP_LOGE(op_name.GetString(), "attr size[%d] should >= [%d]", (int)src.size(), count);
        return FAILED;
    }
    for (int i = 0; i < count; ++i) {
        dst[offset + i] = src[i];
    }
    return SUCCESS;
}

static void SetIntListValue(const ge::onnx::AttributeProto& attr, std::vector<int32_t>& int_list)
{
    for (auto i = 0; i < attr.ints_size(); ++i) {
        int_list.push_back(attr.ints(i));
    }
}

static void GetPadList(const ge::onnx::AttributeProto& attr, std::vector<int32_t>& pad_list)
{
    unsigned int len = attr.ints_size();
    for (unsigned int i = 0; i < len / 2; i++) {
        pad_list.push_back(attr.ints(i));
        pad_list.push_back(attr.ints(i + len / 2));
    }
}

static void SetPadsAttr(std::vector<int32_t>& pad_list, int out_len, ge::Operator& op)
{
    if (!pad_list.empty()) {
        for (int i = static_cast<int>(pad_list.size()); i < kLen2 * out_len; ++i) {
            auto it = pad_list.begin();
            pad_list.insert(it, 0);
        }
    }
    op.SetAttr("pads", pad_list);
}

static void SetSingleValueAttr(const ge::onnx::AttributeProto& attr, ge::Operator& op)
{
    if (attr.name() == "group" && attr.type() == ge::onnx::AttributeProto::INT) {
        op.SetAttr("groups", attr.i());
    } else if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
        op.SetAttr("auto_pad", attr.s());
        is_set_auto_pad = true;
    } else if (attr.name() == "fixed_shift_value" && attr.type() == ge::onnx::AttributeProto::INT) {
        op.SetAttr("fixed_shift_value", attr.i());
    }
}

static Status SetAttrToOpConvTranspose(const ge::onnx::NodeProto* node, ge::Operator& op)
{
    ge::AscendString op_name;
    if (op.GetName(op_name) != ge::GRAPH_SUCCESS) {
        OP_LOGE("", "failed to get op_name");
        return FAILED;
    }
    std::vector<int32_t> strides_list = {1, 1};
    std::vector<int32_t> dilations_list = {1, 1};
    std::vector<int32_t> pad_list;
    std::vector<int32_t> out_pads_list;
    std::vector<int32_t> out_shape_list;
    bool is_trans_2d = false;
    bool is_have_kenel_shape = false;
    int dim_size = 4;
    for (const auto& attr : node->attribute()) {
        if (attr.type() == ge::onnx::AttributeProto::INTS) {
            if (attr.name() == "strides") {
                SetIntListValue(attr, strides_list);
                if (attr.ints_size() == 1) {
                    strides_list.push_back(attr.ints(0));
                }
                op.SetAttr("strides", strides_list);
            } else if (attr.name() == "dilations") {
                if (attr.ints_size() == 1) {
                    dilations_list.push_back(1);
                }
                SetIntListValue(attr, dilations_list);
                op.SetAttr("dilations", dilations_list);
            } else if (attr.name() == "pads") {
                unsigned int len = attr.ints_size();
                if (len & 1) {
                    OP_LOGE(op_name.GetString(), "The length of pads is odd, failed to transform.");
                    return FAILED;
                }
                if (attr.ints_size() == ONNX_1D_ATTR_PAD_LEN) {
                    pad_list.push_back(0);
                    pad_list.push_back(0);
                }
                GetPadList(attr, pad_list);
            } else if (attr.name() == "output_padding") {
                if (attr.ints_size() == 1) {
                    out_pads_list.push_back(0);
                }
                SetIntListValue(attr, out_pads_list);
            } else if (attr.name() == "output_shape") {
                SetIntListValue(attr, out_shape_list);
                is_set_output_shape = true;
            } else if (attr.name() == "kernel_shape") {
                int len = attr.ints_size();
                is_have_kenel_shape = true;
                is_trans_2d = len == 1;
                dim_size = len >= kLen3 ? INPUT_5D : INPUT_4D;
            }
        } else {
            SetSingleValueAttr(attr, op);
        }
    }

    if (!is_have_kenel_shape) {
        OP_LOGE(op_name.GetString(), "attr kernel_shape must have value");
        return FAILED;
    }

    int out_len = dim_size - kLen2;
    if (!out_pads_list.empty()) {
        std::vector<int32_t> out_pads_list_new(out_len + kLen2, 0);
        if (AttrUpdate(out_pads_list_new, out_pads_list, kIndex, out_len, op_name) != SUCCESS) {
            return FAILED;
        }
        op.SetAttr("output_padding", out_pads_list_new);
    }

    if (!out_shape_list.empty()) {
        if ((int)out_shape_list.size() < out_len) {
            OP_LOGE(op_name.GetString(), "attr output shape size[%d] should >= [%d]", (int)out_shape_list.size(),
                    out_len);
            return FAILED;
        }
        int offset = (int)out_shape_list.size() - out_len;
        std::vector<int32_t> out_shape_list_new(out_shape_list.begin() + offset, out_shape_list.end());
        op.SetAttr("output_shape", out_shape_list_new);
    }
    SetPadsAttr(pad_list, out_len, op);

    bool is_set_auto_pad_attr = is_set_output_shape && !is_set_auto_pad;
    if (is_set_auto_pad_attr) {
        op.SetAttr("auto_pad", "SAME_LOWER");
    }

    op.SetAttr("dim_size", dim_size);
    op.SetAttr("trans_2d", is_trans_2d);

    return SUCCESS;
}

static Status ParseParamsConvTranspose(const Message* op_src, ge::Operator& op)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE("ConvTranspose", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int n = node->input_size();
    op.SetAttr("input_num", n);
    op.SetAttr("name", node->name());
    op.DynamicInputRegister("args", n);
    op.DynamicOutputRegister("output", 1);
    op.SetAttr("original_type", "ai.onnx::11::ConvTranspose");

    if (SetAttrToOpConvTranspose(node, op) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

static Status SetFormatConvTranspose(ge::Operator& op, const int& dims)
{
    if (dims == INPUT_4D) {
        auto ret_x = ChangeFormatFromOnnx(op, 1, ge::FORMAT_NCHW, true);
        if (ret_x != ge::GRAPH_SUCCESS) {
            OP_LOGE("ConvTranspose", "failed to update fmap format.");
            return FAILED;
        }
        auto ret_w = ChangeFormatFromOnnx(op, kIndex, ge::FORMAT_NCHW, true);
        if (ret_w != ge::GRAPH_SUCCESS) {
            OP_LOGE("ConvTranspose", "failed to update filter format.");
            return FAILED;
        }
        auto ret_y = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCHW, false);
        if (ret_y != ge::GRAPH_SUCCESS) {
            OP_LOGE("ConvTranspose", "failed to update output format.");
            return FAILED;
        }
    } else if (dims == INPUT_5D) {
        auto ret_x = ChangeFormatFromOnnx(op, 1, ge::FORMAT_NCDHW, true);
        if (ret_x != ge::GRAPH_SUCCESS) {
            OP_LOGE("ConvTranspose", "failed to update fmap format.");
            return FAILED;
        }
        auto ret_w = ChangeFormatFromOnnx(op, kIndex, ge::FORMAT_NCDHW, true);
        if (ret_w != ge::GRAPH_SUCCESS) {
            OP_LOGE("ConvTranspose", "failed to update filter format.");
            return FAILED;
        }
        auto ret_y = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCDHW, false);
        if (ret_y != ge::GRAPH_SUCCESS) {
            OP_LOGE("ConvTranspose", "failed to update output format.");
            return FAILED;
        }
    } else {
        OP_LOGE("ConvTranspose", "The input tensor is not 4D/5D, failed to set format.");
        return FAILED;
    }
    return SUCCESS;
}

static Status GetConvTransposeAttr(const ge::Operator& op, ConvTransposeAttr& convTransposeAttr)
{
    op.GetAttr("strides", convTransposeAttr.strides);
    op.GetAttr("pads", convTransposeAttr.pads);
    op.GetAttr("dilations", convTransposeAttr.dilations);
    op.GetAttr("auto_pad", convTransposeAttr.auto_pad);
    std::string pad_mode = convTransposeAttr.auto_pad;
    op.GetAttr("output_shape", convTransposeAttr.output_shape);
    op.GetAttr("trans_2d", convTransposeAttr.trans_2d);
    auto ret_output_padding = op.GetAttr("output_padding", convTransposeAttr.output_padding);

    if (op.GetAttr("dim_size", convTransposeAttr.dim_size) != SUCCESS) {
        OP_LOGE("ConvTranspose", "failed to get dim size from op");
        return FAILED;
    }

    if (op.GetAttr("input_num", convTransposeAttr.input_num) != SUCCESS) {
        OP_LOGE("ConvTranspose", "failed to get number of input from op.");
        return FAILED;
    }
    if (op.GetAttr("groups", convTransposeAttr.groups) != SUCCESS)
        convTransposeAttr.groups = 1;

    if (op.GetAttr("fixed_shift_value", convTransposeAttr.fixed_shift_value) != SUCCESS) {
        convTransposeAttr.fixed_shift_value = 0;
    }

    if (op.GetAttr("data_format", convTransposeAttr.data_format) != SUCCESS) {
        std::string data_format = convTransposeAttr.dim_size == INPUT_5D ? "NCDHW" : "NCHW";
        convTransposeAttr.data_format = data_format;
    }
    if (ret_output_padding != SUCCESS) {
        if (convTransposeAttr.dim_size == INPUT_5D) {
            std::vector<int64_t> output_padding_list = {0, 0, 0, 0, 0};
            convTransposeAttr.output_padding = output_padding_list;
        } else {
            std::vector<int64_t> output_padding_list = {0, 0, 0, 0};
            convTransposeAttr.output_padding = output_padding_list;
        }
    }
    std::vector<int64_t> strides_list_default = {1, 1, 1, 1};
    std::vector<int64_t> dilations_list_default = {1, 1, 1, 1};
    std::vector<int64_t> pad_list_default = {0, 0, 0, 0};
    std::vector<int64_t> input_size = {0, 0, 0, 0};
    if (convTransposeAttr.dim_size == INPUT_5D) {
        strides_list_default.push_back(1);
        dilations_list_default.push_back(1);
        pad_list_default.push_back(0);
        pad_list_default.push_back(0);
        input_size.push_back(0);
    }
    convTransposeAttr.input_size = input_size;
    if ((int)convTransposeAttr.strides.size() != convTransposeAttr.dim_size)
        convTransposeAttr.strides = strides_list_default;
    if ((int)convTransposeAttr.dilations.size() != convTransposeAttr.dim_size)
        convTransposeAttr.dilations = dilations_list_default;
    if ((int)convTransposeAttr.pads.size() == 0)
        convTransposeAttr.pads = pad_list_default;
    return SUCCESS;
}

static Status ParseOpToGraphConvTranspose(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "failed to get name from op.");
        return FAILED;
    }

    ConvTransposeAttr tbeAttr;
    if (GetConvTransposeAttr(op, tbeAttr) != SUCCESS) {
        return FAILED;
    }

    ge::Operator dataX = op::Data(ori_name + "_dataX").set_attr_index(0);
    ge::Operator dataW = op::Data(ori_name + "_dataW").set_attr_index(1);
    std::vector<Operator> inputs{dataX, dataW};
    std::vector<std::pair<Operator, std::vector<size_t>>> outputs;
    ge::Operator convTranspose;
    ge::Operator dataB;

    std::vector<int64_t> dims = {(int)tbeAttr.input_size.size()};
    auto input_size_tensor = Vec2Tensor(tbeAttr.input_size, dims, ge::DT_INT64);
    auto const_input_size = op::Const(ori_name + "_Const").set_attr_value(input_size_tensor);

    if (tbeAttr.dim_size == INPUT_4D) {
        if (tbeAttr.trans_2d) {
            ge::Operator::OpListInt axes = {2};
            dataX = op::Unsqueeze(ori_name + "_UnsqueezeX").set_input_x(dataX).set_attr_axes(axes);
            dataW = op::Unsqueeze(ori_name + "_UnsqueezeW").set_input_x(dataW).set_attr_axes(axes);
        }
        switch (tbeAttr.input_num) {
            case INPUT_NUM_2:
                convTranspose = op::Conv2DTranspose(ori_name + "_Conv2DTranspose")
                                    .set_input_x(dataX)
                                    .set_input_filter(dataW)
                                    .set_input_input_size(const_input_size)
                                    .set_attr_strides(tbeAttr.strides)
                                    .set_attr_pads(tbeAttr.pads)
                                    .set_attr_dilations(tbeAttr.dilations)
                                    .set_attr_groups(tbeAttr.groups)
                                    .set_attr_output_padding(tbeAttr.output_padding)
                                    .set_attr_data_format(tbeAttr.data_format);
                break;
            case INPUT_NUM_3:
                dataB = op::Data(ori_name + "_dataB").set_attr_index(INPUT_NUM_3 - 1);
                inputs.push_back(dataB);
                convTranspose = op::Conv2DTranspose(ori_name + "_Conv2DTranspose")
                                    .set_input_x(dataX)
                                    .set_input_filter(dataW)
                                    .set_input_input_size(const_input_size)
                                    .set_input_bias(dataB)
                                    .set_attr_strides(tbeAttr.strides)
                                    .set_attr_pads(tbeAttr.pads)
                                    .set_attr_dilations(tbeAttr.dilations)
                                    .set_attr_groups(tbeAttr.groups)
                                    .set_attr_output_padding(tbeAttr.output_padding)
                                    .set_attr_data_format(tbeAttr.data_format);
                break;
            default:
                OP_LOGE("ConvTranspose", "the num of inputs is incorrect.");
                return FAILED;
        }
        if (SetFormatConvTranspose(convTranspose, tbeAttr.dim_size) != SUCCESS) {
            return FAILED;
        }
        if (tbeAttr.trans_2d) {
            ge::Operator::OpListInt axis = {2};
            convTranspose = op::Squeeze(ori_name + "_SqueezeY").set_input_x(convTranspose).set_attr_axis(axis);
        }
    } else if (tbeAttr.dim_size == INPUT_5D) {
        switch (tbeAttr.input_num) {
            case INPUT_NUM_2:
                convTranspose = op::Conv3DTranspose(ori_name + "_Conv3DTranspose")
                                    .set_input_x(dataX)
                                    .set_input_filter(dataW)
                                    .set_input_input_size(const_input_size)
                                    .set_attr_strides(tbeAttr.strides)
                                    .set_attr_pads(tbeAttr.pads)
                                    .set_attr_dilations(tbeAttr.dilations)
                                    .set_attr_groups(tbeAttr.groups)
                                    .set_attr_output_padding(tbeAttr.output_padding)
                                    .set_attr_data_format(tbeAttr.data_format);
                break;
            case INPUT_NUM_3:
                dataB = op::Data(ori_name + "_dataB").set_attr_index(INPUT_NUM_3 - 1);
                inputs.push_back(dataB);
                convTranspose = op::Conv3DTranspose(ori_name + "_Conv3DTranspose")
                                    .set_input_x(dataX)
                                    .set_input_filter(dataW)
                                    .set_input_bias(dataB)
                                    .set_input_input_size(const_input_size)
                                    .set_attr_strides(tbeAttr.strides)
                                    .set_attr_pads(tbeAttr.pads)
                                    .set_attr_dilations(tbeAttr.dilations)
                                    .set_attr_groups(tbeAttr.groups)
                                    .set_attr_output_padding(tbeAttr.output_padding)
                                    .set_attr_data_format(tbeAttr.data_format);
                break;
            default:
                OP_LOGE("ConvTranspose", "the num of inputs is incorrect.");
                return FAILED;
        }
        if (SetFormatConvTranspose(convTranspose, tbeAttr.dim_size) != SUCCESS) {
            return FAILED;
        }
    } else {
        OP_LOGE("ConvTranspose", "just support 4D or 5D input, failed to transform.");
        return FAILED;
    }

    convTranspose.SetAttr("auto_pad", tbeAttr.auto_pad);
    convTranspose.SetAttr("output_shape", tbeAttr.output_shape);
    convTranspose.SetAttr("fixed_shift_value", tbeAttr.fixed_shift_value);
    outputs.emplace_back(convTranspose, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::ConvTranspose"), ge::AscendString("ai.onnx::9::ConvTranspose"),
                   ge::AscendString("ai.onnx::10::ConvTranspose"), ge::AscendString("ai.onnx::11::ConvTranspose"),
                   ge::AscendString("ai.onnx::12::ConvTranspose"), ge::AscendString("ai.onnx::13::ConvTranspose"),
                   ge::AscendString("ai.onnx::14::ConvTranspose"), ge::AscendString("ai.onnx::15::ConvTranspose"),
                   ge::AscendString("ai.onnx::16::ConvTranspose"), ge::AscendString("ai.onnx::17::ConvTranspose"),
                   ge::AscendString("ai.onnx::18::ConvTranspose")})
    .ParseParamsFn(ParseParamsConvTranspose)
    .ParseOpToGraphFn(ParseOpToGraphConvTranspose)
    .ImplyType(ImplyType::TVM);
} // namespace domi
