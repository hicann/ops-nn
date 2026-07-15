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
static const int ONNX_1D_ATTR_LEN = 1;
static const int ONNX_1D_ATTR_PAD_LEN = 2;

struct ConvAttr {
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads;
    int64_t groups;
    std::string data_format;
    int dim_size;
    int input_num;
    bool trans_2d = false;
    std::string auto_pad = "NOTSET";
};

static Status SetAttrToOp(const ge::onnx::NodeProto* node, ge::Operator& op)
{
    std::vector<int32_t> strides_list = {1, 1};
    std::vector<int32_t> dilations_list = {1, 1};
    std::vector<int32_t> pad_list;
    bool is_trans_2d = false;
    bool is_have_kernel_shape = false;
    int dim_size = INPUT_4D;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
            if (attr.ints_size() == ONNX_1D_ATTR_LEN) {
                strides_list.push_back(1);
                is_trans_2d = true;
            }
            for (auto i = 0; i < attr.ints_size(); ++i) {
                strides_list.push_back(attr.ints(i));
            }
            op.SetAttr("strides", strides_list);
        } else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
            if (attr.ints_size() == ONNX_1D_ATTR_LEN) {
                dilations_list.push_back(1);
                is_trans_2d = true;
            }
            for (auto i = 0; i < attr.ints_size(); ++i) {
                dilations_list.push_back(attr.ints(i));
            }
            op.SetAttr("dilations", dilations_list);
        } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
            unsigned int len = attr.ints_size();
            if (len & 1) {
                OP_LOGE("Conv", "The value lenth of pads is odd, transform failed.");
                return FAILED;
            }
            if (attr.ints_size() == ONNX_1D_ATTR_PAD_LEN) {
                pad_list.push_back(0);
                pad_list.push_back(0);
                is_trans_2d = true;
            }
            for (unsigned int i = 0; i < len / 2; i++) {
                pad_list.push_back(attr.ints(i));
                pad_list.push_back(attr.ints(i + len / 2));
            }
            op.SetAttr("pads", pad_list);
        } else if (attr.name() == "group" && attr.type() == ge::onnx::AttributeProto::INT) {
            op.SetAttr("groups", attr.i());
        } else if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
            op.SetAttr("auto_pad", attr.s());
        } else if (attr.name() == "kernel_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
            is_have_kernel_shape = true;
            is_trans_2d = attr.ints_size() >= 2 ? false : true;
            dim_size = attr.ints_size() > 2 ? INPUT_5D : INPUT_4D;
        }
    }

    if (!is_have_kernel_shape) {
        if (strides_list.size() == 2 && dilations_list.size() == 2 && pad_list.empty()) {
            OP_LOGE(GetOpName(op).c_str(), "node must have attr (kernel_shape,pads,strides,dilations) at least one");
            return FAILED;
        }
        dim_size = (strides_list.size() == 5 || pad_list.size() == 6 || dilations_list.size() == 5) ? 5 : 4;
    }

    op.SetAttr("dim_size", dim_size);
    op.SetAttr("trans_2d", is_trans_2d);

    return SUCCESS;
}

static Status ParseParamsConv(const Message* op_src, ge::Operator& op)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE("Conv", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int n = node->input_size();
    op.SetAttr("input_num", n);
    op.SetAttr("name", node->name());
    op.DynamicInputRegister("args", n);
    op.DynamicOutputRegister("output", 1);
    op.SetAttr("original_type", "ai.onnx::11::Conv");

    if (SetAttrToOp(node, op) != SUCCESS) {
        OP_LOGE("Conv", "set attr to operator failed");
        return FAILED;
    }

    return SUCCESS;
}

static Status SetFormat(ge::Operator& op, const int& dims)
{
    if (dims == INPUT_4D) {
        auto ret_x = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCHW, true);
        if (ret_x != ge::GRAPH_SUCCESS) {
            OP_LOGE("Conv", "update fmap format failed.");
            return FAILED;
        }
        auto ret_w = ChangeFormatFromOnnx(op, 1, ge::FORMAT_NCHW, true);
        if (ret_w != ge::GRAPH_SUCCESS) {
            OP_LOGE("Conv", "update filter format failed.");
            return FAILED;
        }
        auto ret_y = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCHW, false);
        if (ret_y != ge::GRAPH_SUCCESS) {
            OP_LOGE("Conv", "update output format failed.");
            return FAILED;
        }
    } else if (dims == INPUT_5D) {
        auto ret_x = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCDHW, true);
        if (ret_x != ge::GRAPH_SUCCESS) {
            OP_LOGE("Conv", "update fmap format failed.");
            return FAILED;
        }
        auto ret_w = ChangeFormatFromOnnx(op, 1, ge::FORMAT_NCDHW, true);
        if (ret_w != ge::GRAPH_SUCCESS) {
            OP_LOGE("Conv", "update filter format failed.");
            return FAILED;
        }
        auto ret_y = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCDHW, false);
        if (ret_y != ge::GRAPH_SUCCESS) {
            OP_LOGE("Conv", "update output format failed.");
            return FAILED;
        }
    } else {
        OP_LOGE("Conv", "The input tensor is not 4D/5D, set format failed.");
        return FAILED;
    }
    return SUCCESS;
}

static Status GetConvAttr(const ge::Operator& op, ConvAttr& convAttr)
{
    auto ret_strides = op.GetAttr("strides", convAttr.strides);
    auto ret_pads = op.GetAttr("pads", convAttr.pads);
    auto ret_dilations = op.GetAttr("dilations", convAttr.dilations);
    op.GetAttr("auto_pad", convAttr.auto_pad);
    std::string pad_mode = convAttr.auto_pad;

    if (ret_strides != SUCCESS && ret_pads != SUCCESS && ret_dilations != SUCCESS) {
        OP_LOGW("Conv",
                "get attr of strides or pads or dilations from op failed, can not distinguish 2D/3D, use default 2D,"
                " please set one of them obviously.");
    }
    if (op.GetAttr("dim_size", convAttr.dim_size) != SUCCESS) {
        OP_LOGE("Conv", "get dim size from op failed");
        return FAILED;
    }
    if (pad_mode != "NOTSET" && pad_mode != "VALID" && convAttr.dim_size == INPUT_5D) {
        OP_LOGE("Conv",
                "The attr of auto_pad is not NOTSET/VALID for 5D, unsupported other value for now, transform failed,"
                "may cause precision error.");
        return FAILED;
    }
    if (op.GetAttr("input_num", convAttr.input_num) != SUCCESS) {
        OP_LOGE("Conv", "get number of input from op failed");
        return FAILED;
    }
    if (op.GetAttr("groups", convAttr.groups) != SUCCESS)
        convAttr.groups = 1;

    if (op.GetAttr("data_format", convAttr.data_format) != SUCCESS) {
        std::string data_format = (convAttr.strides.size() == 5 || convAttr.pads.size() == 6 ||
                                   convAttr.dilations.size() == 5) ?
                                      "NCDHW" :
                                      "NCHW";
        convAttr.data_format = data_format;
    }

    if (op.GetAttr("trans_2d", convAttr.trans_2d) != SUCCESS) {
        OP_LOGW("Conv", "get the flag of convert 1d to 2d failed, use default.");
    }

    std::vector<int64_t> strides_list_default = {1, 1, 1, 1};
    std::vector<int64_t> dilations_list_default = {1, 1, 1, 1};
    std::vector<int64_t> pad_list_default = {0, 0, 0, 0};
    if (convAttr.dim_size == INPUT_5D) {
        strides_list_default.push_back(1);
        dilations_list_default.push_back(1);
        pad_list_default.push_back(0);
        pad_list_default.push_back(0);
    }

    if (convAttr.strides.size() == 2) {
        convAttr.strides = strides_list_default;
    }
    if (convAttr.dilations.size() == 2) {
        convAttr.dilations = dilations_list_default;
    }
    if (convAttr.pads.size() == 0) {
        convAttr.pads = pad_list_default;
    }
    return SUCCESS;
}

static Status ParseOpToGraphConv(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    ConvAttr tbeAttr;
    if (GetConvAttr(op, tbeAttr) != SUCCESS) {
        OP_LOGE("Conv", "get attr value failed.");
        return FAILED;
    }

    ge::Operator dataX = op::Data(ori_name + "_dataX").set_attr_index(0);
    ge::Operator dataW = op::Data(ori_name + "_dataW").set_attr_index(1);
    std::vector<Operator> inputs{dataX, dataW};
    std::vector<std::pair<Operator, std::vector<size_t>>> outputs;
    ge::Operator conv;
    ge::Operator dataB;
    ge::Operator squeeze;
    if (tbeAttr.dim_size == INPUT_4D) {
        if (tbeAttr.trans_2d) {
            ge::Operator::OpListInt axes = {2};
            dataX = op::Unsqueeze(ori_name + "_UnsqueezeX").set_input_x(dataX).set_attr_axes(axes);
            dataW = op::Unsqueeze(ori_name + "_UnsqueezeW").set_input_x(dataW).set_attr_axes(axes);
        }
        switch (tbeAttr.input_num) {
            case INPUT_NUM_2:
                conv = op::Conv2D(ori_name + "_Conv2D")
                           .set_input_x(dataX)
                           .set_input_filter(dataW)
                           .set_attr_strides(tbeAttr.strides)
                           .set_attr_pads(tbeAttr.pads)
                           .set_attr_dilations(tbeAttr.dilations)
                           .set_attr_groups(tbeAttr.groups)
                           .set_attr_data_format(tbeAttr.data_format);
                break;
            case INPUT_NUM_3:
                dataB = op::Data(ori_name + "_dataB").set_attr_index(INPUT_NUM_3 - 1);
                inputs.push_back(dataB);
                conv = op::Conv2D(ori_name + "_Conv2D")
                           .set_input_x(dataX)
                           .set_input_filter(dataW)
                           .set_input_bias(dataB)
                           .set_attr_strides(tbeAttr.strides)
                           .set_attr_pads(tbeAttr.pads)
                           .set_attr_dilations(tbeAttr.dilations)
                           .set_attr_groups(tbeAttr.groups)
                           .set_attr_data_format(tbeAttr.data_format);
                break;
            default:
                OP_LOGE("Conv", "the num of inputs is incorrect.");
                return FAILED;
        }
        if (SetFormat(conv, tbeAttr.dim_size) != SUCCESS) {
            OP_LOGE("Conv", "set format for input and output failed.");
            return FAILED;
        }
        if (tbeAttr.trans_2d) {
            ge::Operator::OpListInt axis = {2};
            squeeze = op::Squeeze(ori_name + "_SqueezeY").set_input_x(conv).set_attr_axis(axis);
        }
    } else if (tbeAttr.dim_size == INPUT_5D) {
        switch (tbeAttr.input_num) {
            case INPUT_NUM_2:
                conv = op::Conv3D(ori_name + "_Conv3D")
                           .set_input_x(dataX)
                           .set_input_filter(dataW)
                           .set_attr_strides(tbeAttr.strides)
                           .set_attr_pads(tbeAttr.pads)
                           .set_attr_dilations(tbeAttr.dilations)
                           .set_attr_groups(tbeAttr.groups)
                           .set_attr_data_format(tbeAttr.data_format);
                break;
            case INPUT_NUM_3:
                dataB = op::Data(ori_name + "_dataB").set_attr_index(INPUT_NUM_3 - 1);
                inputs.push_back(dataB);
                conv = op::Conv3D(ori_name + "_Conv3D")
                           .set_input_x(dataX)
                           .set_input_filter(dataW)
                           .set_input_bias(dataB)
                           .set_attr_strides(tbeAttr.strides)
                           .set_attr_pads(tbeAttr.pads)
                           .set_attr_dilations(tbeAttr.dilations)
                           .set_attr_groups(tbeAttr.groups)
                           .set_attr_data_format(tbeAttr.data_format);
                break;
            default:
                OP_LOGE("Conv", "the num of inputs is incorrect.");
                return FAILED;
        }
        if (SetFormat(conv, tbeAttr.dim_size) != SUCCESS) {
            OP_LOGE("Conv", "set format for input and output failed.");
            return FAILED;
        }
    } else {
        OP_LOGE("Conv", "just support 4D or 5D input, transform failed.");
        return FAILED;
    }

    conv.SetAttr("auto_pad", tbeAttr.auto_pad);

    auto outputOp = tbeAttr.trans_2d ? squeeze : conv;
    outputs.emplace_back(outputOp, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Conv", "ai.onnx::9::Conv", "ai.onnx::10::Conv", "ai.onnx::11::Conv",
                   "ai.onnx::12::Conv", "ai.onnx::13::Conv", "ai.onnx::14::Conv", "ai.onnx::15::Conv",
                   "ai.onnx::16::Conv", "ai.onnx::17::Conv", "ai.onnx::18::Conv"})
    .ParseParamsFn(ParseParamsConv)
    .ParseOpToGraphFn(ParseOpToGraphConv)
    .ImplyType(ImplyType::TVM);
} // namespace domi
