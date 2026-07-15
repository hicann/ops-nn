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
#include "pooling/avg_pool3_d/op_graph/avg_pool3_d_proto.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
struct AvgPoolAttr {
    std::string auto_pad = "NOTSET";
    int64_t ceil_mode = 0;
    int64_t count_include_pad = 0;
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
};

struct AvgTbeAttr {
    bool trans_2d = false;
    std::string padding_mode = "NOTSET";
    int64_t ceil_mode = 0;
    int64_t exclusive = 0;
    std::vector<int64_t> ksize;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
};

static void AvgMaybeChangeAttr(std::vector<int64_t>& value, int64_t length, int64_t num, bool transform_2d)
{
    if (value.empty()) {
        value = std::vector<int64_t>(length, num);
    } else if (length == 4 && num != 0) {
        value.resize(length);
        value[3] = transform_2d ? 1 : value[1];
        value[2] = value[0];
        value[1] = 1;
        value[0] = 1;
    } else if (length == 4 && num == 0 && transform_2d) {
        value.resize(length);
        value[3] = 0;
        value[2] = 0;
    }
}

static Status AvgUpdateAttrFromOnnx(const NodeProto* node, AvgPoolAttr& node_attr)
{
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
            node_attr.auto_pad = attr.s();
            if (node_attr.auto_pad == "SAME_LOWER") {
                OP_LOGW("AveragePool", "Current auto_pad not surpport SAME_LOWER, please ignore");
            }
        }

        if (attr.name() == "ceil_mode" && attr.type() == ge::onnx::AttributeProto::INT) {
            node_attr.ceil_mode = attr.i();
        }

        if (attr.name() == "count_include_pad" && attr.type() == ge::onnx::AttributeProto::INT) {
            node_attr.count_include_pad = attr.i();
        }

        if (attr.name() == "kernel_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                node_attr.kernel_shape.push_back(attr.ints(i));
            }
        }
        if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                node_attr.strides.push_back(attr.ints(i));
            }
        }
        if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
            unsigned int len = attr.ints_size();
            if (len & 1) {
                OP_LOGE("AveragePool",
                        "the length of pads must be even, such as [x1_begin, x2_begin...x1_end, x2_end,...]");
                return FAILED;
            }
            for (unsigned int i = 0; i < len / 2; i++) {
                node_attr.pads.push_back(attr.ints(i));
                node_attr.pads.push_back(attr.ints(i + len / 2));
            }
        }
    }
    return SUCCESS;
}

static Status ParseParamsAveragePool(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("AveragePool", "reinterpret_cast op_src to NodeProto failed.");
        return FAILED;
    }

    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    op_dest.SetAttr("original_type", "ai.onnx::11::AveragePool");

    AvgPoolAttr node_attr;
    if (AvgUpdateAttrFromOnnx(node, node_attr) != SUCCESS) {
        return FAILED;
    }

    int64_t dims = node_attr.kernel_shape.size();
    if (dims != 1 && dims != 2 && dims != 3) {
        OP_LOGE("AveragePool", "Only support 1D/2D/3D, but the length of kernel_shape is %ld", dims);
        return FAILED;
    }

    std::map<string, string> padding_mode = {
        {"NOTSET", "CALCULATED"}, {"SAME_UPPER", "SAME"}, {"SAME_LOWER", "SAME"}, {"VALID", "VALID"}};
    bool trans = false;
    if (dims == 1) {
        dims = 2;
        trans = true;
    }

    const int len_size = 2;
    AvgMaybeChangeAttr(node_attr.kernel_shape, dims == len_size ? dims + len_size : dims, 1, trans);
    op_dest.SetAttr("ksize", node_attr.kernel_shape);

    AvgMaybeChangeAttr(node_attr.strides, dims == len_size ? dims + len_size : dims, 1, trans);
    op_dest.SetAttr("strides", node_attr.strides);

    op_dest.SetAttr("padding_mode", padding_mode[node_attr.auto_pad]);
    op_dest.SetAttr("dims", dims);

    AvgMaybeChangeAttr(node_attr.pads, dims * 2, 0, trans);
    op_dest.SetAttr("pads", node_attr.pads);

    op_dest.SetAttr("ceil_mode", node_attr.ceil_mode);
    op_dest.SetAttr("exclusive", node_attr.count_include_pad);
    op_dest.SetAttr("trans_2d", trans);
    op_dest.SetAttr("name", node->name());
    return SUCCESS;
}

static Status AvgUpdateTbeAttrFromOp(const Operator& op, AvgTbeAttr& tbe_attr)
{
    if (op.GetAttr("ceil_mode", tbe_attr.ceil_mode) != SUCCESS) {
        OP_LOGE("AveragePool", "get ceil_mode from op failed");
        return FAILED;
    };
    if (op.GetAttr("padding_mode", tbe_attr.padding_mode) != SUCCESS) {
        OP_LOGE("AveragePool", "get padding_mode from op failed");
        return FAILED;
    };
    if (op.GetAttr("ksize", tbe_attr.ksize) != SUCCESS) {
        OP_LOGE("AveragePool", "get ksize from op failed");
        return FAILED;
    };
    if (op.GetAttr("strides", tbe_attr.strides) != SUCCESS) {
        OP_LOGE("AveragePool", "get strides from op failed");
        return FAILED;
    };
    if (op.GetAttr("pads", tbe_attr.pads) != SUCCESS) {
        OP_LOGE("AveragePool", "get pads from op failed");
        return FAILED;
    };
    if (op.GetAttr("exclusive", tbe_attr.exclusive) != SUCCESS) {
        OP_LOGE("AveragePool", "get exclusive from op failed");
        return FAILED;
    };
    if (op.GetAttr("trans_2d", tbe_attr.trans_2d) != SUCCESS) {
        OP_LOGW("AveragePool", "get trans_2d from op failed, use default.");
    };
    return SUCCESS;
}

static Status AvgUpdateFormat(Operator& op, Format format)
{
    ge::TensorDesc orgTensorX = op.GetInputDesc("x");
    orgTensorX.SetOriginFormat(format);
    orgTensorX.SetFormat(format);
    auto ret = op.UpdateInputDesc("x", orgTensorX);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update input x format failed.");
        return FAILED;
    }
    OP_LOGD(GetOpName(op).c_str(), "update input x format success, now is %d", op.GetInputDesc("x").GetFormat());

    ge::TensorDesc orgTensorY = op.GetOutputDesc("y");
    orgTensorY.SetOriginFormat(format);
    orgTensorY.SetFormat(format);
    ret = op.UpdateOutputDesc("y", orgTensorY);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update output y format failed.");
        return FAILED;
    }
    OP_LOGD(GetOpName(op).c_str(), "update output y format success, now is %d", op.GetOutputDesc("y").GetFormat());
    return SUCCESS;
}

static Status ParseOpToGraphAveragePool(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    int dims = 0;
    if (op.GetAttr("dims", dims) != SUCCESS) {
        OP_LOGE("AveragePool", "get dims from op failed");
        return FAILED;
    }

    AvgTbeAttr tbe_attr;
    if (AvgUpdateTbeAttrFromOp(op, tbe_attr) != SUCCESS) {
        return FAILED;
    }

    ge::Operator data0 = op::Data(ori_name + "_data0").set_attr_index(0);
    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t>>> outputs;

    const int len_dim = 2;
    if (dims != len_dim) {
        auto avgpool3d = op::AvgPool3D(ori_name + "_AvgPool3D")
                             .set_input_x(data0)
                             .set_attr_ksize(tbe_attr.ksize)
                             .set_attr_strides(tbe_attr.strides)
                             .set_attr_pads(tbe_attr.pads)
                             .set_attr_count_include_pad(tbe_attr.exclusive != 0)
                             .set_attr_ceil_mode(tbe_attr.ceil_mode)
                             .set_attr_data_format("NCDHW");
        if (AvgUpdateFormat(avgpool3d, ge::FORMAT_NCDHW) != SUCCESS) {
            return FAILED;
        }
        outputs.emplace_back(avgpool3d, std::vector<std::size_t>{0});
    } else {
        if (tbe_attr.trans_2d) {
            ge::Operator::OpListInt axes = {3};
            data0 = op::Unsqueeze(ori_name + "_UnsqueezeX").set_input_x(data0).set_attr_axes(axes);
        }
        ge::Operator avgpoolv2 = op::AvgPoolV2(ori_name + "_AvgPoolV2")
                                     .set_input_x(data0)
                                     .set_attr_ksize(tbe_attr.ksize)
                                     .set_attr_strides(tbe_attr.strides)
                                     .set_attr_padding_mode(tbe_attr.padding_mode)
                                     .set_attr_pads(tbe_attr.pads)
                                     .set_attr_ceil_mode(tbe_attr.ceil_mode != 0)
                                     .set_attr_exclusive(tbe_attr.exclusive != 1)
                                     .set_attr_data_format("NCHW");
        if (tbe_attr.trans_2d) {
            ge::Operator::OpListInt axis = {3};
            avgpoolv2 = op::Squeeze(ori_name + "_SqueezeAvgpoolv2").set_input_x(avgpoolv2).set_attr_axis(axis);
        }
        if (AvgUpdateFormat(avgpoolv2, ge::FORMAT_NCHW) != SUCCESS) {
            return FAILED;
        }
        outputs.emplace_back(avgpoolv2, std::vector<std::size_t>{0});
    }

    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::AveragePool", "ai.onnx::9::AveragePool", "ai.onnx::10::AveragePool",
                   "ai.onnx::11::AveragePool", "ai.onnx::12::AveragePool", "ai.onnx::13::AveragePool",
                   "ai.onnx::14::AveragePool", "ai.onnx::15::AveragePool", "ai.onnx::16::AveragePool",
                   "ai.onnx::17::AveragePool", "ai.onnx::18::AveragePool"})
    .ParseParamsFn(ParseParamsAveragePool)
    .ParseOpToGraphFn(ParseOpToGraphAveragePool)
    .ImplyType(ImplyType::TVM);
} //  namespace domi
