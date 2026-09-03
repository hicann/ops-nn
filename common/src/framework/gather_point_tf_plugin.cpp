/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_point_tf_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>
#include <map>
#include "register/register.h"
#include "framework/plugin_util.h"
#include "graph/operator.h"
#include "stub_ops.h"
#include "index/gather_v2/op_graph/gather_v2_proto.h"
#include "log/log.h"

namespace domi {
using namespace ge;

static Status ParseParamsGatherPoint(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int n = static_cast<int>(op_dest.GetInputsSize());
    OP_LOGI(GetOpName(op_dest).c_str(), "ParseParamsGatherPoint input_size = %d", n);
    // 2.set original_type
    op_dest.SetAttr("original_type", std::string("GatherPoint"));
    // 3.set attr if needed
    op_dest.SetAttr("name", GetOpName(op_dest));

    return SUCCESS;
}

static Status ParseOpToGraphGatherPoint(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed");
        return FAILED;
    }

    ge::Operator data_0 = op::Data("inp").set_attr_index(0);
    ge::Operator data_1 = op::Data("idx").set_attr_index(1);
    int32_t axis_val = 1;

    TensorDesc tensor1_desc(ge::Shape(), FORMAT_ND, DT_INT32);
    ge::Tensor const_value(tensor1_desc, (uint8_t*)&axis_val, sizeof(axis_val));
    auto const_op = op::Const("const_data").set_attr_value(const_value);
    int batch_dims = 1;
    auto GatherV2 = op::GatherV2(ori_name.c_str())
                        .set_input_x(data_0)
                        .set_input_indices(data_1)
                        .set_input_axis(const_op)
                        .set_attr_batch_dims(batch_dims);
    std::vector<ge::Operator> inputs{data_0, data_1, const_op};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(GatherV2, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);

    return SUCCESS;
}

// register GatherPoint op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("GatherPoint")
    .ParseParamsByOperatorFn(ParseParamsGatherPoint)
    .ParseOpToGraphFn(ParseOpToGraphGatherPoint)
    .ImplyType(ImplyType::TVM);
} // namespace domi
