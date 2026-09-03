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
 * \file gather_point_grad_tf_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>
#include <map>
#include "register/register.h"
#include "framework/plugin_util.h"
#include "graph/operator.h"
#include "stub_ops.h"
#include "index/scatter_update/op_graph/scatter_update_proto.h"
#include "log/log.h"

namespace domi {
using namespace ge;

static Status ParseParamsGatherPointGrad(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int n = static_cast<int>(op_dest.GetInputsSize());
    OP_LOGI(GetOpName(op_dest).c_str(), "ParseParamsGatherPointGrad input_size = %d", n);
    // 2.set original_type
    op_dest.SetAttr("original_type", std::string("GatherPointGrad"));
    // 3.set attr if needed
    op_dest.SetAttr("name", GetOpName(op_dest));

    return SUCCESS;
}

static Status ParseOpToGraphGatherPointGrad(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed");
        return FAILED;
    }

    ge::Operator data_0 = op::Data("inp").set_attr_index(0);
    ge::Operator data_1 = op::Data("idx").set_attr_index(1);
    ge::Operator data_2 = op::Data("out_g").set_attr_index(2);
    auto use_locking = false;

    auto ScatterUpdate = op::ScatterUpdate(ori_name.c_str())
                             .set_input_var(data_0)
                             .set_input_indices(data_1)
                             .set_input_updates(data_2)
                             .set_attr_use_locking(use_locking);
    std::vector<ge::Operator> inputs{data_0, data_1, data_2};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(ScatterUpdate, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);

    return SUCCESS;
}

// register GatherPointGrad op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("GatherPointGrad")
    .ParseParamsByOperatorFn(ParseParamsGatherPointGrad)
    .ParseOpToGraphFn(ParseOpToGraphGatherPointGrad)
    .ImplyType(ImplyType::TVM);
} // namespace domi
