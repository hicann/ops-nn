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
 * \file batch_matmul_tf_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>
#include "register/register.h"
#include "log/log.h"
#include "error_util.h"
#include "graph/operator.h"
#include "stub_ops.h"
#include "matmul/batch_mat_mul_v3/op_graph/batch_mat_mul_v3_proto.h"

namespace domi {
using namespace ge;

static Status AutoMappingFnBatchMatMul(const ge::Operator& op_src, ge::Operator& op)
{
    ge::AscendString op_name;
    OP_LOGE_IF(op.GetName(op_name) != ge::GRAPH_SUCCESS, FAILED, "", "failed to get op_name");

    Status ret = AutoMappingByOpFn(op_src, op);
    OP_LOGE_IF(ret != SUCCESS, FAILED, op_name.GetString(), "tensorflow plugin parsing failed.");
    bool transposeA = false;
    OP_LOGE_IF(op.GetAttr("adj_x", transposeA) != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(),
               "failed to get adj_x.");
    bool transposeB = false;
    OP_LOGE_IF(op.GetAttr("adj_y", transposeB) != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(),
               "failed to get adj_y.");
    op.SetAttr("adj_x1", transposeA);
    op.SetAttr("adj_x2", transposeB);

    ge::AscendString op_type;
    OP_LOGE_IF(op_src.GetOpType(op_type) != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(), "failed to get op_type");
    if (string(op_type.GetString()) != "BatchMatMulV3") {
        OP_LOGD(op_name.GetString(), "op[BatchMatMul] tensorflow plugin parsing[AutoMapping] succeeded.");
        return SUCCESS;
    }

    // Set original_type
    op.SetAttr("original_type", "BatchMatMulV3");
    ge::DataType data_type;
    OP_LOGE_IF(op.GetAttr("Tout", data_type) != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(),
               "failed to getAttr Tout.");
    op.SetAttr("dst_type", static_cast<int>(data_type));

    OP_LOGD(op_name.GetString(), "op[BatchMatMulV3] tensorflow plugin parsing[AutoMapping] succeeded.");
    return SUCCESS;
}

static Status ParseOpToGraphBatchMatMulV3(const ge::Operator& op, ge::Graph& graph)
{
    ge::AscendString op_name;
    OP_LOGE_IF(op.GetName(op_name) != ge::GRAPH_SUCCESS, FAILED, "", "failed to get op_name");
    OP_LOGD(op_name.GetString(), "op[BatchMatMulV3] tensorflow plugin ParseOpToGraph start.");
    bool transpose_x1 = false;
    OP_LOGE_IF(op.GetAttr("adj_x1", transpose_x1) != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(),
               "failed to get adj_x1.");
    bool transpose_x2 = false;
    OP_LOGE_IF(op.GetAttr("adj_x2", transpose_x2) != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(),
               "failed to get adj_x2.");

    auto data_1 = op::Data("x1");
    (void)data_1.set_attr_index(0);
    auto data_2 = op::Data("x2");
    (void)data_2.set_attr_index(1);
    std::vector<ge::Operator> inputs{data_1, data_2};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indices;
    auto batch_matmul = op::BatchMatMulV3();
    (void)batch_matmul.set_input_x1(data_1);
    (void)batch_matmul.set_input_x2(data_2);
    (void)batch_matmul.set_attr_adj_x1(transpose_x1);
    (void)batch_matmul.set_attr_adj_x2(transpose_x2);

    int dstType = 1;
    OP_LOGE_IF(op.GetAttr("dst_type", dstType) != ge::GRAPH_SUCCESS, FAILED, op_name.GetString(),
               "failed to get dst_type.");

    auto cast = op::Cast();
    (void)cast.set_input_x(batch_matmul);
    (void)cast.set_attr_dst_type(dstType);
    output_indices.emplace_back(cast, vector<std::size_t>{0});

    graph.SetInputs(inputs).SetOutputs(output_indices);
    OP_LOGD(op_name.GetString(), "op[BatchMatMulV3] tensorflow plugin ParseOpToGraph success.");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("BatchMatMulV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType(std::vector<ge::AscendString>{"BatchMatMul", "BatchMatMulV2"})
    .ParseParamsByOperatorFn(AutoMappingFnBatchMatMul)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("BatchMatMulV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BatchMatMulV3")
    .ParseParamsByOperatorFn(AutoMappingFnBatchMatMul)
    .ParseOpToGraphFn(ParseOpToGraphBatchMatMulV3)
    .ImplyType(ImplyType::TVM);
} // namespace domi
