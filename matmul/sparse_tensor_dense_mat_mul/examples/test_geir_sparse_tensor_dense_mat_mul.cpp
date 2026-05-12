/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_sparse_tensor_dense_mat_mul.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <map>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "array_ops.h"
#include "../op_graph/sparse_tensor_dense_mat_mul_proto.h"

using std::map;
using std::vector;
using namespace ge;

static Tensor MakeFp32Tensor(const vector<int64_t>& shape, const vector<float>& vals)
{
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    desc.SetPlacement(kPlacementHost);
    size_t n = vals.size();
    auto* buf = new float[n]{};
    for (size_t i = 0; i < n; ++i) {
        buf[i] = vals[i];
    }
    return Tensor(desc, (uint8_t*)buf, n * sizeof(float));
}

static Tensor MakeInt64Tensor(const vector<int64_t>& shape, const vector<int64_t>& vals)
{
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, DT_INT64);
    desc.SetPlacement(kPlacementHost);
    size_t n = vals.size();
    auto* buf = new int64_t[n]{};
    for (size_t i = 0; i < n; ++i) {
        buf[i] = vals[i];
    }
    return Tensor(desc, (uint8_t*)buf, n * sizeof(int64_t));
}

int main()
{
    // 1. Init GE
    map<AscendString, AscendString> geOpts = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    auto ret = GEInitialize(geOpts);
    if (ret != SUCCESS) {
        printf("GEInitialize failed\n");
        return -1;
    }

    // 2. Build graph
    Graph graph("test_spmm_graph");
    vector<Tensor> inputTensors;
    vector<Operator> graphInputs;
    auto op1 = op::SparseTensorDenseMatMul("spmm");

    // x1_indices: int64 [3,2] -> [[0,0],[0,2],[1,1]]
    {
        auto d = op::Data("x1_indices").set_attr_index(0);
        TensorDesc desc(ge::Shape({3, 2}), FORMAT_ND, DT_INT64);
        desc.SetPlacement(kPlacementHost);
        auto t = MakeInt64Tensor({3, 2}, {0, 0, 0, 2, 1, 1});
        d.update_input_desc_x(desc);
        d.update_output_desc_y(desc);
        graph.AddOp(d);
        op1.set_input_x1_indices(d);
        inputTensors.push_back(t);
        graphInputs.push_back(d);
    }

    // x1_values: float32 [3] -> [1,3,5]
    {
        auto d = op::Data("x1_values").set_attr_index(0);
        TensorDesc desc(ge::Shape({3}), FORMAT_ND, DT_FLOAT);
        desc.SetPlacement(kPlacementHost);
        auto t = MakeFp32Tensor({3}, {1.f, 3.f, 5.f});
        d.update_input_desc_x(desc);
        d.update_output_desc_y(desc);
        graph.AddOp(d);
        op1.set_input_x1_values(d);
        inputTensors.push_back(t);
        graphInputs.push_back(d);
    }

    // x1_shape: int64 [2] -> [2,3] (const, ValueDepend)
    {
        auto c = op::Const("x1_shape");
        TensorDesc desc(ge::Shape({2}), FORMAT_ND, DT_INT64);
        desc.SetPlacement(kPlacementHost);
        auto t = MakeInt64Tensor({2}, {2, 3});
        c.SetAttr("value", t);
        c.update_output_desc_y(desc);
        graph.AddOp(c);
        op1.set_input_x1_shape(c);
        op1.update_input_desc_x1_shape(desc);
        graphInputs.push_back(c);
    }

    // x2: float32 [3,2] -> [[1,2],[3,4],[5,6]]
    {
        auto d = op::Data("x2").set_attr_index(0);
        TensorDesc desc(ge::Shape({3, 2}), FORMAT_ND, DT_FLOAT);
        desc.SetPlacement(kPlacementHost);
        auto t = MakeFp32Tensor({3, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
        d.update_input_desc_x(desc);
        d.update_output_desc_y(desc);
        graph.AddOp(d);
        op1.set_input_x2(d);
        inputTensors.push_back(t);
        graphInputs.push_back(d);
    }

    // y
    op1.update_output_desc_y(TensorDesc(ge::Shape({2, 2}), FORMAT_ND, DT_FLOAT));
    op1.set_attr_adjoint_a(false);
    op1.set_attr_adjoint_b(false);

    graph.AddOp(op1);
    graph.SetInputs(graphInputs).SetOutputs({op1});

    // 3. Run
    map<AscendString, AscendString> sessOpts;
    auto* session = new Session(sessOpts);
    if (!session) {
        printf("Create session failed\n");
        GEFinalize();
        return -1;
    }
    map<AscendString, AscendString> graphOpts;
    ret = session->AddGraph(0, graph, graphOpts);
    if (ret != SUCCESS) {
        printf("AddGraph failed\n");
        delete session;
        GEFinalize();
        return -1;
    }

    vector<Tensor> output;
    ret = session->RunGraph(0, inputTensors, output);
    if (ret != SUCCESS) {
        printf("RunGraph failed: %s\n", std::string(ge::GEGetErrorMsgV2().GetString()).c_str());
        delete session;
        GEFinalize();
        return -1;
    }

    // 4. Print result
    auto* data = (float*)output[0].GetData();
    int64_t n = output[0].GetTensorDesc().GetShape().GetShapeSize();
    printf("Output y (float32, shape=[2,2]):\n");
    for (int64_t i = 0; i < n; ++i) {
        printf(" y[%ld] = %f\n", i, data[i]);
    }
    printf("Expected: [16, 20, 15, 20]\n");

    delete session;
    GEFinalize();

    return 0;
}