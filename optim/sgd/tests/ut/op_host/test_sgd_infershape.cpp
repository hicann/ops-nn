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
 * \file test_sgd_infershape.cpp
 * \brief SGD InferShape 单元测试
 *
 * 覆盖面：
 *   - 正常推导（rank 2 / rank 1 / rank 8 边界）
 *   - UNKNOWN_RANK(-2) 透传
 *   - rank-0 拒绝、rank 9 拒绝（对齐 910B/910C 的 1~8）
 *   - 属性非法拒绝：nesterov && dampening != 0、weight_decay < 0
 *
 * 注：InferDataType 只在图场景使用，已按交付件划分挪到 op_graph/sgd_graph_infer.cpp。
 * op_graph UT 模块只链 graph_plugin_obj，不含 tests/ut/common 的 infershape 公共对象，
 * 故此处不再覆盖 —— 与仓内其他把 InferDataType 放 op_graph 的算子做法一致。
 */

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include <vector>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"
#include "../../../op_graph/sgd_proto.h"

class SGD : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SGD SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SGD TearDown" << std::endl; }
};

namespace {
constexpr size_t SGD_INPUT_NUM = 6;
constexpr size_t SGD_OUTPUT_NUM = 1;

// 按图原型顺序：parameters / gradient / learning_rate / accum / momentum / stat
ge::graphStatus RunSgdInferShape(gert::StorageShape& bigShape, gert::StorageShape& scalarShape, float dampening,
                                 float weightDecay, bool nesterov)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SGD")->infer_shape;
    if (inferShapeFunc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(SGD_INPUT_NUM, SGD_OUTPUT_NUM)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&bigShape, &bigShape, &scalarShape, &bigShape, &scalarShape, &bigShape})
                      .OutputShapes({&bigShape})
                      .NodeAttrs({{"dampening", Ops::NN::AnyValue::CreateFrom<float>(dampening)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom<float>(weightDecay)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(nesterov)}})
                      .Build();
    return inferShapeFunc(holder.GetContext<gert::InferShapeContext>());
}
} // namespace

TEST_F(SGD, sgd_infershape_registered)
{
    // op type 必须是全大写 SGD —— 与 canndev 的 GE op type 及 ini 段名 [SGD] 一致
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SGD"), nullptr);
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("SGD");
    ASSERT_NE(impl->infer_shape, nullptr);
    // InferDataType 现注册在 op_graph/sgd_graph_infer.cpp（仅图场景交付件），
    // 不在本 UT 的链接范围内，故此处不再断言 impl->infer_datatype。
}

TEST_F(SGD, sgd_infershape_normal_rank2)
{
    gert::StorageShape bigShape = {{96, 256}, {96, 256}};
    gert::StorageShape scalarShape = {{1}, {1}};
    ASSERT_EQ(RunSgdInferShape(bigShape, scalarShape, 0.0f, 0.0f, false), ge::GRAPH_SUCCESS);
}

TEST_F(SGD, sgd_infershape_normal_rank1_min_boundary)
{
    gert::StorageShape bigShape = {{33}, {33}};
    gert::StorageShape scalarShape = {{1}, {1}};
    ASSERT_EQ(RunSgdInferShape(bigShape, scalarShape, 0.0f, 0.0f, false), ge::GRAPH_SUCCESS);
}

TEST_F(SGD, sgd_infershape_normal_rank8_max_boundary)
{
    gert::StorageShape bigShape = {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}};
    gert::StorageShape scalarShape = {{1}, {1}};
    ASSERT_EQ(RunSgdInferShape(bigShape, scalarShape, 0.0f, 0.0f, false), ge::GRAPH_SUCCESS);
}

TEST_F(SGD, sgd_infershape_unknown_rank_passthrough)
{
    // UNKNOWN_RANK 在 GE 下表现为 dims == {-2}，必须【透传】而非按 rank 拒绝
    gert::StorageShape bigShape = {{-2}, {-2}};
    gert::StorageShape scalarShape = {{1}, {1}};
    ASSERT_EQ(RunSgdInferShape(bigShape, scalarShape, 0.0f, 0.0f, false), ge::GRAPH_SUCCESS);
}

TEST_F(SGD, sgd_infershape_rank0_rejected)
{
    // rank-0 标量被拒 —— 对齐 910B/910C（canndev var_dims.size() == 0 判非法）
    gert::StorageShape bigShape = {{}, {}};
    gert::StorageShape scalarShape = {{1}, {1}};
    ASSERT_EQ(RunSgdInferShape(bigShape, scalarShape, 0.0f, 0.0f, false), ge::GRAPH_FAILED);
}

TEST_F(SGD, sgd_infershape_rank9_rejected)
{
    // rank > 8 被拒（kMaxDimNum = 8）
    gert::StorageShape bigShape = {{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}};
    gert::StorageShape scalarShape = {{1}, {1}};
    ASSERT_EQ(RunSgdInferShape(bigShape, scalarShape, 0.0f, 0.0f, false), ge::GRAPH_FAILED);
}

TEST_F(SGD, sgd_infershape_nesterov_with_nonzero_dampening_rejected)
{
    // nesterov == true 时 dampening 必须为 0
    gert::StorageShape bigShape = {{96, 256}, {96, 256}};
    gert::StorageShape scalarShape = {{1}, {1}};
    ASSERT_EQ(RunSgdInferShape(bigShape, scalarShape, 0.5f, 0.0f, true), ge::GRAPH_FAILED);
}

TEST_F(SGD, sgd_infershape_nesterov_with_zero_dampening_ok)
{
    gert::StorageShape bigShape = {{96, 256}, {96, 256}};
    gert::StorageShape scalarShape = {{1}, {1}};
    ASSERT_EQ(RunSgdInferShape(bigShape, scalarShape, 0.0f, 0.01f, true), ge::GRAPH_SUCCESS);
}

TEST_F(SGD, sgd_infershape_negative_weight_decay_rejected)
{
    // weight_decay 必须 >= 0
    gert::StorageShape bigShape = {{96, 256}, {96, 256}};
    gert::StorageShape scalarShape = {{1}, {1}};
    ASSERT_EQ(RunSgdInferShape(bigShape, scalarShape, 0.0f, -0.01f, false), ge::GRAPH_FAILED);
}
