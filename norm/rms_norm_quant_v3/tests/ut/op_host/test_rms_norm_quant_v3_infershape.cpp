/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include "infershape_test_util.h"
#include "../../../op_graph/rms_norm_quant_v3_proto.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class RmsNormQuantV3Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmsNormQuantV3InferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RmsNormQuantV3InferShapeTest TearDown" << std::endl; }

    void CheckInferDataType(ge::DataType dstType, bool outputRstd, bool useDefaultAttrs = false)
    {
        const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormQuantV3");
        ASSERT_NE(opImpl, nullptr);
        auto dataTypeFunc = opImpl->infer_datatype;
        ASSERT_NE(dataTypeFunc, nullptr);

        gert::InferDataTypeContextFaker faker;
        faker.IrInputNum(3)
            .NodeIoNum(3, 3)
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND);
        if (!useDefaultAttrs) {
            faker.NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6F)},
                             {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                             {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(dstType))},
                             {"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(outputRstd)}});
        }
        auto contextHolder = faker.Build();
        auto* context = contextHolder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        for (size_t index = 0; index < 3; ++index) {
            context->SetOutputDataType(index, ge::DT_UNDEFINED);
            ASSERT_EQ(context->GetOutputDataType(index), ge::DT_UNDEFINED);
        }

        ASSERT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), dstType);
        EXPECT_EQ(context->GetOutputDataType(1), dstType);
        // A registered output needs a valid dtype even when output_rstd disables its computation.
        EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
    }
};

TEST_F(RmsNormQuantV3Test, empty_input_validates_dtype_combinations_before_ge_skips_tiling)
{
    gert::StorageShape x = {{0, 64}, {0, 64}};
    gert::StorageShape gamma = {{64}, {64}};
    gert::StorageShape scale = {{1}, {1}};
    gert::StorageShape y1;
    gert::StorageShape y2;
    gert::StorageShape rstd;
    auto check = [&](const std::vector<ge::DataType>& types) {
        gert::InferShapeContextFaker faker;
        faker.SetOpType("RmsNormQuantV3")
            .NodeIoNum(7, 3)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1}, {1, 1, 1})
            .InputShapes({&x, &gamma, &scale, &scale, &scale, &scale, &gamma})
            .OutputShapes({&y1, &y2, &rstd});
        for (size_t index = 0; index < types.size(); ++index) {
            faker.NodeInputTd(index, types[index], ge::FORMAT_ND, ge::FORMAT_ND);
        }
        auto holder = faker.Build();
        const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormQuantV3");
        EXPECT_NE(impl, nullptr);
        return impl == nullptr ? ge::GRAPH_FAILED : impl->infer_shape(holder.GetContext<gert::InferShapeContext>());
    };
    for (auto xType : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        const std::vector<ge::DataType> types = {xType,        xType,        ge::DT_FLOAT, ge::DT_FLOAT,
                                                 ge::DT_FLOAT, ge::DT_FLOAT, xType};
        EXPECT_EQ(check(types), ge::GRAPH_SUCCESS);
    }
    for (auto xType : {ge::DT_FLOAT16, ge::DT_BF16}) {
        const std::vector<ge::DataType> types = {xType, xType, xType, xType, ge::DT_INT8, ge::DT_INT8, xType};
        EXPECT_EQ(check(types), ge::GRAPH_SUCCESS);
    }
    const std::vector<ge::DataType> validTypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_FLOAT,
                                                  ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16};
    for (size_t index : {1U, 2U, 3U, 4U, 5U, 6U}) {
        SCOPED_TRACE(index);
        auto types = validTypes;
        types[index] = (index == 4 || index == 5) ? ge::DT_INT8 : ge::DT_BF16;
        EXPECT_EQ(check(types), ge::GRAPH_FAILED);
    }
    auto types = validTypes;
    types[4] = ge::DT_INT32;
    EXPECT_EQ(check(types), ge::GRAPH_FAILED);
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_shape_case1)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({64}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    op.SetAttr("output_rstd", false);
    std::vector<int64_t> expected_y1_shape = {8, 64};
    std::vector<int64_t> expected_y2_shape = {1};
    std::vector<int64_t> expected_rstd_shape = {};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y1_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y2_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_shape_case2)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({64}, ge::DT_FLOAT));
    op.UpdateInputDesc("scales2", create_desc({64}, ge::DT_FLOAT));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("div_mode", true);
    op.SetAttr("dst_type", 2);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "div_mode", "dst_type", "output_rstd"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_rstd_shape = {8, 1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_shape_with_scalar_scales2_uses_x_shape)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({2, 16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("scales2", create_desc({}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), (std::vector<int64_t>{2, 16}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), (std::vector<int64_t>{2, 16}));
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_dtype_output_rstd_true)
{
    for (auto dstType : {ge::DT_INT8, ge::DT_INT4, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8}) {
        SCOPED_TRACE("dst_type: " + std::to_string(static_cast<int>(dstType)));
        CheckInferDataType(dstType, true);
    }
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_dtype_output_rstd_false)
{
    for (auto dstType : {ge::DT_INT8, ge::DT_INT4, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8}) {
        SCOPED_TRACE("dst_type: " + std::to_string(static_cast<int>(dstType)));
        CheckInferDataType(dstType, false);
    }
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_dtype_default_attrs) { CheckInferDataType(ge::DT_INT8, false, true); }

TEST_F(RmsNormQuantV3Test, empty_int4_input_preserves_even_last_axis_requirement)
{
    for (int64_t lastDim : {0, 63, 64, -1}) {
        for (auto dstType : {ge::DT_INT4, ge::DT_INT8}) {
            for (bool outputRstd : {false, true}) {
                SCOPED_TRACE(std::to_string(lastDim) + ":" + std::to_string(dstType));
                ge::op::RmsNormQuantV3 op;
                op.UpdateInputDesc("x", create_desc({0, lastDim}, ge::DT_FLOAT16));
                op.UpdateInputDesc("gamma", create_desc({lastDim}, ge::DT_FLOAT16));
                op.UpdateInputDesc("scales1", create_desc({1}, ge::DT_FLOAT));
                op.SetAttr("epsilon", 1e-6F);
                op.SetAttr("div_mode", true);
                op.SetAttr("dst_type", static_cast<int64_t>(dstType));
                op.SetAttr("output_rstd", outputRstd);
                Runtime2TestParam param{{"epsilon", "div_mode", "dst_type", "output_rstd"}, {}, {}};
                const auto expected = dstType == ge::DT_INT4 && lastDim == 63 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
                EXPECT_EQ(InferShapeTest(op, param), expected);
            }
        }
    }
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_shape_m_not0_n_0_true)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({176, 0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({0}, ge::DT_FLOAT));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("div_mode", true);
    op.SetAttr("dst_type", 2);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "div_mode", "dst_type", "output_rstd"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y1_shape = {176, 0};
    std::vector<int64_t> expected_y2_shape = {0};
    std::vector<int64_t> expected_rstd_shape = {176, 1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y1_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y2_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_shape_m_0_n_not0_true)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({0, 100}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({100}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({100}, ge::DT_FLOAT));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("div_mode", true);
    op.SetAttr("dst_type", 2);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "div_mode", "dst_type", "output_rstd"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y1_shape = {0, 100};
    std::vector<int64_t> expected_y2_shape = {0};
    std::vector<int64_t> expected_rstd_shape = {0, 1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y1_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y2_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_infer_m_0_n_0_true)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({0, 0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({0}, ge::DT_FLOAT));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("div_mode", true);
    op.SetAttr("dst_type", 2);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "div_mode", "dst_type", "output_rstd"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y1_shape = {0, 0};
    std::vector<int64_t> expected_y2_shape = {0};
    std::vector<int64_t> expected_rstd_shape = {0, 1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y1_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y2_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormQuantV3Test, RmsNormQuantV3_unknown_rank_without_scales2_keeps_dummy_y2)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({-2}, ge::DT_FLOAT));
    op.SetAttr("output_rstd", false);

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), (std::vector<int64_t>{-2}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), (std::vector<int64_t>{1}));
}

TEST_F(RmsNormQuantV3Test, empty_batch_makes_all_registered_outputs_empty)
{
    for (bool outputRstd : {false, true}) {
        ge::op::RmsNormQuantV3 op;
        op.UpdateInputDesc("x", create_desc({2, 0, 64}, ge::DT_FLOAT16));
        op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
        op.UpdateInputDesc("scales1", create_desc({1}, ge::DT_FLOAT));
        op.SetAttr("output_rstd", outputRstd);
        Runtime2TestParam param{{"epsilon", "div_mode", "dst_type", "output_rstd"}, {}, {}};
        ASSERT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
        for (size_t output = 0; output < 3; ++output) {
            EXPECT_EQ(op.GetOutputDesc(output).GetShape().GetShapeSize(), 0);
        }
    }
}

TEST_F(RmsNormQuantV3Test, empty_multi_axis_reduction_infers_nonempty_rstd)
{
    ge::op::RmsNormQuantV3 op;
    op.UpdateInputDesc("x", create_desc({3, 0, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({0, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({1}, ge::DT_FLOAT));
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "div_mode", "dst_type", "output_rstd"}, {}, {}};
    ASSERT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetShapeSize(), 0);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetShapeSize(), 0);
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), (std::vector<int64_t>{3, 1, 1}));
}
