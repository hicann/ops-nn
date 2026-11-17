/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <initializer_list>
#include <vector>
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"
#include "../../../op_graph/dynamic_mx_quant_proto.h"

namespace {
class DynamicMxQuantNVFP4 : public testing::Test {};

void CheckShape(std::initializer_list<int64_t> shape, const std::vector<int64_t>& expectedScale,
                ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS, int64_t axis = -1, int64_t blockSize = 16)
{
    ge::op::DynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc(shape, ge::DT_FLOAT16));
    op.SetAttr("axis", axis);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("blocksize", blockSize);
    op.SetAttr("scale_alg", 3);
    op.SetAttr("dst_type_max", 0.0f);
    op.SetAttr("max_low_bound", 0.0f);
    Runtime2TestParam param{
        {"axis", "round_mode", "dst_type", "blocksize", "scale_alg", "dst_type_max", "max_low_bound"}, {}, {}};
    ASSERT_EQ(InferShapeTest(op, param), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>(shape));
        EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expectedScale);
    }
}

TEST_F(DynamicMxQuantNVFP4, shape_block16_and_scale_pair_padding)
{
    CheckShape({2, 16}, {2, 1, 2});
    CheckShape({2, 18}, {2, 1, 2});
    CheckShape({2, 48}, {2, 2, 2});
    CheckShape({2, 66}, {2, 3, 2});
    CheckShape({1, 2, 3, 4, 5, 6, 64}, {1, 2, 3, 4, 5, 6, 2, 2}, ge::GRAPH_SUCCESS, 6);
}

TEST_F(DynamicMxQuantNVFP4, shape_unknown_dimension_and_rank)
{
    CheckShape({2, -1}, {2, -1, 2});
    CheckShape({-2}, {-2});
}

TEST_F(DynamicMxQuantNVFP4, shape_reject_non_tail_axis) { CheckShape({32, 64}, {}, ge::GRAPH_FAILED, 0); }

TEST_F(DynamicMxQuantNVFP4, shape_reject_odd_tail) { CheckShape({2, 17}, {}, ge::GRAPH_FAILED); }

TEST_F(DynamicMxQuantNVFP4, shape_reject_wrong_blocksize) { CheckShape({2, 64}, {}, ge::GRAPH_FAILED, -1, 32); }

void CheckDtype(ge::DataType inputType, ge::DataType outputType, int64_t scaleAlg,
                ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicMxQuant");
    ASSERT_NE(impl, nullptr);
    auto infer = impl->infer_datatype;
    ASSERT_NE(infer, nullptr);
    const auto expectedOutputType = outputType;
    ge::DataType outputScale = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(1)
                      .NodeIoNum(1, 2)
                      .NodeInputTd(0, inputType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outputType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                  {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(outputType)},
                                  {"blocksize", Ops::NN::AnyValue::CreateFrom<int64_t>(scaleAlg == 3 ? 16 : 32)},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(scaleAlg)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0)},
                                  {"max_low_bound", Ops::NN::AnyValue::CreateFrom<float>(0)}})
                      .InputDataTypes({&inputType})
                      .OutputDataTypes({&outputType, &outputScale})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(infer(context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(context->GetOutputDataType(0), expectedOutputType);
        EXPECT_EQ(context->GetOutputDataType(1), scaleAlg == 3 ? ge::DT_FLOAT8_E4M3FN : ge::DT_FLOAT8_E8M0);
    }
}

TEST_F(DynamicMxQuantNVFP4, dtype_fp16_and_bf16_to_e2m1_e4m3)
{
    CheckDtype(ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, 3);
    CheckDtype(ge::DT_BF16, ge::DT_FLOAT4_E2M1, 3);
}

TEST_F(DynamicMxQuantNVFP4, dtype_legacy_algorithms_keep_e8m0_scale)
{
    CheckDtype(ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, 0);
    CheckDtype(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, 1);
    CheckDtype(ge::DT_FLOAT, ge::DT_FLOAT4_E2M1, 2);
}

TEST_F(DynamicMxQuantNVFP4, dtype_reject_fp32_and_other_destinations)
{
    CheckDtype(ge::DT_FLOAT, ge::DT_FLOAT4_E2M1, 3, ge::GRAPH_FAILED);
    CheckDtype(ge::DT_FLOAT16, ge::DT_FLOAT4_E1M2, 3, ge::GRAPH_FAILED);
    CheckDtype(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, 3, ge::GRAPH_FAILED);
}
} // namespace
