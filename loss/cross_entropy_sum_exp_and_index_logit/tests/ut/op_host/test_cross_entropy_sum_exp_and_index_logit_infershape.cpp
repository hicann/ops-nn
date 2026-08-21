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
 * \file test_cross_entropy_sum_exp_and_index_logit_infershape.cpp
 * \brief CrossEntropySumExpAndIndexLogit infershape UT — 图模式 InferShapeTest 驱动，
 *        覆盖 2D/3D 正常 shape 推导与输出 dtype 推导。
 *        （infershape 中 nullptr 输入/输出 shape 的异常分支在 UT 框架下不可达，
 *        参考 softmax_cross_entropy_with_logits / mx_to_block_mx_quant 等仓库内惯例不做覆盖）
 */

#include <iostream>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "../../../op_graph/cross_entropy_sum_exp_and_index_logit_proto.h"

using namespace std;
using namespace ge;

namespace {
constexpr size_t CE_OUTPUT_NUM = 5;
constexpr size_t CE_NODE_INPUT_NUM = 3;

class CrossEntropySumExpAndIndexLogitInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CrossEntropySumExpAndIndexLogit InferShape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CrossEntropySumExpAndIndexLogit InferShape TearDown" << std::endl; }
};

// 2D logits [N, V_local]：predicted/sum_exp/offset/mask 与 target 同 shape，exp_logits 与 logits 同 shape
TEST_F(CrossEntropySumExpAndIndexLogitInferShape, ce_infershape_2d_fp32)
{
    ge::op::CrossEntropySumExpAndIndexLogit op;
    op.UpdateInputDesc("vocab_parallel_logits", create_desc({4, 32}, ge::DT_FLOAT));
    op.UpdateInputDesc("target", create_desc({4}, ge::DT_INT32));
    op.UpdateInputDesc("global_logits_max", create_desc({4}, ge::DT_FLOAT));
    op.SetAttr("vocab_start_index", 0);
    op.SetAttr("vocab_end_index", 32);
    Runtime2TestParam param{{"vocab_start_index", "vocab_end_index"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedTargetShape = {4};
    std::vector<int64_t> expectedLogitsShape = {4, 32};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expectedTargetShape); // predicted_logits
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expectedTargetShape); // sum_exp_logits
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), expectedLogitsShape); // exp_logits
    EXPECT_EQ(op.GetOutputDesc(3).GetShape().GetDims(), expectedTargetShape); // target_offset
    EXPECT_EQ(op.GetOutputDesc(4).GetShape().GetDims(), expectedTargetShape); // target_mask
}

// 3D logits [S, B, V_local]：同上，target 为 [S, B]
TEST_F(CrossEntropySumExpAndIndexLogitInferShape, ce_infershape_3d_bf16)
{
    ge::op::CrossEntropySumExpAndIndexLogit op;
    op.UpdateInputDesc("vocab_parallel_logits", create_desc({2, 3, 16}, ge::DT_BF16));
    op.UpdateInputDesc("target", create_desc({2, 3}, ge::DT_INT32));
    op.UpdateInputDesc("global_logits_max", create_desc({2, 3}, ge::DT_BF16));
    op.SetAttr("vocab_start_index", 0);
    op.SetAttr("vocab_end_index", 16);
    Runtime2TestParam param{{"vocab_start_index", "vocab_end_index"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedTargetShape = {2, 3};
    std::vector<int64_t> expectedLogitsShape = {2, 3, 16};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expectedTargetShape); // predicted_logits
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expectedTargetShape); // sum_exp_logits
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), expectedLogitsShape); // exp_logits
    EXPECT_EQ(op.GetOutputDesc(3).GetShape().GetDims(), expectedTargetShape); // target_offset
    EXPECT_EQ(op.GetOutputDesc(4).GetShape().GetDims(), expectedTargetShape); // target_mask
}

// InferDataType：predicted/sum_exp/exp 固定 FLOAT，offset/mask 固定 INT32（与输入 dtype 无关）
TEST_F(CrossEntropySumExpAndIndexLogitInferShape, ce_inferdatatype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("CrossEntropySumExpAndIndexLogit"), nullptr);
    auto inferDatatypeFunc = gert::OpImplRegistry::GetInstance()
                                 .GetOpImpl("CrossEntropySumExpAndIndexLogit")
                                 ->infer_datatype;
    ASSERT_NE(inferDatatypeFunc, nullptr);

    std::vector<ge::DataType> outputDtypes(CE_OUTPUT_NUM);
    std::vector<void*> outputDtypeRefs;
    for (size_t i = 0; i < outputDtypes.size(); ++i) {
        outputDtypeRefs.push_back(&outputDtypes[i]);
    }

    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType("CrossEntropySumExpAndIndexLogit")
                      .NodeIoNum(CE_NODE_INPUT_NUM, CE_OUTPUT_NUM)
                      .IrInstanceNum({1, 1, 1})
                      .OutputDataTypes(outputDtypeRefs)
                      .Build();

    gert::InferDataTypeContext* context = holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDatatypeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT); // predicted_logits
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT); // sum_exp_logits
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT); // exp_logits
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_INT32); // target_offset
    EXPECT_EQ(context->GetOutputDataType(4), ge::DT_INT32); // target_mask
}
} // namespace
