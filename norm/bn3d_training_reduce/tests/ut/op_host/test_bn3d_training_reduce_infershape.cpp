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
 * \file test_bn3d_training_reduce_infershape.cpp
 * \brief arch35 核心路径 UT —— InferShape
 *
 * 公开契约：Ascend 950 的 custom 候选 CheckSupport 仅放行 NCDHW / NDHWC origin；
 * NDC1HWC0 origin 对该候选返回失败。平台仍可能改选 built-in 同名候选，本组 UT
 * 不证明平台级拒绝。内部 InferShape 能力继续保留，通道轴与归约轴完全由
 * **origin format** 决定，不按 shape 数值猜测：
 *   - NCDHW    : rank 2~5，C = dim1        → 输出 [C]
 *   - NDHWC    : 仅 rank 5，C = dim4       → 输出 [C]
 *   - NDC1HWC0 : 仅 rank 6                 → 输出 [1,1,C1,1,1,C0]
 *   - 其余（含 FORMAT_ND、origin FORMAT_NCHW）一律 GRAPH_FAILED
 *
 * InferDataType（输出恒 DT_FLOAT）已按交付件划分挪到
 * op_graph/bn3d_training_reduce_graph_infer.cpp。op_graph UT 模块只链 graph_plugin_obj，
 * 不含 tests/ut/common 的 infershape 公共对象，暂无法调 InferDataTypeTest；与仓内其他把
 * InferDataType 放 op_graph 的算子（in_training_reduce_v2 / bn_infer_grad / in_infer_v2）
 * 保持一致，此处不再覆盖。其行为由 GEIR 图模式用例覆盖。
 *
 * 本组用例的价值：真机 GE 图模式下本算子以自定义 vendor 装在已含内置 proto 的 CANN 之上，
 * GE 走 CallInferFuncV1 命中内置 libopsproto.so 的实现，本目录的 runtime InferShape 被遮蔽，
 * 无法在真机通路中验证；UT 直接链接本仓源码，是该实现唯一的验证手段。
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "../../../op_graph/bn3d_training_reduce_proto.h"
#include "../../../op_host/bn3d_training_reduce_check_support.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class BN3DTrainingReduceInferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BN3DTrainingReduceInferTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BN3DTrainingReduceInferTest TearDown" << std::endl; }
};

namespace {
ge::graphStatus RunCheckSupport(ge::Format originFormat, const std::vector<int64_t>& inputShape,
                                ge::AscendString& result)
{
    using namespace ge;
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    auto testOp = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(testOp, x, inputShape, DT_FLOAT, originFormat, shapeRange);
    return ops::CheckSupport4BN3DTrainingReduce(testOp, result);
}
} // namespace

// ---------------------------------------------------------------------------
// custom 候选 origin-format 策略：NCDHW / NDHWC 放行，NDC1HWC0 返回不支持。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, check_support_ncdhw_public_origin)
{
    ge::AscendString result;
    EXPECT_EQ(RunCheckSupport(ge::FORMAT_NCDHW, {2, 3, 4, 4, 8}, result), ge::GRAPH_SUCCESS);
    EXPECT_NE(std::string(result.GetString()).find(R"("isSupported": "True")"), std::string::npos);
}

TEST_F(BN3DTrainingReduceInferTest, check_support_ndhwc_public_origin)
{
    ge::AscendString result;
    EXPECT_EQ(RunCheckSupport(ge::FORMAT_NDHWC, {2, 4, 4, 8, 3}, result), ge::GRAPH_SUCCESS);
    EXPECT_NE(std::string(result.GetString()).find(R"("isSupported": "True")"), std::string::npos);
}

TEST_F(BN3DTrainingReduceInferTest, check_support_ndc1hwc0_public_origin_disabled)
{
    ge::AscendString result;
    EXPECT_EQ(RunCheckSupport(ge::FORMAT_NDC1HWC0, {2, 3, 2, 4, 4, 16}, result), ge::GRAPH_FAILED);
    EXPECT_NE(std::string(result.GetString()).find(R"("isSupported": "False")"), std::string::npos);
}

// ---------------------------------------------------------------------------
// origin NCDHW rank 5：[2,3,4,4,8] → sum/square_sum [3]（只保留 C 轴）
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_ncdhw_rank5_001)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({2, 3, 4, 4, 8});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;
    std::vector<int64_t> expected_output_shape = vector<int64_t>({3});

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT, FORMAT_NCDHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc(0).GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(test_op.GetOutputDesc(1).GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// origin NCDHW rank 4：NCDHW 支持 rank 2~5，rank 4 同样取 dim1 作为 C
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_ncdhw_rank4_002)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({2, 7, 8, 8});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;
    std::vector<int64_t> expected_output_shape = vector<int64_t>({7});

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT16, FORMAT_NCDHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc(0).GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(test_op.GetOutputDesc(1).GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// origin NCDHW rank 2：下边界，R0 为空乘积 1，输出仍是 [C]
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_ncdhw_rank2_003)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({4, 5});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;
    std::vector<int64_t> expected_output_shape = vector<int64_t>({5});

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_BF16, FORMAT_NCDHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc(0).GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// origin NCDHW rank 1：越下边界 → GRAPH_FAILED
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_ncdhw_rank1_failed_004)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({8});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT, FORMAT_NCDHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------
// origin NCDHW rank 6：越上边界 → GRAPH_FAILED
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_ncdhw_rank6_failed_005)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({2, 3, 4, 5, 6, 7});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT, FORMAT_NCDHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------
// origin NDHWC rank 5：C 在最后一维（dim4），输出 [C]。
// 与 NCDHW 同 shape 但取不同轴，可证明分派确实依赖 format 而非 shape 数值。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_ndhwc_rank5_006)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({2, 3, 4, 4, 8});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;
    std::vector<int64_t> expected_output_shape = vector<int64_t>({8});

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT, FORMAT_NDHWC, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc(0).GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(test_op.GetOutputDesc(1).GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// origin NDHWC rank 4：NDHWC 仅支持 rank 5 → GRAPH_FAILED
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_ndhwc_rank4_failed_007)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({2, 3, 4, 8});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT, FORMAT_NDHWC, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------
// 内部保留能力：origin NDC1HWC0 rank 6 的 InferShape 仍可推导私有输出形态。
// 上面的 CheckSupport 用例只确认 custom 候选拒绝，不覆盖平台多候选回退。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_ndc1hwc0_rank6_008)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({2, 4, 3, 8, 8, 16});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;
    std::vector<int64_t> expected_output_shape = vector<int64_t>({1, 1, 3, 1, 1, 16});

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT16, FORMAT_NDC1HWC0, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc(0).GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(test_op.GetOutputDesc(1).GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// 内部保留能力：origin NDC1HWC0 仍只接受 rank 6。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_ndc1hwc0_rank5_failed_009)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({2, 4, 3, 8, 16});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT16, FORMAT_NDC1HWC0, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------
// origin FORMAT_ND → GRAPH_FAILED。
// ND 不携带布局语义，无法确定 C 轴，不得按 shape 数值猜测放行。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_nd_failed_010)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({2, 3, 4, 4, 8});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT, FORMAT_ND, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------
// origin FORMAT_NCHW → GRAPH_FAILED。
// NCHW 只可能是 origin NCDHW rank 4 的 storage 形态；作为 origin 不在支持面内，
// canndev 的 legacy(reduce_ops.cc) 与 runtime(bn_3d_training.cc) 两版实现同样拒绝。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_origin_nchw_failed_011)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({2, 3, 8, 8});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT, FORMAT_NCHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------
// unknown rank + NCDHW：输出 [-1]（rank 1、dim 未知）
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_unknown_rank_ncdhw_012)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({-2});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;
    std::vector<int64_t> expected_output_shape = vector<int64_t>({-1});

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT, FORMAT_NCDHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc(0).GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(test_op.GetOutputDesc(1).GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// unknown rank + NDHWC：同样输出 [-1]
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_unknown_rank_ndhwc_013)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({-2});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;
    std::vector<int64_t> expected_output_shape = vector<int64_t>({-1});

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT, FORMAT_NDHWC, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc(0).GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// 内部保留能力：unknown rank + NDC1HWC0 输出 [1,1,-1,1,1,-1]。
//
// 这是本实现相对 canndev 唯一一处有意偏离：canndev 的 NDC1HWC0 分支漏写
// !is_unknown_rank 短路（NDHWC / NCDHW 两个分支都有），unknown rank 时 dimNum=1
// 必然失败，与 op_info 中 dynamicRankSupport.flag=true 自相矛盾。此处修正该缺陷。
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_unknown_rank_ndc1hwc0_014)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({-2});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x;
    std::vector<int64_t> expected_output_shape = vector<int64_t>({1, 1, -1, 1, 1, -1});

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT16, FORMAT_NDC1HWC0, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc(0).GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(test_op.GetOutputDesc(1).GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// 动态 shape（含 -1 但 rank 已知）：C 轴已知时如实透传
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_dynamic_dim_ncdhw_015)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({-1, 16, -1, -1, -1});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{1, 8}, {16, 16}, {1, 32}, {1, 32}, {1, 32}};
    std::vector<int64_t> expected_output_shape = vector<int64_t>({16});

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT, FORMAT_NCDHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc(0).GetShape().GetDims(), expected_output_shape);
}

// ---------------------------------------------------------------------------
// 动态 shape：C 轴本身未知（-1）时输出 [-1]
// ---------------------------------------------------------------------------
TEST_F(BN3DTrainingReduceInferTest, infer_shape_dynamic_c_unknown_016)
{
    using namespace ge;
    auto input_x_shape = vector<int64_t>({4, -1, 8, 8, 8});
    std::vector<std::pair<int64_t, int64_t>> shape_range_x = {{4, 4}, {1, 64}, {8, 8}, {8, 8}, {8, 8}};
    std::vector<int64_t> expected_output_shape = vector<int64_t>({-1});

    auto test_op = op::BN3DTrainingReduce("BN3DTrainingReduce");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, DT_FLOAT, FORMAT_NCDHW, shape_range_x);

    EXPECT_EQ(InferShapeTest(test_op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(test_op.GetOutputDesc(0).GetShape().GetDims(), expected_output_shape);
}
