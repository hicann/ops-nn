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
 * \file test_sparse_apply_adagrad_v2_infershape.cpp
 * \brief SparseApplyAdagradV2 InferShape UT
 *
 * 算子接口（与 op_host/sparse_apply_adagrad_v2_def.cpp / op_graph/sparse_apply_adagrad_v2_proto.h 一致）：
 *   6 输入: var, accum, lr, epsilon, grad, indices
 *   2 输出: var, accum   —— InferShape: out_var=var, out_accum=accum（原地更新）
 *
 * 覆盖场景：
 *   S001  正常 rank2 [4,3]                      -> 两输出均 [4,3]
 *   S002  正常 rank3 [8,4,3]                    -> 两输出均 [8,4,3]
 *   S003  Unknown shape 全 -1  {-1,-1}          -> 两输出均 {-1,-1}
 *   S004  Unknown shape 部分 -1 {4,-1}          -> 两输出均 {4,-1}
 *   S005  Unknown shape lr/epsilon 为 {-1}      -> 两输出均 {-1,-1}（标量动态 shape 放宽）
 *   S006  Unknown rank {-2}                     -> 两输出均 {-2}
 *   S007  负例：accum 与 var 维度值不一致         -> GRAPH_FAILED（校验仍生效）
 *   S008  负例：var 秩 < 2                       -> GRAPH_FAILED（校验仍生效）
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infershape_case_executor.h"

namespace SparseApplyAdagradV2UT {

using namespace gert;

static const std::string OP_NAME = "SparseApplyAdagradV2";

class SparseApplyAdagradV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SparseApplyAdagradV2Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SparseApplyAdagradV2Infershape TearDown" << std::endl; }
};

// 构造 6 输入 + 2 输出的 InferShape 上下文。
// var/accum/grad 取 varShape；lr/epsilon 取 scalarShape；indices 取 indicesShape。
static InfershapeContextPara MakeInfershapePara(const gert::StorageShape& varShape,
                                                const gert::StorageShape& accumShape, const gert::StorageShape& lrShape,
                                                const gert::StorageShape& epsilonShape,
                                                const gert::StorageShape& gradShape,
                                                const gert::StorageShape& indicesShape)
{
    std::vector<InfershapeContextPara::TensorDescription> inputs;
    inputs.emplace_back(varShape, ge::DT_FLOAT, ge::FORMAT_ND);     // 0 var
    inputs.emplace_back(accumShape, ge::DT_FLOAT, ge::FORMAT_ND);   // 1 accum
    inputs.emplace_back(lrShape, ge::DT_FLOAT, ge::FORMAT_ND);      // 2 lr
    inputs.emplace_back(epsilonShape, ge::DT_FLOAT, ge::FORMAT_ND); // 3 epsilon
    inputs.emplace_back(gradShape, ge::DT_FLOAT, ge::FORMAT_ND);    // 4 grad
    inputs.emplace_back(indicesShape, ge::DT_INT32, ge::FORMAT_ND); // 5 indices

    gert::StorageShape outShape({}, {});
    std::vector<InfershapeContextPara::TensorDescription> outputs;
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND); // var
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND); // accum

    return InfershapeContextPara(OP_NAME, inputs, outputs);
}

// S001 正常 rank2
TEST_F(SparseApplyAdagradV2Infershape, S001_Rank2_OutputsEqualInput)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 3}, {2, 3});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, scalar, scalar, grad, indices);
    std::vector<std::vector<int64_t>> expect = {{4, 3}, {4, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S002 正常 rank3
TEST_F(SparseApplyAdagradV2Infershape, S002_Rank3_OutputsEqualInput)
{
    gert::StorageShape var({8, 4, 3}, {8, 4, 3});
    gert::StorageShape accum({8, 4, 3}, {8, 4, 3});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 4, 3}, {2, 4, 3});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, scalar, scalar, grad, indices);
    std::vector<std::vector<int64_t>> expect = {{8, 4, 3}, {8, 4, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S003 Unknown shape（全 -1）：所有 dim 值未知
TEST_F(SparseApplyAdagradV2Infershape, S003_UnknownShape_AllDimUnknown)
{
    gert::StorageShape var({-1, -1}, {-1, -1});
    gert::StorageShape accum({-1, -1}, {-1, -1});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({-1, -1}, {-1, -1});
    gert::StorageShape indices({-1}, {-1});
    auto para = MakeInfershapePara(var, accum, scalar, scalar, grad, indices);
    std::vector<std::vector<int64_t>> expect = {{-1, -1}, {-1, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S004 Unknown shape（部分 -1）：第一维已知，第二维未知
TEST_F(SparseApplyAdagradV2Infershape, S004_UnknownShape_PartialDimUnknown)
{
    gert::StorageShape var({4, -1}, {4, -1});
    gert::StorageShape accum({4, -1}, {4, -1});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({-1, -1}, {-1, -1});
    gert::StorageShape indices({-1}, {-1});
    auto para = MakeInfershapePara(var, accum, scalar, scalar, grad, indices);
    std::vector<std::vector<int64_t>> expect = {{4, -1}, {4, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S005 Unknown shape：lr/epsilon 为 {-1}（1D 未知值），应视为合法标量
TEST_F(SparseApplyAdagradV2Infershape, S005_UnknownShape_ScalarLrEpsilonUnknown)
{
    gert::StorageShape var({-1, -1}, {-1, -1});
    gert::StorageShape accum({-1, -1}, {-1, -1});
    gert::StorageShape scalarUnknown({-1}, {-1});
    gert::StorageShape grad({-1, -1}, {-1, -1});
    gert::StorageShape indices({-1}, {-1});
    auto para = MakeInfershapePara(var, accum, scalarUnknown, scalarUnknown, grad, indices);
    std::vector<std::vector<int64_t>> expect = {{-1, -1}, {-1, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S006 Unknown rank（-2）：秩未知，输出透传为 {-2}
TEST_F(SparseApplyAdagradV2Infershape, S006_UnknownRank_PassThrough)
{
    gert::StorageShape unknownRank({-2}, {-2});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({-2}, {-2});
    gert::StorageShape indices({-2}, {-2});
    auto para = MakeInfershapePara(unknownRank, unknownRank, scalar, scalar, grad, indices);
    std::vector<std::vector<int64_t>> expect = {{-2}, {-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S007 负例：accum 与 var 维度值不一致（非 -1），应报错
TEST_F(SparseApplyAdagradV2Infershape, S007_Negative_AccumShapeMismatch)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 5}, {4, 5});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 3}, {2, 3});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, scalar, scalar, grad, indices);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// S008 负例：var 秩 < 2，应报错
TEST_F(SparseApplyAdagradV2Infershape, S008_Negative_VarRankLessThan2)
{
    gert::StorageShape var({4}, {4});
    gert::StorageShape accum({4}, {4});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2}, {2});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, scalar, scalar, grad, indices);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// S009 var/accum 不一致 + 多个输入含 -2：var={-2}, accum={-1,-1}, lr={-2}, epsilon={1},
//    grad={-2}, indices={1}。任一输入为 -2 即输出 unknown rank，跳过 var/accum 一致性校验
TEST_F(SparseApplyAdagradV2Infershape, S009_AnyInputUnknownRank_VarAccumInconsistent_Case1)
{
    gert::StorageShape var({-2}, {-2});
    gert::StorageShape accum({-1, -1}, {-1, -1});
    gert::StorageShape lr({-2}, {-2});
    gert::StorageShape epsilon({1}, {1});
    gert::StorageShape grad({-2}, {-2});
    gert::StorageShape indices({1}, {1});
    auto para = MakeInfershapePara(var, accum, lr, epsilon, grad, indices);
    std::vector<std::vector<int64_t>> expect = {{-2}, {-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S010 var/accum 不一致 + grad 含 -2：var={-1,-1}, accum={256,256}, lr={1},
//    epsilon={-1}, grad={-2}, indices={-1}。grad 为 -2 即输出 unknown rank
TEST_F(SparseApplyAdagradV2Infershape, S010_AnyInputUnknownRank_VarAccumInconsistent_Case2)
{
    gert::StorageShape var({-1, -1}, {-1, -1});
    gert::StorageShape accum({256, 256}, {256, 256});
    gert::StorageShape lr({1}, {1});
    gert::StorageShape epsilon({-1}, {-1});
    gert::StorageShape grad({-2}, {-2});
    gert::StorageShape indices({-1}, {-1});
    auto para = MakeInfershapePara(var, accum, lr, epsilon, grad, indices);
    std::vector<std::vector<int64_t>> expect = {{-2}, {-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

} // namespace SparseApplyAdagradV2UT
