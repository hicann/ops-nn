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
 * \file test_apply_ftrl_infershape.cpp
 * \brief ApplyFtrl InferShape UT（迭代一 - 核心路径 + 异常路径）
 *
 * 验证 op_host/apply_ftrl_infershape.cpp（DESIGN s8 / spec outputs）：
 *   - var_out.shape = var.shape（input0）。
 *   - MED-3 shape 一致性：accum/linear/grad 必须与 var 完全一致（DimNum + 逐维），
 *     否则 GRAPH_FAILED（对应 spec boundary「shape 不一致 → shape_mismatch」）。
 *   - scalar 输入（lr/l1/l2/lr_power，0-D 或 1 元素 1-D）不参与 InferShape（spec boundary）。
 *
 * 真值源：docs/spec.yaml（outputs / shape_constraints / boundary）+ docs/DESIGN.md s8。
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

namespace ApplyFtrlUT {

using namespace gert;

static const std::string OP_NAME = "ApplyFtrl";

class ApplyFtrlInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyFtrlInfershapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ApplyFtrlInfershapeTest TearDown" << std::endl; }
};

static gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape ss;
    ss.MutableOriginShape().SetDimNum(0);
    ss.MutableStorageShape().SetDimNum(0);
    for (auto d : dims) {
        ss.MutableOriginShape().AppendDim(d);
        ss.MutableStorageShape().AppendDim(d);
    }
    return ss;
}

// 8 输入(var/accum/linear/grad + lr/l1/l2/lr_power) + 1 输出(var)。
// 允许自定义每个张量与 scalar 的 shape（用于 shape 不一致 / 0-D / 1-D scalar 场景）。
static InfershapeContextPara MakePara(
    const std::vector<int64_t>& varShape,
    const std::vector<int64_t>& accumShape,
    const std::vector<int64_t>& linearShape,
    const std::vector<int64_t>& gradShape,
    const std::vector<int64_t>& scalarShape = {1},
    ge::DataType dtype = ge::DT_FLOAT16)
{
    std::vector<InfershapeContextPara::TensorDescription> inputs;
    inputs.emplace_back(MakeShape(varShape),    dtype, ge::FORMAT_ND);  // var
    inputs.emplace_back(MakeShape(accumShape),  dtype, ge::FORMAT_ND);  // accum
    inputs.emplace_back(MakeShape(linearShape), dtype, ge::FORMAT_ND);  // linear
    inputs.emplace_back(MakeShape(gradShape),   dtype, ge::FORMAT_ND);  // grad
    inputs.emplace_back(MakeShape(scalarShape), dtype, ge::FORMAT_ND);  // lr
    inputs.emplace_back(MakeShape(scalarShape), dtype, ge::FORMAT_ND);  // l1
    inputs.emplace_back(MakeShape(scalarShape), dtype, ge::FORMAT_ND);  // l2
    inputs.emplace_back(MakeShape(scalarShape), dtype, ge::FORMAT_ND);  // lr_power

    std::vector<InfershapeContextPara::TensorDescription> outputs;
    outputs.emplace_back(MakeShape({}), dtype, ge::FORMAT_ND);          // var_out

    return InfershapeContextPara(OP_NAME, inputs, outputs);
}

static InfershapeContextPara MakeUniformPara(
    const std::vector<int64_t>& shape, const std::vector<int64_t>& scalarShape = {1},
    ge::DataType dtype = ge::DT_FLOAT16)
{
    return MakePara(shape, shape, shape, shape, scalarShape, dtype);
}

// -------------------------------------------------------------------
// S001: 基础 2D shape → var_out.shape = var.shape
// -------------------------------------------------------------------
TEST_F(ApplyFtrlInfershapeTest, S001_BasicShape2D)
{
    auto para = MakeUniformPara({32, 16});
    std::vector<std::vector<int64_t>> expect = {{32, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// -------------------------------------------------------------------
// S002: 1D shape
// -------------------------------------------------------------------
TEST_F(ApplyFtrlInfershapeTest, S002_1D_Shape)
{
    auto para = MakeUniformPara({1024});
    std::vector<std::vector<int64_t>> expect = {{1024}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// -------------------------------------------------------------------
// S003: 动态 shape（含 -1）→ 输出 shape 仍等于 var
// -------------------------------------------------------------------
TEST_F(ApplyFtrlInfershapeTest, S003_DynamicShape)
{
    auto para = MakeUniformPara({1, -1, -1, 64});
    std::vector<std::vector<int64_t>> expect = {{1, -1, -1, 64}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// -------------------------------------------------------------------
// S004: scalar 输入为 0-D（rank=0）→ 不影响 InferShape，输出=var
//   （spec boundary「标量输入为 0-D」）
// -------------------------------------------------------------------
TEST_F(ApplyFtrlInfershapeTest, S004_Scalar0D_IsValid)
{
    auto para = MakeUniformPara({2, 3}, /*scalarShape=*/{});
    std::vector<std::vector<int64_t>> expect = {{2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// -------------------------------------------------------------------
// S005: scalar 输入为 1 元素 1-D（[1]）→ 输出=var
//   （spec boundary「标量输入为 1 元素 1-D」）
// -------------------------------------------------------------------
TEST_F(ApplyFtrlInfershapeTest, S005_Scalar1D_IsValid)
{
    auto para = MakeUniformPara({2, 3}, /*scalarShape=*/{1});
    std::vector<std::vector<int64_t>> expect = {{2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// -------------------------------------------------------------------
// 异常路径：四张量 shape 不一致 → GRAPH_FAILED（MED-3 / spec shape_mismatch）
// -------------------------------------------------------------------
TEST_F(ApplyFtrlInfershapeTest, S006_Err_AccumShapeMismatch)
{
    auto para = MakePara({32, 4, 4, 4}, {32, 4, 4, 8}, {32, 4, 4, 4}, {32, 4, 4, 4});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(ApplyFtrlInfershapeTest, S007_Err_LinearShapeMismatch)
{
    auto para = MakePara({16, 16}, {16, 16}, {16, 32}, {16, 16});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(ApplyFtrlInfershapeTest, S008_Err_GradShapeMismatch)
{
    auto para = MakePara({1024}, {1024}, {1024}, {512});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===================================================================
// 迭代三：InferShape 全覆盖补齐
//   rank 差异 mismatch / rank=8 合法 / rank>8 不强校验 / 空 Tensor / 非标量端口 / rank-0。
// 真值源：op_host/apply_ftrl_infershape.cpp + spec outputs/boundary + DESIGN s8。
// ===================================================================

// S009: shape 不一致 — rank（DimNum）差异（元素数相同但 DimNum 不同）→ GRAPH_FAILED。
TEST_F(ApplyFtrlInfershapeTest, S009_Err_ShapeMismatch_RankDiff)
{
    auto para = MakePara({2, 3}, {6}, {2, 3}, {2, 3});  // accum rank1 vs var rank2
    ExecuteTestCase(para, ge::GRAPH_FAILED);

    auto para2 = MakePara({4, 4}, {4, 4}, {4, 4}, {16});  // grad rank1
    ExecuteTestCase(para2, ge::GRAPH_FAILED);
}

// S010: rank=8（spec rank_range 上界）合法 → 输出 = var.shape。
TEST_F(ApplyFtrlInfershapeTest, S010_Rank8_MaxValid)
{
    auto para = MakeUniformPara({2, 3, 1, 1, 1, 1, 1, 4});
    std::vector<std::vector<int64_t>> expect = {{2, 3, 1, 1, 1, 1, 1, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S011: rank>8（rank-9）— infershape 不做 rank<=8 强校验（DESIGN s8：rank 上界由 aclnn/GE 兜底）；
//   四张量 shape 一致 → 输出 = var.shape（rank-9）。文档化 op_host 行为，非缺陷。
TEST_F(ApplyFtrlInfershapeTest, S011_RankOver8_NotEnforcedByOpHost)
{
    auto para = MakeUniformPara({2, 3, 1, 1, 1, 1, 1, 1, 1});  // rank-9
    std::vector<std::vector<int64_t>> expect = {{2, 3, 1, 1, 1, 1, 1, 1, 1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S012: 空 Tensor（{0,3}）→ 输出 = var.shape（infershape 直通，spec boundary「空 Tensor 直通」）。
TEST_F(ApplyFtrlInfershapeTest, S012_EmptyTensor)
{
    auto para = MakeUniformPara({0, 3});
    std::vector<std::vector<int64_t>> expect = {{0, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S013: 标量端口非标量形态 — infershape 只读 input0..3，不读 lr/l1/l2/lr_power → 仍成功，输出=var。
//   （scalar 形态校验由 aclnn CheckParams 兜底，DESIGN s8）。
TEST_F(ApplyFtrlInfershapeTest, S013_NonScalarScalarInputs_NotEnforced)
{
    auto para = MakePara({16, 16}, {16, 16}, {16, 16}, {16, 16}, /*scalarShape=*/{2, 3});
    std::vector<std::vector<int64_t>> expect = {{16, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S014: rank-0（标量张量，spec rank_range 下界）→ 四张量 rank0 一致 → 输出 rank0。
TEST_F(ApplyFtrlInfershapeTest, S014_Rank0_Tensors)
{
    auto para = MakePara({}, {}, {}, {});
    std::vector<std::vector<int64_t>> expect = {{}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

}  // namespace ApplyFtrlUT
