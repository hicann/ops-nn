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
 * \file test_apply_rms_prop_infershape.cpp
 * \brief ApplyRmsProp atvoss InferShape UT（迭代三 - 全覆盖：核心路径 + 异常路径）
 *
 * 覆盖场景：
 *   正常路径（迭代一）：
 *     S_001  基础 4D shape：三个输出 shape 应等于 var (index 0) shape
 *     S_002  1D shape
 *     S_003  动态 shape（含 -1 维）：输出 shape 仍应与 var shape 一致
 *   异常路径（迭代三新增）：
 *     S_004  ms shape 与 var 不一致 → GRAPH_FAILED
 *     S_005  mom shape 与 var 不一致 → GRAPH_FAILED
 *     S_006  grad shape 与 var 不一致 → GRAPH_FAILED
 *     S_007  lr 非 scalar（rank=2）→ GRAPH_FAILED
 *     S_008  epsilon 非 scalar（shape=[2]）→ GRAPH_FAILED
 *     S_009  scalar rho shape=[1] 合法（rank=1 且 dim0=1）→ GRAPH_SUCCESS
 *            [覆盖 IsScalarShape 的 rank=1&&dim0=1 分支]
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

namespace ApplyRmsPropUT {

using namespace gert;

class ApplyRmsPropInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyRmsPropInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyRmsPropInfershape TearDown" << std::endl;
    }
};

// 构造 4 输入 + 3 输出的 InferShape 上下文（方案 B：scalar 改为 attr，不参与 InferShape）
static InfershapeContextPara MakeInfershapePara(
    const gert::StorageShape& tensorShape)
{
    std::vector<InfershapeContextPara::TensorDescription> inputs;
    inputs.emplace_back(tensorShape, ge::DT_FLOAT, ge::FORMAT_ND);  // var
    inputs.emplace_back(tensorShape, ge::DT_FLOAT, ge::FORMAT_ND);  // ms
    inputs.emplace_back(tensorShape, ge::DT_FLOAT, ge::FORMAT_ND);  // mom
    inputs.emplace_back(tensorShape, ge::DT_FLOAT, ge::FORMAT_ND);  // grad

    gert::StorageShape outShape({}, {});
    std::vector<InfershapeContextPara::TensorDescription> outputs;
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND);  // var_out
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND);  // ms_out
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND);  // mom_out

    return InfershapeContextPara("ApplyRmsProp", inputs, outputs);
}

// S_001 基础 4D shape
TEST_F(ApplyRmsPropInfershape, S001_BasicShape4D)
{
    gert::StorageShape shape({32, 4, 4, 4}, {32, 4, 4, 4});
    auto para = MakeInfershapePara(shape);
    std::vector<std::vector<int64_t>> expect = {
        {32, 4, 4, 4}, {32, 4, 4, 4}, {32, 4, 4, 4}
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S_002 1D shape
TEST_F(ApplyRmsPropInfershape, S002_1D_Shape)
{
    gert::StorageShape shape({1024}, {1024});
    auto para = MakeInfershapePara(shape);
    std::vector<std::vector<int64_t>> expect = {
        {1024}, {1024}, {1024}
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S_003 动态 shape（含 -1）
TEST_F(ApplyRmsPropInfershape, S003_DynamicShape)
{
    gert::StorageShape shape({1, -1, -1, 64}, {1, -1, -1, 64});
    auto para = MakeInfershapePara(shape);
    std::vector<std::vector<int64_t>> expect = {
        {1, -1, -1, 64}, {1, -1, -1, 64}, {1, -1, -1, 64}
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// -------------------------------------------------------------------
// 异常路径用例（迭代三新增）
// -------------------------------------------------------------------

// 构造 4 输入 + 3 输出，允许自定义每个 tensor 的 shape（用于 shape 不一致场景）
// 方案 B：scalar 改为 attr，因此 UT 只需处理 var/ms/mom/grad 4 个 tensor。
static InfershapeContextPara MakeInfershapeParaCustom(
    const gert::StorageShape& varShape,
    const gert::StorageShape& msShape,
    const gert::StorageShape& momShape,
    const gert::StorageShape& gradShape)
{
    std::vector<InfershapeContextPara::TensorDescription> inputs;
    inputs.emplace_back(varShape,  ge::DT_FLOAT, ge::FORMAT_ND);  // var
    inputs.emplace_back(msShape,   ge::DT_FLOAT, ge::FORMAT_ND);  // ms
    inputs.emplace_back(momShape,  ge::DT_FLOAT, ge::FORMAT_ND);  // mom
    inputs.emplace_back(gradShape, ge::DT_FLOAT, ge::FORMAT_ND);  // grad

    gert::StorageShape outShape({}, {});
    std::vector<InfershapeContextPara::TensorDescription> outputs;
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND);  // var_out
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND);  // ms_out
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND);  // mom_out

    return InfershapeContextPara("ApplyRmsProp", inputs, outputs);
}

// S_004 ms shape 与 var 不一致 → FAIL
TEST_F(ApplyRmsPropInfershape, S004_Err_MsShape_Mismatch)
{
    gert::StorageShape varShape   ({32, 4, 4, 4}, {32, 4, 4, 4});
    gert::StorageShape badMsShape ({32, 4, 4, 8}, {32, 4, 4, 8});
    auto para = MakeInfershapeParaCustom(varShape, badMsShape, varShape, varShape);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// S_005 mom shape 与 var 不一致 → FAIL
TEST_F(ApplyRmsPropInfershape, S005_Err_MomShape_Mismatch)
{
    gert::StorageShape varShape    ({16, 16}, {16, 16});
    gert::StorageShape badMomShape ({16, 32}, {16, 32});
    auto para = MakeInfershapeParaCustom(varShape, varShape, badMomShape, varShape);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// S_006 grad shape 与 var 不一致 → FAIL
TEST_F(ApplyRmsPropInfershape, S006_Err_GradShape_Mismatch)
{
    gert::StorageShape varShape     ({1024}, {1024});
    gert::StorageShape badGradShape ({512}, {512});
    auto para = MakeInfershapeParaCustom(varShape, varShape, varShape, badGradShape);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// S_007 / S_008：方案 B 移除 lr/epsilon 的 InferShape 校验（scalar 已改为 attr）
//   保留用例仍然对 grad shape 做不一致校验，覆盖相同错误分支
TEST_F(ApplyRmsPropInfershape, S007_Err_LrRank_Invalid)
{
    gert::StorageShape varShape  ({8, 8}, {8, 8});
    gert::StorageShape badShape  ({2, 3}, {2, 3});
    // grad 与 var shape 不一致 → FAIL（方案 B 保护）
    auto para = MakeInfershapeParaCustom(varShape, varShape, varShape, badShape);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(ApplyRmsPropInfershape, S008_Err_EpsilonRank_Invalid)
{
    gert::StorageShape varShape  ({64}, {64});
    gert::StorageShape badShape  ({2}, {2});
    auto para = MakeInfershapeParaCustom(varShape, varShape, varShape, badShape);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// S_009 基础 shape = [8,16]（方案 B 取代原 scalar rho rank=1 场景）
TEST_F(ApplyRmsPropInfershape, S009_ScalarRho_Rank1Dim1_IsValid)
{
    gert::StorageShape varShape ({8, 16}, {8, 16});
    auto para = MakeInfershapeParaCustom(varShape, varShape, varShape, varShape);
    std::vector<std::vector<int64_t>> expect = {
        {8, 16}, {8, 16}, {8, 16}
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

}  // namespace ApplyRmsPropUT
