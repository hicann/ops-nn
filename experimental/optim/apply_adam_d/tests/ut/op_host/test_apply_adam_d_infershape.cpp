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
 * \file test_apply_adam_d_infershape.cpp
 * \brief ApplyAdamD InferShape UT（全覆盖：输出 var/m/v shape = 对应输入 shape）
 *
 * 算子接口（与 op_host/apply_adam_d_def.cpp / op_graph/apply_adam_d_proto.h 一致）：
 *   10 输入: var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
 *   3 输出:  var, m, v   —— InferShape4ApplyAdamD: out_var=var, out_m=m, out_v=v（逐个拷贝）
 *
 * 覆盖场景（DESIGN §2.4 / spec.boundary_conditions: rank1/rank2/高rank/单元素/动态）：
 *   S001  rank2 [2,3]                  -> 三输出均 [2,3]
 *   S002  rank1 [16]                   -> 三输出均 [16]
 *   S003  高 rank4 [2,3,4,5]           -> 三输出均 [2,3,4,5]
 *   S004  rank1 单元素 [1]             -> 三输出均 [1]
 *   S005  高 rank5 [4,5,6,7,8] + fp16  -> 三输出均 [4,5,6,7,8]（dtype 无关，拷贝逻辑）
 *   S006  动态 shape [-1,32]           -> 三输出均 [-1,32]
 *   S007  逐输入独立拷贝验证            -> var/m/v 取不同 shape，三输出各自匹配对应输入
 *         [覆盖 CopyShapeInput2Output 三次调用的独立性：OUT_VAR<-var, OUT_M<-m, OUT_V<-v]
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

namespace ApplyAdamDUT {

using namespace gert;

static const std::string OP_NAME = "ApplyAdamD";

class ApplyAdamDInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyAdamDInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyAdamDInfershape TearDown" << std::endl; }
};

// 构造 10 输入 + 3 输出的 InferShape 上下文。
// var/m/v/grad 取 tensorShape；6 个标量取 [1]。InferShape 仅读 index 0/1/2。
static InfershapeContextPara MakeInfershapePara(const gert::StorageShape& tensorShape,
                                                ge::DataType dtype = ge::DT_FLOAT)
{
    gert::StorageShape scalarShape({1}, {1});
    std::vector<InfershapeContextPara::TensorDescription> inputs;
    inputs.emplace_back(tensorShape, dtype, ge::FORMAT_ND); // 0 var
    inputs.emplace_back(tensorShape, dtype, ge::FORMAT_ND); // 1 m
    inputs.emplace_back(tensorShape, dtype, ge::FORMAT_ND); // 2 v
    inputs.emplace_back(scalarShape, dtype, ge::FORMAT_ND); // 3 beta1_power
    inputs.emplace_back(scalarShape, dtype, ge::FORMAT_ND); // 4 beta2_power
    inputs.emplace_back(scalarShape, dtype, ge::FORMAT_ND); // 5 lr
    inputs.emplace_back(scalarShape, dtype, ge::FORMAT_ND); // 6 beta1
    inputs.emplace_back(scalarShape, dtype, ge::FORMAT_ND); // 7 beta2
    inputs.emplace_back(scalarShape, dtype, ge::FORMAT_ND); // 8 epsilon
    inputs.emplace_back(tensorShape, dtype, ge::FORMAT_ND); // 9 grad

    gert::StorageShape outShape({}, {});
    std::vector<InfershapeContextPara::TensorDescription> outputs;
    outputs.emplace_back(outShape, dtype, ge::FORMAT_ND); // var
    outputs.emplace_back(outShape, dtype, ge::FORMAT_ND); // m
    outputs.emplace_back(outShape, dtype, ge::FORMAT_ND); // v

    return InfershapeContextPara(OP_NAME, inputs, outputs);
}

// 构造允许 var/m/v 各自不同 shape 的上下文（用于"逐输入独立拷贝"验证）。
static InfershapeContextPara MakeInfershapeParaCustom(const gert::StorageShape& varShape,
                                                      const gert::StorageShape& mShape,
                                                      const gert::StorageShape& vShape,
                                                      const gert::StorageShape& gradShape)
{
    gert::StorageShape scalarShape({1}, {1});
    std::vector<InfershapeContextPara::TensorDescription> inputs;
    inputs.emplace_back(varShape, ge::DT_FLOAT, ge::FORMAT_ND);    // 0 var
    inputs.emplace_back(mShape, ge::DT_FLOAT, ge::FORMAT_ND);      // 1 m
    inputs.emplace_back(vShape, ge::DT_FLOAT, ge::FORMAT_ND);      // 2 v
    inputs.emplace_back(scalarShape, ge::DT_FLOAT, ge::FORMAT_ND); // 3 beta1_power
    inputs.emplace_back(scalarShape, ge::DT_FLOAT, ge::FORMAT_ND); // 4 beta2_power
    inputs.emplace_back(scalarShape, ge::DT_FLOAT, ge::FORMAT_ND); // 5 lr
    inputs.emplace_back(scalarShape, ge::DT_FLOAT, ge::FORMAT_ND); // 6 beta1
    inputs.emplace_back(scalarShape, ge::DT_FLOAT, ge::FORMAT_ND); // 7 beta2
    inputs.emplace_back(scalarShape, ge::DT_FLOAT, ge::FORMAT_ND); // 8 epsilon
    inputs.emplace_back(gradShape, ge::DT_FLOAT, ge::FORMAT_ND);   // 9 grad

    gert::StorageShape outShape({}, {});
    std::vector<InfershapeContextPara::TensorDescription> outputs;
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND);
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND);
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND);

    return InfershapeContextPara(OP_NAME, inputs, outputs);
}

// S001 rank2
TEST_F(ApplyAdamDInfershape, S001_Rank2_OutputsEqualInput)
{
    gert::StorageShape shape({2, 3}, {2, 3});
    auto para = MakeInfershapePara(shape);
    std::vector<std::vector<int64_t>> expect = {{2, 3}, {2, 3}, {2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S002 rank1
TEST_F(ApplyAdamDInfershape, S002_Rank1_OutputsEqualInput)
{
    gert::StorageShape shape({16}, {16});
    auto para = MakeInfershapePara(shape);
    std::vector<std::vector<int64_t>> expect = {{16}, {16}, {16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S003 高 rank4
TEST_F(ApplyAdamDInfershape, S003_Rank4_OutputsEqualInput)
{
    gert::StorageShape shape({2, 3, 4, 5}, {2, 3, 4, 5});
    auto para = MakeInfershapePara(shape);
    std::vector<std::vector<int64_t>> expect = {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S004 rank1 单元素
TEST_F(ApplyAdamDInfershape, S004_Rank1_SingleElement)
{
    gert::StorageShape shape({1}, {1});
    auto para = MakeInfershapePara(shape);
    std::vector<std::vector<int64_t>> expect = {{1}, {1}, {1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S005 高 rank5 + fp16（infershape 与 dtype 无关，验证拷贝逻辑）
TEST_F(ApplyAdamDInfershape, S005_Rank5_Fp16_OutputsEqualInput)
{
    gert::StorageShape shape({4, 5, 6, 7, 8}, {4, 5, 6, 7, 8});
    auto para = MakeInfershapePara(shape, ge::DT_FLOAT16);
    std::vector<std::vector<int64_t>> expect = {{4, 5, 6, 7, 8}, {4, 5, 6, 7, 8}, {4, 5, 6, 7, 8}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S006 动态 shape（含 -1）
TEST_F(ApplyAdamDInfershape, S006_DynamicShape)
{
    gert::StorageShape shape({-1, 32}, {-1, 32});
    auto para = MakeInfershapePara(shape);
    std::vector<std::vector<int64_t>> expect = {{-1, 32}, {-1, 32}, {-1, 32}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// S007 逐输入独立拷贝：var/m/v 各取不同 shape，三输出各自匹配对应输入
//   覆盖 CopyShapeInput2Output 三次独立调用（OUT_VAR<-var, OUT_M<-m, OUT_V<-v）
TEST_F(ApplyAdamDInfershape, S007_PerInput_IndependentCopy)
{
    gert::StorageShape varShape({2, 3}, {2, 3});
    gert::StorageShape mShape({4}, {4});
    gert::StorageShape vShape({5, 6}, {5, 6});
    auto para = MakeInfershapeParaCustom(varShape, mShape, vShape, varShape);
    std::vector<std::vector<int64_t>> expect = {{2, 3}, {4}, {5, 6}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

} // namespace ApplyAdamDUT
