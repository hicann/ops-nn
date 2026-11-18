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
 * \file test_sparse_apply_adagrad_infershape.cpp
 * \brief SparseApplyAdagrad InferShape UT
 */

#include <gtest/gtest.h>
#include <iostream>
#include "kernel_run_context_facker.h"
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infershape_case_executor.h"

namespace SparseApplyAdagradUT {

using namespace gert;

static const std::string OP_NAME = "SparseApplyAdagrad";

class SparseApplyAdagradInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SparseApplyAdagradInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SparseApplyAdagradInfershape TearDown" << std::endl; }
};

static InfershapeContextPara MakeInfershapePara(const gert::StorageShape& varShape,
                                                const gert::StorageShape& accumShape, const gert::StorageShape& lrShape,
                                                const gert::StorageShape& gradShape,
                                                const gert::StorageShape& indicesShape)
{
    std::vector<InfershapeContextPara::TensorDescription> inputs;
    inputs.emplace_back(varShape, ge::DT_FLOAT, ge::FORMAT_ND);
    inputs.emplace_back(accumShape, ge::DT_FLOAT, ge::FORMAT_ND);
    inputs.emplace_back(lrShape, ge::DT_FLOAT, ge::FORMAT_ND);
    inputs.emplace_back(gradShape, ge::DT_FLOAT, ge::FORMAT_ND);
    inputs.emplace_back(indicesShape, ge::DT_INT32, ge::FORMAT_ND);

    gert::StorageShape outShape({}, {});
    std::vector<InfershapeContextPara::TensorDescription> outputs;
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND);
    outputs.emplace_back(outShape, ge::DT_FLOAT, ge::FORMAT_ND);

    return InfershapeContextPara(OP_NAME, inputs, outputs);
}

TEST_F(SparseApplyAdagradInfershape, Rank2OutputsEqualInput)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 3}, {2, 3});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, scalar, grad, indices);
    std::vector<std::vector<int64_t>> expect = {{4, 3}, {4, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

TEST_F(SparseApplyAdagradInfershape, UnknownShapeOutputsEqualInput)
{
    gert::StorageShape var({-1, -1}, {-1, -1});
    gert::StorageShape accum({-1, -1}, {-1, -1});
    gert::StorageShape scalar({-1}, {-1});
    gert::StorageShape grad({-1, -1}, {-1, -1});
    gert::StorageShape indices({-1}, {-1});
    auto para = MakeInfershapePara(var, accum, scalar, grad, indices);
    std::vector<std::vector<int64_t>> expect = {{-1, -1}, {-1, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

TEST_F(SparseApplyAdagradInfershape, AnyInputUnknownRankOutputsUnknownRank)
{
    gert::StorageShape unknownRank({-2}, {-2});
    gert::StorageShape scalar({}, {});
    auto para = MakeInfershapePara(unknownRank, unknownRank, scalar, unknownRank, unknownRank);
    std::vector<std::vector<int64_t>> expect = {{-2}, {-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

TEST_F(SparseApplyAdagradInfershape, NegativeAccumShapeMismatch)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 5}, {4, 5});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 3}, {2, 3});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, scalar, grad, indices);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeGradAndIndicesMismatch)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({3, 3}, {3, 3});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, scalar, grad, indices);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeVarRankOne)
{
    gert::StorageShape var({4}, {4});
    gert::StorageShape accum({4}, {4});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2}, {2});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, scalar, grad, indices);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeAccumDimNumMismatch)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3, 2}, {4, 3, 2});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 3}, {2, 3});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, scalar, grad, indices);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeLrNotScalar)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    gert::StorageShape lr({2}, {2});
    gert::StorageShape grad({2, 3}, {2, 3});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, lr, grad, indices);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeIndicesNotOneDim)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 3}, {2, 3});
    gert::StorageShape indices({2, 2}, {2, 2});
    auto para = MakeInfershapePara(var, accum, scalar, grad, indices);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeGradDimNumMismatch)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 3, 1}, {2, 3, 1});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, scalar, grad, indices);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeGradTailMismatch)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 5}, {2, 5});
    gert::StorageShape indices({2}, {2});
    auto para = MakeInfershapePara(var, accum, scalar, grad, indices);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 负向：通过双参数 IrInstanceNum 将目标输入/输出的实例数置 0，
// 使 GetInputShape/GetOutputShape 按下标越界规则返回空指针（mse_loss_grad_v2 同款做法）
// 注意：目标位之后的实例数必须一并置 0，否则后续实例会前移占位导致错位
static ge::graphStatus InferSparseApplyAdagradShapeWithNullInstance(const std::vector<uint32_t>& inputInstanceNum,
                                                                    const std::vector<uint32_t>& outputInstanceNum,
                                                                    gert::StorageShape* var, gert::StorageShape* accum,
                                                                    gert::StorageShape* lr, gert::StorageShape* grad,
                                                                    gert::StorageShape* indices)
{
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_NAME.c_str());
    if (impl == nullptr || impl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_NAME.c_str())
                      .IrInstanceNum(inputInstanceNum, outputInstanceNum)
                      .InputShapes({var, accum, lr, grad, indices})
                      .Build();
    return impl->infer_shape(holder.GetContext<gert::InferShapeContext>());
}

TEST_F(SparseApplyAdagradInfershape, NegativeNullVarInput)
{
    EXPECT_EQ(InferSparseApplyAdagradShapeWithNullInstance({0, 0, 0, 0, 0}, {1, 1}, nullptr, nullptr, nullptr, nullptr,
                                                           nullptr),
              ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeNullAccumInput)
{
    gert::StorageShape var({4, 3}, {4, 3});
    EXPECT_EQ(
        InferSparseApplyAdagradShapeWithNullInstance({1, 0, 0, 0, 0}, {1, 1}, &var, nullptr, nullptr, nullptr, nullptr),
        ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeNullLrInput)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    EXPECT_EQ(
        InferSparseApplyAdagradShapeWithNullInstance({1, 1, 0, 0, 0}, {1, 1}, &var, &accum, nullptr, nullptr, nullptr),
        ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeNullGradInput)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    gert::StorageShape scalar({}, {});
    EXPECT_EQ(
        InferSparseApplyAdagradShapeWithNullInstance({1, 1, 1, 0, 0}, {1, 1}, &var, &accum, &scalar, nullptr, nullptr),
        ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeNullIndicesInput)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 3}, {2, 3});
    EXPECT_EQ(
        InferSparseApplyAdagradShapeWithNullInstance({1, 1, 1, 1, 0}, {1, 1}, &var, &accum, &scalar, &grad, nullptr),
        ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeNullVarOutput)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 3}, {2, 3});
    gert::StorageShape indices({2}, {2});
    EXPECT_EQ(
        InferSparseApplyAdagradShapeWithNullInstance({1, 1, 1, 1, 1}, {0, 0}, &var, &accum, &scalar, &grad, &indices),
        ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, NegativeNullAccumOutput)
{
    gert::StorageShape var({4, 3}, {4, 3});
    gert::StorageShape accum({4, 3}, {4, 3});
    gert::StorageShape scalar({}, {});
    gert::StorageShape grad({2, 3}, {2, 3});
    gert::StorageShape indices({2}, {2});
    EXPECT_EQ(
        InferSparseApplyAdagradShapeWithNullInstance({1, 1, 1, 1, 1}, {1, 0}, &var, &accum, &scalar, &grad, &indices),
        ge::GRAPH_FAILED);
}

static ge::graphStatus InferSparseApplyAdagradDtype(ge::DataType varDtype, ge::DataType accumDtype,
                                                    ge::DataType lrDtype, ge::DataType gradDtype,
                                                    ge::DataType indicesDtype)
{
    std::array<ge::DataType, 5> inputs = {varDtype, accumDtype, lrDtype, gradDtype, indicesDtype};
    std::array<ge::DataType, 2> outputs = {ge::DT_UNDEFINED, ge::DT_UNDEFINED};
    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType(OP_NAME.c_str())
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1})
                      .InputDataTypes({&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]})
                      .OutputDataTypes({&outputs[0], &outputs[1]})
                      .Build();
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_NAME.c_str());
    if (impl == nullptr || impl->infer_datatype == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return impl->infer_datatype(holder.GetContext<gert::InferDataTypeContext>());
}

TEST_F(SparseApplyAdagradInfershape, InferDtypePositive)
{
    EXPECT_EQ(InferSparseApplyAdagradDtype(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32),
              ge::GRAPH_SUCCESS);
}

TEST_F(SparseApplyAdagradInfershape, InferDtypeNegativeAccumMismatch)
{
    EXPECT_EQ(InferSparseApplyAdagradDtype(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32),
              ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, InferDtypeNegativeLrMismatch)
{
    EXPECT_EQ(InferSparseApplyAdagradDtype(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32),
              ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, InferDtypeNegativeGradMismatch)
{
    EXPECT_EQ(InferSparseApplyAdagradDtype(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32),
              ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, InferDtypeNegativeVarNotFloat)
{
    EXPECT_EQ(InferSparseApplyAdagradDtype(ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32),
              ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdagradInfershape, InferDtypeNegativeIndicesNotInt32)
{
    EXPECT_EQ(InferSparseApplyAdagradDtype(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64),
              ge::GRAPH_FAILED);
}

} // namespace SparseApplyAdagradUT
