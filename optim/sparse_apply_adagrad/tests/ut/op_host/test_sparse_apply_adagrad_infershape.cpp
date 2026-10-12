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
#include "infer_shape_context_faker.h"
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

} // namespace SparseApplyAdagradUT
