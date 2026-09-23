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
 * \file test_turbo_quant_infershape.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

namespace {
constexpr int64_t TQ_N_CENT = 16;

// gert::StorageShape has no constructor taking a std::vector, so the dims are appended one by one.
void AppendShape(gert::StorageShape& shape, const std::vector<int64_t>& dims)
{
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
}

void RunInferShape(const std::vector<int64_t>& latentDims, const std::vector<int64_t>& expectYDims,
                   const std::vector<int64_t>& expectScaleDims, ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS)
{
    gert::StorageShape latentShape;
    gert::StorageShape centShape;
    gert::StorageShape yShape;
    gert::StorageShape scaleShape;
    AppendShape(latentShape, latentDims);
    AppendShape(centShape, {TQ_N_CENT});

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("TurboQuant")
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&latentShape, &centShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("TurboQuant");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(opImpl->infer_shape(context), expectedStatus);
    if (expectedStatus != ge::GRAPH_SUCCESS) {
        return;
    }
    // y stores headDim / 2 packed bytes; scale stores one fp16 norm per token.
    ASSERT_NE(context->GetOutputShape(0), nullptr);
    ASSERT_NE(context->GetOutputShape(1), nullptr);
    EXPECT_EQ(ut_util::ToVector(*context->GetOutputShape(0)), expectYDims);
    EXPECT_EQ(ut_util::ToVector(*context->GetOutputShape(1)), expectScaleDims);
}
} // namespace

class TurboQuantInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "turbo_quant infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "turbo_quant infershape TearDown" << std::endl; }
};

TEST_F(TurboQuantInferShapeTest, head_dim_512) { RunInferShape({128, 512}, {128, 256}, {128}); }

TEST_F(TurboQuantInferShapeTest, single_token) { RunInferShape({1, 512}, {1, 256}, {1}); }

// The runtime kernel currently supports only headDim=512, so shape inference rejects other known sizes.
TEST_F(TurboQuantInferShapeTest, unsupported_head_dim_128) { RunInferShape({8, 128}, {}, {}, ge::GRAPH_FAILED); }

TEST_F(TurboQuantInferShapeTest, unsupported_head_dim_256) { RunInferShape({8, 256}, {}, {}, ge::GRAPH_FAILED); }

TEST_F(TurboQuantInferShapeTest, unsupported_head_dim_1024) { RunInferShape({8, 1024}, {}, {}, ge::GRAPH_FAILED); }

TEST_F(TurboQuantInferShapeTest, unsupported_odd_head_dim) { RunInferShape({8, 513}, {}, {}, ge::GRAPH_FAILED); }

TEST_F(TurboQuantInferShapeTest, dynamic_token_dim) { RunInferShape({-1, 512}, {-1, 256}, {-1}); }

TEST_F(TurboQuantInferShapeTest, dynamic_head_dim) { RunInferShape({8, -1}, {8, -1}, {8}); }

TEST_F(TurboQuantInferShapeTest, invalid_rank) { RunInferShape({4, 8, 512}, {}, {}, ge::GRAPH_FAILED); }
