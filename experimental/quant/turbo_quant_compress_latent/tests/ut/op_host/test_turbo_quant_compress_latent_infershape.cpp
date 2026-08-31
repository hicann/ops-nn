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
 * \file test_turbo_quant_compress_latent_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <vector>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/turbo_quant_compress_latent_proto.h"

using namespace ge;

class TurboQuantCompressLatentInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "turbo_quant_compress_latent infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "turbo_quant_compress_latent infershape TearDown" << std::endl; }
};

static void RunInferShape(const std::vector<int64_t>& latentDims, const std::vector<int64_t>& expectSlotDims,
                          int64_t outputMode = 0, bool setOutputMode = true)
{
    ge::op::TurboQuantCompressLatent op;
    if (setOutputMode) {
        op.SetAttr("output_mode", outputMode);
    }

    ge::TensorDesc latentDesc;
    ge::Shape latentShape(latentDims);
    latentDesc.SetDataType(ge::DT_FLOAT);
    latentDesc.SetShape(latentShape);
    latentDesc.SetOriginShape(latentShape);
    op.UpdateInputDesc("latent", latentDesc);

    ge::TensorDesc centDesc;
    ge::Shape centShape({16});
    centDesc.SetDataType(ge::DT_FLOAT);
    centDesc.SetShape(centShape);
    centDesc.SetOriginShape(centShape);
    op.UpdateInputDesc("centroids", centDesc);

    Runtime2TestParam param;
    param.attrs = {"output_mode"};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expectSlotDims);
}

// slotSize = alignUp(headDim / 2 + 2, 64); the production shape is [N, 512] -> [N, 320].
TEST_F(TurboQuantCompressLatentInferShapeTest, head_dim_512) { RunInferShape({128, 512}, {128, 320}); }

TEST_F(TurboQuantCompressLatentInferShapeTest, single_token) { RunInferShape({1, 512}, {1, 320}); }

TEST_F(TurboQuantCompressLatentInferShapeTest, default_output_mode) { RunInferShape({8, 512}, {8, 320}, 0, false); }

TEST_F(TurboQuantCompressLatentInferShapeTest, compact_corrected) { RunInferShape({128, 512}, {128, 258}, 1); }

// The slot-size derivation itself is head-dim generic even though tiling currently only enables 512.
TEST_F(TurboQuantCompressLatentInferShapeTest, head_dim_128) { RunInferShape({8, 128}, {8, 128}); }

TEST_F(TurboQuantCompressLatentInferShapeTest, head_dim_256) { RunInferShape({8, 256}, {8, 192}); }

TEST_F(TurboQuantCompressLatentInferShapeTest, head_dim_1024) { RunInferShape({8, 1024}, {8, 576}); }

TEST_F(TurboQuantCompressLatentInferShapeTest, dynamic_token_dim) { RunInferShape({-1, 512}, {-1, 320}); }

TEST_F(TurboQuantCompressLatentInferShapeTest, dynamic_head_dim) { RunInferShape({8, -1}, {8, -1}); }

TEST_F(TurboQuantCompressLatentInferShapeTest, invalid_output_mode)
{
    ge::op::TurboQuantCompressLatent op;
    op.SetAttr("output_mode", 2);
    ge::TensorDesc latentDesc(ge::Shape({8, 512}), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::TensorDesc centDesc(ge::Shape({16}), ge::FORMAT_ND, ge::DT_FLOAT);
    op.UpdateInputDesc("latent", latentDesc);
    op.UpdateInputDesc("centroids", centDesc);
    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(TurboQuantCompressLatentInferShapeTest, invalid_rank)
{
    ge::op::TurboQuantCompressLatent op;

    ge::TensorDesc latentDesc;
    ge::Shape latentShape({4, 8, 512});
    latentDesc.SetDataType(ge::DT_FLOAT);
    latentDesc.SetShape(latentShape);
    latentDesc.SetOriginShape(latentShape);
    op.UpdateInputDesc("latent", latentDesc);

    ge::TensorDesc centDesc;
    ge::Shape centShape({16});
    centDesc.SetDataType(ge::DT_FLOAT);
    centDesc.SetShape(centShape);
    centDesc.SetOriginShape(centShape);
    op.UpdateInputDesc("centroids", centDesc);

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}
