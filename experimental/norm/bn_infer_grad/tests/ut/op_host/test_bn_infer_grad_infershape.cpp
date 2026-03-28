/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/*!
 * \file test_bn_infer_grad_infershape.cpp
 * \brief BnInferGrad InferShape UT - Iteration 1 & 3
 *
 * Iteration 1: Core path tests for NCHW fp32.
 * Verifies output shape = input grads shape, output dtype = input grads dtype.
 *
 * Iteration 3: Boundary cases for NHWC, NC1HWC0, fp16/bf16, empty tensor, channel=1.
 */

#include <gtest/gtest.h>
#include "infershape_case_executor.h"

using TensorDesc = gert::InfershapeContextPara::TensorDescription;
using OpAttr = gert::InfershapeContextPara::OpAttr;

// Helper: Create an InfershapeContextPara for BnInferGrad
// Inputs: grads (4D), scale (1D, C), batch_variance (1D, C)
// Output: x_backprop (same shape as grads)
// Attr: epsilon (float)
static gert::InfershapeContextPara MakeInfershapePara(
    const gert::StorageShape& gradsShape,
    int64_t channelSize,
    ge::DataType dtype = ge::DT_FLOAT,
    ge::Format format = ge::FORMAT_NCHW,
    float epsilon = 0.0001f)
{
    // Attributes: epsilon (index=0)
    std::vector<OpAttr> attrs = {
        OpAttr("epsilon", Ops::Math::AnyValue::CreateFrom<float>(epsilon)),
    };

    // scale shape: (C,)
    gert::StorageShape scaleShape({channelSize}, {channelSize});
    // batch_variance shape: (C,)
    gert::StorageShape varianceShape({channelSize}, {channelSize});

    std::vector<TensorDesc> inputs = {
        TensorDesc(gradsShape, dtype, format),
        TensorDesc(scaleShape, ge::DT_FLOAT, ge::FORMAT_ND),
        TensorDesc(varianceShape, ge::DT_FLOAT, ge::FORMAT_ND),
    };

    // Output shape placeholder (will be inferred)
    gert::StorageShape outputPlaceholder({1}, {1});
    std::vector<TensorDesc> outputs = {
        TensorDesc(outputPlaceholder, dtype, format),
    };

    return gert::InfershapeContextPara("BnInferGrad", inputs, outputs, attrs);
}

// ---------- Iteration 1: Core Path Tests (NCHW, fp32) ----------

// TC-IS-001: NCHW fp32, shape=(2,3,4,4) -> output (2,3,4,4)
TEST(BnInferGradInferShape, NCHW_FP32_Basic)
{
    gert::StorageShape gradsShape({2, 3, 4, 4}, {2, 3, 4, 4});
    auto para = MakeInfershapePara(gradsShape, 3);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3, 4, 4}});
}

// TC-IS-002: NCHW fp32, shape=(1,1,1,1) -> output (1,1,1,1)
TEST(BnInferGradInferShape, NCHW_FP32_SingleElement)
{
    gert::StorageShape gradsShape({1, 1, 1, 1}, {1, 1, 1, 1});
    auto para = MakeInfershapePara(gradsShape, 1);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 1, 1, 1}});
}

// TC-IS-003: NCHW fp32, large shape=(8,64,32,32) -> output (8,64,32,32)
TEST(BnInferGradInferShape, NCHW_FP32_LargeShape)
{
    gert::StorageShape gradsShape({8, 64, 32, 32}, {8, 64, 32, 32});
    auto para = MakeInfershapePara(gradsShape, 64);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{8, 64, 32, 32}});
}

// TC-IS-004: NCHW fp32, shape=(4,128,1,1) -> output (4,128,1,1) (spatial = 1x1)
TEST(BnInferGradInferShape, NCHW_FP32_SpatialOne)
{
    gert::StorageShape gradsShape({4, 128, 1, 1}, {4, 128, 1, 1});
    auto para = MakeInfershapePara(gradsShape, 128);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{4, 128, 1, 1}});
}

// TC-IS-005: NCHW fp32, shape=(1,256,8,8) -> output (1,256,8,8) (batch = 1)
TEST(BnInferGradInferShape, NCHW_FP32_BatchOne)
{
    gert::StorageShape gradsShape({1, 256, 8, 8}, {1, 256, 8, 8});
    auto para = MakeInfershapePara(gradsShape, 256);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 256, 8, 8}});
}

// TC-IS-006: NCHW fp32, non-square spatial shape=(2,3,4,5) -> output (2,3,4,5)
TEST(BnInferGradInferShape, NCHW_FP32_NonSquareSpatial)
{
    gert::StorageShape gradsShape({2, 3, 4, 5}, {2, 3, 4, 5});
    auto para = MakeInfershapePara(gradsShape, 3);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3, 4, 5}});
}

// ==========================================================================
// Iteration 3: NHWC format InferShape
// ==========================================================================

// TC-IS-007: NHWC fp32, shape=(2,4,4,3) -> output (2,4,4,3)
TEST(BnInferGradInferShape, NHWC_FP32_Basic)
{
    gert::StorageShape gradsShape({2, 4, 4, 3}, {2, 4, 4, 3});
    auto para = MakeInfershapePara(gradsShape, 3, ge::DT_FLOAT, ge::FORMAT_NHWC);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 4, 4, 3}});
}

// TC-IS-008: NC1HWC0 fp32, shape=(2,4,8,8,16) -> output (2,4,8,8,16)
TEST(BnInferGradInferShape, NC1HWC0_FP32_Basic)
{
    gert::StorageShape gradsShape({2, 4, 8, 8, 16}, {2, 4, 8, 8, 16});
    auto para = MakeInfershapePara(gradsShape, 64, ge::DT_FLOAT, ge::FORMAT_NC1HWC0);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 4, 8, 8, 16}});
}

// ==========================================================================
// Iteration 3: fp16/bf16 InferShape
// ==========================================================================

// TC-IS-009: NCHW fp16, shape=(2,3,4,4) -> output (2,3,4,4)
TEST(BnInferGradInferShape, NCHW_FP16_Basic)
{
    gert::StorageShape gradsShape({2, 3, 4, 4}, {2, 3, 4, 4});
    auto para = MakeInfershapePara(gradsShape, 3, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3, 4, 4}});
}

// TC-IS-010: NCHW bf16, shape=(2,3,4,4) -> output (2,3,4,4)
TEST(BnInferGradInferShape, NCHW_BF16_Basic)
{
    gert::StorageShape gradsShape({2, 3, 4, 4}, {2, 3, 4, 4});
    auto para = MakeInfershapePara(gradsShape, 3, ge::DT_BF16, ge::FORMAT_NCHW);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3, 4, 4}});
}

// TC-IS-011: NHWC fp16, shape=(2,4,4,64) -> output (2,4,4,64)
TEST(BnInferGradInferShape, NHWC_FP16_Basic)
{
    gert::StorageShape gradsShape({2, 4, 4, 64}, {2, 4, 4, 64});
    auto para = MakeInfershapePara(gradsShape, 64, ge::DT_FLOAT16, ge::FORMAT_NHWC);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 4, 4, 64}});
}

// TC-IS-012: NC1HWC0 bf16, shape=(2,4,8,8,16) -> output (2,4,8,8,16)
TEST(BnInferGradInferShape, NC1HWC0_BF16_Basic)
{
    gert::StorageShape gradsShape({2, 4, 8, 8, 16}, {2, 4, 8, 8, 16});
    auto para = MakeInfershapePara(gradsShape, 64, ge::DT_BF16, ge::FORMAT_NC1HWC0);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 4, 8, 8, 16}});
}

// ==========================================================================
// Iteration 3: Boundary - empty tensor
// ==========================================================================

// TC-IS-013: NCHW fp32, shape=(0,3,4,4) -> output (0,3,4,4) (empty tensor, batch=0)
TEST(BnInferGradInferShape, NCHW_FP32_EmptyBatchZero)
{
    gert::StorageShape gradsShape({0, 3, 4, 4}, {0, 3, 4, 4});
    auto para = MakeInfershapePara(gradsShape, 3);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{0, 3, 4, 4}});
}

// TC-IS-014: NCHW fp32, shape=(2,0,4,4) -> output (2,0,4,4) (empty tensor, channel=0)
TEST(BnInferGradInferShape, NCHW_FP32_EmptyChannelZero)
{
    gert::StorageShape gradsShape({2, 0, 4, 4}, {2, 0, 4, 4});
    auto para = MakeInfershapePara(gradsShape, 0);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 0, 4, 4}});
}

// ==========================================================================
// Iteration 3: Boundary - channel=1
// ==========================================================================

// TC-IS-015: NCHW fp32, channel=1, shape=(4,1,8,8) -> output (4,1,8,8)
TEST(BnInferGradInferShape, NCHW_FP32_ChannelOne)
{
    gert::StorageShape gradsShape({4, 1, 8, 8}, {4, 1, 8, 8});
    auto para = MakeInfershapePara(gradsShape, 1);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{4, 1, 8, 8}});
}

// TC-IS-016: NHWC fp16, channel=1, shape=(4,8,8,1) -> output (4,8,8,1)
TEST(BnInferGradInferShape, NHWC_FP16_ChannelOne)
{
    gert::StorageShape gradsShape({4, 8, 8, 1}, {4, 8, 8, 1});
    auto para = MakeInfershapePara(gradsShape, 1, ge::DT_FLOAT16, ge::FORMAT_NHWC);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{4, 8, 8, 1}});
}
