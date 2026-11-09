/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/hard_sigmoid_grad_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "graph/types.h"
#include "platform/platform_info.h"

class HardSigmoidGradProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HardSigmoidGrad Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "HardSigmoidGrad Proto Test TearDown" << std::endl; }
};

static void SetUpPlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

TEST_F(HardSigmoidGradProtoTest, infershape_1d_fp32)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_shape;

    gert::Shape grad_shape = {128};
    gert::Shape self_shape = {128};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto* ctx = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(ctx), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ctx->GetOutputShape(0)->GetDimNum(), 1);
    ASSERT_EQ(ctx->GetOutputShape(0)->GetDim(0), 128);
}

TEST_F(HardSigmoidGradProtoTest, infershape_4d_bf16)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_shape;

    gert::Shape grad_shape = {2, 3, 4, 5};
    gert::Shape self_shape = {2, 3, 4, 5};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto* ctx = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(ctx), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ctx->GetOutputShape(0)->GetShapeSize(), 120);
}

TEST_F(HardSigmoidGradProtoTest, infershape_shape_mismatch_rejected)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_shape;

    gert::Shape grad_shape = {2, 3};
    gert::Shape self_shape = {3, 2};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(HardSigmoidGradProtoTest, infershape_rank_mismatch_rejected)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_shape;

    gert::Shape grad_shape = {8};
    gert::Shape self_shape = {2, 4};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(HardSigmoidGradProtoTest, infershape_rank9_rejected)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_shape;

    gert::Shape grad_shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::Shape self_shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(HardSigmoidGradProtoTest, infershape_rank8_boundary_success)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_shape;

    gert::Shape grad_shape = {1, 1, 1, 1, 1, 1, 1, 2};
    gert::Shape self_shape = {1, 1, 1, 1, 1, 1, 1, 2};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto* ctx = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(ctx), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ctx->GetOutputShape(0)->GetDimNum(), 8);
}

TEST_F(HardSigmoidGradProtoTest, infershape_dynamic_dim_success)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_shape;

    gert::Shape grad_shape = {ge::UNKNOWN_DIM, 4};
    gert::Shape self_shape = {ge::UNKNOWN_DIM, 4};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto* ctx = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(ctx), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ctx->GetOutputShape(0)->GetDimNum(), 2);
    ASSERT_EQ(ctx->GetOutputShape(0)->GetDim(0), ge::UNKNOWN_DIM);
}

TEST_F(HardSigmoidGradProtoTest, infershape_unknown_rank_deferred)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_shape;

    // Only grads carries the unknown-rank marker; the declared output source
    // (input_x) is known, so the output shape follows input_x and grads is
    // re-checked against it by the tiling stage at runtime.
    gert::Shape grad_shape = {ge::UNKNOWN_DIM_NUM};
    gert::Shape self_shape = {2, 4};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto* ctx = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(ctx), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ctx->GetOutputShape(0)->GetDimNum(), 2);
    ASSERT_EQ(ctx->GetOutputShape(0)->GetDim(0), 2);
    ASSERT_EQ(ctx->GetOutputShape(0)->GetDim(1), 4);
}

TEST_F(HardSigmoidGradProtoTest, infershape_unknown_rank_input_x_deferred)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_shape;

    // The declared output source (input_x) itself is unknown-rank: the output
    // must be marked unknown-rank and deferred to runtime.
    gert::Shape grad_shape = {2, 4};
    gert::Shape self_shape = {ge::UNKNOWN_DIM_NUM};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&grad_shape, &self_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto* ctx = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(ctx), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ctx->GetOutputShape(0)->GetDimNum(), 1);
    ASSERT_EQ(ctx->GetOutputShape(0)->GetDim(0), ge::UNKNOWN_DIM_NUM);
}

TEST_F(HardSigmoidGradProtoTest, inferdtype_same_dtype_fp32)
{
    SetUpPlatform();
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType grad_ref = ge::DT_FLOAT;
    ge::DataType self_ref = ge::DT_FLOAT;
    ge::DataType output_ref = ge::DT_FLOAT;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(2)
                      .NodeIoNum(2, 1)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&grad_ref, &self_ref})
                      .OutputDataTypes({&output_ref})
                      .Build();
    auto ctx = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(dataTypeFunc(ctx), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST_F(HardSigmoidGradProtoTest, inferdtype_follows_input_x)
{
    SetUpPlatform();
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    // Proto declares y with the same dtype as input_x (input 1); a wider grads
    // input must not leak its dtype into the output metadata.
    ge::DataType grad_ref = ge::DT_FLOAT;
    ge::DataType self_ref = ge::DT_FLOAT16;
    ge::DataType output_ref = ge::DT_FLOAT16;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(2)
                      .NodeIoNum(2, 1)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&grad_ref, &self_ref})
                      .OutputDataTypes({&output_ref})
                      .Build();
    auto ctx = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(dataTypeFunc(ctx), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(HardSigmoidGradProtoTest, inferdtype_bf16_input_x)
{
    SetUpPlatform();
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoidGrad")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType grad_ref = ge::DT_BF16;
    ge::DataType self_ref = ge::DT_BF16;
    ge::DataType output_ref = ge::DT_BF16;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(2)
                      .NodeIoNum(2, 1)
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&grad_ref, &self_ref})
                      .OutputDataTypes({&output_ref})
                      .Build();
    auto ctx = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(dataTypeFunc(ctx), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetOutputDataType(0), ge::DT_BF16);
}
