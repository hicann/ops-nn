/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv2d_v2_tuning_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "log/log.h"
#include "array_ops.h"
#include "tests/ut/common/ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_info.h"

#include "../../../op_host/op_tiling/arch35/conv2d_v2_tuning_tiling.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base_utils.h"

using namespace std;
using namespace ge;
using namespace tuningtiling;

namespace {

int64_t InferOutDim(int64_t inputDim, int64_t kernel, int64_t stride, int64_t dilation, int64_t padFront,
                    int64_t padBack)
{
    return (inputDim + padFront + padBack - dilation * (kernel - 1) - 1) / stride + 1;
}

struct TilingContextBuildParams {
    vector<int64_t> fmapShape;
    vector<int64_t> weightShape;
    bool hasBias;
    ge::DataType fmapDtype;
    ge::DataType weightDtype;
    vector<int64_t> strides;
    vector<int64_t> pads;
    vector<int64_t> dilations;
    int64_t groups;
    int32_t opImplMode;
    bool includeOpImplMode;
};

gert::KernelRunContextHolder BuildTilingContext(const TilingContextBuildParams& params)
{
    int64_t N = params.fmapShape[0];
    int64_t Co = params.weightShape[0];
    int64_t kH = params.weightShape[2];
    int64_t kW = params.weightShape[3];
    int64_t sH = params.strides[2];
    int64_t sW = params.strides[3];
    int64_t dH = params.dilations[2];
    int64_t dW = params.dilations[3];
    int64_t pT = params.pads[0];
    int64_t pB = params.pads[1];
    int64_t pL = params.pads[2];
    int64_t pR = params.pads[3];

    int64_t Ho = InferOutDim(params.fmapShape[2], kH, sH, dH, pT, pB);
    int64_t Wo = InferOutDim(params.fmapShape[3], kW, sW, dW, pL, pR);

    gert::StorageShape fmapSs = {{N, params.fmapShape[1], params.fmapShape[2], params.fmapShape[3]},
                                 {N, params.fmapShape[1], params.fmapShape[2], params.fmapShape[3]}};
    gert::StorageShape weightSs = {{Co, params.weightShape[1], kH, kW}, {Co, params.weightShape[1], kH, kW}};
    gert::StorageShape biasSs = {{Co}, {Co}};
    gert::StorageShape outputSs = {{N, Co, Ho, Wo}, {N, Co, Ho, Wo}};

    vector<void*> inputShapes;
    vector<uint32_t> irInstanceNum;
    if (params.hasBias) {
        inputShapes = {&fmapSs, &weightSs, &biasSs, nullptr};
        irInstanceNum = {1, 1, 1, 1};
    } else {
        inputShapes = {&fmapSs, &weightSs, nullptr, nullptr};
        irInstanceNum = {1, 1, 0, 1};
    }
    vector<void*> outputShapes = {&outputSs};

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    vector<pair<string, Ops::NN::AnyValue>> attrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(params.strides)},
        {"pads", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(params.pads)},
        {"dilations", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(params.dilations)},
        {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(params.groups)},
        {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
        {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
    };

    if (params.includeOpImplMode) {
        attrs.emplace_back("op_impl_mode",
                           Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(params.opImplMode)));
    }

    ge::DataType biasDtype = params.hasBias ? params.fmapDtype : params.fmapDtype;

    static fe::PlatFormInfos platformInfo;
    static bool platformInit = false;
    if (!platformInit) {
        platformInfo.Init();
        platformInit = true;
    }
    static optiling::conv_ops_tiling::ConvTilingParseInfo compileInfo;

    string compileInfoStr = R"({"hardware_info":
        {"BT_SIZE": 4096, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 253952,
        "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096,
        "BT_SIZE": 4096, "L0C_SIZE": 262144, "CORE_NUM": 32, "cube_core_cnt": 32, "vector_core_cnt": 64,
        "core_type_list": "CubeCore,VectorCore"}})";
    static map<string, string> socInfos, aicoreSpec, intrinsics;
    static bool resInit = false;
    if (!resInit) {
        GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
        aicoreSpec.insert({"fb0_size", "4096"});
        resInit = true;
    }
    static map<string, string> socVersionInfos = {{"NpuArch", "3510"}};

    return gert::TilingContextFaker()
        .SetOpType("Conv2DV2")
        .NodeIoNum(4, 1)
        .IrInstanceNum(irInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes(outputShapes)
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, params.fmapDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(1, params.weightDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, params.fmapDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, params.fmapDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeAttrs(attrs)
        .TilingData(tilingData.get())
        .Workspace(wsSize)
        .Build();
}

TilingContextBuildParams MakeStandardFp16Params()
{
    TilingContextBuildParams p;
    p.fmapShape = {1, 64, 32, 32};
    p.weightShape = {128, 64, 1, 1};
    p.hasBias = true;
    p.fmapDtype = ge::DT_FLOAT16;
    p.weightDtype = ge::DT_FLOAT16;
    p.strides = {1, 1, 1, 1};
    p.pads = {0, 0, 0, 0};
    p.dilations = {1, 1, 1, 1};
    p.groups = 1;
    p.opImplMode = 0;
    p.includeOpImplMode = false;
    return p;
}

} // namespace

class Conv2DV2TuningTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F(Conv2DV2TuningTilingTest, GetAttrsInfo_StandardFp16Params)
{
    auto holder = BuildTilingContext(MakeStandardFp16Params());
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();
    GetAttrsInfo(ctx, args);

    EXPECT_EQ(args->groups, 1);
    EXPECT_EQ(args->strideH, 1);
    EXPECT_EQ(args->strideW, 1);
    EXPECT_EQ(args->dilationH, 1);
    EXPECT_EQ(args->dilationW, 1);
    EXPECT_EQ(args->padTop, 0);
    EXPECT_EQ(args->padBottom, 0);
    EXPECT_EQ(args->padLeft, 0);
    EXPECT_EQ(args->padRight, 0);
    EXPECT_EQ(args->aDtype, ge::DT_FLOAT16);
    EXPECT_EQ(args->aFormat, ge::FORMAT_NCHW);
    EXPECT_EQ(args->cDtype, ge::DT_FLOAT16);
    EXPECT_EQ(args->cFormat, ge::FORMAT_NCHW);
    EXPECT_EQ(args->aShapeN, 1);
    EXPECT_EQ(args->aShapeH, 32);
    EXPECT_EQ(args->aShapeW, 32);
    EXPECT_EQ(args->cShapeH, 32);
    EXPECT_EQ(args->cShapeW, 32);
}

TEST_F(Conv2DV2TuningTilingTest, GetAttrsInfo_Stride2)
{
    auto p = MakeStandardFp16Params();
    p.strides = {1, 1, 2, 2};
    p.fmapShape = {1, 64, 32, 32};
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();
    GetAttrsInfo(ctx, args);

    EXPECT_EQ(args->strideH, 2);
    EXPECT_EQ(args->strideW, 2);
    EXPECT_EQ(args->cShapeH, 16);
    EXPECT_EQ(args->cShapeW, 16);
}

TEST_F(Conv2DV2TuningTilingTest, GetAttrsInfo_WithPadding)
{
    auto p = MakeStandardFp16Params();
    p.pads = {1, 2, 3, 4};
    p.fmapShape = {1, 64, 32, 32};
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();
    GetAttrsInfo(ctx, args);

    EXPECT_EQ(args->padTop, 1);
    EXPECT_EQ(args->padBottom, 2);
    EXPECT_EQ(args->padLeft, 3);
    EXPECT_EQ(args->padRight, 4);
    EXPECT_EQ(args->cShapeH, 35);
    EXPECT_EQ(args->cShapeW, 39);
}

TEST_F(Conv2DV2TuningTilingTest, GetAttrsInfo_Dilation2)
{
    auto p = MakeStandardFp16Params();
    p.dilations = {1, 1, 2, 2};
    p.fmapShape = {1, 64, 32, 32};
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();
    GetAttrsInfo(ctx, args);

    EXPECT_EQ(args->dilationH, 2);
    EXPECT_EQ(args->dilationW, 2);
}

TEST_F(Conv2DV2TuningTilingTest, GetAttrsInfo_Groups4)
{
    auto p = MakeStandardFp16Params();
    p.groups = 4;
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();
    GetAttrsInfo(ctx, args);

    EXPECT_EQ(args->groups, 4);
}

TEST_F(Conv2DV2TuningTilingTest, GetAttrsInfo_FloatInput)
{
    auto p = MakeStandardFp16Params();
    p.fmapDtype = ge::DT_FLOAT;
    p.includeOpImplMode = true;
    p.opImplMode = 0;
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();
    GetAttrsInfo(ctx, args);

    EXPECT_EQ(args->aDtype, ge::DT_FLOAT);
    EXPECT_EQ(args->cDtype, ge::DT_FLOAT);
    EXPECT_FALSE(args->hf32Flag);
}

TEST_F(Conv2DV2TuningTilingTest, GetAttrsInfo_Hf32Enabled)
{
    auto p = MakeStandardFp16Params();
    p.fmapDtype = ge::DT_FLOAT;
    p.includeOpImplMode = true;
    p.opImplMode = 0x40;
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();
    GetAttrsInfo(ctx, args);

    EXPECT_TRUE(args->hf32Flag);
}

TEST_F(Conv2DV2TuningTilingTest, GetBiasInfo_WithBias)
{
    auto p = MakeStandardFp16Params();
    p.hasBias = true;
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();

    GetBiasInfo(ctx, args, tuningtiling::INPUT_BIAS_INDEX);

    EXPECT_TRUE(args->biasFlag);
    EXPECT_EQ(args->biasDtype, ge::DT_FLOAT16);
}

TEST_F(Conv2DV2TuningTilingTest, GetBiasInfo_WithoutBias)
{
    auto p = MakeStandardFp16Params();
    p.hasBias = false;
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();

    GetBiasInfo(ctx, args, tuningtiling::INPUT_BIAS_INDEX);

    EXPECT_FALSE(args->biasFlag);
    EXPECT_EQ(args->biasDtype, ge::DT_FLOAT16);
}

TEST_F(Conv2DV2TuningTilingTest, GetFilterInfo_Kernel1x1)
{
    auto p = MakeStandardFp16Params();
    p.weightShape = {128, 64, 1, 1};
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();

    GetFilterInfo(ctx, args, tuningtiling::INPUT_B_INDEX);

    EXPECT_EQ(args->bShapeN, 128);
    EXPECT_EQ(args->bShapeC, 64);
    EXPECT_EQ(args->bShapeH, 1);
    EXPECT_EQ(args->bShapeW, 1);
    EXPECT_EQ(args->bDtype, ge::DT_FLOAT16);
    EXPECT_EQ(args->bFormat, ge::FORMAT_NCHW);
}

TEST_F(Conv2DV2TuningTilingTest, GetFilterInfo_Kernel3x3)
{
    auto p = MakeStandardFp16Params();
    p.weightShape = {64, 32, 3, 3};
    p.fmapShape = {1, 32, 14, 14};
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();

    GetFilterInfo(ctx, args, tuningtiling::INPUT_B_INDEX);

    EXPECT_EQ(args->bShapeN, 64);
    EXPECT_EQ(args->bShapeC, 32);
    EXPECT_EQ(args->bShapeH, 3);
    EXPECT_EQ(args->bShapeW, 3);
}

TEST_F(Conv2DV2TuningTilingTest, TilingForConv2DV2Input_ValidContext)
{
    auto p = MakeStandardFp16Params();
    p.hasBias = true;
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<void> inputArgs = nullptr;
    size_t size = 0;

    bool result = TilingForConv2DV2Input(ctx, inputArgs, size);

    EXPECT_TRUE(result);
    EXPECT_NE(inputArgs, nullptr);
    EXPECT_EQ(size, sizeof(Conv2DV2InputArgs));

    auto parsed = static_pointer_cast<Conv2DV2InputArgs>(inputArgs);
    EXPECT_EQ(parsed->aDtype, ge::DT_FLOAT16);
    EXPECT_EQ(parsed->strideH, 1);
    EXPECT_EQ(parsed->groups, 1);
    EXPECT_TRUE(parsed->biasFlag);
}

TEST_F(Conv2DV2TuningTilingTest, TilingForConv2DV2Input_WithoutBias)
{
    auto p = MakeStandardFp16Params();
    p.hasBias = false;
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<void> inputArgs = nullptr;
    size_t size = 0;

    bool result = TilingForConv2DV2Input(ctx, inputArgs, size);

    EXPECT_TRUE(result);
    EXPECT_NE(inputArgs, nullptr);
    static_pointer_cast<Conv2DV2InputArgs>(inputArgs);
    EXPECT_FALSE(static_pointer_cast<Conv2DV2InputArgs>(inputArgs)->biasFlag);
}

TEST_F(Conv2DV2TuningTilingTest, TilingForConv2DV2Input_NullContext)
{
    shared_ptr<void> inputArgs = nullptr;
    size_t size = 0;

    bool result = TilingForConv2DV2Input(nullptr, inputArgs, size);

    EXPECT_FALSE(result);
}

TEST_F(Conv2DV2TuningTilingTest, GetFilterInfo_DepthwiseConv)
{
    auto p = MakeStandardFp16Params();
    p.weightShape = {64, 1, 3, 3};
    p.groups = 64;
    p.fmapShape = {1, 64, 28, 28};
    auto holder = BuildTilingContext(p);
    auto ctx = holder.GetContext<gert::TilingContext>();
    shared_ptr<Conv2DV2InputArgs> args = make_shared<Conv2DV2InputArgs>();

    GetFilterInfo(ctx, args, tuningtiling::INPUT_B_INDEX);

    EXPECT_EQ(args->bShapeN, 64);
    EXPECT_EQ(args->bShapeC, 1);
    EXPECT_EQ(args->bShapeH, 3);
    EXPECT_EQ(args->bShapeW, 3);
}
