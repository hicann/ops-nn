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
 * \file test_conv3d_v2_tuning_tiling.cpp
 * \brief UT for conv3d_v2_tuning_tiling.cpp
 */

#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "tests/ut/common/ut_op_util.h"
#include "platform/platform_info.h"
#include "test_cube_util.h"
#include "../../../op_host/op_tiling/arch35/conv3d_v2_tuning_tiling.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base_utils.h"

using namespace std;
using namespace ge;
using namespace ut_util;

namespace {

static std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeTuningAttrs(const std::vector<int64_t>& strides,
                                                                              const std::vector<int64_t>& pads,
                                                                              const std::vector<int64_t>& dilations,
                                                                              int64_t groups)
{
    return {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
        {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
        {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
    };
}

static gert::KernelRunContextHolder BuildBaseTilingContext(
    ge::DataType fmapDtype, ge::Format fmapFormat, const std::vector<int64_t>& fmapShape, ge::DataType weightDtype,
    ge::Format weightFormat, const std::vector<int64_t>& weightShape, ge::DataType biasDtype, bool hasBias,
    ge::DataType outputDtype, ge::Format outputFormat, const std::vector<int64_t>& outputShape,
    const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs)
{
    gert::StorageShape fmapSs = {{fmapShape[0], fmapShape[1], fmapShape[2], fmapShape[3], fmapShape[4]},
                                 {fmapShape[0], fmapShape[1], fmapShape[2], fmapShape[3], fmapShape[4]}};
    gert::StorageShape weightSs = {{weightShape[0], weightShape[1], weightShape[2], weightShape[3], weightShape[4]},
                                   {weightShape[0], weightShape[1], weightShape[2], weightShape[3], weightShape[4]}};
    gert::StorageShape biasSs = {{outputShape[0]}, {outputShape[0]}};
    gert::StorageShape outputSs = {{outputShape[0], outputShape[1], outputShape[2], outputShape[3], outputShape[4]},
                                   {outputShape[0], outputShape[1], outputShape[2], outputShape[3], outputShape[4]}};

    std::vector<void*> inputShapes;
    if (hasBias) {
        inputShapes = {&fmapSs, &weightSs, &biasSs, nullptr};
    } else {
        inputShapes = {&fmapSs, &weightSs, nullptr, nullptr};
    }
    std::vector<void*> outputShapes = {&outputSs};

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    static string compileInfoStr = R"({"hardware_info":
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
    static fe::PlatFormInfos platformInfo;
    static bool pInit = false;
    if (!pInit) {
        platformInfo.Init();
        pInit = true;
    }
    static optiling::conv_ops_tiling::ConvTilingParseInfo compileInfo;

    return gert::TilingContextFaker()
        .SetOpType("Conv3DV2")
        .NodeIoNum(4, 1)
        .IrInstanceNum({1, 1, 1, 1})
        .InputShapes(inputShapes)
        .OutputShapes(outputShapes)
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, fmapDtype, fmapFormat, fmapFormat)
        .NodeInputTd(1, weightDtype, weightFormat, weightFormat)
        .NodeInputTd(2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtype, outputFormat, outputFormat)
        .NodeAttrs(attrs)
        .TilingData(tilingData.get())
        .Workspace(wsSize)
        .Build();
}

class Conv3DV2TuningTilingTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// GetAttrsInfo Tests
// ============================================================================

TEST_F(Conv3DV2TuningTilingTest, GetAttrsInfo_NCDHW)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 16, 8, 16, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {16, 16, 1, 1, 1}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {1, 16, 8, 16, 16},
                                         MakeTuningAttrs({1, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}, {1, 1, 11, 12, 13}, 2));
    auto ctx = holder.GetContext<gert::TilingContext>();
    auto args = std::make_shared<tuningtiling::Conv3DV2InputArgs>();
    tuningtiling::GetAttrsInfo(ctx, args);

    EXPECT_EQ(args->strideD, 2);
    EXPECT_EQ(args->strideH, 3);
    EXPECT_EQ(args->strideW, 4);
    EXPECT_EQ(args->dilationD, 11);
    EXPECT_EQ(args->dilationH, 12);
    EXPECT_EQ(args->dilationW, 13);
    EXPECT_EQ(args->padHead, 5);
    EXPECT_EQ(args->padTail, 6);
    EXPECT_EQ(args->padTop, 7);
    EXPECT_EQ(args->padBottom, 8);
    EXPECT_EQ(args->padLeft, 9);
    EXPECT_EQ(args->padRight, 10);
    EXPECT_EQ(args->groups, 2);
}

TEST_F(Conv3DV2TuningTilingTest, GetAttrsInfo_NDHWC)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 8, 16, 16, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_DHWCN, {16, 8, 1, 1, 1}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NDHWC, {1, 8, 16, 16, 16},
                                         MakeTuningAttrs({1, 2, 3, 4, 1}, {5, 6, 7, 8, 9, 10}, {1, 11, 12, 13, 1}, 3));
    auto ctx = holder.GetContext<gert::TilingContext>();
    auto args = std::make_shared<tuningtiling::Conv3DV2InputArgs>();
    tuningtiling::GetAttrsInfo(ctx, args);

    EXPECT_EQ(args->strideD, 2);
    EXPECT_EQ(args->strideH, 3);
    EXPECT_EQ(args->strideW, 4);
    EXPECT_EQ(args->dilationD, 11);
    EXPECT_EQ(args->dilationH, 12);
    EXPECT_EQ(args->dilationW, 13);
    EXPECT_EQ(args->padHead, 5);
    EXPECT_EQ(args->padTail, 6);
    EXPECT_EQ(args->padTop, 7);
    EXPECT_EQ(args->padBottom, 8);
    EXPECT_EQ(args->padLeft, 9);
    EXPECT_EQ(args->padRight, 10);
    EXPECT_EQ(args->groups, 3);
}

TEST_F(Conv3DV2TuningTilingTest, GetOutputInfo_NCDHW)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 16, 8, 16, 32}, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {32, 16, 1, 1, 1}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {1, 32, 8, 16, 32},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1));
    auto ctx = holder.GetContext<gert::TilingContext>();
    auto args = std::make_shared<tuningtiling::Conv3DV2InputArgs>();
    tuningtiling::GetOutputInfo(ctx, args);

    EXPECT_EQ(args->cDtype, ge::DT_FLOAT16);
    EXPECT_EQ(args->cFormat, ge::FORMAT_NCDHW);
    EXPECT_EQ(args->cShapeD, 8);
    EXPECT_EQ(args->cShapeH, 16);
    EXPECT_EQ(args->cShapeW, 32);
}

TEST_F(Conv3DV2TuningTilingTest, GetOutputInfo_NDHWC)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 8, 16, 32, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_DHWCN, {16, 8, 1, 1, 1}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NDHWC, {1, 8, 16, 32, 16},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1));
    auto ctx = holder.GetContext<gert::TilingContext>();
    auto args = std::make_shared<tuningtiling::Conv3DV2InputArgs>();
    tuningtiling::GetOutputInfo(ctx, args);

    EXPECT_EQ(args->cDtype, ge::DT_FLOAT16);
    EXPECT_EQ(args->cFormat, ge::FORMAT_NDHWC);
    EXPECT_EQ(args->cShapeD, 8);
    EXPECT_EQ(args->cShapeH, 16);
    EXPECT_EQ(args->cShapeW, 32);
}

TEST_F(Conv3DV2TuningTilingTest, GetFmapInfo_NCDHW)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 16, 8, 16, 32}, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {32, 16, 1, 1, 1}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {1, 32, 8, 16, 32},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1));
    auto ctx = holder.GetContext<gert::TilingContext>();
    auto args = std::make_shared<tuningtiling::Conv3DV2InputArgs>();
    tuningtiling::GetFmapInfo(ctx, args, 0);

    EXPECT_EQ(args->aDtype, ge::DT_FLOAT16);
    EXPECT_EQ(args->aFormat, ge::FORMAT_NCDHW);
    EXPECT_EQ(args->aShapeN, 1);
    EXPECT_EQ(args->aShapeD, 8);
    EXPECT_EQ(args->aShapeH, 16);
    EXPECT_EQ(args->aShapeW, 32);
}

TEST_F(Conv3DV2TuningTilingTest, GetFmapInfo_NDHWC)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 8, 16, 32, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_DHWCN, {16, 8, 1, 1, 1}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NDHWC, {1, 8, 16, 32, 16},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1));
    auto ctx = holder.GetContext<gert::TilingContext>();
    auto args = std::make_shared<tuningtiling::Conv3DV2InputArgs>();
    tuningtiling::GetFmapInfo(ctx, args, 0);

    EXPECT_EQ(args->aDtype, ge::DT_FLOAT16);
    EXPECT_EQ(args->aFormat, ge::FORMAT_NDHWC);
    EXPECT_EQ(args->aShapeN, 1);
    EXPECT_EQ(args->aShapeD, 8);
    EXPECT_EQ(args->aShapeH, 16);
    EXPECT_EQ(args->aShapeW, 32);
}

TEST_F(Conv3DV2TuningTilingTest, GetBiasInfo_WithBias)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 16, 8, 16, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {16, 16, 1, 1, 1}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {1, 16, 8, 16, 16},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1));
    auto ctx = holder.GetContext<gert::TilingContext>();
    auto args = std::make_shared<tuningtiling::Conv3DV2InputArgs>();
    tuningtiling::GetBiasInfo(ctx, args, 2);

    EXPECT_TRUE(args->biasFlag);
    EXPECT_EQ(args->biasDtype, ge::DT_FLOAT);
}

TEST_F(Conv3DV2TuningTilingTest, GetBiasInfo_WithoutBias)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 16, 8, 16, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {16, 16, 1, 1, 1}, ge::DT_FLOAT16, false, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {1, 16, 8, 16, 16},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1));
    auto ctx = holder.GetContext<gert::TilingContext>();
    auto args = std::make_shared<tuningtiling::Conv3DV2InputArgs>();
    tuningtiling::GetBiasInfo(ctx, args, 2);

    EXPECT_FALSE(args->biasFlag);
    EXPECT_EQ(args->biasDtype, ge::DT_FLOAT16);
}

TEST_F(Conv3DV2TuningTilingTest, GetFilterInfo_NCDHW)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 16, 8, 16, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {16, 16, 3, 3, 3}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {1, 16, 6, 14, 14},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1));
    auto ctx = holder.GetContext<gert::TilingContext>();
    auto args = std::make_shared<tuningtiling::Conv3DV2InputArgs>();
    tuningtiling::GetFilterInfo(ctx, args, 1);

    EXPECT_EQ(args->bDtype, ge::DT_FLOAT16);
    EXPECT_EQ(args->bFormat, ge::FORMAT_NCDHW);
    EXPECT_EQ(args->bShapeN, 16);
    EXPECT_EQ(args->bShapeC, 16);
    EXPECT_EQ(args->bShapeD, 3);
    EXPECT_EQ(args->bShapeH, 3);
    EXPECT_EQ(args->bShapeW, 3);
}

TEST_F(Conv3DV2TuningTilingTest, GetFilterInfo_DHWCN)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 8, 16, 16, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_DHWCN, {16, 8, 3, 3, 3}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NDHWC, {1, 8, 14, 14, 16},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1));
    auto ctx = holder.GetContext<gert::TilingContext>();
    auto args = std::make_shared<tuningtiling::Conv3DV2InputArgs>();
    tuningtiling::GetFilterInfo(ctx, args, 1);

    EXPECT_EQ(args->bDtype, ge::DT_FLOAT16);
    EXPECT_EQ(args->bFormat, ge::FORMAT_DHWCN);
    EXPECT_EQ(args->bShapeN, 3);
    EXPECT_EQ(args->bShapeC, 3);
    EXPECT_EQ(args->bShapeD, 16);
    EXPECT_EQ(args->bShapeH, 8);
    EXPECT_EQ(args->bShapeW, 3);
}

TEST_F(Conv3DV2TuningTilingTest, GetFilterInfo_NDHWC)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 8, 16, 16, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_NDHWC, {16, 8, 3, 3, 3}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NDHWC, {1, 8, 14, 14, 16},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1));
    auto ctx = holder.GetContext<gert::TilingContext>();
    auto args = std::make_shared<tuningtiling::Conv3DV2InputArgs>();
    tuningtiling::GetFilterInfo(ctx, args, 1);

    EXPECT_EQ(args->bDtype, ge::DT_FLOAT16);
    EXPECT_EQ(args->bFormat, ge::FORMAT_NDHWC);
    EXPECT_EQ(args->bShapeN, 16);
    EXPECT_EQ(args->bShapeC, 3);
    EXPECT_EQ(args->bShapeD, 8);
    EXPECT_EQ(args->bShapeH, 3);
    EXPECT_EQ(args->bShapeW, 3);
}

TEST_F(Conv3DV2TuningTilingTest, TilingForConv3DV2Input_WithBias_NCDHW)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 16, 8, 16, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {16, 16, 1, 1, 1}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {1, 16, 8, 16, 16},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1));
    auto ctx = holder.GetContext<gert::TilingContext>();
    std::shared_ptr<void> inputArgs = nullptr;
    size_t size = 0;
    bool result = tuningtiling::TilingForConv3DV2Input(ctx, inputArgs, size);

    EXPECT_TRUE(result);
    EXPECT_NE(inputArgs, nullptr);
    EXPECT_EQ(size, sizeof(tuningtiling::Conv3DV2InputArgs));
}

TEST_F(Conv3DV2TuningTilingTest, TilingForConv3DV2Input_WithoutBias_NDHWC)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 8, 16, 16, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_DHWCN, {16, 8, 1, 1, 1}, ge::DT_FLOAT16, false, ge::DT_FLOAT16,
                                         ge::FORMAT_NDHWC, {1, 8, 16, 16, 16},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1));
    auto ctx = holder.GetContext<gert::TilingContext>();
    std::shared_ptr<void> inputArgs = nullptr;
    size_t size = 0;
    bool result = tuningtiling::TilingForConv3DV2Input(ctx, inputArgs, size);

    EXPECT_TRUE(result);
    EXPECT_NE(inputArgs, nullptr);
    EXPECT_FALSE(std::static_pointer_cast<tuningtiling::Conv3DV2InputArgs>(inputArgs)->biasFlag);
}

TEST_F(Conv3DV2TuningTilingTest, TilingForConv3DV2Input_GroupConv)
{
    auto holder = BuildBaseTilingContext(ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 6, 8, 16, 16}, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {18, 2, 1, 1, 1}, ge::DT_FLOAT, true, ge::DT_FLOAT16,
                                         ge::FORMAT_NCDHW, {1, 18, 8, 16, 16},
                                         MakeTuningAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 3));
    auto ctx = holder.GetContext<gert::TilingContext>();
    std::shared_ptr<void> inputArgs = nullptr;
    size_t size = 0;
    bool result = tuningtiling::TilingForConv3DV2Input(ctx, inputArgs, size);

    EXPECT_TRUE(result);
    EXPECT_NE(inputArgs, nullptr);
    auto parsed = std::static_pointer_cast<tuningtiling::Conv3DV2InputArgs>(inputArgs);
    EXPECT_EQ(parsed->groups, 3);
}
} // namespace
