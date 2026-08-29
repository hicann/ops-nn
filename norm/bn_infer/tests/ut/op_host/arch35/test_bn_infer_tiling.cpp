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
 * \file test_bn_infer_tiling.cpp
 * \brief BNInfer arch35 tiling UT.
 */

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "tiling_context_faker.h"
#include "ut_op_common.h"
#include "../../../../op_host/arch35/bn_infer_tiling.h"

using namespace ge;
using namespace optiling;
using namespace std;

namespace {
constexpr const char* COMPILE_INFO = R"({
    "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64,
        "socVersion": "Ascend950"
    }
})";

int64_t GetChannelLen(const vector<int64_t>& shape, ge::Format format)
{
    if (format == ge::FORMAT_NHWC || format == ge::FORMAT_NDHWC) {
        return shape.back();
    }
    return shape.size() > 1 ? shape[1] : 0;
}

gert::StorageShape MakeStorageShape(const vector<int64_t>& dims)
{
    gert::StorageShape storageShape;
    auto& originShape = storageShape.MutableOriginShape();
    auto& realShape = storageShape.MutableStorageShape();
    for (auto dim : dims) {
        originShape.AppendDim(dim);
        realShape.AppendDim(dim);
    }
    return storageShape;
}

ge::graphStatus RunBNInferTiling(const vector<int64_t>& xDims, ge::DataType xDtype, ge::Format xFormat,
                                 uint64_t* tilingKey = nullptr, uint64_t* blockDim = nullptr)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("BNInfer");
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(COMPILE_INFO, socInfos, aicoreSpec, intrinsics, socVersion);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    BNInferCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.blockSize = 32;
    compileInfo.vectorLength = 256;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    if (param == nullptr || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t channelLen = GetChannelLen(xDims, xFormat);
    vector<int64_t> paramDims = {channelLen};
    gert::StorageShape xShape = MakeStorageShape(xDims);
    gert::StorageShape yShape = MakeStorageShape(xDims);
    gert::StorageShape scaleShape = MakeStorageShape(paramDims);
    gert::StorageShape offsetShape = MakeStorageShape(paramDims);
    gert::StorageShape meanShape = MakeStorageShape(paramDims);
    gert::StorageShape varianceShape = MakeStorageShape(paramDims);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("BNInfer")
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1})
                      .InputShapes({&xShape, &scaleShape, &offsetShape, &meanShape, &varianceShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, xFormat, xFormat)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, xDtype, xFormat, xFormat)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5f)}})
                      .TilingData(param.get())
                      .Workspace(workspace)
                      .Build();

    auto tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    auto status = opImpl->tiling(tilingContext);
    if (status == ge::GRAPH_SUCCESS) {
        if (tilingKey != nullptr) {
            *tilingKey = tilingContext->GetTilingKey();
        }
        if (blockDim != nullptr) {
            *blockDim = tilingContext->GetBlockDim();
        }
    }
    return status;
}
} // namespace

class BNInferTilingTest : public testing::Test {};

TEST_F(BNInferTilingTest, ndRegularTilingKey910000)
{
    uint64_t tilingKey = 0;
    uint64_t blockDim = 0;
    auto status = RunBNInferTiling({2, 3, 4}, ge::DT_FLOAT, ge::FORMAT_ND, &tilingKey, &blockDim);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 910000);
    EXPECT_GT(blockDim, 0);
}

TEST_F(BNInferTilingTest, ndSmallAB1FallsBackToGeneralTiling)
{
    uint64_t tilingKey = 0;
    auto status = RunBNInferTiling({256, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, &tilingKey);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 910000);
}

TEST_F(BNInferTilingTest, ndSmallAB1FloatFallsBackToGeneralTiling)
{
    uint64_t tilingKey = 0;
    auto status = RunBNInferTiling({256, 4, 4}, ge::DT_FLOAT, ge::FORMAT_ND, &tilingKey);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 910000);
}

TEST_F(BNInferTilingTest, nhwcRegularTilingKey900000)
{
    uint64_t tilingKey = 0;
    auto status = RunBNInferTiling({2, 4, 5, 16}, ge::DT_BF16, ge::FORMAT_NHWC, &tilingKey);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 900000);
}

TEST_F(BNInferTilingTest, nhwcUnalignedTilingKey901000)
{
    uint64_t tilingKey = 0;
    auto status = RunBNInferTiling({2, 4, 5, 3}, ge::DT_BF16, ge::FORMAT_NHWC, &tilingKey);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 901000);
}

TEST_F(BNInferTilingTest, ndhwcUnalignedTilingKey901000)
{
    uint64_t tilingKey = 0;
    auto status = RunBNInferTiling({1, 3, 4, 5, 8}, ge::DT_BF16, ge::FORMAT_NDHWC, &tilingKey);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 901000);
}

TEST_F(BNInferTilingTest, nhwcContinuousATilingKey901000)
{
    uint64_t tilingKey = 0;
    auto status = RunBNInferTiling({1, 257, 257, 256}, ge::DT_FLOAT16, ge::FORMAT_NHWC, &tilingKey);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 901000);
}

TEST_F(BNInferTilingTest, nhwcFloatContinuousATilingKey901000)
{
    uint64_t tilingKey = 0;
    auto status = RunBNInferTiling({2, 17, 65, 96}, ge::DT_FLOAT, ge::FORMAT_NHWC, &tilingKey);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 901000);
}

TEST_F(BNInferTilingTest, rejectLastChannelWhenChannelCacheExceedsUb)
{
    auto status = RunBNInferTiling({1, 1, 1, 4096}, ge::DT_FLOAT, ge::FORMAT_NHWC);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(BNInferTilingTest, nhwcSmallATilingKey902000)
{
    uint64_t tilingKey = 0;
    auto status = RunBNInferTiling({1, 257, 257, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, &tilingKey);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 902000);
}

TEST_F(BNInferTilingTest, nhwcBfloat16SmallAFallsBackToRegularTiling)
{
    uint64_t tilingKey = 0;
    auto status = RunBNInferTiling({1, 257, 257, 3}, ge::DT_BF16, ge::FORMAT_NHWC, &tilingKey);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 901000);
}

TEST_F(BNInferTilingTest, rejectEmptyND)
{
    auto status = RunBNInferTiling({0, 3, 4}, ge::DT_FLOAT, ge::FORMAT_ND);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(BNInferTilingTest, rejectEmptyLastChannel)
{
    auto status = RunBNInferTiling({2, 4, 5, 0}, ge::DT_FLOAT16, ge::FORMAT_NHWC);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(BNInferTilingTest, rejectEmptyOuterLastChannel)
{
    auto status = RunBNInferTiling({0, 4, 5, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(BNInferTilingTest, rejectEmptyEveryPhysicalAxis)
{
    struct EmptyShapeCase {
        vector<int64_t> dims;
        ge::Format format;
    };
    const vector<EmptyShapeCase> cases = {
        {{0, 3, 4}, ge::FORMAT_ND},          {{2, 0, 4}, ge::FORMAT_ND},          {{0, 0, 4}, ge::FORMAT_ND},
        {{0, 3, 4, 5}, ge::FORMAT_NCHW},     {{2, 0, 4, 5}, ge::FORMAT_NCHW},     {{2, 3, 0, 5}, ge::FORMAT_NCHW},
        {{2, 3, 4, 0}, ge::FORMAT_NCHW},     {{0, 0, 4, 5}, ge::FORMAT_NCHW},     {{0, 3, 4, 5, 6}, ge::FORMAT_NCDHW},
        {{2, 0, 4, 5, 6}, ge::FORMAT_NCDHW}, {{2, 3, 0, 5, 6}, ge::FORMAT_NCDHW}, {{2, 3, 4, 0, 6}, ge::FORMAT_NCDHW},
        {{2, 3, 4, 5, 0}, ge::FORMAT_NCDHW}, {{0, 0, 4, 5, 6}, ge::FORMAT_NCDHW}, {{0, 3, 4, 5}, ge::FORMAT_NHWC},
        {{2, 0, 4, 5}, ge::FORMAT_NHWC},     {{2, 3, 0, 5}, ge::FORMAT_NHWC},     {{2, 3, 4, 0}, ge::FORMAT_NHWC},
        {{0, 0, 4, 5}, ge::FORMAT_NHWC},     {{0, 3, 4, 5, 6}, ge::FORMAT_NDHWC}, {{2, 0, 4, 5, 6}, ge::FORMAT_NDHWC},
        {{2, 3, 0, 5, 6}, ge::FORMAT_NDHWC}, {{2, 3, 4, 0, 6}, ge::FORMAT_NDHWC}, {{2, 3, 4, 5, 0}, ge::FORMAT_NDHWC},
        {{0, 0, 4, 5, 6}, ge::FORMAT_NDHWC},
    };

    for (const auto& testCase : cases) {
        EXPECT_EQ(RunBNInferTiling(testCase.dims, ge::DT_FLOAT, testCase.format), ge::GRAPH_FAILED)
            << "unexpected status for format " << static_cast<int>(testCase.format);
    }
}

TEST_F(BNInferTilingTest, rejectEmptyOneDimensionalInput)
{
    EXPECT_NE(RunBNInferTiling({0}, ge::DT_FLOAT, ge::FORMAT_ND), ge::GRAPH_SUCCESS);
}

TEST_F(BNInferTilingTest, rejectUnsupportedNc1hwc0Format)
{
    EXPECT_NE(RunBNInferTiling({2, 1, 4, 4, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0), ge::GRAPH_SUCCESS);
}

TEST_F(BNInferTilingTest, rejectUnsupportedKnownShapeNullFormat)
{
    EXPECT_NE(RunBNInferTiling({2, 3, 4, 4}, ge::DT_FLOAT16, ge::FORMAT_NULL), ge::GRAPH_SUCCESS);
}
