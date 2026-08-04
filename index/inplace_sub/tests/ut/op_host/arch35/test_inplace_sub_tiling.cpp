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
 * \file test_inplace_sub_tiling.cpp
 * \brief the ut of inplace_sub_tiling
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>

#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "tiling_context_faker.h"
#include "ut_op_common.h"
#include "../../../../op_host/arch35/inplace_sub_tiling.h"
#include "../../../../op_kernel/arch35/inplace_sub_tiling_data.h"

using namespace ge;
using namespace optiling;
using namespace std;

namespace {
constexpr const char* COMPILE_INFO = R"({
    "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
    }
})";

ge::graphStatus RunInplaceSubTiling(gert::StorageShape xShape, gert::StorageShape indicesShape,
                                    gert::StorageShape vShape, ge::DataType xDtype, ge::DataType indicesDtype,
                                    ge::DataType vDtype, InplaceSubTilingData* outTilingData = nullptr)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceSub");
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO, socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    InplaceSubCompileInfo compileInfo;
    compileInfo.core_num = 64;
    compileInfo.ub_size = 253952;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    if (param == nullptr || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &indicesShape, &vShape})
                      .OutputShapes({&xShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, vDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(workspace)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto status = opImpl->tiling(tilingContext);
    if (status == ge::GRAPH_SUCCESS && outTilingData != nullptr) {
        auto tilingData = reinterpret_cast<const InplaceSubTilingData*>(tilingContext->GetRawTilingData()->GetData());
        *outTilingData = *tilingData;
    }
    return status;
}
} // namespace

class InplaceSubTilingTest : public testing::Test {};

TEST_F(InplaceSubTilingTest, staticShapeSuccess)
{
    InplaceSubTilingData tilingData;
    auto status = RunInplaceSubTiling({{64, 128}, {64, 128}}, {{16}, {16}}, {{16, 128}, {16, 128}}, ge::DT_FLOAT16,
                                      ge::DT_INT32, ge::DT_FLOAT16, &tilingData);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingData.needCoreNum, 10);
    EXPECT_EQ(tilingData.n, 64);
    EXPECT_EQ(tilingData.k, 16);
    EXPECT_EQ(tilingData.innerSize, 128);
    EXPECT_EQ(tilingData.perCoreN, 7);
}

TEST_F(InplaceSubTilingTest, staticShapeFloat32Success)
{
    InplaceSubTilingData tilingData;
    auto status = RunInplaceSubTiling({{8, 4}, {8, 4}}, {{2}, {2}}, {{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::DT_INT32,
                                      ge::DT_FLOAT, &tilingData);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingData.n, 8);
    EXPECT_EQ(tilingData.k, 2);
    EXPECT_EQ(tilingData.innerSize, 4);
}

TEST_F(InplaceSubTilingTest, staticShapeInt32Success)
{
    InplaceSubTilingData tilingData;
    auto status = RunInplaceSubTiling({{8, 4}, {8, 4}}, {{2}, {2}}, {{2, 4}, {2, 4}}, ge::DT_INT32, ge::DT_INT32,
                                      ge::DT_INT32, &tilingData);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingData.n, 8);
    EXPECT_EQ(tilingData.k, 2);
    EXPECT_EQ(tilingData.innerSize, 4);
}

TEST_F(InplaceSubTilingTest, staticShapeComplex64Success)
{
    auto status = RunInplaceSubTiling({{8, 4}, {8, 4}}, {{2}, {2}}, {{2, 4}, {2, 4}}, ge::DT_COMPLEX64, ge::DT_INT32,
                                      ge::DT_COMPLEX64);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(InplaceSubTilingTest, staticShapeUint16Success)
{
    auto status = RunInplaceSubTiling({{8, 4}, {8, 4}}, {{2}, {2}}, {{2, 4}, {2, 4}}, ge::DT_UINT16, ge::DT_INT32,
                                      ge::DT_UINT16);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(InplaceSubTilingTest, staticShapeComplex32Success)
{
    auto status = RunInplaceSubTiling({{8, 4}, {8, 4}}, {{2}, {2}}, {{2, 4}, {2, 4}}, ge::DT_COMPLEX32, ge::DT_INT32,
                                      ge::DT_COMPLEX32);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(InplaceSubTilingTest, acceptAscend950SupportedDtypes)
{
    for (ge::DataType dtype :
         {ge::DT_COMPLEX64, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32,
          ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_COMPLEX32}) {
        auto status = RunInplaceSubTiling({{4, 6}, {4, 6}}, {{2}, {2}}, {{2, 6}, {2, 6}}, dtype, ge::DT_INT32, dtype);
        EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    }
}

TEST_F(InplaceSubTilingTest, emptyFirstDimSuccess)
{
    InplaceSubTilingData tilingData;
    auto status = RunInplaceSubTiling({{0, 4}, {0, 4}}, {{0}, {0}}, {{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::DT_INT32,
                                      ge::DT_FLOAT, &tilingData);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingData.needCoreNum, 1);
    EXPECT_EQ(tilingData.n, 0);
    EXPECT_EQ(tilingData.k, 0);
    EXPECT_EQ(tilingData.innerSize, 4);
    EXPECT_EQ(tilingData.perCoreN, 0);
}

TEST_F(InplaceSubTilingTest, rejectEmptyFirstDimWithNonEmptyIndices)
{
    auto status = RunInplaceSubTiling({{0, 4}, {0, 4}}, {{1}, {1}}, {{1, 4}, {1, 4}}, ge::DT_FLOAT, ge::DT_INT32,
                                      ge::DT_FLOAT);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(InplaceSubTilingTest, emptyTailDimSuccess)
{
    InplaceSubTilingData tilingData;
    auto status = RunInplaceSubTiling({{4, 0}, {4, 0}}, {{2}, {2}}, {{2, 0}, {2, 0}}, ge::DT_FLOAT16, ge::DT_INT32,
                                      ge::DT_FLOAT16, &tilingData);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingData.n, 4);
    EXPECT_EQ(tilingData.k, 2);
    EXPECT_EQ(tilingData.innerSize, 0);
}

TEST_F(InplaceSubTilingTest, emptyIndicesSuccess)
{
    InplaceSubTilingData tilingData;
    auto status = RunInplaceSubTiling({{4, 4}, {4, 4}}, {{0}, {0}}, {{0, 4}, {0, 4}}, ge::DT_INT32, ge::DT_INT32,
                                      ge::DT_INT32, &tilingData);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingData.n, 4);
    EXPECT_EQ(tilingData.k, 0);
    EXPECT_EQ(tilingData.innerSize, 4);
}

TEST_F(InplaceSubTilingTest, rejectInvalidIndicesRank)
{
    auto status = RunInplaceSubTiling({{4, 6}, {4, 6}}, {{2, 1}, {2, 1}}, {{2, 6}, {2, 6}}, ge::DT_FLOAT, ge::DT_INT32,
                                      ge::DT_FLOAT);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(InplaceSubTilingTest, rejectMismatchedValueDtype)
{
    auto status = RunInplaceSubTiling({{4, 6}, {4, 6}}, {{2}, {2}}, {{2, 6}, {2, 6}}, ge::DT_FLOAT, ge::DT_INT32,
                                      ge::DT_FLOAT16);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(InplaceSubTilingTest, rejectInt32OverflowShape)
{
    constexpr int64_t overflowDim = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    auto status = RunInplaceSubTiling({{overflowDim, 1}, {overflowDim, 1}}, {{1}, {1}}, {{1, 1}, {1, 1}}, ge::DT_FLOAT,
                                      ge::DT_INT32, ge::DT_FLOAT);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(InplaceSubTilingTest, rejectNegativeTailDim)
{
    auto status = RunInplaceSubTiling({{4, -1}, {4, -1}}, {{2}, {2}}, {{2, -1}, {2, -1}}, ge::DT_FLOAT, ge::DT_INT32,
                                      ge::DT_FLOAT);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(InplaceSubTilingTest, rejectInnerSizeOverflow)
{
    constexpr int64_t largeDim = std::numeric_limits<int64_t>::max() / 2 + 1;
    auto status = RunInplaceSubTiling({{2, largeDim, 3}, {2, largeDim, 3}}, {{1}, {1}},
                                      {{1, largeDim, 3}, {1, largeDim, 3}}, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
