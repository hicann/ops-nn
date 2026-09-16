/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/update_tensor_desc_tiling_data.h"

using namespace ut_util;
using namespace ge;

namespace {
struct UpdateTensorDescCompileInfo {};

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint32_t blockDim = 0;
    size_t workspaceSize = static_cast<size_t>(-1);
    UpdateTensorDescTilingData tilingData{};
};

TilingResult RunTilingCase(const std::vector<int64_t>& xDims, ge::DataType xDtype, const std::vector<int64_t>& yDims,
                           ge::DataType yDtype, const std::vector<int64_t>& shapeAttr)
{
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::string compileInfoStr = R"({"hardware_info":{"UB_SIZE":262144,"CORE_NUM":64}})";
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    UpdateTensorDescCompileInfo compileInfo;
    TilingResult result;

    gert::StorageShape xShape;
    for (const int64_t dim : xDims) {
        xShape.MutableOriginShape().AppendDim(dim);
        xShape.MutableStorageShape().AppendDim(dim);
    }
    gert::StorageShape yShape;
    for (const int64_t dim : yDims) {
        yShape.MutableOriginShape().AppendDim(dim);
        yShape.MutableStorageShape().AppendDim(dim);
    }

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType("UpdateTensorDesc")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum(std::vector<uint32_t>{1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"shape", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(shapeAttr)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        ADD_FAILURE() << "Failed to create the tiling context";
        return result;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    std::map<std::string, std::string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);

    std::string opType("UpdateTensorDesc");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        ADD_FAILURE() << "UpdateTensorDesc tiling callbacks are not registered";
        return result;
    }
    auto tilingFunc = opImpl->tiling;

    result.status = tilingFunc(tilingContext);
    result.blockDim = tilingContext->GetBlockDim();
    if (tilingContext->GetWorkspaceSizes(1) != nullptr) {
        result.workspaceSize = tilingContext->GetWorkspaceSizes(1)[0];
    }
    auto* rawTilingData = tilingContext->GetRawTilingData();
    if (result.status == ge::GRAPH_SUCCESS && rawTilingData != nullptr &&
        rawTilingData->GetDataSize() >= sizeof(UpdateTensorDescTilingData)) {
        std::memcpy(&result.tilingData, rawTilingData->GetData(), sizeof(UpdateTensorDescTilingData));
    }
    return result;
}
} // namespace

class UpdateTensorDescTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UpdateTensorDescTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "UpdateTensorDescTiling TearDown" << std::endl; }
};

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test1)
{
    // 常规场景：rank3 attr、FP32 输入、单核 RMW
    const auto r = RunTilingCase({2, 3}, ge::DT_FLOAT, {4, 8, 4}, ge::DT_INT64, {4, 8, 4});
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.blockDim, 1U);
    EXPECT_EQ(r.workspaceSize, 0U);
    EXPECT_EQ(r.tilingData.rank, 3);
    EXPECT_EQ(r.tilingData.shape[0], 4);
    EXPECT_EQ(r.tilingData.shape[1], 8);
    EXPECT_EQ(r.tilingData.shape[2], 4);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test2)
{
    // rank1 最小 numel 场景 + rank0 输入（scalar x）
    const auto r = RunTilingCase({}, ge::DT_INT64, {128}, ge::DT_INT64, {128});
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.blockDim, 1U);
    EXPECT_EQ(r.tilingData.rank, 1);
    EXPECT_EQ(r.tilingData.shape[0], 128);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test3)
{
    // x rank8（上限）+ attr rank5
    const auto r = RunTilingCase({2, 2, 2, 2, 2, 2, 2, 2}, ge::DT_UINT8, {2, 2, 2, 2, 8}, ge::DT_INT64,
                                 {2, 2, 2, 2, 8});
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingData.rank, 5);
    EXPECT_EQ(r.tilingData.shape[4], 8);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test4)
{
    // 非法场景：x dtype 不在支持的 12 种 dtype 集合内（bfloat16）
    const auto r = RunTilingCase({2, 3}, ge::DT_BF16, {4, 8, 4}, ge::DT_INT64, {4, 8, 4});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test5)
{
    // 非法场景：y dtype 非 INT64
    const auto r = RunTilingCase({2, 3}, ge::DT_FLOAT, {4, 8, 4}, ge::DT_FLOAT, {4, 8, 4});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test6)
{
    // 非法场景：rank(x) > 8
    const auto r = RunTilingCase({2, 2, 2, 2, 2, 2, 2, 2, 2}, ge::DT_FLOAT, {4, 8, 4}, ge::DT_INT64, {4, 8, 4});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test7)
{
    // 非法场景：rank(attr shape) = 0
    const auto r = RunTilingCase({2, 3}, ge::DT_FLOAT, {128}, ge::DT_INT64, {});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test8)
{
    // 非法场景：attr shape 存在负维度
    const auto r = RunTilingCase({2, 3}, ge::DT_FLOAT, {4, -8, 4}, ge::DT_INT64, {4, -8, 4});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test9)
{
    // 非法场景：numel(attr shape) < 128
    const auto r = RunTilingCase({2, 3}, ge::DT_FLOAT, {4, 4}, ge::DT_INT64, {4, 4});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test10)
{
    // 非法场景：numel(y TensorDesc) < 128
    const auto r = RunTilingCase({2, 3}, ge::DT_FLOAT, {8}, ge::DT_INT64, {128});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test11)
{
    // 动态 shape：x desc 含 -1（未知维），tiling 仅校验 rank(x)，x 为占位输入不影响 tiling
    const auto r = RunTilingCase({-1, -1}, ge::DT_FLOAT, {4, 8, 4}, ge::DT_INT64, {4, 8, 4});
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.blockDim, 1U);
    EXPECT_EQ(r.tilingData.rank, 3);
    EXPECT_EQ(r.tilingData.shape[2], 4);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test12)
{
    // 动态 shape：x desc 为 [-2]（未知秩），rank(x)=1 未超限，不影响 tiling
    const auto r = RunTilingCase({-2}, ge::DT_FLOAT, {128}, ge::DT_INT64, {128});
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingData.rank, 1);
    EXPECT_EQ(r.tilingData.shape[0], 128);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test13)
{
    // 非法场景：attr shape 含 -1（未知维），host 侧校验链 fail-fast 拒绝
    const auto r = RunTilingCase({-1, -1}, ge::DT_FLOAT, {-1, 128}, ge::DT_INT64, {-1, 128});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescTiling, update_tensor_desc_tiling_test14)
{
    // 非法场景：attr shape 为 [-2]（未知秩），host 侧校验链 fail-fast 拒绝
    const auto r = RunTilingCase({-2}, ge::DT_FLOAT, {-2}, ge::DT_INT64, {-2});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}
