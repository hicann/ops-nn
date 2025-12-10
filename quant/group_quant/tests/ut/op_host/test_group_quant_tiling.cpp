/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include "log/log.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

struct GroupQuantCompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

class GroupQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupQuantTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tilingData)
{
    auto data = tilingData->GetData();
    string result;
    for (size_t i = 0; i < tilingData->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tilingData->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

static void InitPlatForm(
    fe::PlatFormInfos& platformInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
    map<string, string>& intrinsics)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    platformInfo.Init();
}

void GroupQuantTilingSimpleTest(
    const int64_t dimS, const int64_t dimE, const int64_t dimH, const ge::DataType xDtype,
    const ge::DataType scaleDtype, const ge::DataType groupIndexDtype, const ge::DataType offsetDtype,
    const ge::DataType yDtype, const int64_t attrDstType, const bool hasOffset, const ge::graphStatus expectStatus,
    const uint64_t expectTilingKey, const int64_t expectBlockDim, const string expectTilingData)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910B"}};
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

    std::initializer_list<int64_t> xShape = {dimS, dimH};
    std::initializer_list<int64_t> scaleShape = {dimE, dimH};
    std::initializer_list<int64_t> groupIndexShape = {
        dimE,
    };
    std::initializer_list<int64_t> offsetShape = {
        1,
    };
    std::initializer_list<int64_t> yShape = {dimS, dimH};
    gert::StorageShape xStorageShape = {xShape, xShape};
    gert::StorageShape scaleStorageShape = {scaleShape, scaleShape};
    gert::StorageShape groupIndexStorageShape = {groupIndexShape, groupIndexShape};
    gert::StorageShape offsetStorageShape = {offsetShape, offsetShape};
    gert::StorageShape outStorageShape = {yShape, yShape};

    GroupQuantCompileInfo compileInfo;
    std::string opType("GroupQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernelHolder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
            .Outputs({&compileInfo})
            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 1024 * 1024);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&xStorageShape, &scaleStorageShape, &groupIndexStorageShape, &offsetStorageShape})
                      .OutputShapes({&outStorageShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(attrDstType)},
                      })
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tilingFunc(tilingContext), expectStatus);

    auto blockDim = tilingContext->GetBlockDim();
    ASSERT_EQ(blockDim, expectBlockDim);

    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    ASSERT_EQ(tilingDataResult, expectTilingData);
}

TEST_F(GroupQuantTiling, tiling_demo_case)
{
    int64_t dimS = 6;
    int64_t dimE = 4;
    int64_t dimH = 4;
    int64_t attrDstType = 2;
    bool hasOffset = true;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    uint64_t expectTilingKey = 0;
    int64_t expectBlockDim = 6;
    string expectTilingData = "6 4 4 1 6 6 1 1 ";
    GroupQuantTilingSimpleTest(
        dimS, dimE, dimH, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT8, attrDstType, hasOffset,
        expectStatus, expectTilingKey, expectBlockDim, expectTilingData);
}

TEST_F(GroupQuantTiling, tiling_net_case_full_s5120_e32_h1024)
{
    int64_t dimS = 5120;
    int64_t dimE = 32;
    int64_t dimH = 1024;
    int64_t attrDstType = 2;
    bool hasOffset = true;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    uint64_t expectTilingKey = 0;
    int64_t expectBlockDim = 48;
    string expectTilingData = "5120 32 1024 1 48 32 107 106 ";
    GroupQuantTilingSimpleTest(
        dimS, dimE, dimH, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT8, attrDstType, hasOffset,
        expectStatus, expectTilingKey, expectBlockDim, expectTilingData);
}

TEST_F(GroupQuantTiling, tiling_net_case_incremental_s80_e32_h1024)
{
    int64_t dimS = 80;
    int64_t dimE = 32;
    int64_t dimH = 1024;
    int64_t attrDstType = 2;
    bool hasOffset = true;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    uint64_t expectTilingKey = 0;
    int64_t expectBlockDim = 48;
    string expectTilingData = "80 32 1024 1 48 32 2 1 ";
    GroupQuantTilingSimpleTest(
        dimS, dimE, dimH, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT8, attrDstType, hasOffset,
        expectStatus, expectTilingKey, expectBlockDim, expectTilingData);
}
