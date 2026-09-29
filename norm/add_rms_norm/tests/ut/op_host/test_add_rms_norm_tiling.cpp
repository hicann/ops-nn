/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <limits>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "test_add_rms_norm_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class AddRmsNormTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddRmsNormTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AddRmsNormTiling TearDown" << std::endl; }
};

struct SplitARTilingDataForTest {
    uint64_t numRow;
    uint64_t numCol;
    uint64_t rBlockFactor;
    uint64_t rBlockNum;
    uint64_t tailR;
    uint64_t binAddQuotient;
    float epsilon;
    float avgFactor;
    uint64_t ubFactor;
    uint64_t blockTailBinAddQuotient;
    uint64_t tailBinAddQuotient;
    uint64_t basicBlockLoop;
    uint64_t mainFoldCount;
    uint64_t tailBasicBlockLoop;
    uint64_t tailMainFoldCount;
    uint64_t numRowAlign;
};

struct TransTilingDataForTest {
    uint64_t numRow;
    uint64_t numCol;
    uint64_t rAligned;
    uint64_t rTileBase;
    uint64_t totalTiles;
    uint64_t tilesPerCore;
    uint64_t tileALen;
    uint64_t tileATail;
    float epsilon;
    float avgFactor;
};

struct Arch35TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    size_t systemWorkspaceSize = 0;
    SplitARTilingDataForTest splitAR{};
    TransTilingDataForTest trans{};
};

Arch35TilingResult RunArch35Tiling(uint64_t numRow, uint64_t numCol, ge::DataType dataType,
                                   int32_t deterministicLevel = 0, uint64_t ubSizeOverride = 0)
{
    int64_t row = static_cast<int64_t>(numRow);
    int64_t col = static_cast<int64_t>(numCol);
    gert::StorageShape inputShapeX1 = {{row, col}, {row, col}};
    gert::StorageShape inputShapeX2 = {{row, col}, {row, col}};
    gert::StorageShape gammaShape = {{col}, {col}};
    gert::StorageShape outputShapeY = {{row, col}, {row, col}};
    gert::StorageShape rstdShape = {{row, 1}, {row, 1}};
    gert::StorageShape outputShapeX = {{row, col}, {row, col}};

    std::map<std::string, std::string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 56}
                          })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::AddRmsNormCompileInfo compileInfo;

    std::string opType("AddRmsNorm");
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr) {
        return {};
    }
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    auto* parseContext = kernelHolder.GetContext<gert::TilingParseContext>();
    EXPECT_TRUE(parseContext->GetPlatformInfo()->Init());
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parseContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    if (ubSizeOverride != 0) {
        // 仅供 Host Tiling 边界 UT 注入平台资源；默认 0 时仍使用真实 Ascend950 compile info。
        compileInfo.totalUbSize = ubSizeOverride;
    }

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    EXPECT_NE(param, nullptr);
    if (param == nullptr) {
        return {};
    }
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&inputShapeX1, &inputShapeX2, &gammaShape})
                      .OutputShapes({&outputShapeY, &rstdShape, &outputShapeX})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .DeterministicLevelInfo(deterministicLevel)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    Arch35TilingResult result;
    result.status = tilingFunc(tilingContext);
    result.tilingKey = tilingContext->GetTilingKey();
    result.blockDim = tilingContext->GetBlockDim();
    result.workspaceSize = *reinterpret_cast<const size_t*>(workspaceSize->GetData());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    result.systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (result.status == ge::GRAPH_SUCCESS && result.tilingKey == 3000) {
        result.splitAR = *reinterpret_cast<const SplitARTilingDataForTest*>(
            tilingContext->GetRawTilingData()->GetData());
    } else if (result.status == ge::GRAPH_SUCCESS && result.tilingKey == 4000) {
        result.trans = *reinterpret_cast<const TransTilingDataForTest*>(tilingContext->GetRawTilingData()->GetData());
    }
    return result;
}

// TEST_F(AddRmsNormTiling, add_rms_norm_tiling_001) {
//   //dlog_setlevel(0,0,0);
//   gert::StorageShape input_shape_x1 = {{24, 1, 2560}, {24, 1, 2560}};
//   gert::StorageShape input_shape_x2 = {{24, 1, 2560}, {24, 1, 2560}};
//   gert::StorageShape gamma_shape = {{2560,}, {2560,}};
//   gert::StorageShape out_shape_y = {{24, 1, 2560}, {24, 1, 2560}};
//   gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};
//   gert::StorageShape out_shape_x = {{24, 1, 2560}, {24, 1, 2560}};

//   string compile_info_string = R"({
//         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                           "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
//                           "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE":
//                           196608, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536,
//                           "L0C_SIZE": 131072, "CORE_NUM": 40}
//                           })";
//   map<string, string> soc_infos;
//   map<string, string> aicore_spec;
//   map<string, string> intrinsics;
//   GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//   // platform info
//   fe::PlatFormInfos platform_info;
//   platform_info.Init();
//     // compile info
//   struct AddRmsNormCompileInfo
//   {
//       uint32_t totalCoreNum = 40;
//       uint64_t totalUbSize = 196608;
//       platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
//   };
//   AddRmsNormCompileInfo compile_info;

//   std::string op_type("AddRmsNorm");
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//   // tilingParseFunc simulate
//   auto kernel_holder = gert::KernelRunContextFaker()
//                     .KernelIONum(2, 1)
//                     .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void
//                     *>(&platform_info)}) .Outputs({nullptr}) .Build();

//   ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//   intrinsics);

//   tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>());

//   // tilingFunc simulate
//   auto param = gert::TilingData::CreateCap(4096);
//   auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//   auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
//   ASSERT_NE(param, nullptr);
//   auto holder = gert::TilingContextFaker()
//                     .NodeIoNum(3, 3)
//                     .IrInstanceNum({1, 1, 1})
//                     .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
//                     .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
//                     .CompileInfo(nullptr)
//                     .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//                     .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
//                     .TilingData(param.get())
//                     .Workspace(ws_size)
//                     .Build();

//   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//   tiling_func(tiling_context);
//   // // workspaces nullptr return failed
//   // EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//   // // todo check tiling result
//   // auto tiling_key = tiling_context->GetTilingKey();
//   // ASSERT_EQ(tiling_key, 13);
//   //dlog_setlevel(0,3,0);
// }

// TEST_F(AddRmsNormTiling, add_rms_norm_tiling_002) {
//   //dlog_setlevel(0,0,0);
//   gert::StorageShape input_shape_x1 = {{24, 1, 25600}, {24, 1, 25600}};
//   gert::StorageShape input_shape_x2 = {{24, 1, 25600}, {24, 1, 25600}};
//   gert::StorageShape gamma_shape = {{25600,}, {25600,}};
//   gert::StorageShape out_shape_y = {{24, 1, 25600}, {24, 1, 25600}};
//   gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};
//   gert::StorageShape out_shape_x = {{24, 1, 25600}, {24, 1, 25600}};

//   string compile_info_string = R"({
//         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                           "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
//                           "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE":
//                           196608, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536,
//                           "L0C_SIZE": 131072, "CORE_NUM": 40}
//                           })";
//   map<string, string> soc_infos;
//   map<string, string> aicore_spec;
//   map<string, string> intrinsics;
//   GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//   // platform info
//   fe::PlatFormInfos platform_info;
//   platform_info.Init();
//     // compile info
//   struct AddRmsNormCompileInfo
//   {
//       uint32_t totalCoreNum = 40;
//       uint64_t totalUbSize = 196608;
//       platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
//   };
//   AddRmsNormCompileInfo compile_info;

//   std::string op_type("AddRmsNorm");
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//   // tilingParseFunc simulate
//   auto kernel_holder = gert::KernelRunContextFaker()
//                     .KernelIONum(2, 1)
//                     .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void
//                     *>(&platform_info)}) .Outputs({&compile_info}) .Build();

//   ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//   intrinsics);

//   ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//   // tilingFunc simulate
//   auto param = gert::TilingData::CreateCap(4096);
//   auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//   auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
//   ASSERT_NE(param, nullptr);
//   auto holder = gert::TilingContextFaker()
//                     .NodeIoNum(3, 3)
//                     .IrInstanceNum({1, 1, 1})
//                     .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
//                     .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
//                     .CompileInfo(&compile_info)
//                     .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//                     .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
//                     .TilingData(param.get())
//                     .Workspace(ws_size)
//                     .Build();

//   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//   // workspaces nullptr return failed
//   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//   // todo check tiling result
//   auto tiling_key = tiling_context->GetTilingKey();
//   ASSERT_EQ(tiling_key, 11);
//   //dlog_setlevel(0,3,0);
// }

// TEST_F(AddRmsNormTiling, add_rms_norm_tiling_003) {
//   //dlog_setlevel(0,0,0);
//   gert::StorageShape input_shape_x1 = {{24, 1, 256}, {24, 1, 256}};
//   gert::StorageShape input_shape_x2 = {{24, 1, 256}, {24, 1, 256}};
//   gert::StorageShape gamma_shape = {{256,}, {256,}};
//   gert::StorageShape out_shape_y = {{24, 1, 256}, {24, 1, 256}};
//   gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};
//   gert::StorageShape out_shape_x = {{24, 1, 256}, {24, 1, 256}};

//   string compile_info_string = R"({
//         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                           "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
//                           "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE":
//                           196608, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536,
//                           "L0C_SIZE": 131072, "CORE_NUM": 40}
//                           })";
//   map<string, string> soc_infos;
//   map<string, string> aicore_spec;
//   map<string, string> intrinsics;
//   GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//   // platform info
//   fe::PlatFormInfos platform_info;
//   platform_info.Init();
//     // compile info
//   struct AddRmsNormCompileInfo
//   {
//       uint32_t totalCoreNum = 40;
//       uint64_t totalUbSize = 196608;
//       platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
//   };
//   AddRmsNormCompileInfo compile_info;

//   std::string op_type("AddRmsNorm");
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//   // tilingParseFunc simulate
//   auto kernel_holder = gert::KernelRunContextFaker()
//                     .KernelIONum(2, 1)
//                     .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void
//                     *>(&platform_info)}) .Outputs({&compile_info}) .Build();

//   ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//   intrinsics);

//   ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//   // tilingFunc simulate
//   auto param = gert::TilingData::CreateCap(4096);
//   auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//   auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
//   ASSERT_NE(param, nullptr);
//   auto holder = gert::TilingContextFaker()
//                     .NodeIoNum(3, 3)
//                     .IrInstanceNum({1, 1, 1})
//                     .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
//                     .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
//                     .CompileInfo(&compile_info)
//                     .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//                     .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
//                     .TilingData(param.get())
//                     .Workspace(ws_size)
//                     .Build();

//   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//   // workspaces nullptr return failed
//   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//   // todo check tiling result
//   auto tiling_key = tiling_context->GetTilingKey();
//   ASSERT_EQ(tiling_key, 23);
//   //dlog_setlevel(0,3,0);
// }

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape_x1 = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape input_shape_x2 = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape gamma_shape = {{
                                          2560,
                                      },
                                      {
                                          2560,
                                      }};
    gert::StorageShape out_shape_y = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};
    gert::StorageShape out_shape_x = {{24, 1, 2560}, {24, 1, 2560}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "2201"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 13);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_002)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape_x1 = {{24, 1, 25600}, {24, 1, 25600}};
    gert::StorageShape input_shape_x2 = {{24, 1, 25600}, {24, 1, 25600}};
    gert::StorageShape gamma_shape = {{
                                          25600,
                                      },
                                      {
                                          25600,
                                      }};
    gert::StorageShape out_shape_y = {{24, 1, 25600}, {24, 1, 25600}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};
    gert::StorageShape out_shape_x = {{24, 1, 25600}, {24, 1, 25600}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "2201"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 11);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_003)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape_x1 = {{24, 1, 256}, {24, 1, 256}};
    gert::StorageShape input_shape_x2 = {{24, 1, 256}, {24, 1, 256}};
    gert::StorageShape gamma_shape = {{
                                          256,
                                      },
                                      {
                                          256,
                                      }};
    gert::StorageShape out_shape_y = {{24, 1, 256}, {24, 1, 256}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};
    gert::StorageShape out_shape_x = {{24, 1, 256}, {24, 1, 256}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "2201"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 23);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_004)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape_x1 = {{4, 1, 8192}, {4, 1, 8192}};
    gert::StorageShape input_shape_x2 = {{4, 1, 8192}, {4, 1, 8192}};
    gert::StorageShape gamma_shape = {{
                                          8192,
                                      },
                                      {
                                          8192,
                                      }};
    gert::StorageShape out_shape_y = {{4, 1, 8192}, {4, 1, 8192}};
    gert::StorageShape rstd_shape = {{4, 1, 1}, {4, 1, 1}};
    gert::StorageShape out_shape_x = {{4, 1, 8192}, {4, 1, 8192}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    // R 工作量不足以满足 2-block/8-tile 准入，回退 SplitD。
    ASSERT_EQ(tiling_key, 2000);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_005)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape_x1 = {{4096, 1, 2304}, {4096, 1, 2304}};
    gert::StorageShape input_shape_x2 = {{4096, 1, 2304}, {4096, 1, 2304}};
    gert::StorageShape gamma_shape = {{
                                          2304,
                                      },
                                      {
                                          2304,
                                      }};
    gert::StorageShape out_shape_y = {{4096, 1, 2304}, {4096, 1, 2304}};
    gert::StorageShape rstd_shape = {{4096, 1, 1}, {4096, 1, 1}};
    gert::StorageShape out_shape_x = {{4096, 1, 2304}, {4096, 1, 2304}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_006)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape_x1 = {{409, 1, 1281}, {409, 1, 1281}};
    gert::StorageShape input_shape_x2 = {{409, 1, 1281}, {409, 1, 1281}};
    gert::StorageShape gamma_shape = {{
                                          1281,
                                      },
                                      {
                                          1281,
                                      }};
    gert::StorageShape out_shape_y = {{409, 1, 1281}, {409, 1, 1281}};
    gert::StorageShape rstd_shape = {{409, 1, 1}, {409, 1, 1}};
    gert::StorageShape out_shape_x = {{409, 1, 1281}, {409, 1, 1281}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "2201"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 32);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_1001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape_x1 = {{1024, 1, 12288}, {1024, 1, 12288}};
    gert::StorageShape input_shape_x2 = {{1024, 1, 12288}, {1024, 1, 12288}};
    gert::StorageShape gamma_shape = {{
                                          12288,
                                      },
                                      {
                                          12288,
                                      }};
    gert::StorageShape out_shape_y = {{1024, 1, 12288}, {1024, 1, 12288}};
    gert::StorageShape rstd_shape = {{1024, 1, 1}, {1024, 1, 1}};
    gert::StorageShape out_shape_x = {{1024, 1, 12288}, {1024, 1, 12288}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 2000);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_1)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape_x1 = {{16, 32, 768}, {16, 32, 768}};
    gert::StorageShape input_shape_x2 = {{16, 32, 768}, {16, 32, 768}};
    gert::StorageShape gamma_shape = {{
                                          768,
                                      },
                                      {
                                          768,
                                      }};
    gert::StorageShape out_shape_y = {{16, 32, 768}, {16, 32, 768}};
    gert::StorageShape rstd_shape = {{16, 32, 1}, {16, 32, 1}};
    gert::StorageShape out_shape_x = {{16, 32, 768}, {16, 32, 768}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_2)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape_x1 = {{16, 128, 768}, {16, 128, 768}};
    gert::StorageShape input_shape_x2 = {{16, 128, 768}, {16, 128, 768}};
    gert::StorageShape gamma_shape = {{
                                          768,
                                      },
                                      {
                                          768,
                                      }};
    gert::StorageShape out_shape_y = {{16, 128, 768}, {16, 128, 768}};
    gert::StorageShape rstd_shape = {{16, 128, 1}, {16, 128, 1}};
    gert::StorageShape out_shape_x = {{16, 128, 768}, {16, 128, 768}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_007)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape_x1 = {{126, 1, 5120}, {126, 1, 5120}};
    gert::StorageShape input_shape_x2 = {{126, 1, 5120}, {126, 1, 5120}};
    gert::StorageShape gamma_shape = {{
                                          5120,
                                      },
                                      {
                                          5120,
                                      }};
    gert::StorageShape out_shape_y = {{126, 1, 5120}, {126, 1, 5120}};
    gert::StorageShape rstd_shape = {{126, 1, 1}, {126, 1, 1}};
    gert::StorageShape out_shape_x = {{126, 1, 5120}, {126, 1, 5120}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "2201"}};
    string compile_info_string = R"({
 	         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
 	                           "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
 	                           "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
 	                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
 	                           "CORE_NUM": 40}
 	                           })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 30);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_transpose_001)
{
    // 64 核上大 A 边界为 8*64*64=32768；A=32769、R=16 进入 Trans。
    gert::StorageShape input_shape_x1 = {{32769, 16}, {32769, 16}};
    gert::StorageShape input_shape_x2 = {{32769, 16}, {32769, 16}};
    gert::StorageShape gamma_shape = {{16}, {16}};
    gert::StorageShape out_shape_y = {{32769, 16}, {32769, 16}};
    gert::StorageShape rstd_shape = {{32769, 1}, {32769, 1}};
    gert::StorageShape out_shape_x = {{32769, 16}, {32769, 16}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 4000);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_split_ar_001)
{
    // 大 R 非整除回归：校验四个静态 DB 分块、tailR 及三类折叠点。
    gert::StorageShape input_shape_x1 = {{1, 294913}, {1, 294913}};
    gert::StorageShape input_shape_x2 = {{1, 294913}, {1, 294913}};
    gert::StorageShape gamma_shape = {{294913}, {294913}};
    gert::StorageShape out_shape_y = {{1, 294913}, {1, 294913}};
    gert::StorageShape rstd_shape = {{1, 1}, {1, 1}};
    gert::StorageShape out_shape_x = {{1, 294913}, {1, 294913}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3000);
    EXPECT_EQ(tiling_context->GetScheduleMode(), 1);
    const auto* splitARTiling = reinterpret_cast<const SplitARTilingDataForTest*>(
        tiling_context->GetRawTilingData()->GetData());
    ASSERT_NE(splitARTiling, nullptr);
    EXPECT_EQ(tiling_context->GetBlockDim(), 4);
    EXPECT_EQ(splitARTiling->rBlockNum, 4);
    EXPECT_EQ((splitARTiling->rBlockNum - 1) * splitARTiling->rBlockFactor + splitARTiling->tailR, 294913);
    EXPECT_EQ(splitARTiling->ubFactor, 8192);
    EXPECT_GE(splitARTiling->rBlockFactor / splitARTiling->ubFactor, 8);
    EXPECT_GT(splitARTiling->binAddQuotient, 0);
    EXPECT_GT(splitARTiling->blockTailBinAddQuotient, 0);
    EXPECT_GT(splitARTiling->tailBinAddQuotient, 0);
    EXPECT_GT(splitARTiling->basicBlockLoop, 0);
    EXPECT_LE(splitARTiling->mainFoldCount, splitARTiling->basicBlockLoop);
    EXPECT_GT(splitARTiling->tailBasicBlockLoop, 0);
    EXPECT_LE(splitARTiling->tailMainFoldCount, splitARTiling->tailBasicBlockLoop);
    EXPECT_EQ(splitARTiling->numRowAlign, 8);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_transpose_batch_invariant)
{
    // batch一致性时R=16不走转置模板: 期望 key=1000 (R全载)
    gert::StorageShape input_shape_x1 = {{64, 16}, {64, 16}};
    gert::StorageShape input_shape_x2 = {{64, 16}, {64, 16}};
    gert::StorageShape gamma_shape = {{16}, {16}};
    gert::StorageShape out_shape_y = {{64, 16}, {64, 16}};
    gert::StorageShape rstd_shape = {{64, 1}, {64, 1}};
    gert::StorageShape out_shape_x = {{64, 16}, {64, 16}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .DeterministicLevelInfo(3)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    // batch 一致性时禁用整个 Trans 家族，走 RFullLoad。
    ASSERT_EQ(tiling_key, 1000);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_transpose_fp32)
{
    // Trans 的 R 性能边界按一行字节数不超过一个 UB block 表达。Ascend 950 block=32B，
    // FP32 每元素 4B，因此 R=8 是正向边界；64 核上 A=32769 选择 key=4000。
    gert::StorageShape input_shape_x1 = {{32769, 8}, {32769, 8}};
    gert::StorageShape input_shape_x2 = {{32769, 8}, {32769, 8}};
    gert::StorageShape gamma_shape = {{8}, {8}};
    gert::StorageShape out_shape_y = {{32769, 8}, {32769, 8}};
    gert::StorageShape rstd_shape = {{32769, 1}, {32769, 1}};
    gert::StorageShape out_shape_x = {{32769, 8}, {32769, 8}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    // FP32 R=8 走转置(4000)。
    ASSERT_EQ(tiling_key, 4000);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_split_ar_batch_invariant)
{
    // R 已超过 RFullLoad 容量，验证 batch 一致性模式仍会禁用 key=3000。
    gert::StorageShape input_shape_x1 = {{1, 294912}, {1, 294912}};
    gert::StorageShape input_shape_x2 = {{1, 294912}, {1, 294912}};
    gert::StorageShape gamma_shape = {{294912}, {294912}};
    gert::StorageShape out_shape_y = {{1, 294912}, {1, 294912}};
    gert::StorageShape rstd_shape = {{1, 1}, {1, 1}};
    gert::StorageShape out_shape_x = {{1, 294912}, {1, 294912}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .DeterministicLevelInfo(3)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    // batch 一致性时禁用 SplitAR，回退 SplitD。
    ASSERT_EQ(tiling_key, 2000);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_transpose_bf16)
{
    // bf16 R=16 且 A=32769 走转置模板: 期望 key=4000。
    gert::StorageShape input_shape_x1 = {{32769, 16}, {32769, 16}};
    gert::StorageShape input_shape_x2 = {{32769, 16}, {32769, 16}};
    gert::StorageShape gamma_shape = {{16}, {16}};
    gert::StorageShape out_shape_y = {{32769, 16}, {32769, 16}};
    gert::StorageShape rstd_shape = {{32769, 1}, {32769, 1}};
    gert::StorageShape out_shape_x = {{32769, 16}, {32769, 16}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 4000);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_transpose_small_r)
{
    // R=8 < 16，A=32769 超过 64 核大 A 门槛：期望 key=4000。
    gert::StorageShape input_shape_x1 = {{32769, 8}, {32769, 8}};
    gert::StorageShape input_shape_x2 = {{32769, 8}, {32769, 8}};
    gert::StorageShape gamma_shape = {{8}, {8}};
    gert::StorageShape out_shape_y = {{32769, 8}, {32769, 8}};
    gert::StorageShape rstd_shape = {{32769, 1}, {32769, 1}};
    gert::StorageShape out_shape_x = {{32769, 8}, {32769, 8}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 4000);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_split_ar_fp32)
{
    // fp32 在每核 32768 元素、每行四个 R block 的性能边界进入 SplitAR。
    gert::StorageShape input_shape_x1 = {{4, 131072}, {4, 131072}};
    gert::StorageShape input_shape_x2 = {{4, 131072}, {4, 131072}};
    gert::StorageShape gamma_shape = {{131072}, {131072}};
    gert::StorageShape out_shape_y = {{4, 131072}, {4, 131072}};
    gert::StorageShape rstd_shape = {{4, 1}, {4, 1}};
    gert::StorageShape out_shape_x = {{4, 131072}, {4, 131072}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3000);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_split_ar_bf16)
{
    // SplitAR bf16 大 R 回归：numRow=2，R=294912，期望 key=3000。
    gert::StorageShape input_shape_x1 = {{2, 294912}, {2, 294912}};
    gert::StorageShape input_shape_x2 = {{2, 294912}, {2, 294912}};
    gert::StorageShape gamma_shape = {{294912}, {294912}};
    gert::StorageShape out_shape_y = {{2, 294912}, {2, 294912}};
    gert::StorageShape rstd_shape = {{2, 1}, {2, 1}};
    gert::StorageShape out_shape_x = {{2, 294912}, {2, 294912}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3000);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_reduce_empty)
{
    // reduce empty: R=0 且 A=16, 期望 key=5000 空模板, 活量 64B < 32KB 单核
    gert::StorageShape input_shape_x1 = {{16, 0}, {16, 0}};
    gert::StorageShape input_shape_x2 = {{16, 0}, {16, 0}};
    gert::StorageShape gamma_shape = {{0}, {0}};
    gert::StorageShape out_shape_y = {{16, 0}, {16, 0}};
    gert::StorageShape rstd_shape = {{16, 1}, {16, 1}};
    gert::StorageShape out_shape_x = {{16, 0}, {16, 0}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 5000);
    ASSERT_EQ(holder.GetContext<gert::TilingContext>()->GetBlockDim(), 1);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_performance_template_boundaries)
{
    using BoundaryCase = std::tuple<const char*, ge::DataType, uint64_t, uint64_t, uint64_t>;
    const std::vector<BoundaryCase> cases = {
        // Trans 只保留大 A 端：A 必须严格大于 8*VL_FP32*numCore。
        {"trans_fp16_a_one", ge::DT_FLOAT16, 1, 16, 1000},
        {"trans_fp16_physical_before", ge::DT_FLOAT16, 2, 16, 1000},
        {"trans_fp16_physical_at", ge::DT_FLOAT16, 3, 16, 1000},
        {"trans_fp16_small_upper", ge::DT_FLOAT16, 168, 16, 1000},
        {"trans_fp16_middle_first", ge::DT_FLOAT16, 169, 16, 1000},
        {"trans_fp16_middle_220", ge::DT_FLOAT16, 220, 16, 1000},
        {"trans_fp16_middle_221", ge::DT_FLOAT16, 221, 16, 1000},
        {"trans_fp16_middle_224", ge::DT_FLOAT16, 224, 16, 1000},
        {"trans_fp16_middle_225", ge::DT_FLOAT16, 225, 16, 1000},
        {"trans_fp16_medium_old_margin", ge::DT_FLOAT16, 14336, 8, 1000},
        {"trans_fp16_medium_before", ge::DT_FLOAT16, 17919, 8, 1000},
        {"trans_fp16_excluded_middle_tiny_r", ge::DT_FLOAT16, 17920, 8, 1000},
        {"trans_fp16_excluded_middle_max_r", ge::DT_FLOAT16, 17920, 16, 1000},
        {"trans_fp16_large_before", ge::DT_FLOAT16, 28671, 16, 1000},
        {"trans_fp16_large_boundary", ge::DT_FLOAT16, 28672, 16, 1000},
        {"trans_fp16_large_first", ge::DT_FLOAT16, 28673, 16, 4000},
        {"trans_fp16_multi_tile", ge::DT_FLOAT16, 100000, 16, 4000},
        {"trans_fp16_unbounded_a", ge::DT_FLOAT16, 10000000, 1, 4000},
        {"trans_fp16_r_out", ge::DT_FLOAT16, 168, 17, 1000},
        {"trans_bf16_multi_tile", ge::DT_BF16, 100000, 8, 4000},
        {"trans_fp32_a_one", ge::DT_FLOAT, 1, 8, 1000},
        {"trans_fp32_physical_before", ge::DT_FLOAT, 2, 8, 1000},
        {"trans_fp32_physical_at", ge::DT_FLOAT, 3, 8, 1000},
        {"trans_fp32_small_upper", ge::DT_FLOAT, 168, 8, 1000},
        {"trans_fp32_middle_first", ge::DT_FLOAT, 169, 8, 1000},
        {"trans_fp32_middle_224", ge::DT_FLOAT, 224, 8, 1000},
        {"trans_fp32_middle_225", ge::DT_FLOAT, 225, 8, 1000},
        {"trans_fp32_medium_old_margin", ge::DT_FLOAT, 14336, 4, 1000},
        {"trans_fp32_medium_before", ge::DT_FLOAT, 17919, 4, 1000},
        {"trans_fp32_excluded_middle_tiny_r", ge::DT_FLOAT, 17920, 4, 1000},
        {"trans_fp32_excluded_middle_max_r", ge::DT_FLOAT, 17920, 8, 1000},
        {"trans_fp32_r_out", ge::DT_FLOAT, 168, 9, 1000},
        {"trans_fp32_large_boundary", ge::DT_FLOAT, 28672, 8, 1000},
        {"trans_fp32_large_first", ge::DT_FLOAT, 28673, 8, 4000},
        {"trans_fp32_multi_tile", ge::DT_FLOAT, 100000, 8, 4000},

        // SplitAR 要求 A<=numCore/2、至少两个 R block，且每核跨 A 累计至少八个 DB tile。
        {"split_fp16_full_last", ge::DT_FLOAT16, 1, 11424, 1000},
        {"split_fp16_before", ge::DT_FLOAT16, 1, 131071, 2000},
        {"split_fp16_at", ge::DT_FLOAT16, 1, 131072, 3000},
        {"split_fp16_last_a", ge::DT_FLOAT16, 28, 131072, 3000},
        {"split_fp16_a_out", ge::DT_FLOAT16, 29, 131072, 2000},
        {"split_fp16_large_r", ge::DT_FLOAT16, 1, 1048577, 3000},
        {"split_bf16_full_last", ge::DT_BF16, 1, 11424, 1000},
        {"split_bf16_before", ge::DT_BF16, 1, 131071, 2000},
        {"split_bf16_at", ge::DT_BF16, 1, 131072, 3000},
        {"split_bf16_last_a", ge::DT_BF16, 28, 131072, 3000},
        {"split_bf16_large_r", ge::DT_BF16, 3, 1048577, 3000},
        {"split_bf16_a_out", ge::DT_BF16, 29, 131072, 2000},
        {"split_fp32_full_last", ge::DT_FLOAT, 1, 6288, 1000},
        {"split_fp32_before", ge::DT_FLOAT, 1, 65535, 2000},
        {"split_fp32_at", ge::DT_FLOAT, 1, 65536, 3000},
        {"split_fp32_last_a", ge::DT_FLOAT, 28, 65536, 3000},
        {"split_fp32_plus_one", ge::DT_FLOAT, 1, 65537, 3000},
        {"split_fp32_large_r", ge::DT_FLOAT, 5, 1048577, 3000},
        {"split_fp32_a_out", ge::DT_FLOAT, 29, 65536, 2000},
    };

    for (const auto& testCase : cases) {
        SCOPED_TRACE(std::get<0>(testCase));
        Arch35TilingResult result = RunArch35Tiling(std::get<2>(testCase), std::get<3>(testCase),
                                                    std::get<1>(testCase));
        EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(result.tilingKey, std::get<4>(testCase));
    }
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_new_template_abi_layout)
{
    // 校验两个新增 key 的关键字段布局。
    Arch35TilingResult transResult = RunArch35Tiling(28673, 16, ge::DT_FLOAT16);
    ASSERT_EQ(transResult.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(transResult.tilingKey, 4000);
    EXPECT_EQ(transResult.trans.numRow, 28673);
    EXPECT_EQ(transResult.trans.numCol, 16);
    EXPECT_EQ(transResult.trans.rAligned, 16);
    EXPECT_EQ(transResult.trans.rTileBase, 16);
    EXPECT_EQ(transResult.trans.totalTiles, 106);
    EXPECT_EQ(transResult.trans.tilesPerCore, 2);
    EXPECT_EQ(transResult.trans.tileALen, 272);
    EXPECT_EQ(transResult.trans.tileATail, 113);
    EXPECT_FLOAT_EQ(transResult.trans.epsilon, 0.01F);
    EXPECT_FLOAT_EQ(transResult.trans.avgFactor, 1.0F / 16.0F);

    Arch35TilingResult splitResult = RunArch35Tiling(2, 262144, ge::DT_FLOAT16);
    ASSERT_EQ(splitResult.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(splitResult.tilingKey, 3000);
    EXPECT_EQ(splitResult.splitAR.numRow, 2);
    EXPECT_EQ(splitResult.splitAR.numCol, 262144);
    EXPECT_GT(splitResult.splitAR.binAddQuotient, 0);
    EXPECT_FLOAT_EQ(splitResult.splitAR.epsilon, 0.01F);
    EXPECT_FLOAT_EQ(splitResult.splitAR.avgFactor, 1.0F / 262144.0F);
    EXPECT_GT(splitResult.splitAR.ubFactor, 0);
    EXPECT_GT(splitResult.splitAR.basicBlockLoop, 0);
    EXPECT_GT(splitResult.splitAR.tailBasicBlockLoop, 0);
    const size_t expectedUserWorkspace = splitResult.splitAR.rBlockNum * splitResult.splitAR.numRowAlign *
                                         sizeof(float);
    EXPECT_EQ(splitResult.workspaceSize, splitResult.systemWorkspaceSize + expectedUserWorkspace);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_split_ar_int64_max)
{
    // 只执行 host tiling，不分配巨型 tensor；验证 R 的 int64_t 上界不溢出且不被截断。
    const uint64_t maxR = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    const std::vector<ge::DataType> dataTypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
    for (ge::DataType dataType : dataTypes) {
        SCOPED_TRACE(dataType);
        Arch35TilingResult result = RunArch35Tiling(1, maxR, dataType);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.tilingKey, 3000);
        EXPECT_EQ(result.splitAR.numRow, 1);
        EXPECT_EQ(result.splitAR.numCol, maxR);
        ASSERT_GE(result.splitAR.rBlockNum, 4);
        EXPECT_EQ(result.blockDim, result.splitAR.rBlockNum);
        EXPECT_EQ((result.splitAR.rBlockNum - 1) * result.splitAR.rBlockFactor + result.splitAR.tailR, maxR);
        EXPECT_GT(result.splitAR.ubFactor, 0);
        EXPECT_LE(result.splitAR.ubFactor, result.splitAR.rBlockFactor);
    }
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_split_ar_total_elements_over_int32)
{
    // A*R=2,147,483,660，刚好越过 INT32_MAX；只执行 Host Tiling，避免为边界测试分配超大张量。
    constexpr uint64_t numRow = 28;
    constexpr uint64_t numCol = 76695845;
    constexpr uint64_t totalElements = numRow * numCol;
    static_assert(totalElements > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));

    Arch35TilingResult result = RunArch35Tiling(numRow, numCol, ge::DT_FLOAT16);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(result.tilingKey, 3000);
    EXPECT_EQ(result.splitAR.numRow, numRow);
    EXPECT_EQ(result.splitAR.numCol, numCol);
    EXPECT_EQ(result.blockDim, 56);
    EXPECT_EQ((result.splitAR.rBlockNum - 1) * result.splitAR.rBlockFactor + result.splitAR.tailR, numCol);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_split_d_rejects_unencodable_int64_max)
{
    // 确定性模式会跳过 SplitAR；旧 key=2000 的 uint32_t ABI 无法表示该 R，应快速失败而非截断或死循环。
    uint64_t maxR = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    Arch35TilingResult result = RunArch35Tiling(1, maxR, ge::DT_FLOAT, 3);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_split_ar_core_and_static_db_strategy)
{
    struct StrategyCase {
        ge::DataType dataType;
        uint64_t numRow;
        uint64_t numCol;
        uint64_t expectedKey;
        uint32_t expectedBlockDim;
        uint64_t expectedUbFactor;
    };
    const std::vector<StrategyCase> cases = {
        // FP32：单个 DB tile 为 4096 元素，两个 8-tile R block 从 65536 开始；A=28
        // 仍可把 56 个 R block 的总工作均摊到每核 8 tile，A=29 超过 numCore/2 后回退。
        {ge::DT_FLOAT, 1, 65535, 2000, 0, 0},
        {ge::DT_FLOAT, 1, 65536, 3000, 2, 4096},
        {ge::DT_FLOAT, 28, 65536, 3000, 56, 4096},
        {ge::DT_FLOAT, 29, 65536, 2000, 0, 0},
        // FP16/BF16 的单个 DB tile 为 8192 元素，对应整行边界为 131072；A 上界相同。
        {ge::DT_FLOAT16, 1, 131071, 2000, 0, 0},
        {ge::DT_FLOAT16, 1, 131072, 3000, 2, 8192},
        {ge::DT_BF16, 28, 131072, 3000, 56, 8192},
        {ge::DT_FLOAT16, 29, 131072, 2000, 0, 0},
        // R 没有上限型门槛，超长行继续按固定 DB tile 循环。
        {ge::DT_FLOAT16, 1, 1048577, 3000, 16, 8192},
        {ge::DT_FLOAT, 1, 1048577, 3000, 32, 4096},
    };

    for (const auto& testCase : cases) {
        SCOPED_TRACE(testCase.numRow);
        Arch35TilingResult result = RunArch35Tiling(testCase.numRow, testCase.numCol, testCase.dataType);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.tilingKey, testCase.expectedKey);
        if (testCase.expectedKey == 3000) {
            EXPECT_EQ(result.blockDim, testCase.expectedBlockDim);
            EXPECT_EQ(result.splitAR.ubFactor, testCase.expectedUbFactor);
            EXPECT_GE(result.splitAR.rBlockNum, 2);
            EXPECT_GE(testCase.numRow *
                          ((result.splitAR.rBlockFactor + result.splitAR.ubFactor - 1) / result.splitAR.ubFactor),
                      8);
            EXPECT_GE(
                testCase.numRow * ((result.splitAR.tailR + result.splitAR.ubFactor - 1) / result.splitAR.ubFactor), 8);
            EXPECT_GE(result.splitAR.numRowAlign, testCase.numRow);
        }
    }
}

TEST_F(AddRmsNormTiling, add_rms_norm_tiling_split_ar_binary_add_limit)
{
    // 当前 Ascend950 的实际 UB 只能得到 FP32=4096、B16=8192，天然小于二分归约上限。
    // 这里人为把 Host compile info 的 UB 放大到 2 MiB：如果只按 UB 容量选 tile，三种
    // dtype 都会选出超过 4*VL_FP32^2 的 ubFactor；修复后必须夹到 16384，并继续通过
    // 跨 tile 循环处理大 R，而不是让 SplitAR 失败回退。
    constexpr uint64_t simulatedUbSize = 2UL * 1024UL * 1024UL;
    constexpr uint64_t binaryAddElementMaxLen = 4UL * 64UL * 64UL;
    constexpr uint64_t firstAdmittedR = 2UL * 8UL * binaryAddElementMaxLen;
    const std::vector<ge::DataType> dataTypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
    for (ge::DataType dataType : dataTypes) {
        SCOPED_TRACE(dataType);
        Arch35TilingResult result = RunArch35Tiling(1, firstAdmittedR, dataType, 0, simulatedUbSize);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.tilingKey, 3000);
        EXPECT_EQ(result.splitAR.ubFactor, binaryAddElementMaxLen);
        EXPECT_EQ(result.splitAR.binAddQuotient, binaryAddElementMaxLen / 2);
        EXPECT_EQ(result.splitAR.rBlockNum, 2);
    }
}

TEST_F(AddRmsNormTiling, add_rms_norm_pre_mode_invalid_gamma_dim)
{
    gert::StorageShape input_shape_x1 = {{2, 64}, {2, 64}};
    gert::StorageShape input_shape_x2 = {{2, 64}, {2, 64}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape out_shape_y = {{2, 64}, {2, 64}};
    gert::StorageShape rstd_shape = {{0}, {0}};
    gert::StorageShape out_shape_x = {{2, 64}, {2, 64}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
             "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                               "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                               "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                               "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                               "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                               "CORE_NUM": 40}
                               })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormCompileInfo compile_info;

    std::string op_type("AddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape_x1, &input_shape_x2, &gamma_shape})
                      .OutputShapes({&out_shape_y, &rstd_shape, &out_shape_x})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
