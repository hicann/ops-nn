/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace std;
using namespace ut_util;
class ModulateTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ModulateTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ModulateTiling TearDown" << std::endl;
  }
};

TEST_F(ModulateTiling, test_modulate_tiling_000) {
    std::string opType("Modulate");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // compile info
    struct ModulateCompileInfo {
      int32_t totalCoreNum = 30;
      uint64_t ubSizePlatForm = 0;
    } compileInfo;
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
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape xShape = {{32, 8, 1024}, {32, 8, 1024}};
    gert::StorageShape scaleShape = {{32, 1, 1024}, {32, 1, 1024}};
    gert::StorageShape shiftShape = {{32, 1, 1024}, {32, 1, 1024}};
    gert::StorageShape outShape = {{32, 8, 1024}, {32, 8, 1024}};

    // tilingParseFunc simulate
    auto holder =
        gert::TilingContextFaker()
          .SetOpType("Modulate")
          .NodeIoNum(3, 1)
          .IrInstanceNum({1, 1, 1})
          .InputShapes({&xShape, &scaleShape, &shiftShape})
          .OutputShapes({&outShape})
          .CompileInfo(&compileInfo)
          .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .TilingData(param.get())
          .Workspace(wsSize)
          .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

TEST_F(ModulateTiling, test_modulate_tiling_001) {
    std::string opType("Modulate");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // compile info
    struct ModulateCompileInfo {
      int32_t totalCoreNum = 30;
      uint64_t ubSizePlatForm = 0;
    } compileInfo;
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
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape xShape = {{32, 8, 5120}, {32, 8, 5120}};
    gert::StorageShape scaleShape = {{32, 1, 5120}, {32, 1, 5120}};
    gert::StorageShape shiftShape = {{32, 1, 5120}, {32, 1, 5120}};
    gert::StorageShape outShape = {{32, 8, 5120}, {32, 8, 5120}};

    // tilingParseFunc simulate
    auto holder =
        gert::TilingContextFaker()
          .SetOpType("Modulate")
          .NodeIoNum(3, 1)
          .IrInstanceNum({1, 1, 1})
          .InputShapes({&xShape, &scaleShape, &shiftShape})
          .OutputShapes({&outShape})
          .CompileInfo(&compileInfo)
          .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .TilingData(param.get())
          .Workspace(wsSize)
          .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

TEST_F(ModulateTiling, test_modulate_tiling_002) {
    std::string opType("Modulate");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // compile info
    struct ModulateCompileInfo {
      int32_t totalCoreNum = 30;
      uint64_t ubSizePlatForm = 0;
    } compileInfo;
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
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape xShape = {{8, 4, 20480}, {8, 4, 20480}};
    gert::StorageShape scaleShape = {{8, 1, 20480}, {8, 1, 20480}};
    gert::StorageShape shiftShape = {{8, 1, 20480}, {8, 1, 20480}};
    gert::StorageShape outShape = {{8, 4, 20480}, {8, 4, 20480}};

    // tilingParseFunc simulate
    auto holder =
        gert::TilingContextFaker()
          .SetOpType("Modulate")
          .NodeIoNum(3, 1)
          .IrInstanceNum({1, 1, 1})
          .InputShapes({&xShape, &scaleShape, &shiftShape})
          .OutputShapes({&outShape})
          .CompileInfo(&compileInfo)
          .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
          .TilingData(param.get())
          .Workspace(wsSize)
          .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}