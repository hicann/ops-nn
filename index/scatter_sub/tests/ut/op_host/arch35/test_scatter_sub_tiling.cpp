/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /* !
 * \file test_scatter_sub_tiling.cpp
 * \brief the ut of scatter_sub_tiling
 */

#include <iostream>
#include <fstream>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/scatter_sub_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class ScatterSubTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScatterSubTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScatterSubTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

struct ScatterSubOpsParamInfos {
  ge::DataType indiceDtype;
  ge::DataType varDtype;
  gert::StorageShape indiceShape;
  gert::StorageShape updateShape;
  gert::StorageShape varShape;
};

static void ExecuteTestCase(const ScatterSubOpsParamInfos& scatterSubOpsParamInfos, int32_t deterministic,
  ge::graphStatus status = ge::GRAPH_SUCCESS)
{
  string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
  map<string, string> socInfos;
  map<string, string> aicoreSpec;
  map<string, string> intrinsics;

  GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

  // platform info
  fe::PlatFormInfos platformInfo;
  platformInfo.Init();

  // compile info
  optiling::ScatterSubCompileInfo compileInfo;
  compileInfo.core_num = 64;
  compileInfo.ub_size = 253952;

  std::string opType("ScatterSub");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
  auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
  auto wsSize = reinterpret_cast<gert::ContinuousVector *>(workspaceSizeHoler.get());
  ASSERT_NE(param, nullptr);
  gert::StorageShape indiceShape = scatterSubOpsParamInfos.indiceShape;
  gert::StorageShape updateShape = scatterSubOpsParamInfos.updateShape;
  gert::StorageShape varShape = scatterSubOpsParamInfos.varShape;
  auto holder = gert::TilingContextFaker()
                  .NodeIoNum(3, 1)
                  .IrInstanceNum({1, 1, 1})
                  .InputShapes({&varShape, &indiceShape, &updateShape})
                  .OutputShapes({&varShape})
                  .CompileInfo(&compileInfo)
                  .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                  .NodeInputTd(0, scatterSubOpsParamInfos.varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeInputTd(1, scatterSubOpsParamInfos.indiceDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeInputTd(2, scatterSubOpsParamInfos.varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(0, scatterSubOpsParamInfos.varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                  .DeterministicInfo(reinterpret_cast<int32_t*>(deterministic))
                  .TilingData(param.get())
                  .Workspace(wsSize)
                  .Build();
  gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tilingContext, nullptr);
  auto infos = tilingContext->GetPlatformInfo();
  ASSERT_NE(infos, nullptr);
  infos->SetPlatformRes("SoCInfo", socInfos);
  infos->SetPlatformRes("AICoreSpec", aicoreSpec);
  infos->SetCoreNumByCoreType("AICore");
  infos->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // workspaces nullptr return failed
  EXPECT_EQ(tilingFunc(tilingContext), status);
}

TEST_F(ScatterSubTiling, ScatterSub_tiling_ascendc_varUint8_indiceInt32_simd_notSupportAtomicAdd) {
  ScatterSubOpsParamInfos scatterSubOpsParamInfos;
  scatterSubOpsParamInfos.indiceDtype = ge::DT_INT32;
  scatterSubOpsParamInfos.varDtype = ge::DT_UINT8;
  scatterSubOpsParamInfos.indiceShape = {{65, 133}, {65, 133}};
  scatterSubOpsParamInfos.updateShape = {{65, 133, 1, 130}, {65, 133, 1, 130}};
  scatterSubOpsParamInfos.varShape = {{191, 1, 130}, {191, 1, 130}};

  ExecuteTestCase(scatterSubOpsParamInfos, 0);
}

TEST_F(ScatterSubTiling, ScatterSub_tiling_ascendc_varUint8_indiceInt64_simd_notSupportAtomicAdd) {
  ScatterSubOpsParamInfos scatterSubOpsParamInfos;
  scatterSubOpsParamInfos.indiceDtype = ge::DT_INT64;
  scatterSubOpsParamInfos.varDtype = ge::DT_UINT8;
  scatterSubOpsParamInfos.indiceShape = {{65, 133}, {65, 133}};
  scatterSubOpsParamInfos.updateShape = {{65, 133, 1, 130}, {65, 133, 1, 130}};
  scatterSubOpsParamInfos.varShape = {{191, 1, 130}, {191, 1, 130}};

  ExecuteTestCase(scatterSubOpsParamInfos, 0);
}

TEST_F(ScatterSubTiling, ScatterSub_tiling_ascendc_varUint8_indiceInt64_simd_exceed_ubBoundary) {
  ScatterSubOpsParamInfos scatterSubOpsParamInfos;
  scatterSubOpsParamInfos.indiceDtype = ge::DT_INT64;
  scatterSubOpsParamInfos.varDtype = ge::DT_UINT8;
  scatterSubOpsParamInfos.indiceShape = {{6}, {6}};
  scatterSubOpsParamInfos.updateShape = {{6, 3, 3, 3, 3, 3, 3, 137}, {6, 3, 3, 3, 3, 3, 3, 137}};
  scatterSubOpsParamInfos.varShape = {{3, 3, 3, 3, 3, 3, 3, 137}, {3, 3, 3, 3, 3, 3, 3, 137}};

  ExecuteTestCase(scatterSubOpsParamInfos, 0);
}

TEST_F(ScatterSubTiling, ScatterSub_tiling_ascendc_varUint8_indiceInt32_simt_notSupportAtomicAdd) {
  ScatterSubOpsParamInfos scatterSubOpsParamInfos;
  scatterSubOpsParamInfos.indiceDtype = ge::DT_INT32;
  scatterSubOpsParamInfos.varDtype = ge::DT_UINT8;
  scatterSubOpsParamInfos.indiceShape = {{65, 133}, {65, 133}};
  scatterSubOpsParamInfos.updateShape = {{}, {}};
  scatterSubOpsParamInfos.varShape = {{191, 1, 2}, {191, 1, 2}};

  ExecuteTestCase(scatterSubOpsParamInfos, 0);
}

TEST_F(ScatterSubTiling, ScatterSub_tiling_ascendc_varUint8_indiceInt64_simt_notSupportAtomicAdd) {
  ScatterSubOpsParamInfos scatterSubOpsParamInfos;
  scatterSubOpsParamInfos.indiceDtype = ge::DT_INT64;
  scatterSubOpsParamInfos.varDtype = ge::DT_UINT8;
  scatterSubOpsParamInfos.indiceShape = {{65, 133}, {65, 133}};
  scatterSubOpsParamInfos.updateShape = {{}, {}};
  scatterSubOpsParamInfos.varShape = {{191, 1, 2}, {191, 1, 2}};

  ExecuteTestCase(scatterSubOpsParamInfos, 0);
}

TEST_F(ScatterSubTiling, ScatterSub_tiling_ascendc_varFp32_indiceInt32_deterministic) {
  ScatterSubOpsParamInfos scatterSubOpsParamInfos;
  scatterSubOpsParamInfos.indiceDtype = ge::DT_INT32;
  scatterSubOpsParamInfos.varDtype = ge::DT_FLOAT;
  scatterSubOpsParamInfos.indiceShape = {{19, 31}, {19, 31}};
  scatterSubOpsParamInfos.updateShape = {{19, 31, 2, 9}, {19, 31, 2, 9}};
  scatterSubOpsParamInfos.varShape = {{129, 2, 9}, {129, 2, 9}};

  ExecuteTestCase(scatterSubOpsParamInfos, 1);
}
