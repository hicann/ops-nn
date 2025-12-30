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
 * \file test_masked_scatter_tiling.cpp
 * \brief the ut of masked_scatter_tiling
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/masked_scatter_tiling_arch35.h"

using namespace std;

class MaskedScatterTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaskedScatterTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaskedScatterTiling TearDown" << std::endl;
    }
};

struct MaskedScatterOpsParamInfos {
    ge::DataType xDtype;
    ge::DataType maskDtype;
    ge::DataType updateDtype;
    ge::DataType yDtype;
    gert::StorageShape xShape;
    gert::StorageShape maskShape;
    gert::StorageShape updateShape;
    gert::StorageShape yShape;
};

template <typename T>
string ToString(void* buf, size_t size) {
  std::string result;
  const T* data = reinterpret_cast<const T*>(buf);
  size_t len = size / sizeof(T);
  for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    result += " ";
  }
  return result;
}

static void ExecuteTestCase(const MaskedScatterOpsParamInfos& opsParamInfos, string expectTilingData,
    ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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
    optiling::MaskedScatterCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    std::string opType("MaskedScatter");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector *>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = opsParamInfos.xShape;
    gert::StorageShape maskShape = opsParamInfos.maskShape;
    gert::StorageShape updateShape = opsParamInfos.updateShape;
    gert::StorageShape yShape = opsParamInfos.yShape;
    auto holder = gert::TilingContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&xShape, &maskShape, &updateShape})
                    .OutputShapes({&yShape})
                    .CompileInfo(&compileInfo)
                    .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                    .NodeInputTd(0, opsParamInfos.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, opsParamInfos.maskDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, opsParamInfos.updateDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, opsParamInfos.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
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
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    auto tilingDataResult = ToString<int64_t>(rawTilingData->GetData(), rawTilingData->GetDataSize());
    EXPECT_EQ(tilingDataResult, expectTilingData);
}

TEST_F(MaskedScatterTiling, MaskedScatter_tiling_ascendc_bit_width_1) {
    MaskedScatterOpsParamInfos opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT8;
    opsParamInfos.maskDtype = ge::DT_BOOL;
    opsParamInfos.updateDtype = opsParamInfos.xDtype;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{64, 32, 16, 1024}, {64, 32, 16, 1024}};
    opsParamInfos.maskShape = opsParamInfos.xShape;
    opsParamInfos.updateShape = {{64, 32, 16, 1024}, {64, 32, 16, 1024}};
    opsParamInfos.yShape = opsParamInfos.xShape;
    string expectTilingData = "524288 524288 22080 24 24 16448 16448 33554432 44160 256 ";

    ExecuteTestCase(opsParamInfos, expectTilingData);
}

TEST_F(MaskedScatterTiling, MaskedScatter_tiling_ascendc_bit_width_2) {
    MaskedScatterOpsParamInfos opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT16;
    opsParamInfos.maskDtype = ge::DT_BOOL;
    opsParamInfos.updateDtype = opsParamInfos.xDtype;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{64, 32, 16, 1024}, {64, 32, 16, 1024}};
    opsParamInfos.maskShape = opsParamInfos.xShape;
    opsParamInfos.updateShape = {{64, 32, 16, 1024}, {64, 32, 16, 1024}};
    opsParamInfos.yShape = opsParamInfos.xShape;
    string expectTilingData = "524288 524288 22080 24 24 16448 16448 33554432 27616 256 ";

    ExecuteTestCase(opsParamInfos, expectTilingData);
}

TEST_F(MaskedScatterTiling, MaskedScatter_tiling_ascendc_bit_width_4) {
    MaskedScatterOpsParamInfos opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.maskDtype = ge::DT_BOOL;
    opsParamInfos.updateDtype = opsParamInfos.xDtype;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{64, 32, 16, 1024}, {64, 32, 16, 1024}};
    opsParamInfos.maskShape = opsParamInfos.xShape;
    opsParamInfos.updateShape = {{64, 32, 16, 1024}, {64, 32, 16, 1024}};
    opsParamInfos.yShape = opsParamInfos.xShape;
    string expectTilingData = "524288 524288 22080 24 24 16448 16448 33554432 0 0 ";

    ExecuteTestCase(opsParamInfos, expectTilingData);
}

TEST_F(MaskedScatterTiling, MaskedScatter_tiling_ascendc_bit_width_8) {
    MaskedScatterOpsParamInfos opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT64;
    opsParamInfos.maskDtype = ge::DT_BOOL;
    opsParamInfos.updateDtype = opsParamInfos.xDtype;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{64, 32, 16, 1024}, {64, 32, 16, 1024}};
    opsParamInfos.maskShape = opsParamInfos.xShape;
    opsParamInfos.updateShape = {{64, 32, 16, 1024}, {64, 32, 16, 1024}};
    opsParamInfos.yShape = opsParamInfos.xShape;
    string expectTilingData = "524288 524288 22080 24 24 16448 16448 33554432 0 0 ";

    ExecuteTestCase(opsParamInfos, expectTilingData);
}

TEST_F(MaskedScatterTiling, MaskedScatter_tiling_ascendc_bit_width_8_big_maskSum) {
    MaskedScatterOpsParamInfos opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT64;
    opsParamInfos.maskDtype = ge::DT_BOOL;
    opsParamInfos.updateDtype = opsParamInfos.xDtype;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{64, 1024, 1024, 1024}, {64, 1024, 1024, 1024}};
    opsParamInfos.maskShape = opsParamInfos.xShape;
    opsParamInfos.updateShape = {{64, 1024, 1024, 1024}, {64, 1024, 1024, 1024}};
    opsParamInfos.yShape = opsParamInfos.xShape;
    string expectTilingData = "1073741824 1073741824 12288 87382 87382 4096 4096 68719476736 0 0 ";

    ExecuteTestCase(opsParamInfos, expectTilingData);
}

TEST_F(MaskedScatterTiling, MaskedScatter_tiling_ascendc_error_inDtype) {
    MaskedScatterOpsParamInfos opsParamInfos;
    opsParamInfos.xDtype = ge::DT_COMPLEX64;
    opsParamInfos.maskDtype = ge::DT_BOOL;
    opsParamInfos.updateDtype = ge::DT_COMPLEX64;
    opsParamInfos.yDtype = ge::DT_COMPLEX64;
    opsParamInfos.xShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    opsParamInfos.maskShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    opsParamInfos.updateShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    opsParamInfos.yShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    string expectTilingData = "";
    ExecuteTestCase(opsParamInfos, expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(MaskedScatterTiling, MaskedScatter_tiling_ascendc_error_maskDtype) {
    MaskedScatterOpsParamInfos opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT;
    opsParamInfos.maskDtype = ge::DT_FLOAT;
    opsParamInfos.updateDtype = ge::DT_FLOAT;
    opsParamInfos.yDtype = ge::DT_FLOAT;
    opsParamInfos.xShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    opsParamInfos.maskShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    opsParamInfos.updateShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    opsParamInfos.yShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    string expectTilingData = "";
    ExecuteTestCase(opsParamInfos, expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(MaskedScatterTiling, MaskedScatter_tiling_ascendc_diff_Dtype) {
    MaskedScatterOpsParamInfos opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT;
    opsParamInfos.maskDtype = ge::DT_BOOL;
    opsParamInfos.updateDtype = ge::DT_INT32;
    opsParamInfos.yDtype = ge::DT_FLOAT;
    opsParamInfos.xShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    opsParamInfos.maskShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    opsParamInfos.updateShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    opsParamInfos.yShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    string expectTilingData = "";
    ExecuteTestCase(opsParamInfos, expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(MaskedScatterTiling, MaskedScatter_tiling_ascendc_error_shape) {
    MaskedScatterOpsParamInfos opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT;
    opsParamInfos.maskDtype = ge::DT_BOOL;
    opsParamInfos.updateDtype = ge::DT_FLOAT;
    opsParamInfos.yDtype = ge::DT_FLOAT;
    opsParamInfos.xShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    opsParamInfos.maskShape = {{60, 14, 16, 1}, {60, 14, 16, 1}};
    opsParamInfos.updateShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    opsParamInfos.yShape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    string expectTilingData = "";
    ExecuteTestCase(opsParamInfos, expectTilingData, ge::GRAPH_FAILED);
}