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
 * \file test_index_fill_d_tiling.cpp
 * \brief
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/index_fill_d_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class IndexFillDTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "IndexFIllDTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IndexFIllDTiling TearDown" << std::endl;
    }
};

struct IndexFillDParamsInfo {
    ge::DataType xDtype;
    gert::StorageShape xShape;
    gert::StorageShape assist1Shape;
    gert::StorageShape assist2Shape;
    gert::StorageShape yShape;
};

template <typename T>
static string ToString(void* buf, size_t size) {
std::string result;
const T* data = reinterpret_cast<const T*>(buf);
size_t len = size / sizeof(T);
for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    result += " ";
}
return result;
}

static void ExecuteTestCase(const IndexFillDParamsInfo& opsParamInfos, string expectTilingData,
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
    optiling::IndexFillDCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    std::string opType("IndexFillD");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector *>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = opsParamInfos.xShape;
    gert::StorageShape assist1Shape = opsParamInfos.assist1Shape;
    gert::StorageShape assist2Shape = opsParamInfos.assist2Shape;
    gert::StorageShape yShape = opsParamInfos.yShape;
    auto holder = gert::TilingContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&xShape, &assist1Shape, &assist2Shape})
                    .OutputShapes({&yShape})
                    .CompileInfo(&compileInfo)
                    .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                    .NodeInputTd(0, opsParamInfos.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, opsParamInfos.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, opsParamInfos.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, opsParamInfos.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
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
    // todo check tiling result
    auto rawTilingData = tilingContext->GetRawTilingData();
    auto tilingDataResult = ToString<int64_t>(rawTilingData->GetData(), rawTilingData->GetDataSize());
    EXPECT_EQ(tilingDataResult, expectTilingData);
}

TEST_F(IndexFillDTiling, IndexFillD_tiling_ascendc_bit_width_1) {
    IndexFillDParamsInfo opsParamInfos;
    opsParamInfos.xDtype = ge::DT_BOOL;
    opsParamInfos.xShape = {{64, 32, 16, 1024}, {64, 32, 16, 1024}};
    opsParamInfos.assist1Shape = opsParamInfos.xShape;
    opsParamInfos.assist2Shape = opsParamInfos.xShape;
    opsParamInfos.yShape = opsParamInfos.xShape;
    string expectTilingData = "524288 524288 31744 16384 16384 17 17 ";

    ExecuteTestCase(opsParamInfos, expectTilingData);
}

TEST_F(IndexFillDTiling, IndexFillD_tiling_ascendc_bit_width_2) {
    IndexFillDParamsInfo opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT;
    opsParamInfos.xShape = {{64, 32, 16, 1024}, {64, 32, 16, 1024}};
    opsParamInfos.assist1Shape = opsParamInfos.xShape;
    opsParamInfos.assist2Shape = opsParamInfos.xShape;
    opsParamInfos.yShape = opsParamInfos.xShape;
    string expectTilingData = "524288 524288 7680 2048 2048 69 69 ";

    ExecuteTestCase(opsParamInfos, expectTilingData);
}

TEST_F(IndexFillDTiling, IndexFillD_tiling_ascendc_bit_width_4) {
    IndexFillDParamsInfo opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.xShape = {{64, 32, 16, 1024}, {64, 32, 16, 1024}};
    opsParamInfos.assist1Shape = opsParamInfos.xShape;
    opsParamInfos.assist2Shape = opsParamInfos.xShape;
    opsParamInfos.yShape = opsParamInfos.xShape;
    string expectTilingData = "524288 524288 7680 2048 2048 69 69 ";

    ExecuteTestCase(opsParamInfos, expectTilingData);
}