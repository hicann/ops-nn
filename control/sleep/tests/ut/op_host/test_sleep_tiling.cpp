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
#include <vector>
#include <memory>
#include <cstring>
#include "log/log.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/tensor.h"
#include "register/op_impl_registry.h"
#include "control/sleep/op_kernel/sleep_tiling_data.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class SleepTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

template <typename T>
void SetConstInput(size_t constIndex, ge::DataType dtype, T* constData, int64_t dataSize,
                   std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& constTensors)
{
    std::unique_ptr<uint8_t[]> inputTensorHolder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(T) * dataSize]);
    auto inputTensor = reinterpret_cast<gert::Tensor*>(inputTensorHolder.get());
    gert::Tensor tensor({{dataSize}, {dataSize}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, dtype, nullptr);
    std::memcpy(inputTensor, &tensor, sizeof(gert::Tensor));
    auto tensorData = reinterpret_cast<T*>(inputTensor + 1);
    for (int64_t i = 0; i < dataSize; i++) {
        tensorData[i] = constData[i];
    }
    inputTensor->SetData(gert::TensorData{tensorData});
    constTensors.push_back(std::make_pair(constIndex, std::move(inputTensorHolder)));
}

static void RunSleepTiling(int64_t cycles, const std::string& socVersion, ge::graphStatus expectRet,
                           int64_t expectCycles = 0, uint32_t expectBlockDim = 0)
{
    map<string, string> socVersionInfos = {{"Short_SoC_version", socVersion}, {"NpuArch", "3510"}};
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                          "UB_SIZE": 245760, "L2_SIZE": 134217728, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 64,
                          "cube_core_cnt": 64, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
                          })";
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1650";

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    gert::StorageShape cyclesShape = {{1}, {1}};
    gert::StorageShape dummyOutShape = {{1}, {1}};
    int64_t cyclesValue = cycles;
    int64_t dummyCompileInfo = 0;

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> constTensors;
    SetConstInput(0, ge::DT_INT64, &cyclesValue, 1, constTensors);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Sleep")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&cyclesShape})
                      .OutputShapes({&dummyOutShape})
                      .CompileInfo(&dummyCompileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(constTensors)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);

    std::string opType("Sleep");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    EXPECT_EQ(tilingFunc(tilingContext), expectRet);
    if (expectRet == ge::GRAPH_SUCCESS) {
        auto* td = tilingContext->GetTilingData<SleepTilingData>();
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->cycles, expectCycles);
        EXPECT_EQ(tilingContext->GetBlockDim(), expectBlockDim);
    }
}

TEST_F(SleepTilingTest, Ascend950_cycles_pass_through)
{
    RunSleepTiling(1650000, "Ascend950", ge::GRAPH_SUCCESS, 1650000, 1U);
}

TEST_F(SleepTilingTest, Ascend950_cycles_minimum) { RunSleepTiling(1, "Ascend950", ge::GRAPH_SUCCESS, 1, 1U); }

TEST_F(SleepTilingTest, cycles_zero_failed) { RunSleepTiling(0, "Ascend950", ge::GRAPH_FAILED); }

TEST_F(SleepTilingTest, cycles_negative_failed) { RunSleepTiling(-100, "Ascend950", ge::GRAPH_FAILED); }

TEST_F(SleepTilingTest, unsupported_soc_failed) { RunSleepTiling(1000000, "Ascend910B", ge::GRAPH_FAILED); }
