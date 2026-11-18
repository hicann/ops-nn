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
 * \file test_single_layer_lstm_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../../op_host/single_layer_lstm_grad_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

struct SingleLayerLstmGradCompileInfo {
    uint32_t aicCoreNum = 20;
    uint32_t aivCoreNum = 40;
    int64_t sysWorkspaceSize = 196608;
    int64_t ubSizePlatForm = 0;
    int64_t l1SizePlatForm = 0;
};

class SingleLayerLstmGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SingleLayerLstmGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SingleLayerLstmGradTiling TearDown" << std::endl; }
};

void TestSingleLayerLstmGradTiling(int64_t batch, int64_t timeStep, int64_t inputSize, int64_t hiddenSize,
                                   ge::DataType dataType, uint64_t expectTilingKey,
                                   const std::string& direction = "UNIDIRECTIONAL",
                                   const std::string& gateOrder = "ijfo",
                                   ge::graphStatus expectStatus = ge::GRAPH_SUCCESS)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape xShape = {{timeStep, batch, inputSize}, {timeStep, batch, inputSize}};
    gert::StorageShape inith0Shape = {{1, batch, hiddenSize}, {1, batch, hiddenSize}};
    gert::StorageShape hShape = {{timeStep, batch, hiddenSize}, {timeStep, batch, hiddenSize}};
    gert::StorageShape wShape = {{4 * hiddenSize, inputSize + hiddenSize}, {4 * hiddenSize, inputSize + hiddenSize}};
    gert::StorageShape bShape = {{4 * hiddenSize}, {4 * hiddenSize}};

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    string COMPILE_INFO_STRING_910B = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 40}})";
    GetPlatFormInfos(COMPILE_INFO_STRING_910B.c_str(), socInfos, aicoreSpec, intrinsics);

    // Platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // Compile info
    SingleLayerLstmGradCompileInfo compileInfo;

    std::string op_type("SingleLayerLstmGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("SingleLayerLstmGrad")
                      .NodeIoNum(16, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &wShape, &bShape, &hShape, &inith0Shape, &inith0Shape, &hShape, &hShape,
                                    &hShape, &inith0Shape, &inith0Shape, &hShape, &hShape, &hShape, &hShape, &hShape})
                      .OutputShapes({&wShape, &bShape, &xShape, &inith0Shape, &inith0Shape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(14, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(15, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"direction", Ops::NN::AnyValue::CreateFrom<string>(direction)},
                                  {"gate_order", Ops::NN::AnyValue::CreateFrom<string>(gateOrder)}})
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
    const ge::graphStatus status = tilingFunc(tilingContext);
    EXPECT_EQ(status, expectStatus);
    if (status != ge::GRAPH_SUCCESS) {
        return;
    }

    auto realTilingKey = tilingContext->GetTilingKey();
    ASSERT_EQ(realTilingKey, expectTilingKey);
    // dlog_setlevel(0, 3, 0);
}

void TestSingleLayerLstmGradDataTiling(int64_t batch, int64_t timeStep, int64_t inputSize, int64_t hiddenSize,
                                       ge::DataType dataType, uint64_t expectTilingKey)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape xShape = {{timeStep, batch, inputSize}, {timeStep, batch, inputSize}};
    gert::StorageShape inith0Shape = {{1, batch, hiddenSize}, {1, batch, hiddenSize}};
    gert::StorageShape hShape = {{timeStep, batch, hiddenSize}, {timeStep, batch, hiddenSize}};
    gert::StorageShape wShape = {{4 * hiddenSize, inputSize + hiddenSize}, {4 * hiddenSize, inputSize + hiddenSize}};
    gert::StorageShape bShape = {{4 * hiddenSize}, {4 * hiddenSize}};

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    string COMPILE_INFO_STRING_910B = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 40}})";
    GetPlatFormInfos(COMPILE_INFO_STRING_910B.c_str(), socInfos, aicoreSpec, intrinsics);

    // Platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // Compile info
    SingleLayerLstmGradCompileInfo compileInfo;

    std::string op_type("SingleLayerLstmGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("SingleLayerLstmGrad")
                      .NodeIoNum(17, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &wShape, &bShape, &hShape, &inith0Shape, &inith0Shape, &hShape, &hShape,
                                    &hShape, &inith0Shape, &inith0Shape, &hShape, &hShape, &hShape, &hShape, &hShape,
                                    &hShape})
                      .OutputShapes({&wShape, &bShape, &xShape, &inith0Shape, &inith0Shape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(14, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(15, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(16, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"direction", Ops::NN::AnyValue::CreateFrom<string>("UNIDIRECTIONAL")},
                                  {"gate_order", Ops::NN::AnyValue::CreateFrom<string>("ijfo")}})
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

    auto realTilingKey = tilingContext->GetTilingKey();
    ASSERT_EQ(realTilingKey, expectTilingKey);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(SingleLayerLstmGradTiling, single_layer_lstm_grad_tilingkey_0)
{
    std::cout << "run case: "
              << "single_layer_lstm_grad_tilingkey_0" << std::endl;
    TestSingleLayerLstmGradTiling(8, 40, 8, 16, ge::DT_FLOAT, 0);
}

TEST_F(SingleLayerLstmGradTiling, single_layer_lstm_grad_tilingkey_seq_0)
{
    std::cout << "run case: "
              << "single_layer_lstm_grad_tilingkey_0" << std::endl;
    TestSingleLayerLstmGradDataTiling(8, 40, 8, 16, ge::DT_FLOAT, 0);
}

TEST_F(SingleLayerLstmGradTiling, rejects_invalid_direction)
{
    TestSingleLayerLstmGradTiling(8, 40, 8, 16, ge::DT_FLOAT, 0, "BIDIRECTIONAL", "ijfo", ge::GRAPH_FAILED);
}

TEST_F(SingleLayerLstmGradTiling, rejects_invalid_gate_order)
{
    TestSingleLayerLstmGradTiling(8, 40, 8, 16, ge::DT_FLOAT, 0, "UNIDIRECTIONAL", "bad", ge::GRAPH_FAILED);
}

TEST_F(SingleLayerLstmGradTiling, rejects_overflowing_generic_workspace_shapes)
{
    optiling::SingleLayerLstmGradPlanRequest request;
    request.timeStep = std::numeric_limits<int64_t>::max();
    request.batch = 2;
    request.inputSize = 1;
    request.hiddenSize = 1;
    request.elemBytes = 4;
    request.aicCoreNum = 8;
    request.ubSizePlatForm = 248 * 1024;
    alignas(8) uint8_t data[4096]{};
    optiling::SingleLayerLstmGradPlanResult result;
    EXPECT_NE(optiling::PlanSingleLayerLstmGrad(request, data, sizeof(data), result), ge::GRAPH_SUCCESS);

    request.timeStep = 1;
    request.batch = 1;
    request.inputSize = std::numeric_limits<int64_t>::max();
    request.hiddenSize = std::numeric_limits<int64_t>::max();
    EXPECT_NE(optiling::PlanSingleLayerLstmGrad(request, data, sizeof(data), result), ge::GRAPH_SUCCESS);
}

// Ascend950 (regbase): BF16 small shapes may use the AIV-only path (key 20000).
// FP16/FP32, seq_length and oversized hidden use the Cube/wide pipeline (key 0).
void TestSingleLayerLstmGradTilingRegbase(int64_t batch, int64_t timeStep, int64_t inputSize, int64_t hiddenSize,
                                          bool hasSeq, ge::DataType dataType, uint64_t expectTilingKey,
                                          bool checkWorkspace = false)
{
    gert::StorageShape xShape = {{timeStep, batch, inputSize}, {timeStep, batch, inputSize}};
    gert::StorageShape inith0Shape = {{1, batch, hiddenSize}, {1, batch, hiddenSize}};
    gert::StorageShape hShape = {{timeStep, batch, hiddenSize}, {timeStep, batch, hiddenSize}};
    gert::StorageShape wShape = {{4 * hiddenSize, inputSize + hiddenSize}, {4 * hiddenSize, inputSize + hiddenSize}};
    gert::StorageShape bShape = {{4 * hiddenSize}, {4 * hiddenSize}};

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    string COMPILE_INFO_STRING_950 = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
    "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    GetPlatFormInfos(COMPILE_INFO_STRING_950.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    SingleLayerLstmGradCompileInfo compileInfo;

    std::string op_type("SingleLayerLstmGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);

    gert::StorageShape emptySeqShape; // 0-dim placeholder == seq_length absent (aclnn behavior)
    gert::StorageShape* seqShape = hasSeq ? &hShape : &emptySeqShape;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("SingleLayerLstmGrad")
                      .NodeIoNum(17, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &wShape, &bShape, &hShape, &inith0Shape, &inith0Shape, &hShape, &hShape,
                                    &hShape, &inith0Shape, &inith0Shape, &hShape, &hShape, &hShape, &hShape, &hShape,
                                    seqShape})
                      .OutputShapes({&wShape, &bShape, &xShape, &inith0Shape, &inith0Shape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(14, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(15, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(16, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"direction", Ops::NN::AnyValue::CreateFrom<string>("UNIDIRECTIONAL")},
                                  {"gate_order", Ops::NN::AnyValue::CreateFrom<string>("ifjo")}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), expectTilingKey);
    if (checkWorkspace) {
        /* Every AscendC operator reserves the library's own workspace, so the total is never zero;
         * what this case pins is the part ON TOP of that reservation. At fp32 dw and db accumulate
         * into the outputs themselves and the path asks for none. At a narrow dtype the fp32
         * accumulators and the per-core forward replay cache live there, and the accumulators
         * alone are 4H x (I + H) floats for dw plus 4H for db -- a floor the case can state from
         * the shapes, without pinning the core count the tiling happens to choose. */
        auto platform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
        const auto reserved = static_cast<int64_t>(platform.GetLibApiWorkSpaceSize());
        auto* wsData = reinterpret_cast<const size_t*>(wsSize->GetData());
        const int64_t privateBytes = static_cast<int64_t>(wsData[0]) - reserved;
        if (dataType == ge::DT_FLOAT) {
            const int64_t dgateBytes = timeStep * batch * 4 * hiddenSize * static_cast<int64_t>(sizeof(float));
            const int64_t xhBytes = timeStep * batch * (inputSize + hiddenSize) * static_cast<int64_t>(sizeof(float));
            const int64_t dxhBytes = batch * (inputSize + hiddenSize) * static_cast<int64_t>(sizeof(float));
            ASSERT_EQ(privateBytes, dgateBytes + xhBytes + dxhBytes);
        } else {
            const int64_t accumulatorBytes = 4 * hiddenSize * (inputSize + hiddenSize + 1) *
                                             static_cast<int64_t>(sizeof(float));
            ASSERT_GE(privateBytes, accumulatorBytes);
        }
    }
}

TEST_F(SingleLayerLstmGradTiling, single_layer_lstm_grad_regbase_small_fp32)
{
    TestSingleLayerLstmGradTilingRegbase(12, 3, 6, 6, false, ge::DT_FLOAT, 0, true);
}

TEST_F(SingleLayerLstmGradTiling, single_layer_lstm_grad_regbase_small_fp16)
{
    TestSingleLayerLstmGradTilingRegbase(49, 2, 998, 10, false, ge::DT_FLOAT16, 0, true);
}

TEST_F(SingleLayerLstmGradTiling, single_layer_lstm_grad_regbase_small_bf16)
{
    TestSingleLayerLstmGradTilingRegbase(12, 3, 6, 6, false, ge::DT_BF16, 20000, true);
}

TEST_F(SingleLayerLstmGradTiling, single_layer_lstm_grad_regbase_seq_fallback)
{
    TestSingleLayerLstmGradTilingRegbase(12, 3, 6, 6, true, ge::DT_FLOAT, 0);
}

// FP32 always uses the Cube/wide path, including shapes that would fit in the small-path UB budget.
TEST_F(SingleLayerLstmGradTiling, single_layer_lstm_grad_regbase_blocked_hidden)
{
    TestSingleLayerLstmGradTilingRegbase(8, 4, 32, 128, false, ge::DT_FLOAT, 0);
}

TEST_F(SingleLayerLstmGradTiling, single_layer_lstm_grad_regbase_big_hidden_fallback)
{
    TestSingleLayerLstmGradTilingRegbase(8, 4, 32, 4096, false, ge::DT_FLOAT, 0);
}
