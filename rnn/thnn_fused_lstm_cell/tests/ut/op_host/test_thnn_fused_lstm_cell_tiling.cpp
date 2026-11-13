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
 * \file test_thnn_fused_lstm_cell_tiling.cpp
 * \brief
 */

#include <atomic>
#include <array>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <fstream>
#include <limits>
#include <optional>
#include <thread>
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

using namespace ut_util;
using namespace std;
using namespace ge;

struct ThnnFusedLstmCellCompileInfo {};

class ThnnFusedLstmCellTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ThnnFusedLstmCellTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ThnnFusedLstmCellTiling TearDown" << std::endl; }
};

std::vector<uint8_t> RunThnnFusedLstmCellTiling(int64_t batchSize, int64_t hiddenSize, ge::DataType dataType)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape commonShape = {{batchSize, hiddenSize}, {batchSize, hiddenSize}};
    gert::StorageShape gatesShape = {{batchSize, 4 * hiddenSize}, {batchSize, 4 * hiddenSize}};
    gert::StorageShape biasShape = {{4 * hiddenSize}, {4 * hiddenSize}};

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
    ThnnFusedLstmCellCompileInfo compileInfo;

    std::string op_type("ThnnFusedLstmCell");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (opImpl == nullptr) {
        ADD_FAILURE() << "ThnnFusedLstmCell implementation is not registered.";
        return {};
    }
    auto tilingFunc = opImpl->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    if (param == nullptr || workspaceSizeHoler == nullptr) {
        ADD_FAILURE() << "Failed to allocate tiling context data.";
        return {};
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ThnnFusedLstmCell")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&gatesShape, &gatesShape, &commonShape, &biasShape, &biasShape})
                      .OutputShapes({&commonShape, &commonShape, &gatesShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        ADD_FAILURE() << "Failed to create tiling context.";
        return {};
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    if (tilingFunc(tilingContext) != ge::GRAPH_SUCCESS) {
        ADD_FAILURE() << "ThnnFusedLstmCell tiling failed.";
        return {};
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr) {
        ADD_FAILURE() << "ThnnFusedLstmCell tiling data is empty.";
        return {};
    }
    auto data = static_cast<uint8_t*>(rawTilingData->GetData());
    return {data, data + rawTilingData->GetDataSize()};
    // dlog_setlevel(0, 3, 0);
}

TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_float_succ)
{
    std::cout << "run case: "
              << "thnn_fused_lstm_cell_float_succ" << std::endl;
    auto tilingData = RunThnnFusedLstmCellTiling(40, 8, ge::DT_FLOAT);
    EXPECT_FALSE(tilingData.empty());
}

static void RunConcurrentTiling(int64_t batchSize, int64_t hiddenSize, std::atomic<int>& ready,
                                std::atomic<bool>& start, const std::vector<uint8_t>& expectedTilingData,
                                std::atomic<bool>& matched)
{
    ready.fetch_add(1);
    while (!start.load()) {
        std::this_thread::yield();
    }
    for (size_t idx = 0; idx < 200; ++idx) {
        auto tilingData = RunThnnFusedLstmCellTiling(batchSize, hiddenSize, ge::DT_FLOAT);
        if (tilingData != expectedTilingData) {
            matched.store(false);
            return;
        }
    }
}

TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_concurrent_tiling)
{
    constexpr size_t thread_num = 4;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> matched{true};
    std::array<std::thread, thread_num> threads;
    std::array<std::vector<uint8_t>, thread_num> expectedTilingData;

    expectedTilingData[0] = RunThnnFusedLstmCellTiling(40, 8, ge::DT_FLOAT);
    expectedTilingData[1] = RunThnnFusedLstmCellTiling(32, 16, ge::DT_FLOAT);
    expectedTilingData[2] = RunThnnFusedLstmCellTiling(24, 24, ge::DT_FLOAT);
    expectedTilingData[3] = RunThnnFusedLstmCellTiling(16, 32, ge::DT_FLOAT);
    for (const auto& data : expectedTilingData) {
        EXPECT_FALSE(data.empty());
    }

    threads[0] = std::thread(RunConcurrentTiling, 40, 8, std::ref(ready), std::ref(start),
                             std::cref(expectedTilingData[0]), std::ref(matched));
    threads[1] = std::thread(RunConcurrentTiling, 32, 16, std::ref(ready), std::ref(start),
                             std::cref(expectedTilingData[1]), std::ref(matched));
    threads[2] = std::thread(RunConcurrentTiling, 24, 24, std::ref(ready), std::ref(start),
                             std::cref(expectedTilingData[2]), std::ref(matched));
    threads[3] = std::thread(RunConcurrentTiling, 16, 32, std::ref(ready), std::ref(start),
                             std::cref(expectedTilingData[3]), std::ref(matched));

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (ready.load() != static_cast<int>(thread_num) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    bool allThreadsReady = ready.load() == static_cast<int>(thread_num);
    start.store(true);
    for (auto& thread : threads) {
        thread.join();
    }
    ASSERT_TRUE(allThreadsReady);
    EXPECT_TRUE(matched.load());
}

// ---------------------------------------------------------------------------
// 负向校验域(跨输入/输出 shape 校验 + GetInfo/FillTilingInfo 边界的 UT 直调承载):
// 每个 case 锚定一条生产校验分支,断言 tilingFunc 拒绝(ge::GRAPH_FAILED)。
// 缺失输入用 IrInstanceNum 对应槽 0 实例 + InputShapes 传 packed 短数组(实例数与数组长度一致,
// 参照仓内 sigmoid_focal_loss/ctc_loss_v3 等可选输入用例构造);
// nullptr 输出不传 shape + IrInstanceNum 输出实例 0(输出链越界规则返回空指针);
// 平台边界(aivNum/UB_SIZE/taskSize)用自定义编译信息 JSON 构造。
// ---------------------------------------------------------------------------

static std::string MakeThnnFusedLstmCellCompileInfoJson(uint64_t ubSize, uint64_t coreNum)
{
    return R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": )" +
           std::to_string(ubSize) + R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": )" +
           std::to_string(coreNum) + R"(}})";
}

static std::optional<ge::graphStatus> RunThnnFusedLstmCellTilingRaw(
    const std::vector<gert::StorageShape*>& inputShapes, const std::vector<gert::StorageShape*>& outputShapes,
    const std::vector<uint32_t>& outputInstanceNum = {1, 1, 1},
    const std::string& compileInfoJson = MakeThnnFusedLstmCellCompileInfoJson(196608, 40),
    const std::vector<uint32_t>& inputInstanceNum = {1, 1, 1, 1, 1})
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoJson.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    ThnnFusedLstmCellCompileInfo compileInfo;

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ThnnFusedLstmCell");
    if (opImpl == nullptr) {
        ADD_FAILURE() << "ThnnFusedLstmCell implementation is not registered.";
        return std::nullopt;
    }
    auto tilingFunc = opImpl->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    if (param == nullptr || workspaceSizeHoler == nullptr) {
        ADD_FAILURE() << "Failed to allocate tiling context data.";
        return std::nullopt;
    }

    // 缺失输入槽由 IrInstanceNum 置 0 实例承载,InputShapes 传 packed 短数组(长度=总实例数,
    // 与声明实例数保持一致; 数组元素含空指针或长度与实例数失配均会被 faker 构建拒绝)
    gert::TilingContextFaker faker;
    faker.SetOpType("ThnnFusedLstmCell")
        .NodeIoNum(5, 3)
        .IrInstanceNum(inputInstanceNum, outputInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes(outputShapes)
        .CompileInfo(&compileInfo)
        .TilingData(param.get())
        .Workspace(wsSize);
    for (size_t idx = 0; idx < inputInstanceNum.size(); ++idx) {
        if (inputInstanceNum[idx] != 0) {
            faker.NodeInputTd(static_cast<int32_t>(idx), ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        }
    }
    for (size_t idx = 0; idx < outputShapes.size(); ++idx) {
        if (outputShapes[idx] != nullptr) {
            faker.NodeOutputTd(static_cast<int32_t>(idx), ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        }
    }
    faker.PlatformInfo(reinterpret_cast<char*>(&platformInfo));
    auto holder = faker.Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        ADD_FAILURE() << "Failed to create tiling context.";
        return std::nullopt;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    return tilingFunc(tilingContext);
}

// 输入模板: cx=(batch,hidden) 为契约基准, gates=(batch,4*hidden), bias=(4*hidden,), 输出 hy/cy=(batch,hidden)
static void MakeValidThnnFusedLstmCellShapes(int64_t batch, int64_t hidden, gert::StorageShape& gatesShape,
                                             gert::StorageShape& cxShape, gert::StorageShape& biasShape,
                                             gert::StorageShape& hyShape, gert::StorageShape& storageShape)
{
    gatesShape = {{batch, 4 * hidden}, {batch, 4 * hidden}};
    cxShape = {{batch, hidden}, {batch, hidden}};
    biasShape = {{4 * hidden}, {4 * hidden}};
    hyShape = {{batch, hidden}, {batch, hidden}};
    storageShape = {{batch, 4 * hidden}, {batch, 4 * hidden}};
}

// GetInfo: cx rank != 2(rank of cx must be 2)
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_cx_rank_invalid)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    cxShape = {{2, 4, 8}, {2, 4, 8}};
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&hyShape, &hyShape, &storageShape});
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// CheckParamsShape: inputGates rank 与 (B,4H) 不一致
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_input_gates_rank_mismatch)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    gert::StorageShape badGatesShape = {{2, 4, 16}, {2, 4, 16}};
    auto status = RunThnnFusedLstmCellTilingRaw({&badGatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&hyShape, &hyShape, &storageShape});
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// CheckParamsShape: inputGates 维度值与 (B,4H) 不一致
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_input_gates_dim_mismatch)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    gert::StorageShape badGatesShape = {{3, 16}, {3, 16}};
    auto status = RunThnnFusedLstmCellTilingRaw({&badGatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&hyShape, &hyShape, &storageShape});
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// CheckParamsShape: 输出 hy rank 与 (B,H) 不一致
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_output_hy_rank_mismatch)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    gert::StorageShape badHyShape = {{2, 4, 3}, {2, 4, 3}};
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&badHyShape, &hyShape, &storageShape});
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// CheckParamsShape: 输出 cy 维度值与 (B,H) 不一致
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_output_cy_dim_mismatch)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    gert::StorageShape badCyShape = {{2, 5}, {2, 5}};
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&hyShape, &badCyShape, &storageShape});
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// CheckParamsShape: cx 维度为负(dimensions of cx must be non-negative)
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_cx_negative_dim)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    cxShape = {{-1, 4}, {-1, 4}};
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&hyShape, &hyShape, &storageShape});
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// GetInfo/CheckParamsShape: cx 输入实例缺失——faker 置该输入槽实例数为 0 后 context 输入槽位右移,
// GetInputShape(2) 返回非 (B,H) 形状,经 rank 校验拒绝(行为契约: cx 缺失/非法必拒);
// GetInfo 的 cxShape==nullptr 防御分支系 faker 域不可达(输入实例 0 时槽位右移返回非空形状)
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_cx_null_shape)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &biasShape, &biasShape},
                                                {&hyShape, &hyShape, &storageShape}, {1, 1, 1},
                                                MakeThnnFusedLstmCellCompileInfoJson(196608, 40), {1, 1, 0, 1, 1});
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// CheckParamsShape: inputBias 输入实例缺失(必选输入空指针防御分支,查询越界返回空指针)
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_input_bias_null_shape)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &cxShape, &biasShape},
                                                {&hyShape, &hyShape, &storageShape}, {1, 1, 1},
                                                MakeThnnFusedLstmCellCompileInfoJson(196608, 40), {1, 1, 1, 0, 1});
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// CheckParamsShape: storage 输出实例缺失(GetOutputShape 越界返回空指针)
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_output_storage_null_shape)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&hyShape, &hyShape}, {1, 1, 0});
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// GetInfo: AIV 核数为 0(Invalid number of aicore)
// 注: GetInfo 的 platformInfo==nullptr 防御分支不经faker构造——TilingContext 的
// platform 槽位由 faker.Build() 填充,不设置 PlatformInfo 时 context 不完整(实测GetContext即空),
// 生产链路 FE 恒提供平台信息,该分支定性为不可达防御代码。
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_aiv_num_zero)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&hyShape, &hyShape, &storageShape}, {1, 1, 1},
                                                MakeThnnFusedLstmCellCompileInfoJson(196608, 0));
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// FillTilingInfo: UB 尺寸过小(UB size is too small for tiling)
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_ub_size_too_small)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&hyShape, &hyShape, &storageShape}, {1, 1, 1},
                                                MakeThnnFusedLstmCellCompileInfoJson(10000, 40));
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// FillTilingInfo: taskSize 计算为 0(Invalid task size 0 from UB size)
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_task_size_zero)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&hyShape, &hyShape, &storageShape}, {1, 1, 1},
                                                MakeThnnFusedLstmCellCompileInfoJson(20480, 40));
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------
// 边界枚举域(corner表承载): B/H 维取值 ∈ {0, 1, 负, MAX, 2^31前缀溢出×后继0, 2^32}。
// tiling 为纯 host 计算,任意大 shape 经 faker 构造零内存分配即可判别行为。
// tiling data 布局与生产 ABI 一致: 6×uint64(B/H/BH/col/taskTotal/taskSingle)
// + 2×uint32(taskSize@偏移48/tailSize@偏移52), 共 56B。
// ---------------------------------------------------------------------------

struct ThnnFusedLstmCellTilingInfoView {
    uint64_t B;
    uint64_t H;
    uint64_t BH;
    uint64_t col;
    uint64_t taskTotal;
    uint64_t taskSingle;
    uint32_t taskSize;
    uint32_t tailSize;
};

static ThnnFusedLstmCellTilingInfoView ParseThnnFusedLstmCellTilingData(const std::vector<uint8_t>& data)
{
    ThnnFusedLstmCellTilingInfoView view{};
    if (data.size() < sizeof(view)) {
        ADD_FAILURE() << "tiling data size " << data.size() << " is smaller than the 56B tiling layout.";
        return view;
    }
    std::memcpy(&view, data.data(), sizeof(view));
    return view;
}

// corner B=0: 合法空batch,tiling 成功(taskTotal=col*B=0)
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_batch_size_zero)
{
    auto tilingData = RunThnnFusedLstmCellTiling(0, 4, ge::DT_FLOAT);
    ASSERT_FALSE(tilingData.empty());
    auto view = ParseThnnFusedLstmCellTilingData(tilingData);
    EXPECT_EQ(view.B, 0U);
    EXPECT_EQ(view.taskTotal, 0U);
}

// corner B=1: 最小非空batch
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_batch_size_one)
{
    auto tilingData = RunThnnFusedLstmCellTiling(1, 4, ge::DT_FLOAT);
    ASSERT_FALSE(tilingData.empty());
    auto view = ParseThnnFusedLstmCellTilingData(tilingData);
    EXPECT_EQ(view.B, 1U);
}

// corner B=MAX: uint64 域 BH=B*H 回绕系定义行为(无符号回绕非UB);该量级输入物理不可达,
// tiling 行为定界=成功产出(B/BH 等字段如实记录回绕值)
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_batch_size_max)
{
    auto tilingData = RunThnnFusedLstmCellTiling(std::numeric_limits<int64_t>::max(), 4, ge::DT_FLOAT);
    ASSERT_FALSE(tilingData.empty());
    auto view = ParseThnnFusedLstmCellTilingData(tilingData);
    EXPECT_EQ(view.B, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
}

// corner B=2^31(前缀溢出×后继0): B 维越过 int32 正值域上限,H=0 合法空行
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_batch_prefix_overflow_hidden_zero)
{
    constexpr int64_t b2p31 = 2147483648LL;
    auto tilingData = RunThnnFusedLstmCellTiling(b2p31, 0, ge::DT_FLOAT);
    ASSERT_FALSE(tilingData.empty());
    auto view = ParseThnnFusedLstmCellTilingData(tilingData);
    EXPECT_EQ(view.B, static_cast<uint64_t>(b2p31));
    EXPECT_EQ(view.H, 0U);
    EXPECT_EQ(view.taskTotal, 0U);
}

// corner H=0: 合法空hidden,tiling 成功(col=0,taskTotal=0)
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_hidden_size_zero)
{
    auto tilingData = RunThnnFusedLstmCellTiling(2, 0, ge::DT_FLOAT);
    ASSERT_FALSE(tilingData.empty());
    auto view = ParseThnnFusedLstmCellTilingData(tilingData);
    EXPECT_EQ(view.H, 0U);
    EXPECT_EQ(view.taskTotal, 0U);
}

// corner H=1: 最小非空hidden
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_hidden_size_one)
{
    auto tilingData = RunThnnFusedLstmCellTiling(2, 1, ge::DT_FLOAT);
    ASSERT_FALSE(tilingData.empty());
    auto view = ParseThnnFusedLstmCellTilingData(tilingData);
    EXPECT_EQ(view.H, 1U);
}

// corner H<0: hidden 维为负,dimensions of cx must be non-negative 拒绝
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_hidden_negative_dim)
{
    gert::StorageShape gatesShape;
    gert::StorageShape cxShape;
    gert::StorageShape biasShape;
    gert::StorageShape hyShape;
    gert::StorageShape storageShape;
    MakeValidThnnFusedLstmCellShapes(2, 4, gatesShape, cxShape, biasShape, hyShape, storageShape);
    cxShape = {{2, -1}, {2, -1}};
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&hyShape, &hyShape, &storageShape});
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// corner H=MAX: hidden 超过 gates 维度可计算上界(4*hidden 溢出 int64),直接拒绝;
// 其余输入 shape 内容无关(拒绝先于跨输入 shape 比较)
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_hidden_size_max_reject)
{
    gert::StorageShape gatesShape = {{1, 4}, {1, 4}};
    gert::StorageShape cxShape = {{1, std::numeric_limits<int64_t>::max()}, {1, std::numeric_limits<int64_t>::max()}};
    gert::StorageShape biasShape = {{4}, {4}};
    gert::StorageShape hyShape = {{1, 1}, {1, 1}};
    gert::StorageShape storageShape = {{1, 4}, {1, 4}};
    auto status = RunThnnFusedLstmCellTilingRaw({&gatesShape, &gatesShape, &cxShape, &biasShape, &biasShape},
                                                {&hyShape, &hyShape, &storageShape});
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, ge::GRAPH_FAILED);
}

// corner H=2^32(uint32 域上界外,尾块计算修复的直接靶点): tailSize 须在 uint64 域取模,
// 与 col(按完整 H 的 ceil 除)保持一致——旧实现先截断 H 到 uint32(=0)再取模,
// tailSize 错误回退整块 taskSize(UB_SIZE=196608 时 taskSize=1280,期望尾块=256),与 col 失配
TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_hidden_size_2p32_tail_consistency)
{
    constexpr int64_t hidden2p32 = 4294967296LL; // 2^32
    auto tilingData = RunThnnFusedLstmCellTiling(2, hidden2p32, ge::DT_FLOAT);
    ASSERT_FALSE(tilingData.empty());
    auto view = ParseThnnFusedLstmCellTilingData(tilingData);
    const uint64_t h = static_cast<uint64_t>(hidden2p32);
    EXPECT_EQ(view.H, h);
    ASSERT_GT(view.taskSize, 0U);
    // col = ceil(H / taskSize),尾块元素数 = H - (col-1)*taskSize = H % taskSize(非整除时)
    const uint64_t expectCol = (h + view.taskSize - 1U) / view.taskSize;
    EXPECT_EQ(view.col, expectCol);
    const uint64_t expectTail = (h % view.taskSize != 0U) ? (h % view.taskSize) : view.taskSize;
    EXPECT_EQ(view.tailSize, static_cast<uint32_t>(expectTail));
    EXPECT_EQ(h - (view.col - 1U) * view.taskSize, static_cast<uint64_t>(view.tailSize));
}
