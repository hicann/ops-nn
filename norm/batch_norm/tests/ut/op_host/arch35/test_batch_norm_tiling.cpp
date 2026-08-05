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
 * \file test_batch_norm_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../../op_host/arch35/batch_norm_tiling.h"
using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t FLOAT16_BYTES_NUM = 2;
constexpr int64_t FLOAT32_BYTES_NUM = 4;
constexpr int64_t X_Y_QUEUE_NUM = 2;
constexpr int64_t SMALL_TENSOR_QUEUE_NUM = 6;

struct BatchNormInferLastChannelTilingDataForUt {
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t usedCoreNums;
    int64_t totalALen;
    int64_t aOuter;
    int64_t bOuter;
    int64_t tileBlockALen;
    int64_t tileBlockATail;
    int64_t tileBlockAPaddingNum;
    int64_t tileBlockBLen;
    int64_t tileBlockBTail;
    float epsilon;
};

struct BatchNormInferTilingDataForUt {
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t usedCoreNums;
    int64_t totalB0Len;
    int64_t totalALen;
    int64_t totalB1Len;
    int64_t b0Outer;
    int64_t aOuter;
    int64_t b1Outer;
    int64_t tileBlockB0Len;
    int64_t tileBlockB0Tail;
    int64_t tileBlockALen;
    int64_t tileBlockATail;
    int64_t tileBlockB1Len;
    int64_t tileBlockB1Tail;
    int64_t tileBlockAPaddingNum;
    float epsilon;
};

int64_t AlignUpForUt(int64_t value, int64_t align) { return ((value + align - 1) / align) * align; }

static void RunBatchNormInferTilingForTest(gert::StorageShape& xShape, ge::Format format, int64_t channel,
                                           uint64_t expectedTilingKey,
                                           BatchNormInferTilingDataForUt* inferTilingData = nullptr,
                                           BatchNormInferLastChannelTilingDataForUt* lastChannelTilingData = nullptr)
{
    gert::StorageShape scaleShape = {{channel}, {channel}};
    gert::StorageShape reserveSpace3Shape = {{1}, {1}};

    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"NpuArch", "3510"}, {"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::BatchNormCompileInfo compileInfo;

    std::string opType("BatchNorm");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    ASSERT_NE(opImpl, nullptr);
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .SetOpType(opType)
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(5, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &scaleShape, &scaleShape, &scaleShape, &scaleShape})
                      .OutputShapes({&xShape, &scaleShape, &scaleShape, &scaleShape, &scaleShape, &reserveSpace3Shape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, format, format)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, format, format)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5f)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                                  {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"exponential_avg_factor", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), expectedTilingKey);
    if (inferTilingData != nullptr) {
        auto rawTilingData = tilingContext->GetRawTilingData();
        ASSERT_NE(rawTilingData, nullptr);
        ASSERT_EQ(rawTilingData->GetDataSize(), sizeof(BatchNormInferTilingDataForUt));
        *inferTilingData = *reinterpret_cast<const BatchNormInferTilingDataForUt*>(rawTilingData->GetData());
    }
    if (lastChannelTilingData != nullptr) {
        auto rawTilingData = tilingContext->GetRawTilingData();
        ASSERT_NE(rawTilingData, nullptr);
        ASSERT_EQ(rawTilingData->GetDataSize(), sizeof(BatchNormInferLastChannelTilingDataForUt));
        *lastChannelTilingData = *reinterpret_cast<const BatchNormInferLastChannelTilingDataForUt*>(
            rawTilingData->GetData());
    }
}
// training 场景（is_training = true）的 tiling 入口，用于校验模板选择。
// tilingDataOut 非空时把 raw tiling data 原样带出，供逐字节比对。
// ubSize 可配，便于构造 UB 不足的降级场景；expectTilingFailure 为 true 时只断言 tiling 失败。
static void RunBatchNormTrainingTilingForTest(gert::StorageShape& xShape, ge::Format format, int64_t channel,
                                              uint64_t expectedTilingKey, std::string* tilingDataOut = nullptr,
                                              int64_t ubSize = 245760, bool expectTilingFailure = false)
{
    gert::StorageShape scaleShape = {{channel}, {channel}};
    gert::StorageShape reserveSpace3Shape = {{1}, {1}};

    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": )" +
                               std::to_string(ubSize) + R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"NpuArch", "3510"}, {"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::BatchNormCompileInfo compileInfo;

    std::string opType("BatchNorm");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    ASSERT_NE(opImpl, nullptr);
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .SetOpType(opType)
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(5, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &scaleShape, &scaleShape, &scaleShape, &scaleShape})
                      .OutputShapes({&xShape, &scaleShape, &scaleShape, &scaleShape, &scaleShape, &reserveSpace3Shape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, format, format)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, format, format)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-4f)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                                  {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"exponential_avg_factor", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    if (expectTilingFailure) {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
        return;
    }
    ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), expectedTilingKey);
    if (tilingDataOut != nullptr) {
        auto rawTilingData = tilingContext->GetRawTilingData();
        ASSERT_NE(rawTilingData, nullptr);
        tilingDataOut->assign(reinterpret_cast<const char*>(rawTilingData->GetData()), rawTilingData->GetDataSize());
    }
}
} // namespace

class BatchNormTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BatchNormTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchNormTiling TearDown" << std::endl; }
};

TEST_F(BatchNormTilingTest, batch_norm_tiling_ra_full_reduce_arch35_test_0)
{
    gert::StorageShape x_shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape scale_shape = {{5}, {5}};
    gert::StorageShape reserve_space_3_shape = {{1}, {1}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version = {{"NpuArch", "3510"}, {"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormCompileInfo compile_info;

    std::string op_type("BatchNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .SetOpType(op_type)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1})
                      .InputShapes({&x_shape, &scale_shape, &scale_shape, &scale_shape, &scale_shape})
                      .OutputShapes(
                          {&x_shape, &scale_shape, &scale_shape, &scale_shape, &scale_shape, &reserve_space_3_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NHWC")}})
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
    ASSERT_EQ(tiling_key, 400000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

TEST_F(BatchNormTilingTest, batch_norm_tiling_infer_nchw_float32_core8_arch35_test_0)
{
    gert::StorageShape x_shape = {{1, 2048, 7, 7}, {1, 2048, 7, 7}};
    gert::StorageShape scale_shape = {{2048}, {2048}};
    gert::StorageShape reserve_space_3_shape = {{1}, {1}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 8, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version = {{"NpuArch", "3510"}, {"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormCompileInfo compile_info;

    std::string op_type("BatchNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .SetOpType(op_type)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(compile_info.coreNum, 8);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1})
                      .InputShapes({&x_shape, &scale_shape, &scale_shape, &scale_shape, &scale_shape})
                      .OutputShapes(
                          {&x_shape, &scale_shape, &scale_shape, &scale_shape, &scale_shape, &reserve_space_3_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-4f)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                                  {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"exponential_avg_factor", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 910000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    ASSERT_GE(tilingData->GetDataSize(), sizeof(BatchNormInferTilingDataForUt));
    const auto* inferTilingData = reinterpret_cast<const BatchNormInferTilingDataForUt*>(tilingData->GetData());
    int64_t usedUbSize = DOUBLE_BUFFER_NUM *
                         (X_Y_QUEUE_NUM * inferTilingData->tileBlockB0Len * inferTilingData->tileBlockALen *
                              inferTilingData->tileBlockB1Len * FLOAT32_BYTES_NUM +
                          SMALL_TENSOR_QUEUE_NUM * inferTilingData->tileBlockALen * FLOAT32_BYTES_NUM);
    ASSERT_LE(usedUbSize, compile_info.ubSize);
}

TEST_F(BatchNormTilingTest, batch_norm_tiling_infer_ncdhw_small_ab1_b0_boundary)
{
    BatchNormInferTilingDataForUt tilingData{};
    constexpr int64_t coreNum = 64;
    constexpr int64_t minSmallAB1B0Len = coreNum * 2;

    gert::StorageShape b0BelowShape = {{minSmallAB1B0Len - 1, 3, 1, 1, 2}, {minSmallAB1B0Len - 1, 3, 1, 1, 2}};
    RunBatchNormInferTilingForTest(b0BelowShape, ge::FORMAT_NCDHW, 3, 910000, &tilingData);
    EXPECT_EQ(tilingData.totalB0Len, minSmallAB1B0Len - 1);
    EXPECT_EQ(tilingData.totalALen, 3);
    EXPECT_EQ(tilingData.totalB1Len, 2);

    gert::StorageShape b0BoundaryShape = {{minSmallAB1B0Len, 3, 1, 1, 2}, {minSmallAB1B0Len, 3, 1, 1, 2}};
    RunBatchNormInferTilingForTest(b0BoundaryShape, ge::FORMAT_NCDHW, 3, 910000, &tilingData);
    EXPECT_EQ(tilingData.totalB0Len, minSmallAB1B0Len);
    EXPECT_EQ(tilingData.totalALen, 3);
    EXPECT_EQ(tilingData.totalB1Len, 2);
}

TEST_F(BatchNormTilingTest, batch_norm_tiling_infer_nd_small_ab1)
{
    gert::StorageShape targetShape = {{65536, 1, 16}, {65536, 1, 16}};
    RunBatchNormInferTilingForTest(targetShape, ge::FORMAT_ND, 1, 911000);
}

TEST_F(BatchNormTilingTest, batch_norm_tiling_infer_nchw_fp16_vector32_aligned_ub_arch35_test_0)
{
    gert::StorageShape x_shape = {{1, 512, 7, 7}, {1, 512, 7, 7}};
    gert::StorageShape scale_shape = {{512}, {512}};
    gert::StorageShape reserve_space_3_shape = {{1}, {1}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 131072, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 1, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version = {{"NpuArch", "3510"}, {"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::BatchNormCompileInfo compile_info;

    std::string op_type("BatchNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .SetOpType(op_type)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    compile_info.vectorLength = 32;
    ASSERT_EQ(compile_info.coreNum, 1);
    ASSERT_EQ(compile_info.ubSize, 131072);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1})
                      .InputShapes({&x_shape, &scale_shape, &scale_shape, &scale_shape, &scale_shape})
                      .OutputShapes(
                          {&x_shape, &scale_shape, &scale_shape, &scale_shape, &scale_shape, &reserve_space_3_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-4f)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                                  {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"exponential_avg_factor", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling_context->GetTilingKey(), 910000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    ASSERT_GE(tilingData->GetDataSize(), sizeof(BatchNormInferTilingDataForUt));
    const auto* inferTilingData = reinterpret_cast<const BatchNormInferTilingDataForUt*>(tilingData->GetData());
    ASSERT_EQ(inferTilingData->tileBlockALen, 233);
    ASSERT_EQ(inferTilingData->tileBlockB0Len, 1);
    ASSERT_EQ(inferTilingData->tileBlockB1Len, 64);

    int64_t xyBufferSize = inferTilingData->tileBlockB0Len * inferTilingData->tileBlockALen *
                           inferTilingData->tileBlockB1Len * FLOAT16_BYTES_NUM;
    int64_t paramBufferSize = inferTilingData->tileBlockALen * FLOAT32_BYTES_NUM;
    int64_t alignedUsedUbSize = DOUBLE_BUFFER_NUM *
                                (X_Y_QUEUE_NUM * AlignUpForUt(xyBufferSize, compile_info.blockSize) +
                                 SMALL_TENSOR_QUEUE_NUM * AlignUpForUt(paramBufferSize, compile_info.blockSize));
    ASSERT_EQ(alignedUsedUbSize, 130816);
    ASSERT_LE(alignedUsedUbSize, compile_info.ubSize);
}

TEST_F(BatchNormTilingTest, batch_norm_tiling_infer_last_channel_small_a_b_boundary)
{
    gert::StorageShape belowShape = {{65535, 8, 1, 1}, {65535, 8, 1, 1}};
    RunBatchNormInferTilingForTest(belowShape, ge::FORMAT_NCHW, 8, 900000);

    gert::StorageShape boundaryShape = {{65536, 8, 1, 1}, {65536, 8, 1, 1}};
    RunBatchNormInferTilingForTest(boundaryShape, ge::FORMAT_NCHW, 8, 900000);

    gert::StorageShape targetShape = {{65537, 8, 1, 1}, {65537, 8, 1, 1}};
    RunBatchNormInferTilingForTest(targetShape, ge::FORMAT_NCHW, 8, 902000);
}

TEST_F(BatchNormTilingTest, batch_norm_tiling_infer_nhwc_continuous_a)
{
    gert::StorageShape aOuterOneShape = {{8192, 1, 8, 33}, {8192, 1, 8, 33}};
    RunBatchNormInferTilingForTest(aOuterOneShape, ge::FORMAT_NHWC, 33, 900000);

    gert::StorageShape targetShapeA84 = {{256, 42, 42, 84}, {256, 42, 42, 84}};
    RunBatchNormInferTilingForTest(targetShapeA84, ge::FORMAT_NHWC, 84, 901000);

    gert::StorageShape targetShapeA168 = {{256, 42, 42, 168}, {256, 42, 42, 168}};
    RunBatchNormInferTilingForTest(targetShapeA168, ge::FORMAT_NHWC, 168, 901000);

    gert::StorageShape targetShapeA336 = {{256, 42, 42, 336}, {256, 42, 42, 336}};
    RunBatchNormInferTilingForTest(targetShapeA336, ge::FORMAT_NHWC, 336, 901000);

    BatchNormInferLastChannelTilingDataForUt tilingData{};
    gert::StorageShape unalignedAShape = {{8193, 1, 8, 65}, {8193, 1, 8, 65}};
    RunBatchNormInferTilingForTest(unalignedAShape, ge::FORMAT_NHWC, 65, 901000, nullptr, &tilingData);
    constexpr int64_t ubSize = 245760;
    constexpr int64_t vlFp32 = 64;
    constexpr int64_t paramNum = 6;
    int64_t paramAlignLen = (tilingData.tileBlockALen + vlFp32 - 1) / vlFp32 * vlFp32;
    int64_t paramBytes = DOUBLE_BUFFER_NUM * paramNum * FLOAT32_BYTES_NUM * paramAlignLen;
    int64_t paramCacheBytes = paramNum * FLOAT32_BYTES_NUM * paramAlignLen;
    int64_t xYBytes = tilingData.tileBlockBLen * tilingData.tileBlockALen * X_Y_QUEUE_NUM * DOUBLE_BUFFER_NUM *
                      FLOAT32_BYTES_NUM;
    EXPECT_LE(paramBytes + paramCacheBytes + xYBytes, ubSize);

    gert::StorageShape smallAShape = {{8192, 1, 8, 8}, {8192, 1, 8, 8}};
    RunBatchNormInferTilingForTest(smallAShape, ge::FORMAT_NHWC, 8, 900000);

    gert::StorageShape aOuterExceedShape = {{256, 16, 16, 512}, {256, 16, 16, 512}};
    RunBatchNormInferTilingForTest(aOuterExceedShape, ge::FORMAT_NHWC, 512, 900000);

    gert::StorageShape beyondContinuousAShape = {{256, 16, 16, 513}, {256, 16, 16, 513}};
    RunBatchNormInferTilingForTest(beyondContinuousAShape, ge::FORMAT_NHWC, 513, 900000);
}

// ND 且 r0 > 1（x 折算成 r1=dim0, a=dim1, r0=其余）时应由 FullReduce(200000) 承接。
// 历史上 FullReduce / RARBlockSplitR 的 IsCapable 只收 NCHW/NCDHW，这类 shape 会落到
// 兜底的 Welford(300000)，而 Welford 不支持 r1*r0 <= vlFp32 的归约长度。
TEST_F(BatchNormTilingTest, batch_norm_tiling_nd_r0_gt_one_full_reduce_arch35)
{
    // r1*r0 = 32 < vlFp32(64)
    gert::StorageShape ndSmallShape = {{8, 512, 4}, {8, 512, 4}};
    RunBatchNormTrainingTilingForTest(ndSmallShape, ge::FORMAT_ND, 512, 200000);

    // r1*r0 = 64 == vlFp32
    gert::StorageShape ndVlShape = {{8, 512, 8}, {8, 512, 8}};
    RunBatchNormTrainingTilingForTest(ndVlShape, ge::FORMAT_ND, 512, 200000);

    // r1*r0 = 40，非 2 的幂
    gert::StorageShape ndNonPow2Shape = {{8, 512, 5}, {8, 512, 5}};
    RunBatchNormTrainingTilingForTest(ndNonPow2Shape, ge::FORMAT_ND, 512, 200000);

    // 4D ND
    gert::StorageShape nd4dShape = {{8, 512, 4, 1}, {8, 512, 4, 1}};
    RunBatchNormTrainingTilingForTest(nd4dShape, ge::FORMAT_ND, 512, 200000);
}

// 同 dims 的 ND 与 NCHW 折算出的 (r1, a, r0) 相同，模板与 tiling data 必须完全一致。
// kernel 侧只消费 tiling data，不感知 format，这条用例把该等价性钉住。
TEST_F(BatchNormTilingTest, batch_norm_tiling_nd_nchw_equivalence_arch35)
{
    gert::StorageShape ndShape = {{8, 512, 4}, {8, 512, 4}};
    gert::StorageShape nchwShape = {{8, 512, 4}, {8, 512, 4}};
    std::string ndTilingData;
    std::string nchwTilingData;
    RunBatchNormTrainingTilingForTest(ndShape, ge::FORMAT_ND, 512, 200000, &ndTilingData);
    RunBatchNormTrainingTilingForTest(nchwShape, ge::FORMAT_NCHW, 512, 200000, &nchwTilingData);
    ASSERT_EQ(ndTilingData, nchwTilingData);

    gert::StorageShape ndBigShape = {{2048, 32, 16}, {2048, 32, 16}};
    gert::StorageShape nchwBigShape = {{2048, 32, 16}, {2048, 32, 16}};
    std::string ndBigTilingData;
    std::string nchwBigTilingData;
    RunBatchNormTrainingTilingForTest(ndBigShape, ge::FORMAT_ND, 32, 300000, &ndBigTilingData);
    RunBatchNormTrainingTilingForTest(nchwBigShape, ge::FORMAT_NCHW, 32, 300000, &nchwBigTilingData);
    ASSERT_EQ(ndBigTilingData, nchwBigTilingData);
}

// r1*r0 足够大、FullReduce 的 UB 判据过不去时，仍应落到 Welford(300000)，
// 确认 Welford 的 IsCapable 守卫没有把它本该承接的场景一起挡掉。
TEST_F(BatchNormTilingTest, batch_norm_tiling_large_reduce_still_welford_arch35)
{
    gert::StorageShape ndShape = {{2048, 32, 16}, {2048, 32, 16}};
    RunBatchNormTrainingTilingForTest(ndShape, ge::FORMAT_ND, 32, 300000);

    gert::StorageShape nchwShape = {{2048, 32, 16}, {2048, 32, 16}};
    RunBatchNormTrainingTilingForTest(nchwShape, ge::FORMAT_NCHW, 32, 300000);
}

// a 小到满足 RARBlockSplitR 的 a * 2 < blockDim、且 r1 * r0 大到 FullReduce 的 UB 判据过不去时，
// 应落到 RARBlockSplitR(250000)。该模板只消费基类折算出的 (r1, a, r0)，
// 因此 ND 与同 dims 的 NCHW 必须落到同一模板。
TEST_F(BatchNormTilingTest, batch_norm_tiling_nd_rar_block_split_r_arch35)
{
    gert::StorageShape ndShape = {{4096, 16, 8}, {4096, 16, 8}};
    gert::StorageShape nchwShape = {{4096, 16, 4, 2}, {4096, 16, 4, 2}};
    std::string ndTilingData;
    std::string nchwTilingData;
    RunBatchNormTrainingTilingForTest(ndShape, ge::FORMAT_ND, 16, 250000, &ndTilingData);
    RunBatchNormTrainingTilingForTest(nchwShape, ge::FORMAT_NCHW, 16, 250000, &nchwTilingData);
    ASSERT_EQ(ndTilingData, nchwTilingData);
}

// UB 不足时 Welford 的固定开销会超过总 UB。ubRemain 若按无符号数相减会下溢成巨大值，
// 进而算出无意义的切分参数却报 tiling 成功；这里断言其明确失败。
// r1 * r0 = 2048 远大于一个 VL，FullReduce 的 UB 判据在该 UB 下过不去，
// RARBlockSplitR 也因 a * 2 >= blockDim 拒收，故会落到 Welford。
TEST_F(BatchNormTilingTest, batch_norm_tiling_welford_ub_not_enough_arch35)
{
    gert::StorageShape ndShape = {{256, 512, 8}, {256, 512, 8}};
    RunBatchNormTrainingTilingForTest(ndShape, ge::FORMAT_ND, 512, 0, nullptr, 1024, true);
}

// ubRemain 为正、但不足一个对齐单位的窗口：a = 512 时固定开销为
// smallUbSize(2304) + binaryAddBufSize(32) + blockSize * BLOCK_RESERVE_NUMBER(224) = 2560，
// UB = 2624 使 ubRemain = 64 > 0，但 ubSize = 64 / 24 = 2、ubSizeAlign 被向下对齐抹成 0。
// 只校验 ubRemain > 0 覆盖不到这一档，故校验放在对齐之后。
TEST_F(BatchNormTilingTest, batch_norm_tiling_welford_ub_align_to_zero_arch35)
{
    gert::StorageShape ndShape = {{256, 512, 8}, {256, 512, 8}};
    RunBatchNormTrainingTilingForTest(ndShape, ge::FORMAT_ND, 512, 0, nullptr, 2624, true);
}

// ubSizeAlign 为正、但 parallelN 不足一个 VL 的窗口。同样的固定开销 2560，UB = 2752 使
// ubRemain = 192、ubSize = 192 / 24 = 8、ubSizeAlign = 8 通过了非零校验；随后 r0 = 8 >= 8
// 走第一分支，parallelN = 8 <= vlFp32(64)，binaryAddQuotient 退化成半个 VL、binaryAddLastNum
// 为 0，归约结果恒为 0。IsCapable 里按 r1 * r0(2048) 判的守卫对这一档无效——parallelN 恒
// 小于等于 r1 * r0，前者大于 vlFp32 推不出后者也大于——故守卫必须落在 parallelN 上。
TEST_F(BatchNormTilingTest, batch_norm_tiling_welford_parallel_n_below_vl_arch35)
{
    gert::StorageShape ndShape = {{256, 512, 8}, {256, 512, 8}};
    RunBatchNormTrainingTilingForTest(ndShape, ge::FORMAT_ND, 512, 0, nullptr, 2752, true);
}
