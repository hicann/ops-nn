/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "rnn/dynamic_augru/op_host/arch35/dynamic_augru_tiling_arch35.h"
#include "test_cube_util.h"

namespace {
struct RuntimeTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    int64_t time = 0;
    int64_t batch = 0;
    int64_t input = 0;
    int64_t hidden = 0;
    size_t workspace = 0;
    uint32_t blocks = 0;
    uint64_t key = 0;
};

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableStorageShape().AppendDim(dim);
        shape.MutableOriginShape().AppendDim(dim);
    }
    return shape;
}

RuntimeTilingResult RunRuntimeTiling(int64_t time, int64_t batch, int64_t input, int64_t hidden,
                                     uint32_t cubeCores = 32, ge::DataType stateType = ge::DT_FLOAT,
                                     uint32_t sequenceMode = 1)
{
    auto xShape = MakeShape({time, batch, input});
    auto weightInputShape = MakeShape({input, 3 * hidden});
    auto weightHiddenShape = MakeShape({hidden, 3 * hidden});
    auto weightAttentionShape = MakeShape({time, batch});
    auto biasInputShape = MakeShape({3 * hidden});
    auto biasHiddenShape = MakeShape({3 * hidden});
    auto sequenceLengthShape = sequenceMode == 2 ? MakeShape({time, batch, hidden}) : MakeShape({batch});
    auto initHShape = MakeShape({1, batch, hidden});
    std::vector<gert::StorageShape> outputShapes(7, MakeShape({time, batch, hidden}));
    std::vector<gert::StorageShape*> outputShapePointers;
    for (auto& shape : outputShapes) {
        outputShapePointers.push_back(&shape);
    }

    const std::string hardwareInfo = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore",
        "socVersion": "Ascend950"}})";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> version;
    GetPlatFormInfos(hardwareInfo.c_str(), socInfos, aicoreSpec, intrinsics, version);
    socInfos["cube_core_cnt"] = std::to_string(cubeCores);
    aicoreSpec["cube_freq"] = "1650";

    fe::PlatFormInfos platformInfo;
    if (!platformInfo.Init()) {
        ADD_FAILURE() << "Failed to initialize the Ascend950 platform fixture";
        return {};
    }
    optiling::DynamicAUGRUCompileInfo compileInfo;
    compileInfo.aicCoreNum = cubeCores;
    compileInfo.aivCoreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.blockSize = 256;
    compileInfo.vectorLength = 256;
    compileInfo.isArch35 = true;
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAUGRU");
    if (opImpl == nullptr) {
        ADD_FAILURE() << "DynamicAUGRU implementation is not registered";
        return {};
    }
    if (opImpl->tiling == nullptr) {
        ADD_FAILURE() << "DynamicAUGRU tiling callback is not registered; build this test with --soc=ascend950";
        return {};
    }

    auto tilingData = gert::TilingData::CreateCap(16384);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (tilingData == nullptr || workspace == nullptr) {
        return {};
    }

    auto holder = gert::TilingContextFaker()
                      .SetOpType("DynamicAUGRU")
                      .NodeIoNum(8, 7)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &weightInputShape, &weightHiddenShape, &weightAttentionShape,
                                    &biasInputShape, &biasHiddenShape,
                                    sequenceMode == 0 ? nullptr : &sequenceLengthShape, &initHShape})
                      .OutputShapes(outputShapePointers)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, stateType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, stateType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, sequenceMode == 2 ? ge::DT_FLOAT16 : ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, stateType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, stateType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, stateType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, stateType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, stateType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, stateType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, stateType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(6, stateType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"direction", Ops::NN::AnyValue::CreateFrom<std::string>("UNIDIRECTIONAL")},
                          {"cell_depth", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"keep_prob", Ops::NN::AnyValue::CreateFrom<float>(1.0F)},
                          {"cell_clip", Ops::NN::AnyValue::CreateFrom<float>(-1.0F)},
                          {"num_proj", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"time_major", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                          {"activation", Ops::NN::AnyValue::CreateFrom<std::string>("tanh")},
                          {"gate_order", Ops::NN::AnyValue::CreateFrom<std::string>("zrh")},
                          {"reset_after", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                          {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                      })
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto* context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr) {
        ADD_FAILURE() << "Failed to create the DynamicAUGRU tiling context";
        return {};
    }
    auto* contextPlatform = context->GetPlatformInfo();
    if (contextPlatform == nullptr) {
        ADD_FAILURE() << "DynamicAUGRU tiling context has no platform information";
        return {};
    }
    contextPlatform->SetPlatformRes("version", version);
    contextPlatform->SetPlatformRes("SoCInfo", socInfos);
    contextPlatform->SetPlatformRes("AICoreSpec", aicoreSpec);
    contextPlatform->SetCoreNumByCoreType("AICore");
    contextPlatform->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    RuntimeTilingResult result;
    result.status = opImpl->tiling(context);
    if (result.status != ge::GRAPH_SUCCESS) {
        return result;
    }
    const auto* dimensions = reinterpret_cast<const int64_t*>(context->GetRawTilingData()->GetData());
    result.time = dimensions[0];
    result.batch = dimensions[1];
    result.input = dimensions[2];
    result.hidden = dimensions[3];
    result.workspace = context->GetWorkspaceSizes(1)[0];
    result.blocks = context->GetBlockDim();
    result.key = context->GetTilingKey();
    return result;
}
} // namespace

TEST(DynamicAUGRUTiling, regenerates_tiling_from_each_runtime_shape)
{
    const auto first = RunRuntimeTiling(3, 2, 5, 7);
    ASSERT_EQ(first.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(first.time, 3);
    EXPECT_EQ(first.batch, 2);
    EXPECT_EQ(first.input, 5);
    EXPECT_EQ(first.hidden, 7);

    const auto second = RunRuntimeTiling(9, 4, 11, 13);
    ASSERT_EQ(second.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(second.time, 9);
    EXPECT_EQ(second.batch, 4);
    EXPECT_EQ(second.input, 11);
    EXPECT_EQ(second.hidden, 13);
    EXPECT_NE(first.workspace, second.workspace);
}

TEST(DynamicAUGRUTiling, rejects_unresolved_compile_shape_at_runtime)
{
    EXPECT_EQ(RunRuntimeTiling(-1, 2, 5, 7).status, ge::GRAPH_FAILED);
}

TEST(DynamicAUGRUTiling, scales_regbase_parallelism_at_runtime_batch_boundaries)
{
    const auto shortSequence = RunRuntimeTiling(127, 128, 64, 128);
    const auto longSequence = RunRuntimeTiling(128, 128, 64, 128);
    const auto smallBatch = RunRuntimeTiling(128, 127, 64, 128);
    const auto largeBatch = RunRuntimeTiling(1, 1024, 64, 128);
    const auto wideRecurrent = RunRuntimeTiling(64, 870, 512, 1024);
    ASSERT_EQ(shortSequence.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(longSequence.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(smallBatch.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(largeBatch.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(shortSequence.blocks, 16U);
    EXPECT_EQ(longSequence.blocks, 16U);
    EXPECT_EQ(smallBatch.blocks, 15U);
    EXPECT_EQ(largeBatch.blocks, 32U);
    ASSERT_EQ(wideRecurrent.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(wideRecurrent.blocks, 32U);
}

TEST(DynamicAUGRUTiling, bounds_regbase_launch_for_all_shapes)
{
    const auto limitedHardware = RunRuntimeTiling(128, 128, 64, 128, 6);
    const auto tinyState = RunRuntimeTiling(128, 128, 4, 4);
    const auto wideShape = RunRuntimeTiling(128, 128, 64, 1024);
    ASSERT_EQ(limitedHardware.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(tinyState.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(wideShape.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(limitedHardware.blocks, 6U);
    EXPECT_EQ(tinyState.blocks, 1U);
    EXPECT_EQ(wideShape.blocks, 32U);
}

TEST(DynamicAUGRUTiling, dtype_profiles_share_algorithm_keys)
{
    for (ge::DataType dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        const auto fast = RunRuntimeTiling(3, 2, 5, 7, 32, dtype);
        const auto wideShape = RunRuntimeTiling(5, 1, 5, 769, 32, dtype);
        ASSERT_EQ(fast.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(wideShape.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(fast.key, 1U);
        EXPECT_EQ(wideShape.key, 1U);
    }
}

TEST(DynamicAUGRUTiling, does_not_launch_empty_batch_tasks)
{
    const auto oneBatch = RunRuntimeTiling(2, 1, 33, 8192);
    const auto fewBatches = RunRuntimeTiling(2, 5, 33, 1024);
    ASSERT_EQ(oneBatch.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(fewBatches.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(oneBatch.blocks, 1U);
    EXPECT_EQ(fewBatches.blocks, 2U);
    EXPECT_EQ(RunRuntimeTiling(2, 128, 33, 128, 0).status, ge::GRAPH_FAILED);
}

TEST(DynamicAUGRUTiling, selects_all_sequence_templates)
{
    for (uint32_t mode : {0U, 1U, 2U}) {
        const auto result = RunRuntimeTiling(3, 2, 5, 7, 32, ge::DT_FLOAT, mode);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(result.key, mode);
    }
}
