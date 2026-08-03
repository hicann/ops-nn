/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file test_apply_adam_w_quant_arch35_tiling.cpp
 * \brief ApplyAdamWQuant arch35 (Ascend950) regbase tiling UT.
 *
 * 覆盖 arch35 regbase tiling 路径:直接调 tiling_func 走 ApplyAdamWQuantRegbaseTiling。验证 3 个 dtype 的 TilingKey
 * (fp32=100/fp16=200/bf16=300)与非法 block_size 拒绝。
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/apply_adam_w_quant_tiling_arch35.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "array_ops.h"
#include "tests/ut/common/ut_op_util.h"
#include "tests/ut/common/any_value.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class ApplyAdamWQuantArch35Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyAdamWQuantArch35Tiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ApplyAdamWQuantArch35Tiling TearDown" << std::endl; }
};

static void Arch35TilingTest(std::initializer_list<int64_t> varShape,    // var/grad/m/v 同 shape
                             std::initializer_list<int64_t> absmaxShape, // absmax_m/absmax_v shape
                             int64_t blockSize, ge::DataType varGradDtype, const ge::graphStatus expectStatus,
                             uint64_t tilingKeyValue, uint64_t expectedLastBlockSize = 0)
{
    std::string opType("ApplyAdamWQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // Ascend950 平台:CORE_NUM 64 / UB 245760,socVersion 触发 IsRegbaseSocVersion=true
    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, socVersion);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::ApplyAdamWQuantRegbaseCompileInfo compile_info;

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape input1Shape = {varShape, varShape};
    gert::StorageShape input2Shape = {varShape, varShape};
    gert::StorageShape input3Shape = {varShape, varShape};
    gert::StorageShape input4Shape = {varShape, varShape};
    gert::StorageShape qmapShape = {{256}, {256}};
    gert::StorageShape input7Shape = {absmaxShape, absmaxShape};
    gert::StorageShape input8Shape = {absmaxShape, absmaxShape};
    gert::StorageShape stepShape = {{1}, {1}};
    gert::StorageShape output4Shape = {absmaxShape, absmaxShape};

    std::vector<uint32_t> irInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(9, 5)
                      .IrInstanceNum(irInstanceNum)
                      .InputShapes({&input1Shape, &input2Shape, &input3Shape, &input4Shape, &qmapShape, &qmapShape,
                                    &input7Shape, &input8Shape, &stepShape})
                      .OutputShapes({&input1Shape, &input3Shape, &input4Shape, &output4Shape, &output4Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varGradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, varGradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, varGradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom<float>(0.1f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom<float>(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom<float>(0.9f)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom<float>(0.9f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.0001f)},
                                  {"gnorm_scale", Ops::NN::AnyValue::CreateFrom<float>(0.9f)},
                                  {"quant_mode", Ops::NN::AnyValue::CreateFrom<std::string>("BLOCKWISE")},
                                  {"block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(blockSize)}})
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    EXPECT_EQ(tiling_func(tiling_context), expectStatus);
    if (expectStatus == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(tiling_context->GetTilingKey(), tilingKeyValue);
        if (expectedLastBlockSize != 0) {
            auto rawTilingData = tiling_context->GetRawTilingData();
            ASSERT_NE(rawTilingData, nullptr);
            ASSERT_GE(rawTilingData->GetDataSize(), sizeof(ApplyAdamWQuantRegbaseTilingData));
            const auto* tilingData = reinterpret_cast<const ApplyAdamWQuantRegbaseTilingData*>(
                rawTilingData->GetData());
            ASSERT_NE(tilingData, nullptr);
            EXPECT_EQ(tilingData->last_block_size, expectedLastBlockSize);
            EXPECT_GT(tilingData->last_core_last_block, 0);
            EXPECT_LE(tilingData->last_core_last_block, tilingData->one_core_do_block_num_per_row);
        }
    }
}

TEST_F(ApplyAdamWQuantArch35Tiling, arch35_tilingkey_100_fp32)
{
    Arch35TilingTest({96, 256}, {96}, 256, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 100);
}

TEST_F(ApplyAdamWQuantArch35Tiling, arch35_tilingkey_200_fp16)
{
    Arch35TilingTest({96, 256}, {96}, 256, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 200);
}

TEST_F(ApplyAdamWQuantArch35Tiling, arch35_tilingkey_300_bf16)
{
    Arch35TilingTest({96, 256}, {96}, 256, ge::DT_BF16, ge::GRAPH_SUCCESS, 300);
}

TEST_F(ApplyAdamWQuantArch35Tiling, arch35_multiblock_fp32)
{
    // 多 block(1024 = 4 blocks per row)覆盖分核路径
    Arch35TilingTest({16, 1024}, {64}, 256, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 100);
}

TEST_F(ApplyAdamWQuantArch35Tiling, arch35_partial_last_block_fp32)
{
    // 100000 = 390 * 256 + 160，kernel 仅应回写尾块的 160 个有效元素。
    Arch35TilingTest({100000}, {391}, 256, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 100, 160);
}

TEST_F(ApplyAdamWQuantArch35Tiling, arch35_unaligned_partial_last_blocks_fp32)
{
    for (int64_t tail : {1, 7, 8, 9, 255}) {
        SCOPED_TRACE(testing::Message() << "tail=" << tail);
        Arch35TilingTest({256 + tail}, {2}, 256, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 100, tail);
    }
}

TEST_F(ApplyAdamWQuantArch35Tiling, arch35_short_absmax_rejected)
{
    // 257 个状态元素需要 2 个 absmax，不能按 floor(size / block_size) 仅提供 1 个。
    Arch35TilingTest({257}, {1}, 256, ge::DT_FLOAT, ge::GRAPH_FAILED, 0);
}

TEST_F(ApplyAdamWQuantArch35Tiling, arch35_invalid_block_size_rejected)
{
    // block_size != 256 应被拒绝
    Arch35TilingTest({96, 256}, {96}, 128, ge::DT_FLOAT, ge::GRAPH_FAILED, 0);
}
