/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../op_host/rms_norm_dynamic_quant_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
const std::string OP_TYPE = "RmsNormDynamicQuant";
constexpr int kInputNum = 4;  // x, gamma, smooth_scales, beta
constexpr int kOutputNum = 2; // y, scale

const std::string COMPILE_INFO_910B = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 40
      }
    })";
} // namespace

class RmsNormDynamicQuantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmsNormDynamicQuantTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "RmsNormDynamicQuantTilingTest TearDown" << std::endl; }
};

// 通用 RunTiling
//   irInstanceNum: 4 个输入的实例数，可选输入不存在填 0 (如 {1,1,0,0})
//   inputShapes:   只填存在的输入 (如 {&x, &gamma})
//   NodeInputTd 全量调用 4 次，faker 内部根据 IrInstanceNum 自动处理多余调用
static ge::graphStatus RunTiling(const std::string& compileInfoJson, const vector<uint32_t>& irInstanceNum,
                                 ge::DataType inDtype, const vector<ge::DataType>& outputDtypes,
                                 const vector<gert::StorageShape*>& inputShapes,
                                 const vector<gert::StorageShape*>& outputShapes,
                                 const vector<pair<string, Ops::NN::AnyValue>>& attrs,
                                 uint32_t* outputTilingKey = nullptr)
{
    map<string, string> socInfos, aicoreSpec, intrinsics;
    GetPlatFormInfos(compileInfoJson.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::RmsNormDynamicQuantCompileInfo compileInfo;

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE.c_str());
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr)
        return ge::GRAPH_FAILED;
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoJson.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    auto* parseCtx = kernelHolder.GetContext<gert::TilingParseContext>();
    EXPECT_TRUE(parseCtx->GetPlatformInfo()->Init());
    parseCtx->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseCtx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    EXPECT_NE(tilingData, nullptr);

    gert::TilingContextFaker faker;
    faker.NodeIoNum(kInputNum, kOutputNum)
        .IrInstanceNum(irInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes(outputShapes)
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtypes[0], ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, outputDtypes[1], ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs(attrs)
        .TilingData(tilingData.get())
        .Workspace(wsSize);

    auto holder = faker.Build();
    auto* tilingCtx = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tilingCtx->GetPlatformInfo(), nullptr);

    ge::graphStatus status = tilingFunc(tilingCtx);
    if (outputTilingKey != nullptr)
        *outputTilingKey = tilingCtx->GetTilingKey();
    return status;
}

// ========================================================================
// {1,1,1,1}: 全部输入存在
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, normal_fp16_all_exist)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape smooth = {{16}, {16}};
    gert::StorageShape beta = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    uint32_t key = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 1, 1}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma, &smooth, &beta},
                        {&y, &scale}, {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}, &key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1u);
}

// ========================================================================
// {1,1,0,0}: smooth=None, beta=None
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, opt_no_smooth_no_beta)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    uint32_t key = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 0, 0}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma}, {&y, &scale},
                        {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}, &key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1u);
}

// ========================================================================
// {1,1,1,0}: smooth=Exist, beta=None
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, opt_has_smooth_no_beta)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape smooth = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    uint32_t key = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 1, 0}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma, &smooth},
                        {&y, &scale}, {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}, &key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1u);
}

// ========================================================================
// {1,1,1,1}: BF16 dtype coverage
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, normal_bf16_all_exist)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape smooth = {{16}, {16}};
    gert::StorageShape beta = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    uint32_t key = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 1, 1}, DT_BF16, {DT_INT8, DT_FLOAT}, {&x, &gamma, &smooth, &beta},
                        {&y, &scale}, {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}, &key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1u);
}

// ========================================================================
// Slice-D tiling: 大列场景触发 SliceD 路径
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, slice_d_large_col)
{
    gert::StorageShape x = {{1, 1, 30000}, {1, 1, 30000}};
    gert::StorageShape gamma = {{30000}, {30000}};
    gert::StorageShape smooth = {{30000}, {30000}};
    gert::StorageShape beta = {{30000}, {30000}};
    gert::StorageShape y = {{1, 1, 30000}, {1, 1, 30000}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    uint32_t key = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 1, 1}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma, &smooth, &beta},
                        {&y, &scale}, {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}, &key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 3u);
}

// ========================================================================
// Error case: gamma 最后一维与 x 不匹配
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, error_gamma_lastdim_mismatch)
{
    gert::StorageShape x = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape gamma = {{2000}, {2000}};
    gert::StorageShape smooth = {{2000}, {2000}};
    gert::StorageShape beta = {{2000}, {2000}};
    gert::StorageShape y = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape scale = {{24, 1}, {24, 1}};
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 1, 1}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma, &smooth, &beta},
                        {&y, &scale}, {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}),
              ge::GRAPH_FAILED);
}
