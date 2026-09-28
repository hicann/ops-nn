/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_augru_grad_tiling.cpp
 * \brief DynamicAUGRUGrad tiling UT：验证tiling数据、blockDim、workspace与非法输入报错
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "rnn/dynamic_augru_grad/op_kernel/arch35/dynamic_augru_grad_tiling_data.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
const string COMPILE_INFO_STRING_950 = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 32, "socVersion": "Ascend950"}
                        })";
} // namespace

class DynamicAUGRUGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DynamicAUGRUGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DynamicAUGRUGradTilingTest TearDown" << std::endl; }
};

struct AugruTilingResult {
    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    DynamicAUGRUGradTilingData tilingData;
};

static void RunTilingCase(int64_t t, int64_t b, int64_t iSize, int64_t hSize, ge::DataType dtype, bool withSeqLen,
                          const std::string& gateOrder, ge::graphStatus expectRet, AugruTilingResult* result = nullptr,
                          bool resetAfter = true, const std::string& direction = "UNIDIRECTIONAL",
                          int64_t cellDepth = 1, int64_t numProj = 0, bool timeMajor = true,
                          ge::DataType seqDtype = ge::DT_INT32, int64_t seqLenDim = -1, float keepProb = -1.0f,
                          float cellClip = -1.0f, int64_t wHiddenLeadingDim = 0)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion;
    GetPlatFormInfos(COMPILE_INFO_STRING_950.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    aicoreSpec["cube_freq"] = "1650"; // matmul tiling依赖cube_freq（GetPlatFormInfos仅置占位串）

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    struct DynamicAUGRUGradCompileInfoStub {
    } compileInfo;

    string opType("DynamicAUGRUGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(COMPILE_INFO_STRING_950.c_str()),
                                     reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xShape = {{t, b, iSize}, {t, b, iSize}};
    gert::StorageShape wInputShape = {{iSize, 3 * hSize}, {iSize, 3 * hSize}};
    gert::StorageShape wHiddenShape = wHiddenLeadingDim == 0 ?
                                          gert::StorageShape{{hSize, 3 * hSize}, {hSize, 3 * hSize}} :
                                          gert::StorageShape{{wHiddenLeadingDim, hSize, 3 * hSize},
                                                             {wHiddenLeadingDim, hSize, 3 * hSize}};
    gert::StorageShape attShape = {{t, b, hSize}, {t, b, hSize}};
    gert::StorageShape bhShape = {{b, hSize}, {b, hSize}};
    gert::StorageShape hShape = {{t, b, hSize}, {t, b, hSize}};
    gert::StorageShape seqLenShape = {{seqLenDim < 0 ? b : seqLenDim}, {seqLenDim < 0 ? b : seqLenDim}};
    gert::StorageShape dwInputShape = {{iSize, 3 * hSize}, {iSize, 3 * hSize}};
    gert::StorageShape dwHiddenShape = {{hSize, 3 * hSize}, {hSize, 3 * hSize}};
    gert::StorageShape dbShape = {{3 * hSize}, {3 * hSize}};
    gert::StorageShape dxShape = {{t, b, iSize}, {t, b, iSize}};
    gert::StorageShape dwAttShape = {{t, b}, {t, b}};

    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());

    // 16输入：x/wi/wh/att/y/init_h/h/dy/dh/update/update_att/reset/new/hidden_new/seq_length(可选)/mask(可选占位)
    // 可选输入缺省时：InputShapes仅传前14项且跳过其NodeInputTd（对0实例索引会越界）
    std::vector<void*> inputShapeRefs = {&xShape, &wInputShape, &wHiddenShape, &attShape, &hShape, &bhShape, &hShape,
                                         &hShape, &bhShape,     &hShape,       &hShape,   &hShape, &hShape,  &hShape};
    if (withSeqLen) {
        inputShapeRefs.push_back(&seqLenShape);
    }
    gert::TilingContextFaker faker;
    auto& chain = faker.SetOpType(opType)
                      .NodeIoNum(16, 7)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, withSeqLen ? 1 : 0, 0})
                      .InputShapes(inputShapeRefs)
                      .OutputShapes(
                          {&dwInputShape, &dwHiddenShape, &dbShape, &dbShape, &dxShape, &bhShape, &dwAttShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    if (withSeqLen) {
        chain.NodeInputTd(14, seqDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = chain.NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(6, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"direction", Ops::NN::AnyValue::CreateFrom<std::string>(direction)},
                                  {"cell_depth", Ops::NN::AnyValue::CreateFrom<int64_t>(cellDepth)},
                                  {"keep_prob", Ops::NN::AnyValue::CreateFrom<float>(keepProb)},
                                  {"cell_clip", Ops::NN::AnyValue::CreateFrom<float>(cellClip)},
                                  {"num_proj", Ops::NN::AnyValue::CreateFrom<int64_t>(numProj)},
                                  {"time_major", Ops::NN::AnyValue::CreateFrom<bool>(timeMajor)},
                                  {"gate_order", Ops::NN::AnyValue::CreateFrom<std::string>(gateOrder)},
                                  {"reset_after", Ops::NN::AnyValue::CreateFrom<bool>(resetAfter)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tilingFunc(tilingContext), expectRet);
    if (expectRet != ge::GRAPH_SUCCESS || result == nullptr) {
        return;
    }
    result->blockDim = tilingContext->GetBlockDim();
    result->workspaceSize = *(tilingContext->GetWorkspaceSizes(1));
    auto* tilingData = reinterpret_cast<const DynamicAUGRUGradTilingData*>(
        tilingContext->GetRawTilingData()->GetData());
    result->tilingData = *tilingData;
}

// 常规形状（带seq_length）：基础字段、blockDim、workspace正确
TEST_F(DynamicAUGRUGradTilingTest, tiling_normal_with_seq_len)
{
    AugruTilingResult ret;
    RunTilingCase(8, 4, 16, 32, ge::DT_FLOAT, true, "zrh", ge::GRAPH_SUCCESS, &ret);
    EXPECT_EQ(ret.tilingData.timeStep, 8);
    EXPECT_EQ(ret.tilingData.batchSize, 4);
    EXPECT_EQ(ret.tilingData.hiddenSize, 32);
    EXPECT_EQ(ret.tilingData.inputSize, 16);
    EXPECT_EQ(ret.tilingData.isSeqLength, 1);
    EXPECT_EQ(ret.tilingData.gateOrder, 0);
    EXPECT_EQ(ret.blockDim, 32U); // AIC核数
    EXPECT_GT(ret.tilingData.hTile, 0);
    EXPECT_GT(ret.tilingData.bTile, 0);
    // workspace = (2*TB*3H + TB*H + 2*B*H) * 4字节
    int64_t tb = 8 * 4;
    int64_t expectWs = (2 * tb * 96 + tb * 32 + 2 * 4 * 32) * 4;
    EXPECT_GE(ret.workspaceSize, static_cast<size_t>(expectWs));
}

TEST_F(DynamicAUGRUGradTilingTest, tiling_weight_hidden_leading_singleton)
{
    AugruTilingResult ret;
    RunTilingCase(1, 4, 16, 32, ge::DT_FLOAT, true, "zrh", ge::GRAPH_SUCCESS, &ret, true, "UNIDIRECTIONAL", 1, 0, true,
                  ge::DT_INT32, -1, -1.0f, -1.0f, 1);
    EXPECT_EQ(ret.tilingData.timeStep, 1);
    EXPECT_EQ(ret.tilingData.hiddenSize, 32);
}

TEST_F(DynamicAUGRUGradTilingTest, tiling_weight_hidden_non_singleton_leading_failed)
{
    RunTilingCase(1, 4, 16, 32, ge::DT_FLOAT, true, "zrh", ge::GRAPH_FAILED, nullptr, true, "UNIDIRECTIONAL", 1, 0,
                  true, ge::DT_INT32, -1, -1.0f, -1.0f, 2);
}

// 不带seq_length：isSeqLength=0
TEST_F(DynamicAUGRUGradTilingTest, tiling_normal_without_seq_len)
{
    AugruTilingResult ret;
    RunTilingCase(4, 8, 32, 64, ge::DT_FLOAT, false, "zrh", ge::GRAPH_SUCCESS, &ret);
    EXPECT_EQ(ret.tilingData.isSeqLength, 0);
    EXPECT_EQ(ret.tilingData.timeStep, 4);
    EXPECT_EQ(ret.tilingData.batchSize, 8);
    EXPECT_EQ(ret.tilingData.hiddenSize, 64);
}

// rzh门序
TEST_F(DynamicAUGRUGradTilingTest, tiling_gate_order_rzh)
{
    AugruTilingResult ret;
    RunTilingCase(4, 8, 32, 64, ge::DT_FLOAT, true, "rzh", ge::GRAPH_SUCCESS, &ret);
    EXPECT_EQ(ret.tilingData.gateOrder, 1);
}

// 非法gate_order：失败
TEST_F(DynamicAUGRUGradTilingTest, tiling_bad_gate_order_failed)
{
    RunTilingCase(4, 8, 32, 64, ge::DT_FLOAT, true, "xxx", ge::GRAPH_FAILED);
}

// reset_after=False：数学硬编码reset_after语义，失败
TEST_F(DynamicAUGRUGradTilingTest, tiling_bad_reset_after_failed)
{
    RunTilingCase(4, 8, 32, 64, ge::DT_FLOAT, true, "zrh", ge::GRAPH_FAILED, nullptr, false);
}

// x的T维为0（非法shape）：失败
TEST_F(DynamicAUGRUGradTilingTest, tiling_x_dim_failed)
{
    RunTilingCase(0, 8, 32, 64, ge::DT_FLOAT, true, "zrh", ge::GRAPH_FAILED);
}

// fp16：tiling成功，workspace包含fp32副本附加区（wHidden/wInput/xFp32/dw结果区）
TEST_F(DynamicAUGRUGradTilingTest, tiling_fp16_workspace)
{
    AugruTilingResult ret;
    RunTilingCase(8, 4, 16, 32, ge::DT_FLOAT16, true, "zrh", ge::GRAPH_SUCCESS, &ret);
    int64_t tb = 8 * 4;
    int64_t expectWs = (2 * tb * 96 + tb * 32 + 2 * 4 * 32 + 2 * 32 * 96 + 2 * 16 * 96 + 2 * tb * 16) * 4;
    EXPECT_GE(ret.workspaceSize, static_cast<size_t>(expectWs));
}

// H非16对齐：tiling成功，hPad=32
TEST_F(DynamicAUGRUGradTilingTest, tiling_hidden_unaligned_hpad)
{
    AugruTilingResult ret;
    RunTilingCase(4, 4, 16, 20, ge::DT_FLOAT, true, "zrh", ge::GRAPH_SUCCESS, &ret);
    EXPECT_EQ(ret.tilingData.hPad, 32);
    EXPECT_EQ(ret.tilingData.hiddenSize, 20);
}

// 大H场景：hTile分块不超过ubLength
TEST_F(DynamicAUGRUGradTilingTest, tiling_large_hidden_split)
{
    AugruTilingResult ret;
    RunTilingCase(4, 4, 16, 1024, ge::DT_FLOAT, true, "zrh", ge::GRAPH_SUCCESS, &ret);
    EXPECT_LE(ret.tilingData.hTile, ret.tilingData.ubLength);
    EXPECT_EQ(ret.tilingData.hTile % 8, 0);
    EXPECT_GE(ret.tilingData.hTile, 8);
}

// 属性拒绝：direction非UNIDIRECTIONAL / cell_depth=2 / num_proj=1 / time_major=false
TEST_F(DynamicAUGRUGradTilingTest, tiling_bad_direction_failed)
{
    RunTilingCase(4, 4, 16, 32, ge::DT_FLOAT, true, "zrh", ge::GRAPH_FAILED, nullptr, true, "BIDIRECTIONAL");
}

TEST_F(DynamicAUGRUGradTilingTest, tiling_bad_cell_depth_failed)
{
    RunTilingCase(4, 4, 16, 32, ge::DT_FLOAT, true, "zrh", ge::GRAPH_FAILED, nullptr, true, "UNIDIRECTIONAL", 2);
}

TEST_F(DynamicAUGRUGradTilingTest, tiling_bad_num_proj_failed)
{
    RunTilingCase(4, 4, 16, 32, ge::DT_FLOAT, true, "zrh", ge::GRAPH_FAILED, nullptr, true, "UNIDIRECTIONAL", 1, 64);
}

TEST_F(DynamicAUGRUGradTilingTest, tiling_bad_time_major_failed)
{
    RunTilingCase(4, 4, 16, 32, ge::DT_FLOAT, true, "zrh", ge::GRAPH_FAILED, nullptr, true, "UNIDIRECTIONAL", 1, 0,
                  false);
}

// keep_prob=1.0（TF侧缺省）：等价无dropout，成功
TEST_F(DynamicAUGRUGradTilingTest, tiling_keep_prob_one_ok)
{
    AugruTilingResult ret;
    RunTilingCase(4, 8, 32, 64, ge::DT_FLOAT, true, "zrh", ge::GRAPH_SUCCESS, &ret, true, "UNIDIRECTIONAL", 1, 0, true,
                  ge::DT_INT32, -1, 1.0f);
    EXPECT_EQ(ret.tilingData.timeStep, 4);
}

// keep_prob非1.0/-1.0：dropout不生效，静默接受会产生错误数值，拒绝
TEST_F(DynamicAUGRUGradTilingTest, tiling_bad_keep_prob_failed)
{
    RunTilingCase(4, 8, 32, 64, ge::DT_FLOAT, true, "zrh", ge::GRAPH_FAILED, nullptr, true, "UNIDIRECTIONAL", 1, 0,
                  true, ge::DT_INT32, -1, 0.5f);
}

// cell_clip非-1.0：clip不生效，静默接受会产生错误数值，拒绝
TEST_F(DynamicAUGRUGradTilingTest, tiling_bad_cell_clip_failed)
{
    RunTilingCase(4, 8, 32, 64, ge::DT_FLOAT, true, "zrh", ge::GRAPH_FAILED, nullptr, true, "UNIDIRECTIONAL", 1, 0,
                  true, ge::DT_INT32, -1, -1.0f, 5.0f);
}

// seq_length违例：dtype非INT32 / 长度非[B]
TEST_F(DynamicAUGRUGradTilingTest, tiling_bad_seq_len_dtype_failed)
{
    RunTilingCase(4, 4, 16, 32, ge::DT_FLOAT, true, "zrh", ge::GRAPH_FAILED, nullptr, true, "UNIDIRECTIONAL", 1, 0,
                  true, ge::DT_FLOAT);
}

TEST_F(DynamicAUGRUGradTilingTest, tiling_bad_seq_len_shape_failed)
{
    RunTilingCase(4, 4, 16, 32, ge::DT_FLOAT, true, "zrh", ge::GRAPH_FAILED, nullptr, true, "UNIDIRECTIONAL", 1, 0,
                  true, ge::DT_INT32, 5);
}

// db内联开关边界：小H（UB装得下累加器）开启，大H关闭
TEST_F(DynamicAUGRUGradTilingTest, tiling_db_inline_boundary)
{
    AugruTilingResult ret;
    RunTilingCase(4, 4, 16, 1024, ge::DT_FLOAT, true, "zrh", ge::GRAPH_SUCCESS, &ret);
    EXPECT_EQ(ret.tilingData.enableDbInline, 1);
    RunTilingCase(4, 4, 16, 6144, ge::DT_FLOAT, true, "zrh", ge::GRAPH_SUCCESS, &ret);
    EXPECT_EQ(ret.tilingData.enableDbInline, 0);
}

// Only a single recurrent K block can overlap prefetch with the projection.
TEST_F(DynamicAUGRUGradTilingTest, tiling_pipeline_switch)
{
    AugruTilingResult ret;
    RunTilingCase(8, 4, 16, 16, ge::DT_FLOAT, true, "zrh", ge::GRAPH_SUCCESS, &ret);
    EXPECT_EQ(ret.tilingData.enablePipeline, 0);
    RunTilingCase(8, 4, 16, 32, ge::DT_FLOAT, true, "zrh", ge::GRAPH_SUCCESS, &ret);
    EXPECT_EQ(ret.tilingData.enablePipeline, 0);
    RunTilingCase(8, 4, 16, 32, ge::DT_FLOAT16, true, "zrh", ge::GRAPH_SUCCESS, &ret);
    EXPECT_EQ(ret.tilingData.enablePipeline, 0);
    RunTilingCase(8, 4, 16, 4096, ge::DT_FLOAT, true, "zrh", ge::GRAPH_SUCCESS, &ret);
    EXPECT_EQ(ret.tilingData.enablePipeline, 0);
}
