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
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"

using namespace ge;
using namespace ut_util;

namespace optiling {
struct SmoothL1LossGradCompileInfo {};
} // namespace optiling

class SmoothL1LossGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SmoothL1LossGradTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SmoothL1LossGradTilingTest TearDown" << std::endl; }
};

static void DoTilingTest(std::initializer_list<int64_t> predictDims, std::initializer_list<int64_t> labelDims,
                         std::initializer_list<int64_t> doutDims, ge::DataType predictDtype, float sigma,
                         ge::graphStatus expectedStatus, ge::DataType labelDtype = ge::DT_UNDEFINED,
                         ge::DataType doutDtype = ge::DT_UNDEFINED, ge::Format predictFormat = ge::FORMAT_ND,
                         ge::Format labelFormat = ge::FORMAT_ND, ge::Format doutFormat = ge::FORMAT_ND,
                         ge::Format outputFormat = ge::FORMAT_ND, ge::DataType outputDtype = ge::DT_UNDEFINED,
                         std::initializer_list<int64_t> outputDims = {}, uint64_t ubSize = 245760,
                         bool withSigmaAttr = true)
{
    std::string opType("SmoothL1LossGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    std::string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": )" +
                                 std::to_string(ubSize) + R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64}
    })";
    std::map<std::string, std::string> socInfos, aicoreSpec, intrinsics;
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::SmoothL1LossGradCompileInfo compileInfo;
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(32);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    gert::StorageShape predictShape = {predictDims, predictDims};
    gert::StorageShape labelShape = {labelDims, labelDims};
    gert::StorageShape doutShape = {doutDims, doutDims};
    const auto effectiveOutputDims = outputDims.size() == 0 ? predictDims : outputDims;
    gert::StorageShape outputShape = {effectiveOutputDims, effectiveOutputDims};
    ge::DataType effectiveLabelDtype = labelDtype == ge::DT_UNDEFINED ? predictDtype : labelDtype;
    ge::DataType effectiveDoutDtype = doutDtype == ge::DT_UNDEFINED ? predictDtype : doutDtype;
    ge::DataType effectiveOutputDtype = outputDtype == ge::DT_UNDEFINED ? predictDtype : outputDtype;

    gert::TilingContextFaker tilingFaker;
    tilingFaker.SetOpType(opType)
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .InputShapes({&predictShape, &labelShape, &doutShape})
        .OutputShapes({&outputShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, predictDtype, predictFormat, predictFormat)
        .NodeInputTd(1, effectiveLabelDtype, labelFormat, labelFormat)
        .NodeInputTd(2, effectiveDoutDtype, doutFormat, doutFormat)
        .NodeOutputTd(0, effectiveOutputDtype, outputFormat, outputFormat);
    if (withSigmaAttr) {
        tilingFaker.NodeAttrs({{"sigma", Ops::NN::AnyValue::CreateFrom<float>(sigma)}});
    }
    auto holder = tilingFaker.TilingData(param.get()).Workspace(wsSize).Build();

    gert::TilingContext* ctx = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(ctx, nullptr);
    ASSERT_NE(ctx->GetPlatformInfo(), nullptr);
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(ctx), expectedStatus);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_fp32_accepted)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_SUCCESS);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_unsupported_dtype_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_INT8, 1.0f, ge::GRAPH_FAILED);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_dtype_mismatch_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_FLOAT16);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_exact_shape_mismatch_rejected)
{
    DoTilingTest({2, 6}, {3, 4}, {2, 6}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_sigma_zero_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 0.0f, ge::GRAPH_FAILED);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_sigma_nonfinite_rejected)
{
    // sigma 非有限值(NaN)拒绝: 属性域契约 (0, +inf) 的 isfinite 守卫
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, std::numeric_limits<float>::quiet_NaN(), ge::GRAPH_FAILED);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_predict_non_nd_format_accepted)
{
    // 格式契约归 def.cpp 注册面声明（Format{ND}）；GE 图编译路径不强制（NCHW 图用例端到端接受），
    // kernel 直调路由二进制匹配拒绝，tiling 不做格式校验
    DoTilingTest({1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, ge::DT_FLOAT, 1.0f, ge::GRAPH_SUCCESS, ge::DT_UNDEFINED,
                 ge::DT_UNDEFINED, ge::FORMAT_NCHW);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_rank_9_accepted)
{
    // rank>8 输入按展平 totalLength 逐元素切分，tiling 不做维度上限校验
    DoTilingTest({1, 1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 2}, ge::DT_FLOAT16,
                 1.0f, ge::GRAPH_SUCCESS);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_total_length_overflow_rejected)
{
    // 2^60 个元素（fp16 每元素 16 bit）超出 int64 切分算术上界（(INT64_MAX-32767)/16），tiling 拒绝
    DoTilingTest({536870912, 536870912, 4}, {536870912, 536870912, 4}, {536870912, 536870912, 4}, ge::DT_FLOAT16, 1.0f,
                 ge::GRAPH_FAILED);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_negative_total_length_rejected)
{
    // 负维构造的元素总数为负：多核切分 coreNum 会取 0，显式拒绝
    DoTilingTest({-2, 4}, {-2, 4}, {-2, 4}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_prefix_overflow_trailing_zero_accepted)
{
    // 前缀两维积 2^58 量级 × 尾维 0：元素总数为 0，走空 Tensor 短路而非 int64 溢出守卫
    DoTilingTest({536870912, 536870912, 0}, {536870912, 536870912, 0}, {536870912, 536870912, 0}, ge::DT_FLOAT16, 1.0f,
                 ge::GRAPH_SUCCESS);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_label_unsupported_dtype_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_INT8);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_dout_unsupported_dtype_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_UNDEFINED, ge::DT_INT8);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_dout_dtype_mismatch_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_UNDEFINED, ge::DT_FLOAT16);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_label_non_nd_format_accepted)
{
    // 格式契约归 def.cpp 注册面声明（Format{ND}）；GE 图编译路径不强制（NCHW 图用例端到端接受），
    // kernel 直调路由二进制匹配拒绝，tiling 不做格式校验
    DoTilingTest({1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, ge::DT_FLOAT, 1.0f, ge::GRAPH_SUCCESS, ge::DT_UNDEFINED,
                 ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_NCHW);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_dout_non_nd_format_accepted)
{
    // 格式契约归 def.cpp 注册面声明（Format{ND}）；GE 图编译路径不强制（NCHW 图用例端到端接受），
    // kernel 直调路由二进制匹配拒绝，tiling 不做格式校验
    DoTilingTest({1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, ge::DT_FLOAT, 1.0f, ge::GRAPH_SUCCESS, ge::DT_UNDEFINED,
                 ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NCHW);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_output_non_nd_format_accepted)
{
    // 输出格式契约归 def.cpp 注册面声明（Format{ND}）；GE 图编译路径不强制，tiling 不做输出格式校验
    DoTilingTest({1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, ge::DT_FLOAT, 1.0f, ge::GRAPH_SUCCESS, ge::DT_UNDEFINED,
                 ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NCHW);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_dout_shape_mismatch_rejected)
{
    DoTilingTest({4}, {4}, {2, 2}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_output_shape_mismatch_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_UNDEFINED, {2, 16});
}

TEST_F(SmoothL1LossGradTilingTest, tiling_output_dtype_mismatch_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_insufficient_ub_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_UNDEFINED, {}, 512);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_rank0_scalar_accepted)
{
    // rank=0 标量输入契约: 展平为 {1} 单元素后正常切分
    DoTilingTest({}, {}, {}, ge::DT_FLOAT, 1.0f, ge::GRAPH_SUCCESS);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_empty_tensor_shortcircuit_accepted)
{
    // 空 Tensor(totalLength==0)短路: SetBlockDim(1) 直接成功, kernel 侧守卫返回
    DoTilingTest({0, 0}, {0, 0}, {0, 0}, ge::DT_FLOAT, 1.0f, ge::GRAPH_SUCCESS);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_sigma_attr_absent_default_accepted)
{
    // sigma 为 OPTIONAL 属性(注册面默认 1.0f): 图省略属性节点时走 DEFAULT_SIGMA 路径, tiling 接受
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_SUCCESS, ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_UNDEFINED, {}, 245760, false);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_parse_registered_noop_success)
{
    // const-tiling 解析注册面: TilingParse 注册在册且为无状态解析, 任意上下文恒 SUCCESS
    std::string opType("SmoothL1LossGrad");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::SmoothL1LossGradCompileInfo compileInfo;
    auto holder = gert::KernelRunContextFaker()
                      .KernelIONum(2, 1)
                      .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platformInfo)})
                      .Outputs({&compileInfo})
                      .Build();
    EXPECT_EQ(opImpl->tiling_parse(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}
