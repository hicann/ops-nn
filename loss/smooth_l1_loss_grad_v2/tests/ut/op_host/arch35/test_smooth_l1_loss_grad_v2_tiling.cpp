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
 * \file test_smooth_l1_loss_grad_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../../../op_host/arch35/smooth_l1_loss_grad_v2_tiling_base.h"

using namespace std;

class SmoothL1LossGradV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SmoothL1LossGradV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SmoothL1LossGradV2Tiling TearDown" << std::endl; }
};

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics,
                         map<string, string>& socVersion)
{
    string compile_info_string = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    GetPlatFormInfos(compile_info_string.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    // platform info
    platFormInfo.Init();
}

static void SmoothL1LossGradV2TilingCase(std::initializer_list<int64_t>& inputShape1,
                                         std::initializer_list<int64_t>& inputShape2,
                                         std::initializer_list<int64_t>& inputShape3,
                                         std::initializer_list<int64_t>& outputShape, ge::DataType inputDtype,
                                         std::string& reduction, float& sigma)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion;
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    optiling::SmoothL1LossGradV2CompileInfo compileInfo;
    std::string opType("SmoothL1LossGradV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoStr = R"({})";
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platFormInfo)})
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

    // tilingFunc simulate
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    gert::StorageShape gradShape = {inputShape1, inputShape1};
    gert::StorageShape input_Shape = {inputShape2, inputShape2};
    gert::StorageShape targetShape = {inputShape3, inputShape3};
    gert::StorageShape youtShape = {outputShape, outputShape};
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&gradShape, &input_Shape, &targetShape})
                      .OutputShapes({&youtShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"sigma", Ops::NN::AnyValue::CreateFrom<float>(sigma)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(SmoothL1LossGradV2Tiling, kl_div_loss_grad_david_tiling1)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {2048, 1, 48};
    std::initializer_list<int64_t> inputShape2 = {2048, 1, 1};
    std::initializer_list<int64_t> inputShape3 = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    std::string reduction = "mean";
    float sigma = 1.0f;
    SmoothL1LossGradV2TilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT /*inputdtype*/,
                                 reduction, sigma);
}

namespace {
// GET_TPL_TILING_KEY(schMode, doutIsScalar): schMode occupies bit 0..15, doutIsScalar occupies bit 16.
// doutIsScalar is declared as ASCENDC_TPL_UINT_DECL(doutIsScalar, 1, UI_LIST, 0, ATTR_IS_TRUE), and UINT
// params are encoded by their UI_LIST index: 0 -> index 0 (bit cleared), 1 -> index 1 (bit set).
constexpr uint64_t SLG_DOUT_IS_SCALAR_MASK = 1UL << 16;
} // namespace

// Status-oriented helper: drives one SmoothL1LossGradV2 tiling invocation with independent per-port dtypes,
// covering the scalar-dout / tensor-dout dag split and all validation failure branches.
static void DoSmoothL1LossGradV2StatusCase(std::initializer_list<int64_t>& predictShape,
                                           std::initializer_list<int64_t>& labelShape,
                                           std::initializer_list<int64_t>& doutShape,
                                           std::initializer_list<int64_t>& outputShape, ge::DataType predictDtype,
                                           ge::DataType labelDtype, ge::DataType doutDtype, ge::DataType outputDtype,
                                           const std::string& reduction, float sigma, ge::graphStatus expectedStatus,
                                           uint64_t& outTilingKey)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion;
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    optiling::SmoothL1LossGradV2CompileInfo compileInfo;
    std::string opType("SmoothL1LossGradV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoStr = R"({})";
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platFormInfo)})
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

    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    gert::StorageShape predictShapeStorage = {predictShape, predictShape};
    gert::StorageShape labelShapeStorage = {labelShape, labelShape};
    gert::StorageShape doutShapeStorage = {doutShape, doutShape};
    gert::StorageShape youtShapeStorage = {outputShape, outputShape};
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&predictShapeStorage, &labelShapeStorage, &doutShapeStorage})
                      .OutputShapes({&youtShapeStorage})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, predictDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, labelDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, doutDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"sigma", Ops::NN::AnyValue::CreateFrom<float>(sigma)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        outTilingKey = tilingContext->GetTilingKey();
    }
}

// fp16, tensor dout, reduction=mean: DoTensorDagOpTiling half instantiation.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_tiling_fp16_tensor_dout_mean)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT16, ge::DT_FLOAT16,
                                   ge::DT_FLOAT16, ge::DT_FLOAT16, "mean", 1.0f, ge::GRAPH_SUCCESS, tilingKey);
    EXPECT_EQ(tilingKey & SLG_DOUT_IS_SCALAR_MASK, 0U);
}

// bf16, tensor dout, reduction=none: DoTensorDagOpTiling bfloat16_t instantiation.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_tiling_bf16_tensor_dout_none)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_BF16, ge::DT_BF16,
                                   ge::DT_BF16, ge::DT_BF16, "none", 1.0f, ge::GRAPH_SUCCESS, tilingKey);
    EXPECT_EQ(tilingKey & SLG_DOUT_IS_SCALAR_MASK, 0U);
}

// fp32, tensor dout, reduction=none.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_tiling_fp32_tensor_dout_none)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "none", 1.0f, ge::GRAPH_SUCCESS, tilingKey);
    EXPECT_EQ(tilingKey & SLG_DOUT_IS_SCALAR_MASK, 0U);
}

// fp32, tensor dout, reduction=sum.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_tiling_fp32_tensor_dout_sum)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "sum", 1.0f, ge::GRAPH_SUCCESS, tilingKey);
    EXPECT_EQ(tilingKey & SLG_DOUT_IS_SCALAR_MASK, 0U);
}

// fp32, scalar dout (shape {1}): DoScalarDagOpTiling path, doutIsScalar bit is set in the tiling key.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_tiling_fp32_scalar_dout)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", 1.0f, ge::GRAPH_SUCCESS, tilingKey);
    EXPECT_NE(tilingKey & SLG_DOUT_IS_SCALAR_MASK, 0U);
}

// fp16, scalar dout, reduction=none: DoScalarDagOpTiling half instantiation.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_tiling_fp16_scalar_dout)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT16, ge::DT_FLOAT16,
                                   ge::DT_FLOAT16, ge::DT_FLOAT16, "none", 1.0f, ge::GRAPH_SUCCESS, tilingKey);
    EXPECT_NE(tilingKey & SLG_DOUT_IS_SCALAR_MASK, 0U);
}

// bf16, scalar dout, reduction=sum: DoScalarDagOpTiling bfloat16_t instantiation.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_tiling_bf16_scalar_dout)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_BF16, ge::DT_BF16,
                                   ge::DT_BF16, ge::DT_BF16, "sum", 1.0f, ge::GRAPH_SUCCESS, tilingKey);
    EXPECT_NE(tilingKey & SLG_DOUT_IS_SCALAR_MASK, 0U);
}

// custom sigma (2.0) is propagated through the scalar dag.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_tiling_custom_sigma)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", 2.0f, ge::GRAPH_SUCCESS, tilingKey);
    EXPECT_EQ(tilingKey & SLG_DOUT_IS_SCALAR_MASK, 0U);
}

// sigma = 0 is accepted (invertSigma becomes NaN inside the tiling data).
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_tiling_sigma_zero)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "none", 0.0f, ge::GRAPH_SUCCESS, tilingKey);
    EXPECT_EQ(tilingKey & SLG_DOUT_IS_SCALAR_MASK, 0U);
}

// scalar predict/label with reduction=mean: EnsureNotScalar maps the scalar to {1}, dimVal=1.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_tiling_scalar_input_mean)
{
    std::initializer_list<int64_t> predictShape = {};
    std::initializer_list<int64_t> labelShape = {};
    std::initializer_list<int64_t> doutShape = {};
    std::initializer_list<int64_t> outputShape = {};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", 1.0f, ge::GRAPH_SUCCESS, tilingKey);
    EXPECT_NE(tilingKey & SLG_DOUT_IS_SCALAR_MASK, 0U); // scalar dout also takes the scalar dag path
}

// negative sigma is rejected by CalcSigma.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_negative_sigma_failed)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", -1.0f, ge::GRAPH_FAILED, tilingKey);
}

// unsupported predict dtype (int32) is rejected by GetShapeAttrsInfo.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_unsupported_dtype_failed)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_INT32, ge::DT_INT32,
                                   ge::DT_INT32, ge::DT_INT32, "mean", 1.0f, ge::GRAPH_FAILED, tilingKey);
}

// predict/label dtype mismatch is rejected.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_label_dtype_mismatch_failed)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT16,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", 1.0f, ge::GRAPH_FAILED, tilingKey);
}

// predict/dout dtype mismatch is rejected.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_dout_dtype_mismatch_failed)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT16, ge::DT_FLOAT, "mean", 1.0f, ge::GRAPH_FAILED, tilingKey);
}

// predict/output dtype mismatch is rejected.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_output_dtype_mismatch_failed)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT16, "mean", 1.0f, ge::GRAPH_FAILED, tilingKey);
}

// invalid reduction string is rejected by CalcReduceMeanCof.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_invalid_reduction_failed)
{
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 1};
    std::initializer_list<int64_t> doutShape = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "avg", 1.0f, ge::GRAPH_FAILED, tilingKey);
}

// empty tensor with reduction=mean is rejected.
TEST_F(SmoothL1LossGradV2Tiling, smooth_l1_loss_grad_v2_mean_empty_tensor_failed)
{
    std::initializer_list<int64_t> predictShape = {2, 0, 4};
    std::initializer_list<int64_t> labelShape = {2, 0, 4};
    std::initializer_list<int64_t> doutShape = {2, 0, 4};
    std::initializer_list<int64_t> outputShape = {2, 0, 4};
    uint64_t tilingKey = 0;
    DoSmoothL1LossGradV2StatusCase(predictShape, labelShape, doutShape, outputShape, ge::DT_FLOAT, ge::DT_FLOAT,
                                   ge::DT_FLOAT, ge::DT_FLOAT, "mean", 1.0f, ge::GRAPH_FAILED, tilingKey);
}
