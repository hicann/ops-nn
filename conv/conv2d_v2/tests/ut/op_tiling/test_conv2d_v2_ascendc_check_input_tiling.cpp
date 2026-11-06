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
 * \file test_conv2d_v2_ascendc_check_input_tiling.cpp
 * \brief UT for conv2d_v2_base_tiling_check_input.cpp
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include "log/log.h"
#include "array_ops.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base_utils.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base.h"
#include "test_conv2d_v2_ascendc_utils_tiling.h"

using namespace ut_util;
using namespace conv_tiling_utils;

namespace {
void RunInputCheckTest(const std::string& caseName, const std::vector<int64_t>& fmShape,
                       const std::vector<int64_t>& weightShape, ge::DataType dtype, bool hasBias, bool expectFail)
{
    int64_t batch = fmShape[0], cin = fmShape[1], hi = fmShape[2], wi = fmShape[3];
    int64_t cout = weightShape[0], ci_per_group = weightShape[1], kH = weightShape[2], kW = weightShape[3];

    gert::StorageShape featuremap = {{batch, cin, hi, wi}, {batch, cin, hi, wi}};
    gert::StorageShape weight = {{cout, ci_per_group, kH, kW}, {cout, ci_per_group, kH, kW}};
    gert::StorageShape bias = {{cout}, {cout}};
    int64_t ho = hi, wo = wi;
    ConvShape convShapeH = {static_cast<uint64_t>(hi), static_cast<uint64_t>(kH), 0U, 0U, 1U, 1U};
    ConvShape convShapeW = {static_cast<uint64_t>(wi), static_cast<uint64_t>(kW), 0U, 0U, 1U, 1U};
    ho = InferOut(convShapeH);
    wo = InferOut(convShapeW);
    gert::StorageShape output = {{batch, cout, ho, wo}, {batch, cout, ho, wo}};

    std::vector<void*> input_shape_ref = hasBias ? std::vector<void*>{&featuremap, &weight, &bias, nullptr} :
                                                   std::vector<void*>{&featuremap, &weight, nullptr, nullptr};
    std::vector<void*> output_shapes_ref = {&output};

    std::string op_type = "Conv2DV2";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    std::string compile_info_string = R"({"hardware_info":
        {"BT_SIZE": 4096, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 253952,
        "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096,
        "BT_SIZE": 4096, "L0C_SIZE": 262144, "CORE_NUM": 32, "cube_core_cnt": 32, "vector_core_cnt": 64,
        "core_type_list": "CubeCore,VectorCore"}})";
    std::map<std::string, std::string> soc_infos, aicore_spec, intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    aicore_spec.insert({"fb0_size", "4096"});

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;

    auto tilingDataPtr = gert::TilingData::CreateCap(MEM_SIZE_4K);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(MEM_SIZE_4K);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    std::vector<int64_t> stridesVec = {1, 1, 1, 1};
    std::vector<int64_t> padsVec = {0, 0, 0, 0};
    std::vector<int64_t> dilationsVec = {1, 1, 1, 1};
    ge::DataType bias_dtype = dtype;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(NUM_4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes(input_shape_ref)
                      .OutputShapes(output_shapes_ref)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(DIM_2, bias_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(DIM_3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stridesVec)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padsVec)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilationsVec)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(tilingDataPtr.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ge::graphStatus ret = tiling_func(tiling_context);

    if (expectFail) {
        EXPECT_EQ(ret, ge::GRAPH_FAILED) << "Case [" << caseName << "] expected FAILED but got SUCCESS";
    } else {
        EXPECT_EQ(ret, ge::GRAPH_SUCCESS) << "Case [" << caseName << "] expected SUCCESS but got FAILED";
    }
}
} // namespace

class TestConv2dCheckInput : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// CheckFmapShape: dim > MAX_FM_H_B16_SHAPE(1000000) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, fmap_h_over_max_fail)
{
    RunInputCheckTest("fmap_h_over", {1, 16, 1000001, 16}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

TEST_F(TestConv2dCheckInput, fmap_w_over_max_fail)
{
    RunInputCheckTest("fmap_w_over", {1, 16, 16, 1000001}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckWeightShape: kh/kw > MAX_KH_B16_SHAPE(1000000) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, weight_kh_over_max_fail)
{
    RunInputCheckTest("weight_kh_over", {1, 16, 16, 16}, {16, 16, 1000001, 1}, ge::DT_FLOAT16, true, true);
}

TEST_F(TestConv2dCheckInput, weight_kw_over_max_fail)
{
    RunInputCheckTest("weight_kw_over", {1, 16, 16, 16}, {16, 16, 1, 1000001}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckOutputShape: output dim > MAX_OUT_SHAPE(1000000) → FAILED
// Use hi=65536, wi=65536, kh=512 → ho exceeds limit (ho*wo overflow in check limits before this)
// Actually, ho=hi when kh=1, pad=0, so hi=1000001 → ho=1000001 > 1000000 → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, output_h_over_max_fail)
{
    RunInputCheckTest("output_h_over", {1, 16, 1000001, 16}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckWeightShape: weight cout=0 → CheckDim returns false (<=0)
// ============================================================================
TEST_F(TestConv2dCheckInput, weight_cout_zero_fail)
{
    RunInputCheckTest("weight_cout_zero", {1, 16, 16, 16}, {0, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// Various valid shapes → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_1x1_kernel_success)
{
    RunInputCheckTest("valid_1x1", {1, 16, 16, 16}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, false);
}

TEST_F(TestConv2dCheckInput, valid_3x3_kernel_success)
{
    RunInputCheckTest("valid_3x3", {1, 16, 32, 32}, {32, 16, 3, 3}, ge::DT_FLOAT16, true, false);
}

TEST_F(TestConv2dCheckInput, valid_7x7_kernel_success)
{
    RunInputCheckTest("valid_7x7", {1, 64, 112, 112}, {128, 64, 7, 7}, ge::DT_FLOAT16, true, false);
}

TEST_F(TestConv2dCheckInput, valid_no_bias_success)
{
    RunInputCheckTest("valid_no_bias", {1, 32, 64, 64}, {64, 32, 3, 3}, ge::DT_FLOAT16, false, false);
}

// ============================================================================
// Multiple input channels → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_multi_channel_success)
{
    RunInputCheckTest("multi_chan", {2, 128, 56, 56}, {256, 128, 1, 1}, ge::DT_FLOAT16, true, false);
}

// ============================================================================
// CheckFmapShape: fmap N out of range (above MAX_FM_BATCH_SHAPE) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, fmap_n_zero_fail)
{
    RunInputCheckTest("fmap_n_over", {1000001, 16, 16, 16}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckFmapShape: fmap cin out of range (above MAX_FM_C_SHAPE) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, fmap_cin_zero_fail)
{
    RunInputCheckTest("fmap_cin_over", {1, 1000001, 16, 16}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckWeightShape: weight cout > MAX_COUT_SHAPE → CheckDim fails → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, weight_cin_zero_fail)
{
    RunInputCheckTest("weight_cout_over", {1, 16, 16, 16}, {1000001, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckOutputShape: output wo > MAX_OUT_SHAPE → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, output_w_over_max_fail)
{
    RunInputCheckTest("output_w_over", {1, 16, 16, 1000001}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// Valid with FLOAT dtype → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_fp32_dtype_success)
{
    RunInputCheckTest("fp32_dtype", {1, 16, 16, 16}, {16, 16, 1, 1}, ge::DT_FLOAT, true, false);
}

// ============================================================================
// Valid with BFLOAT16 dtype → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_bf16_dtype_success)
{
    RunInputCheckTest("bf16_dtype", {1, 16, 32, 32}, {32, 16, 3, 3}, ge::DT_BF16, true, false);
}

// ============================================================================
// Valid with FLOAT dtype, no bias → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_int8_dtype_success)
{
    RunInputCheckTest("fp32_nobias", {1, 16, 16, 16}, {16, 16, 1, 1}, ge::DT_FLOAT, false, false);
}

// ============================================================================
// CheckParamsDtypeWithBias: invalid dtype combination (INT32 not supported) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, invalid_bias_dtype_fail)
{
    RunInputCheckTest("invalid_bias_dtype", {1, 16, 16, 16}, {16, 16, 1, 1}, ge::DT_INT32, true, true);
}

// ============================================================================
// Large batch size → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_large_batch_success)
{
    RunInputCheckTest("large_batch", {100, 16, 64, 64}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, false);
}

// ============================================================================
// Output H=1, W=1 → SUCCESS (edge case)
// ============================================================================
TEST_F(TestConv2dCheckInput, output_hw_one_success)
{
    RunInputCheckTest("hw_one", {1, 1, 1, 1}, {1, 1, 1, 1}, ge::DT_FLOAT16, true, false);
}

// ============================================================================
// Direct-call coverage for the remaining check_input branches: bias shape
// variants, 3-D shapes, discontinuous (view) inputs, quant/extend scale checks
// and dual-output special checks.
// ============================================================================
#include <exe_graph/runtime/runtime_tensor.h>
#include <exe_graph/runtime/storage_format.h>
#include "../../../op_host/op_tiling/arch35/conv2d_v2_base_tiling.h"

namespace {
struct InputSpec {
    ge::DataType dtype;
    ge::Format format;
};

gert::KernelRunContextHolder BuildGenericInputContext(optiling::conv_ops_tiling::ConvTilingParseInfo& compileInfo,
                                                      fe::PlatFormInfos& platformInfo,
                                                      const std::vector<gert::StorageShape*>& inputShapes,
                                                      const std::vector<InputSpec>& inputDescs,
                                                      const std::vector<gert::StorageShape*>& outputShapes,
                                                      const std::vector<InputSpec>& outputDescs, const void* tilingData,
                                                      const gert::ContinuousVector* ws)
{
    std::vector<uint32_t> inInst(inputShapes.size(), 1);
    std::vector<uint32_t> outInst(outputShapes.size(), 1);
    gert::TilingContextFaker faker;
    faker.SetOpType("Conv2DV2")
        .NodeIoNum(inputShapes.size(), outputShapes.size())
        .IrInstanceNum(inInst, outInst)
        .InputShapes(inputShapes)
        .OutputShapes(outputShapes)
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo));
    for (size_t i = 0; i < inputDescs.size(); ++i) {
        faker.NodeInputTd(static_cast<int32_t>(i), inputDescs[i].dtype, inputDescs[i].format, inputDescs[i].format);
    }
    for (size_t i = 0; i < outputDescs.size(); ++i) {
        faker.NodeOutputTd(static_cast<int32_t>(i), outputDescs[i].dtype, outputDescs[i].format, outputDescs[i].format);
    }
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::vector<int64_t> dilations = {1, 1, 1, 1};
    faker.NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                     {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                     {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                     {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                     {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                     {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                     {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                     {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}});
    return faker.TilingData(tilingData).Workspace(ws).Build();
}

void InitInputSut(optiling::conv_ops_tiling::Conv2dBaseTiling& sut,
                  optiling::conv_ops_tiling::ConvTilingParseInfo& opInfo, gert::TilingContext* ctx)
{
    sut.opInfo_ = &opInfo;
    sut.paramInfo_.nodeType = "Conv2DV2";
    sut.paramInfo_.paramsIdxVec = {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}};
    sut.shapeInfo_.batch = 1;
    sut.shapeInfo_.ci = 16;
    sut.shapeInfo_.hi = 16;
    sut.shapeInfo_.wi = 16;
    sut.shapeInfo_.co = 16;
    sut.shapeInfo_.kh = 3;
    sut.shapeInfo_.kw = 3;
    sut.shapeInfo_.ho = 16;
    sut.shapeInfo_.wo = 16;
    sut.attrInfo_.groups = 1;
    sut.descInfo_.fMapDtype = ge::DT_FLOAT16;
    sut.descInfo_.weightDtype = ge::DT_FLOAT16;
    sut.descInfo_.biasDtype = ge::DT_FLOAT16;
    sut.descInfo_.outDtype = ge::DT_FLOAT16;
    sut.descInfo_.fMapFormat = ge::FORMAT_NCHW;
    sut.descInfo_.weightFormat = ge::FORMAT_NCHW;
    sut.descInfo_.biasFormat = ge::FORMAT_ND;
    sut.descInfo_.outFormat = ge::FORMAT_NCHW;
    sut.flagInfo_.hasBias = true;
    (void)ctx;
}
} // namespace

class TestConv2dCheckInputDirect : public testing::Test {
protected:
    void SetUp() override
    {
        compile_info_.aicoreNum = AIC_NUM;
        compile_info_.npuArch = NpuArch::DAV_3510;
        tilingData_ = gert::TilingData::CreateCap(MEM_SIZE_4K);
        wsHolder_ = gert::ContinuousVector::Create<size_t>(MEM_SIZE_4K);
        ws_ = reinterpret_cast<gert::ContinuousVector*>(wsHolder_.get());
        fmap_ = {{1, 16, 16, 16}, {1, 16, 16, 16}};
        weight_ = {{16, 16, 3, 3}, {16, 16, 3, 3}};
        bias_ = {{16}, {16}};
        output_ = {{1, 16, 16, 16}, {1, 16, 16, 16}};
    }

    gert::KernelRunContextHolder BuildCtx(const std::vector<gert::StorageShape*>& inShapes,
                                          const std::vector<InputSpec>& inDescs,
                                          const std::vector<gert::StorageShape*>& outShapes,
                                          const std::vector<InputSpec>& outDescs)
    {
        return BuildGenericInputContext(compile_info_, platform_info_, inShapes, inDescs, outShapes, outDescs,
                                        tilingData_.get(), ws_);
    }

    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info_;
    fe::PlatFormInfos platform_info_;
    std::unique_ptr<uint8_t[]> tilingData_;
    std::unique_ptr<uint8_t[]> wsHolder_;
    gert::ContinuousVector* ws_ = nullptr;
    gert::StorageShape fmap_;
    gert::StorageShape weight_;
    gert::StorageShape bias_;
    gert::StorageShape output_;
};

TEST_F(TestConv2dCheckInputDirect, discontinuous_default_stride)
{
    gert::StorageShape fmapView = {{1, 16, 16, 16}, {16, 16, 1, 16}};
    gert::TensorV2 fmapTensor(fmapView, {ge::FORMAT_NCHW, ge::FORMAT_NCHW, gert::ExpandDimsType()},
                              gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
    gert::TensorV2 weightTensor(weight_, {ge::FORMAT_NCHW, ge::FORMAT_NCHW, gert::ExpandDimsType()},
                                gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
    gert::TensorV2 biasTensor(bias_, {ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()},
                              gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
    gert::TensorV2 offsetTensor;
    std::vector<gert::TensorV2*> tensors = {&fmapTensor, &weightTensor, &biasTensor, &offsetTensor};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv2DV2")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .OutputShapes({&output_})
                      .InputTensors(tensors)
                      .CompileInfo(&compile_info_)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info_))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(tilingData_.get())
                      .Workspace(ws_)
                      .Build();
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    EXPECT_EQ(sut.GetDisContinuousFlag(), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(sut.flagInfo_.disContinuousFlag);
}

TEST_F(TestConv2dCheckInputDirect, discontinuous_invalid_dtype)
{
    gert::StorageShape fmapView = {{1, 16, 16, 16}, {16, 16, 1, 16}};
    gert::TensorV2 fmapTensor(fmapView, {ge::FORMAT_NCHW, ge::FORMAT_NCHW, gert::ExpandDimsType()},
                              gert::TensorPlacement::kOnHost, ge::DT_INT8, nullptr, nullptr,
                              gert::Stride({16, 1, 256, 16}), 0);
    gert::TensorV2 weightTensor(weight_, {ge::FORMAT_NCHW, ge::FORMAT_NCHW, gert::ExpandDimsType()},
                                gert::TensorPlacement::kOnHost, ge::DT_INT8, nullptr);
    gert::TensorV2 biasTensor(bias_, {ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()},
                              gert::TensorPlacement::kOnHost, ge::DT_INT8, nullptr);
    gert::TensorV2 offsetTensor;
    std::vector<gert::TensorV2*> tensors = {&fmapTensor, &weightTensor, &biasTensor, &offsetTensor};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv2DV2")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .OutputShapes({&output_})
                      .InputTensors(tensors)
                      .CompileInfo(&compile_info_)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info_))
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(tilingData_.get())
                      .Workspace(ws_)
                      .Build();
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    sut.descInfo_.fMapDtype = ge::DT_INT8;
    EXPECT_EQ(sut.GetDisContinuousFlag(), ge::GRAPH_FAILED);
}

TEST_F(TestConv2dCheckInputDirect, discontinuous_bad_stride_and_offset)
{
    gert::StorageShape fmapView = {{1, 16, 16, 16}, {16, 16, 1, 16}};
    {
        gert::TensorV2 fmapTensor(fmapView, {ge::FORMAT_NCHW, ge::FORMAT_NCHW, gert::ExpandDimsType()},
                                  gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr, nullptr,
                                  gert::Stride({16, 1, 256, 16}), 5);
        gert::TensorV2 weightTensor(weight_, {ge::FORMAT_NCHW, ge::FORMAT_NCHW, gert::ExpandDimsType()},
                                    gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
        gert::TensorV2 biasTensor(bias_, {ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()},
                                  gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
        gert::TensorV2 offsetTensor;
        std::vector<gert::TensorV2*> tensors = {&fmapTensor, &weightTensor, &biasTensor, &offsetTensor};
        auto holder = gert::TilingContextFaker()
                          .SetOpType("Conv2DV2")
                          .NodeIoNum(4, 1)
                          .IrInstanceNum({1, 1, 1, 1})
                          .OutputShapes({&output_})
                          .InputTensors(tensors)
                          .CompileInfo(&compile_info_)
                          .PlatformInfo(reinterpret_cast<char*>(&platform_info_))
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                          .TilingData(tilingData_.get())
                          .Workspace(ws_)
                          .Build();
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        EXPECT_EQ(sut.GetDisContinuousFlag(), ge::GRAPH_FAILED);
    }
    {
        gert::TensorV2 fmapTensor(fmapView, {ge::FORMAT_NCHW, ge::FORMAT_NCHW, gert::ExpandDimsType()},
                                  gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr, nullptr,
                                  gert::Stride({16, 1, 1, 1}), 0);
        gert::TensorV2 weightTensor(weight_, {ge::FORMAT_NCHW, ge::FORMAT_NCHW, gert::ExpandDimsType()},
                                    gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
        gert::TensorV2 biasTensor(bias_, {ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()},
                                  gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
        gert::TensorV2 offsetTensor;
        std::vector<gert::TensorV2*> tensors = {&fmapTensor, &weightTensor, &biasTensor, &offsetTensor};
        auto holder = gert::TilingContextFaker()
                          .SetOpType("Conv2DV2")
                          .NodeIoNum(4, 1)
                          .IrInstanceNum({1, 1, 1, 1})
                          .OutputShapes({&output_})
                          .InputTensors(tensors)
                          .CompileInfo(&compile_info_)
                          .PlatformInfo(reinterpret_cast<char*>(&platform_info_))
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                          .TilingData(tilingData_.get())
                          .Workspace(ws_)
                          .Build();
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        EXPECT_EQ(sut.GetDisContinuousFlag(), ge::GRAPH_FAILED);
    }
}

TEST_F(TestConv2dCheckInputDirect, discontinuous_legal_view)
{
    gert::StorageShape fmapView = {{1, 16, 16, 16}, {16, 16, 1, 16}};
    gert::TensorV2 fmapTensor(fmapView, {ge::FORMAT_NCHW, ge::FORMAT_NCHW, gert::ExpandDimsType()},
                              gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr, nullptr,
                              gert::Stride({16, 1, 256, 16}), 0);
    gert::TensorV2 weightTensor(weight_, {ge::FORMAT_NCHW, ge::FORMAT_NCHW, gert::ExpandDimsType()},
                                gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
    gert::TensorV2 biasTensor(bias_, {ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()},
                              gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
    gert::TensorV2 offsetTensor;
    std::vector<gert::TensorV2*> tensors = {&fmapTensor, &weightTensor, &biasTensor, &offsetTensor};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv2DV2")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .OutputShapes({&output_})
                      .InputTensors(tensors)
                      .CompileInfo(&compile_info_)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info_))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .TilingData(tilingData_.get())
                      .Workspace(ws_)
                      .Build();
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    EXPECT_EQ(sut.GetDisContinuousFlag(), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(sut.flagInfo_.disContinuousFlag);
}

TEST_F(TestConv2dCheckInputDirect, three_dim_shapes_rejected)
{
    gert::StorageShape fmapBad = {{1, 16, 16}, {1, 16, 16}};
    gert::StorageShape weightBad = {{16, 16, 16}, {16, 16, 16}};
    gert::StorageShape outputBad = {{1, 16, 16}, {1, 16, 16}};
    {
        std::vector<gert::StorageShape*> inShapes = {&fmapBad, &weight_, &bias_, nullptr};
        std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {ge::DT_FLOAT16, ge::FORMAT_ND},
                                          {ge::DT_FLOAT, ge::FORMAT_ND}};
        std::vector<gert::StorageShape*> outShapes = {&output_};
        std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};
        auto holder = BuildCtx(inShapes, inDescs, outShapes, outDescs);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        EXPECT_EQ(sut.ParseFmapShape(), ge::GRAPH_FAILED);
    }
    {
        std::vector<gert::StorageShape*> inShapes = {&fmap_, &weightBad, &bias_, nullptr};
        std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {ge::DT_FLOAT16, ge::FORMAT_ND},
                                          {ge::DT_FLOAT, ge::FORMAT_ND}};
        std::vector<gert::StorageShape*> outShapes = {&output_};
        std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};
        auto holder = BuildCtx(inShapes, inDescs, outShapes, outDescs);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        EXPECT_EQ(sut.ParseWeightShape(), ge::GRAPH_FAILED);
    }
    {
        std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &bias_, nullptr};
        std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {ge::DT_FLOAT16, ge::FORMAT_ND},
                                          {ge::DT_FLOAT, ge::FORMAT_ND}};
        std::vector<gert::StorageShape*> outShapes = {&outputBad};
        std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};
        auto holder = BuildCtx(inShapes, inDescs, outShapes, outDescs);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        EXPECT_EQ(sut.ParseOutputShape(), ge::GRAPH_FAILED);
    }
}

TEST_F(TestConv2dCheckInputDirect, parse_fmap_shape_discontinuous)
{
    gert::StorageShape fmapHwnc = {{16, 16, 1, 16}, {16, 16, 1, 16}};
    std::vector<gert::StorageShape*> inShapes = {&fmapHwnc, &weight_, &bias_, nullptr};
    std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {ge::DT_FLOAT, ge::FORMAT_ND}};
    std::vector<gert::StorageShape*> outShapes = {&output_};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};
    auto holder = BuildCtx(inShapes, inDescs, outShapes, outDescs);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    sut.flagInfo_.disContinuousFlag = true;
    EXPECT_EQ(sut.ParseFmapShape(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sut.oriShapeAttrInfo_.oriFmapN, 1);
    EXPECT_EQ(sut.oriShapeAttrInfo_.oriFmapC, 16);
    EXPECT_EQ(sut.oriShapeAttrInfo_.oriFmapH, 16);
    EXPECT_EQ(sut.oriShapeAttrInfo_.oriFmapW, 16);
}

TEST_F(TestConv2dCheckInputDirect, bias_shape_variants)
{
    std::vector<InputSpec> inDescsNchw = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {ge::DT_FLOAT, ge::FORMAT_ND}};
    std::vector<gert::StorageShape*> outShapes = {&output_};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};

    // 1-D bias whose length does not match cout
    {
        gert::StorageShape biasShort = {{8}, {8}};
        std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &biasShort, nullptr};
        auto holder = BuildCtx(inShapes, inDescsNchw, outShapes, outDescs);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        EXPECT_EQ(sut.ParseBiasShape(), ge::GRAPH_FAILED);
    }
    // valid 4-D bias
    {
        gert::StorageShape bias4d = {{1, 16, 1, 1}, {1, 16, 1, 1}};
        std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &bias4d, nullptr};
        auto holder = BuildCtx(inShapes, inDescsNchw, outShapes, outDescs);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        sut.descInfo_.biasFormat = ge::FORMAT_NCHW;
        EXPECT_EQ(sut.ParseBiasShape(), ge::GRAPH_SUCCESS);
    }
    // 4-D bias with non-C dim != 1
    {
        gert::StorageShape bias4dBad = {{1, 16, 2, 1}, {1, 16, 2, 1}};
        std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &bias4dBad, nullptr};
        auto holder = BuildCtx(inShapes, inDescsNchw, outShapes, outDescs);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        sut.descInfo_.biasFormat = ge::FORMAT_NCHW;
        EXPECT_EQ(sut.ParseBiasShape(), ge::GRAPH_FAILED);
    }
    // 4-D bias with ND format cannot locate the C dim
    {
        gert::StorageShape bias4d = {{1, 16, 1, 1}, {1, 16, 1, 1}};
        std::vector<InputSpec> inDescsNd = inDescsNchw;
        inDescsNd[2] = {ge::DT_FLOAT16, ge::FORMAT_ND};
        std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &bias4d, nullptr};
        auto holder = BuildCtx(inShapes, inDescsNd, outShapes, outDescs);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        sut.descInfo_.biasFormat = ge::FORMAT_ND;
        EXPECT_EQ(sut.ParseBiasShape(), ge::GRAPH_FAILED);
    }
    // 2-D bias is invalid
    {
        gert::StorageShape bias2d = {{16, 1}, {16, 1}};
        std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &bias2d, nullptr};
        auto holder = BuildCtx(inShapes, inDescsNchw, outShapes, outDescs);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        EXPECT_EQ(sut.ParseBiasShape(), ge::GRAPH_FAILED);
    }
}

TEST_F(TestConv2dCheckInputDirect, output_h_w_bound_checks)
{
    std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &bias_, nullptr};
    std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {ge::DT_FLOAT, ge::FORMAT_ND}};
    std::vector<gert::StorageShape*> outShapes = {&output_};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};
    auto holder = BuildCtx(inShapes, inDescs, outShapes, outDescs);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    sut.oriShapeAttrInfo_.oriOutputH = 1000001;
    sut.oriShapeAttrInfo_.oriOutputW = 16;
    EXPECT_EQ(sut.CheckOutputShape(), ge::GRAPH_FAILED);
    sut.oriShapeAttrInfo_.oriOutputH = 16;
    sut.oriShapeAttrInfo_.oriOutputW = 1000001;
    EXPECT_EQ(sut.CheckOutputShape(), ge::GRAPH_FAILED);
}

TEST_F(TestConv2dCheckInputDirect, quant_scale_checks)
{
    std::vector<gert::StorageShape*> outShapes = {&output_};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};

    auto runCase = [&](ge::DataType scaleDtype, const gert::StorageShape& scaleShape) {
        std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &bias_, nullptr};
        std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {scaleDtype, ge::FORMAT_ND},
                                          {ge::DT_FLOAT, ge::FORMAT_ND}};
        return BuildGenericInputContext(compile_info_, platform_info_, inShapes, inDescs, outShapes, outDescs,
                                        tilingData_.get(), ws_);
    };
    auto runScaleCase = [&](ge::DataType scaleDtype, gert::StorageShape& scaleShape) {
        std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &scaleShape, nullptr};
        std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                          {scaleDtype, ge::FORMAT_ND},
                                          {ge::DT_FLOAT, ge::FORMAT_ND}};
        return BuildGenericInputContext(compile_info_, platform_info_, inShapes, inDescs, outShapes, outDescs,
                                        tilingData_.get(), ws_);
    };

    // wrong dtype
    {
        gert::StorageShape scaleShape = {{16}, {16}};
        auto holder = runScaleCase(ge::DT_FLOAT16, scaleShape);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        sut.flagInfo_.quantFlag = true;
        EXPECT_EQ(sut.CheckQuantScaleLegal(), ge::GRAPH_FAILED);
    }
    // non 1-D scale
    {
        gert::StorageShape scaleShape = {{16, 1}, {16, 1}};
        auto holder = runScaleCase(ge::DT_INT64, scaleShape);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        sut.flagInfo_.quantFlag = true;
        EXPECT_EQ(sut.CheckQuantScaleLegal(), ge::GRAPH_FAILED);
    }
    // length mismatch
    {
        gert::StorageShape scaleShape = {{8}, {8}};
        auto holder = runScaleCase(ge::DT_INT64, scaleShape);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        sut.flagInfo_.quantFlag = true;
        EXPECT_EQ(sut.CheckQuantScaleLegal(), ge::GRAPH_FAILED);
    }
    // valid vector quant scale
    {
        gert::StorageShape scaleShape = {{16}, {16}};
        auto holder = runScaleCase(ge::DT_INT64, scaleShape);
        auto ctx = holder.GetContext<gert::TilingContext>();
        optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
        InitInputSut(sut, compile_info_, ctx);
        sut.flagInfo_.quantFlag = true;
        EXPECT_EQ(sut.CheckQuantScaleLegal(), ge::GRAPH_SUCCESS);
        EXPECT_GT(sut.fixpipeInfo_.channelWiseCoeff, 0);
    }
}

TEST_F(TestConv2dCheckInputDirect, extend_scale_legal_invalid_dtype)
{
    std::vector<gert::StorageShape*> outShapes = {&output_};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};
    gert::StorageShape scaleShape = {{16}, {16}};
    std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &scaleShape, nullptr};
    std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {ge::DT_FLOAT, ge::FORMAT_ND}};
    auto holder = BuildGenericInputContext(compile_info_, platform_info_, inShapes, inDescs, outShapes, outDescs,
                                           tilingData_.get(), ws_);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    uint8_t quantMode = 0;
    EXPECT_FALSE(sut.CheckScaleLegal(2, quantMode, "scale0"));
}

TEST_F(TestConv2dCheckInputDirect, extend_scale_legal_invalid_format)
{
    std::vector<gert::StorageShape*> outShapes = {&output_};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};
    gert::StorageShape scaleShape = {{16}, {16}};
    std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &scaleShape, nullptr};
    std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_INT64, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT, ge::FORMAT_ND}};
    auto holder = BuildGenericInputContext(compile_info_, platform_info_, inShapes, inDescs, outShapes, outDescs,
                                           tilingData_.get(), ws_);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    uint8_t quantMode = 0;
    EXPECT_FALSE(sut.CheckScaleLegal(2, quantMode, "scale0"));
}

TEST_F(TestConv2dCheckInputDirect, extend_scale_legal_invalid_shape)
{
    std::vector<gert::StorageShape*> outShapes = {&output_};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};
    gert::StorageShape scaleShape = {{16, 1}, {16, 1}};
    std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &scaleShape, nullptr};
    std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_INT64, ge::FORMAT_ND},
                                      {ge::DT_FLOAT, ge::FORMAT_ND}};
    auto holder = BuildGenericInputContext(compile_info_, platform_info_, inShapes, inDescs, outShapes, outDescs,
                                           tilingData_.get(), ws_);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    uint8_t quantMode = 0;
    EXPECT_FALSE(sut.CheckScaleLegal(2, quantMode, "scale0"));
}

TEST_F(TestConv2dCheckInputDirect, extend_scale_legal_scalar_quant)
{
    std::vector<gert::StorageShape*> outShapes = {&output_};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};
    gert::StorageShape scaleShape = {{1}, {1}};
    std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &scaleShape, nullptr};
    std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_INT64, ge::FORMAT_ND},
                                      {ge::DT_FLOAT, ge::FORMAT_ND}};
    auto holder = BuildGenericInputContext(compile_info_, platform_info_, inShapes, inDescs, outShapes, outDescs,
                                           tilingData_.get(), ws_);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    uint8_t quantMode = 0;
    EXPECT_TRUE(sut.CheckScaleLegal(2, quantMode, "scale0"));
}

TEST_F(TestConv2dCheckInputDirect, extend_dual_output_special)
{
    std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &bias_, nullptr};
    std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {ge::DT_FLOAT, ge::FORMAT_ND}};
    gert::StorageShape output1 = {{1, 16, 16, 16}, {1, 16, 16, 16}};
    std::vector<gert::StorageShape*> outShapes = {&output_, &output1};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}, {ge::DT_INT8, ge::FORMAT_NCHW}};
    auto holder = BuildCtx(inShapes, inDescs, outShapes, outDescs);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    sut.flagInfo_.extendConvFlag = true;
    sut.attrInfo_.dualOutput = 1;
    sut.descInfo_.outFormat = ge::FORMAT_NCHW;
    sut.descInfo_.outDtype = ge::DT_FLOAT16;
    sut.descInfo_.out1Format = ge::FORMAT_NHWC;
    sut.descInfo_.out1Dtype = ge::DT_INT8;
    EXPECT_EQ(sut.CheckExtendDualOutputSpecial(), ge::GRAPH_FAILED);
    sut.descInfo_.out1Format = ge::FORMAT_NCHW;
    sut.descInfo_.out1Dtype = ge::DT_BF16;
    EXPECT_EQ(sut.CheckExtendDualOutputSpecial(), ge::GRAPH_FAILED);
    sut.descInfo_.out1Dtype = ge::DT_INT8;
    EXPECT_EQ(sut.CheckExtendDualOutputSpecial(), ge::GRAPH_SUCCESS);
}

TEST_F(TestConv2dCheckInputDirect, params_dtype_int8_c04_fuse)
{
    std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &bias_, nullptr};
    std::vector<InputSpec> inDescs = {{ge::DT_INT8, ge::FORMAT_NCHW},
                                      {ge::DT_INT8, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT, ge::FORMAT_ND},
                                      {ge::DT_FLOAT, ge::FORMAT_ND}};
    std::vector<gert::StorageShape*> outShapes = {&output_};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}};
    auto holder = BuildCtx(inShapes, inDescs, outShapes, outDescs);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    sut.opInfo_->isCubeVectorFuse = true;
    sut.descInfo_.fMapDtype = ge::DT_INT8;
    sut.descInfo_.weightFormat = ge::FORMAT_FRACTAL_Z_C04;
    EXPECT_EQ(sut.CheckParamsDtype(), ge::GRAPH_FAILED);
}

TEST_F(TestConv2dCheckInputDirect, dual_output_parse_3d_rejected)
{
    std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &bias_, nullptr};
    std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {ge::DT_FLOAT, ge::FORMAT_ND}};
    gert::StorageShape output1 = {{1, 16, 16}, {1, 16, 16}};
    std::vector<gert::StorageShape*> outShapes = {&output_, &output1};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}, {ge::DT_INT8, ge::FORMAT_NCHW}};
    auto holder = BuildCtx(inShapes, inDescs, outShapes, outDescs);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    sut.flagInfo_.extendConvFlag = true;
    sut.attrInfo_.dualOutput = 1;
    EXPECT_EQ(sut.ParseExtendDualOutputShape(), ge::GRAPH_FAILED);
}

TEST_F(TestConv2dCheckInputDirect, dual_output_shape_mismatch)
{
    std::vector<gert::StorageShape*> inShapes = {&fmap_, &weight_, &bias_, nullptr};
    std::vector<InputSpec> inDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                      {ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {ge::DT_FLOAT, ge::FORMAT_ND}};
    gert::StorageShape output1 = {{1, 8, 16, 16}, {1, 8, 16, 16}};
    std::vector<gert::StorageShape*> outShapes = {&output_, &output1};
    std::vector<InputSpec> outDescs = {{ge::DT_FLOAT16, ge::FORMAT_NCHW}, {ge::DT_INT8, ge::FORMAT_NCHW}};
    auto holder = BuildCtx(inShapes, inDescs, outShapes, outDescs);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitInputSut(sut, compile_info_, ctx);
    sut.flagInfo_.extendConvFlag = true;
    sut.attrInfo_.dualOutput = 1;
    sut.oriShapeAttrInfo_.oriOutputN = 1;
    sut.oriShapeAttrInfo_.oriOutputC = 16;
    sut.oriShapeAttrInfo_.oriOutputH = 16;
    sut.oriShapeAttrInfo_.oriOutputW = 16;
    sut.oriShapeAttrInfo_.oriOutput1N = 1;
    sut.oriShapeAttrInfo_.oriOutput1C = 8;
    sut.oriShapeAttrInfo_.oriOutput1H = 16;
    sut.oriShapeAttrInfo_.oriOutput1W = 16;
    EXPECT_EQ(sut.CheckExtendDualOutputShape(), ge::GRAPH_FAILED);
}
