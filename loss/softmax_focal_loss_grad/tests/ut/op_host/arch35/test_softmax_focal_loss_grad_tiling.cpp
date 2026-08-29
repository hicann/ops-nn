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
 * \file test_softmax_focal_loss_grad_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../../op_kernel/arch35/softmax_focal_loss_grad_tiling_data.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class SoftmaxFocalLossGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SoftmaxFocalLossGradTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SoftmaxFocalLossGradTiling TearDown" << std::endl; }
};

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics)
{
    string hardwareInfo = R"({
        "hardware_info": {"UB_SIZE": 253952, "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(hardwareInfo.c_str(), socInfos, aicoreSpec, intrinsics);
    platFormInfo.Init();
}

// 末尾四个参数用于负向用例: 覆盖单个输入的 shape/dtype, 缺省表示与 pred 一致
static ge::graphStatus DoSoftmaxFocalLossGradTilingCase(
    std::initializer_list<int64_t> shape, ge::DataType dtype, bool hasWeight, float gamma, float alpha,
    std::string reduction, uint64_t& tilingKey, uint32_t& blockDim, std::initializer_list<int64_t> targetShapeOv = {},
    std::initializer_list<int64_t> doutShapeOv = {}, std::initializer_list<int64_t> weightShapeOv = {},
    ge::DataType doutDtypeOv = ge::DT_UNDEFINED, float* reductionCoefOut = nullptr, bool withReductionAttr = true,
    float* gammaOut = nullptr, float* alphaOut = nullptr)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics);

    struct SoftmaxFocalLossGradCompileInfo {};
    SoftmaxFocalLossGradCompileInfo compileInfo;

    std::string opType("SoftmaxFocalLossGrad");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    string compileInfoStr = R"({"device_id": null})";
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platFormInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    EXPECT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape predShape = {shape, shape};
    gert::StorageShape targetShape = targetShapeOv.size() > 0 ? gert::StorageShape{targetShapeOv, targetShapeOv} :
                                                                gert::StorageShape{shape, shape};
    gert::StorageShape doutShape = doutShapeOv.size() > 0 ? gert::StorageShape{doutShapeOv, doutShapeOv} :
                                                            gert::StorageShape{shape, shape};
    gert::StorageShape weightShape = weightShapeOv.size() > 0 ? gert::StorageShape{weightShapeOv, weightShapeOv} :
                                                                gert::StorageShape{shape, shape};
    gert::StorageShape gradShape = {shape, shape};
    ge::DataType doutDtype = doutDtypeOv == ge::DT_UNDEFINED ? dtype : doutDtypeOv;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    EXPECT_NE(param, nullptr);

    // reduction 是 OPTIONAL 属性, withReductionAttr=false 模拟调用方完全不下发的场景
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"alpha", Ops::NN::AnyValue::CreateFrom<float>(alpha)}, {"gamma", Ops::NN::AnyValue::CreateFrom<float>(gamma)}};
    if (withReductionAttr) {
        attrList.emplace_back("reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction));
    }

    // 缺省 optional 输入时: NodeIoNum 只算实到输入, IrInstanceNum 该位置 0, InputShapes 不传 nullptr
    auto buildHolder = [&]() {
        if (hasWeight) {
            return gert::TilingContextFaker()
                .SetOpType("SoftmaxFocalLossGrad")
                .NodeIoNum(4, 1)
                .IrInstanceNum({1, 1, 1, 1})
                .InputShapes({&predShape, &targetShape, &doutShape, &weightShape})
                .OutputShapes({&gradShape})
                .CompileInfo(&compileInfo)
                .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, doutDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs(attrList)
                .TilingData(param.get())
                .Workspace(wsSize)
                .Build();
        }
        return gert::TilingContextFaker()
            .SetOpType("SoftmaxFocalLossGrad")
            .NodeIoNum(3, 1)
            .IrInstanceNum({1, 1, 1, 0})
            .InputShapes({&predShape, &targetShape, &doutShape})
            .OutputShapes({&gradShape})
            .CompileInfo(&compileInfo)
            .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
            .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, doutDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(attrList)
            .TilingData(param.get())
            .Workspace(wsSize)
            .Build();
    };
    auto holder = buildHolder();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tilingContext, nullptr);
    EXPECT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto ret = tilingFunc(tilingContext);
    if (ret == ge::GRAPH_SUCCESS) {
        tilingKey = tilingContext->GetTilingKey();
        blockDim = tilingContext->GetBlockDim();
        auto raw = tilingContext->GetRawTilingData();
        if (raw != nullptr && raw->GetDataSize() >= sizeof(SoftmaxFocalLossGradArch35TilingData)) {
            const auto* td = reinterpret_cast<const SoftmaxFocalLossGradArch35TilingData*>(raw->GetData());
            if (reductionCoefOut != nullptr) {
                *reductionCoefOut = td->reductionCoef;
            }
            if (gammaOut != nullptr) {
                *gammaOut = td->gamma;
            }
            if (alphaOut != nullptr) {
                *alphaOut = td->alpha;
            }
        }
    }
    return ret;
}

TEST_F(SoftmaxFocalLossGradTiling, test_tiling_fp32_with_weight)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey, blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 32U);
}

TEST_F(SoftmaxFocalLossGradTiling, test_tiling_fp16_with_weight)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(
        DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT16, true, 2.0f, 0.25f, "mean", tilingKey, blockDim),
        ge::GRAPH_SUCCESS);
}

TEST_F(SoftmaxFocalLossGradTiling, test_tiling_no_weight)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(
        DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, false, 2.0f, 0.25f, "mean", tilingKey, blockDim),
        ge::GRAPH_SUCCESS);
}

TEST_F(SoftmaxFocalLossGradTiling, test_tiling_multi_chunk_big_d)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(
        DoSoftmaxFocalLossGradTilingCase({4, 40000}, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey, blockDim),
        ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 4U);
}

TEST_F(SoftmaxFocalLossGradTiling, test_tiling_core_saturated)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({1000, 96}, ge::DT_FLOAT, true, 2.0f, 0.25f, "sum", tilingKey, blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 63U); // CeilDiv(1000,64)=16 行/核 -> 实际用 63 核
}

// 空张量(任一维长度为 0)按 A2 拒收: 归约轴(类别维)语义失效, 不静默返回空结果
TEST_F(SoftmaxFocalLossGradTiling, test_tiling_empty_tensor)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({0, 64}, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey, blockDim),
              ge::GRAPH_FAILED);
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 0}, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey, blockDim),
              ge::GRAPH_FAILED);
}

TEST_F(SoftmaxFocalLossGradTiling, test_tiling_invalid_dtype)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_INT8, true, 2.0f, 0.25f, "mean", tilingKey, blockDim),
              ge::GRAPH_FAILED);
}

TEST_F(SoftmaxFocalLossGradTiling, test_tiling_target_shape_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({8, 64}, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey, blockDim,
                                               {8, 32}),
              ge::GRAPH_FAILED);
}

TEST_F(SoftmaxFocalLossGradTiling, test_tiling_dout_shape_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({8, 64}, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey, blockDim,
                                               {}, {8, 32}),
              ge::GRAPH_FAILED);
}

TEST_F(SoftmaxFocalLossGradTiling, test_tiling_weight_shape_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({8, 64}, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey, blockDim,
                                               {}, {}, {8, 32}),
              ge::GRAPH_FAILED);
}

// GE 图通路会把 dout 的 dtype 归一到与 pred 一致, 该校验只能在 tiling 层验证
TEST_F(SoftmaxFocalLossGradTiling, test_tiling_dout_dtype_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({8, 64}, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey, blockDim,
                                               {}, {}, {}, ge::DT_FLOAT16),
              ge::GRAPH_FAILED);
}

// reduction 缺省值是 "mean": 属性未下发时必须与显式传 "mean" 得到同一个系数,
// 不能落到 1.0(那是 "none"/"sum" 的语义)。A2 的 softmax_focal_loss_grad.py 签名默认同为 mean。
TEST_F(SoftmaxFocalLossGradTiling, test_tiling_reduction_default_is_mean)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    float coefAbsent = 0.0f;
    float coefExplicit = 0.0f;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey, blockDim,
                                               {}, {}, {}, ge::DT_UNDEFINED, &coefAbsent, false),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey, blockDim,
                                               {}, {}, {}, ge::DT_UNDEFINED, &coefExplicit, true),
              ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(coefAbsent, 1.0f / (32.0f * 128.0f));
    EXPECT_FLOAT_EQ(coefAbsent, coefExplicit);
}

// "none" / "sum" 不缩放, 与 A2 一致
TEST_F(SoftmaxFocalLossGradTiling, test_tiling_reduction_none_and_sum_no_scaling)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    float coefNone = 0.0f;
    float coefSum = 0.0f;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 2.0f, 0.25f, "none", tilingKey, blockDim,
                                               {}, {}, {}, ge::DT_UNDEFINED, &coefNone),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 2.0f, 0.25f, "sum", tilingKey, blockDim,
                                               {}, {}, {}, ge::DT_UNDEFINED, &coefSum),
              ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(coefNone, 1.0f);
    EXPECT_FLOAT_EQ(coefSum, 1.0f);
}

// 大小写不敏感, 对齐 A2 的 reduction.lower(): "MEAN"/"Mean" 与 "mean" 同义
TEST_F(SoftmaxFocalLossGradTiling, test_tiling_reduction_case_insensitive)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    float coefUpper = 0.0f;
    float coefMixed = 0.0f;
    float coefSumUpper = 0.0f;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 2.0f, 0.25f, "MEAN", tilingKey, blockDim,
                                               {}, {}, {}, ge::DT_UNDEFINED, &coefUpper),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 2.0f, 0.25f, "Mean", tilingKey, blockDim,
                                               {}, {}, {}, ge::DT_UNDEFINED, &coefMixed),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 2.0f, 0.25f, "SUM", tilingKey, blockDim,
                                               {}, {}, {}, ge::DT_UNDEFINED, &coefSumUpper),
              ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(coefUpper, 1.0f / (32.0f * 128.0f));
    EXPECT_FLOAT_EQ(coefMixed, 1.0f / (32.0f * 128.0f));
    EXPECT_FLOAT_EQ(coefSumUpper, 1.0f);
}

// 支持范围约束对齐 A2 的 check_dtype: 取值不在 none/mean/sum 内直接失败, 不静默按 1.0 计算
TEST_F(SoftmaxFocalLossGradTiling, test_tiling_reduction_invalid_value_rejected)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 2.0f, 0.25f, "avg", tilingKey, blockDim),
              ge::GRAPH_FAILED);
    EXPECT_EQ(
        DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 2.0f, 0.25f, "means", tilingKey, blockDim),
        ge::GRAPH_FAILED);
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 2.0f, 0.25f, "", tilingKey, blockDim),
              ge::GRAPH_FAILED);
}

// 属性顺序必须与 A2 的 REG_OP 一致(alpha 在前、gamma 在后)。用非对称取值断言 tilingData,
// 否则索引写反时 gamma/alpha 被互换喂入, 而只看返回码/blockDim/tilingKey 的用例照样全绿。
TEST_F(SoftmaxFocalLossGradTiling, test_tiling_attr_order_alpha_before_gamma)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    float gamma = 0.0f;
    float alpha = 0.0f;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({32, 128}, ge::DT_FLOAT, true, 3.0f, 0.75f, "mean", tilingKey, blockDim,
                                               {}, {}, {}, ge::DT_UNDEFINED, nullptr, true, &gamma, &alpha),
              ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(gamma, 3.0f);
    EXPECT_FLOAT_EQ(alpha, 0.75f);
}
