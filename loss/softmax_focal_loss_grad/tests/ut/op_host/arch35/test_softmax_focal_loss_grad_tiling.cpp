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
static ge::graphStatus DoSoftmaxFocalLossGradTilingCase(std::initializer_list<int64_t> shape, ge::DataType dtype,
                                                        bool hasWeight, float gamma, float alpha, std::string reduction,
                                                        uint64_t& tilingKey, uint32_t& blockDim,
                                                        std::initializer_list<int64_t> targetShapeOv = {},
                                                        std::initializer_list<int64_t> doutShapeOv = {},
                                                        std::initializer_list<int64_t> weightShapeOv = {},
                                                        ge::DataType doutDtypeOv = ge::DT_UNDEFINED)
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
                .NodeAttrs({{"gamma", Ops::NN::AnyValue::CreateFrom<float>(gamma)},
                            {"alpha", Ops::NN::AnyValue::CreateFrom<float>(alpha)},
                            {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
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
            .NodeAttrs({{"gamma", Ops::NN::AnyValue::CreateFrom<float>(gamma)},
                        {"alpha", Ops::NN::AnyValue::CreateFrom<float>(alpha)},
                        {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
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

TEST_F(SoftmaxFocalLossGradTiling, test_tiling_empty_tensor)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossGradTilingCase({0, 64}, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey, blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 1U);
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
