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
 * \file test_softmax_focal_loss_tiling.cpp
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
#include "../../../../op_kernel/arch35/softmax_focal_loss_tiling_data.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class SoftmaxFocalLossTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SoftmaxFocalLossTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SoftmaxFocalLossTiling TearDown" << std::endl; }
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

// 末尾两个参数用于负向用例: 覆盖单个输入的 shape, 缺省表示与 pred 一致
static ge::graphStatus DoSoftmaxFocalLossTilingCase(std::initializer_list<int64_t> shape, ge::DataType predDtype,
                                                    ge::DataType weightDtype, bool hasWeight, float gamma, float alpha,
                                                    std::string reduction, uint64_t& tilingKey, uint32_t& blockDim,
                                                    std::initializer_list<int64_t> targetShapeOv = {},
                                                    std::initializer_list<int64_t> weightShapeOv = {},
                                                    bool withReductionAttr = true, float* gammaOut = nullptr,
                                                    float* alphaOut = nullptr)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics);

    struct SoftmaxFocalLossCompileInfo {};
    SoftmaxFocalLossCompileInfo compileInfo;

    std::string opType("SoftmaxFocalLoss");
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
    gert::StorageShape weightShape = weightShapeOv.size() > 0 ? gert::StorageShape{weightShapeOv, weightShapeOv} :
                                                                gert::StorageShape{shape, shape};
    gert::StorageShape yShape = {shape, shape};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    EXPECT_NE(param, nullptr);

    // reduction 是 OPTIONAL 属性, withReductionAttr=false 模拟调用方完全不下发的场景
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"gamma", Ops::NN::AnyValue::CreateFrom<float>(gamma)}, {"alpha", Ops::NN::AnyValue::CreateFrom<float>(alpha)}};
    if (withReductionAttr) {
        attrList.emplace_back("reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction));
    }

    // 缺省 optional 输入时: NodeIoNum 只算实到输入, IrInstanceNum 该位置 0, InputShapes 不传 nullptr
    auto buildHolder = [&]() {
        if (hasWeight) {
            return gert::TilingContextFaker()
                .SetOpType("SoftmaxFocalLoss")
                .NodeIoNum(3, 1)
                .IrInstanceNum({1, 1, 1})
                .InputShapes({&predShape, &targetShape, &weightShape})
                .OutputShapes({&yShape})
                .CompileInfo(&compileInfo)
                .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                .NodeInputTd(0, predDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, weightDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, predDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs(attrList)
                .TilingData(param.get())
                .Workspace(wsSize)
                .Build();
        }
        return gert::TilingContextFaker()
            .SetOpType("SoftmaxFocalLoss")
            .NodeIoNum(2, 1)
            .IrInstanceNum({1, 1, 0})
            .InputShapes({&predShape, &targetShape})
            .OutputShapes({&yShape})
            .CompileInfo(&compileInfo)
            .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
            .NodeInputTd(0, predDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, predDtype, ge::FORMAT_ND, ge::FORMAT_ND)
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
        if (raw != nullptr && raw->GetDataSize() >= sizeof(SoftmaxFocalLossArch35TilingData)) {
            const auto* td = reinterpret_cast<const SoftmaxFocalLossArch35TilingData*>(raw->GetData());
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

TEST_F(SoftmaxFocalLossTiling, test_tiling_fp32_with_weight)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 128}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "none", tilingKey,
                                           blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 32U);
}

TEST_F(SoftmaxFocalLossTiling, test_tiling_fp16_weight_fp32)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 128}, ge::DT_FLOAT16, ge::DT_FLOAT, true, 2.0f, 0.25f, "none",
                                           tilingKey, blockDim),
              ge::GRAPH_SUCCESS);
}

TEST_F(SoftmaxFocalLossTiling, test_tiling_no_weight)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 128}, ge::DT_FLOAT, ge::DT_FLOAT, false, 2.0f, 0.25f, "none", tilingKey,
                                           blockDim),
              ge::GRAPH_SUCCESS);
}

TEST_F(SoftmaxFocalLossTiling, test_tiling_multi_chunk_big_d)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({4, 40000}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "none", tilingKey,
                                           blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 4U);
}

TEST_F(SoftmaxFocalLossTiling, test_tiling_core_saturated)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({1000, 96}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "none", tilingKey,
                                           blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 63U); // CeilDiv(1000,64)=16 行/核 -> 实际用 63 核
}

TEST_F(SoftmaxFocalLossTiling, test_tiling_empty_tensor)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    // 空张量按 A2 拒收: 类别维语义失效, 不静默返回空结果
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({0, 64}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "none", tilingKey,
                                           blockDim),
              ge::GRAPH_FAILED);
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 0}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "none", tilingKey,
                                           blockDim),
              ge::GRAPH_FAILED);
}

TEST_F(SoftmaxFocalLossTiling, test_tiling_invalid_dtype)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 128}, ge::DT_INT8, ge::DT_FLOAT, true, 2.0f, 0.25f, "none", tilingKey,
                                           blockDim),
              ge::GRAPH_FAILED);
}

TEST_F(SoftmaxFocalLossTiling, test_tiling_target_shape_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({8, 64}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "none", tilingKey,
                                           blockDim, {8, 32}),
              ge::GRAPH_FAILED);
}

TEST_F(SoftmaxFocalLossTiling, test_tiling_weight_shape_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({8, 64}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "none", tilingKey,
                                           blockDim, {}, {8, 32}),
              ge::GRAPH_FAILED);
}

// reduction 仅支持 "none"(与 A2 一致): 输出 shape 恒等于 pred, 装不下 mean/sum 的标量结果,
// 传入其他取值必须报错而不是静默按 none 计算; 大小写不敏感
TEST_F(SoftmaxFocalLossTiling, test_tiling_reduction_only_none)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 128}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "none", tilingKey,
                                           blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 128}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "NONE", tilingKey,
                                           blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 128}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "mean", tilingKey,
                                           blockDim),
              ge::GRAPH_FAILED);
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 128}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "sum", tilingKey,
                                           blockDim),
              ge::GRAPH_FAILED);
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 128}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "avg", tilingKey,
                                           blockDim),
              ge::GRAPH_FAILED);
}

// 属性顺序必须与 A2 的 REG_OP 一致(前向是 gamma 在前、alpha 在后, 与反向相反)。用非对称取值断言 tilingData。
TEST_F(SoftmaxFocalLossTiling, test_tiling_attr_order_gamma_before_alpha)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    float gamma = 0.0f;
    float alpha = 0.0f;
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 128}, ge::DT_FLOAT, ge::DT_FLOAT, true, 3.0f, 0.75f, "none", tilingKey,
                                           blockDim, {}, {}, true, &gamma, &alpha),
              ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(gamma, 3.0f);
    EXPECT_FLOAT_EQ(alpha, 0.75f);
}

// 缺省值取 "none": 不下发该属性时按缺省处理, 应正常通过(而不是落到不支持的取值上)
TEST_F(SoftmaxFocalLossTiling, test_tiling_reduction_absent_defaults_to_none)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoSoftmaxFocalLossTilingCase({32, 128}, ge::DT_FLOAT, ge::DT_FLOAT, true, 2.0f, 0.25f, "none", tilingKey,
                                           blockDim, {}, {}, false),
              ge::GRAPH_SUCCESS);
}
