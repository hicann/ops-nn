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
#include <map>
#include <vector>

#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

#include "../../../op_kernel/elu_grad_v2_tiling_data.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
struct TilingCase {
    string name;
    ge::DataType inputType;
    ge::DataType outputType;
    vector<int64_t> shape;
    float alpha;
    float scale;
    float inputScale;
    bool isResult;
};

struct EluGradV2TilingCompileInfo {};

gert::StorageShape MakeStorageShape(const vector<int64_t>& shape)
{
    switch (shape.size()) {
        case 0:
            return gert::StorageShape({}, {});
        case 1:
            return gert::StorageShape({shape[0]}, {shape[0]});
        case 2:
            return gert::StorageShape({shape[0], shape[1]}, {shape[0], shape[1]});
        case 3:
            return gert::StorageShape({shape[0], shape[1], shape[2]}, {shape[0], shape[1], shape[2]});
        case 4:
            return gert::StorageShape({shape[0], shape[1], shape[2], shape[3]},
                                      {shape[0], shape[1], shape[2], shape[3]});
        case 5:
            return gert::StorageShape({shape[0], shape[1], shape[2], shape[3], shape[4]},
                                      {shape[0], shape[1], shape[2], shape[3], shape[4]});
        default:
            return gert::StorageShape({}, {});
    }
}

string GetCompileInfo()
{
    return R"({
        "hardware_info": {
            "BT_SIZE": 0,
            "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false,
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true,
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 48
        }
    })";
}

void RunTilingCase(const TilingCase& param)
{
    gert::StorageShape input0Shape = MakeStorageShape(param.shape);
    gert::StorageShape input1Shape = MakeStorageShape(param.shape);
    gert::StorageShape outputShape = MakeStorageShape(param.shape);

    string compileInfoString = GetCompileInfo();
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    EluGradV2TilingCompileInfo compileInfo;

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("EluGradV2");
    ASSERT_NE(opImpl, nullptr);
    auto tilingFunc = opImpl->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    auto* parseContext = kernelHolder.GetContext<gert::TilingParseContext>();
    ASSERT_NE(parseContext, nullptr);
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(tilingData, nullptr);
    ASSERT_NE(wsSize, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("EluGradV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input0Shape, &input1Shape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, param.inputType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.inputType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.outputType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(param.alpha)},
                                  {"scale", Ops::NN::AnyValue::CreateFrom<float>(param.scale)},
                                  {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(param.inputScale)},
                                  {"is_result", Ops::NN::AnyValue::CreateFrom<bool>(param.isResult)}})
                      .TilingData(tilingData.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS) << param.name;

    uint64_t totalLength = 1U;
    for (auto dim : param.shape) {
        totalLength *= static_cast<uint64_t>(dim);
    }

    auto* rawTilingData = tilingContext->GetRawTilingData();
    ASSERT_NE(rawTilingData, nullptr);
    auto* tilingDataPtr = reinterpret_cast<const EluGradV2TilingData*>(rawTilingData->GetData());
    ASSERT_NE(tilingDataPtr, nullptr);
    EXPECT_EQ(tilingDataPtr->totalLength, totalLength) << param.name;
    EXPECT_FLOAT_EQ(tilingDataPtr->alpha, param.alpha) << param.name;
    EXPECT_FLOAT_EQ(tilingDataPtr->scale, param.scale) << param.name;
    EXPECT_FLOAT_EQ(tilingDataPtr->inputScale, param.inputScale) << param.name;
    EXPECT_EQ(static_cast<bool>(tilingDataPtr->isResult), param.isResult) << param.name;
    EXPECT_GT(tilingContext->GetBlockDim(), 0U) << param.name;
    if (totalLength > 0U) {
        EXPECT_GT(tilingDataPtr->tileDataNum, 0U) << param.name;
    }
}
} // namespace

class EluGradV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "EluGradV2TilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "EluGradV2TilingTest TearDown" << std::endl; }
};

TEST_F(EluGradV2TilingTest, elu_grad_v2_float32_exp_success)
{
    RunTilingCase({"float32_exp", ge::DT_FLOAT, ge::DT_FLOAT, {1, 2, 8, 16}, 1.3F, 0.7F, 1.1F, false});
}

TEST_F(EluGradV2TilingTest, elu_grad_v2_float16_result_success)
{
    RunTilingCase({"float16_result", ge::DT_FLOAT16, ge::DT_FLOAT16, {1, 2, 8, 16}, 1.2F, 0.8F, 1.0F, true});
}

TEST_F(EluGradV2TilingTest, elu_grad_v2_bfloat16_exp_success)
{
    RunTilingCase({"bfloat16_exp", ge::DT_BF16, ge::DT_BF16, {1, 2, 8, 16}, 1.0F, 1.0F, 1.0F, false});
}

TEST_F(EluGradV2TilingTest, elu_grad_v2_empty_tensor_success)
{
    RunTilingCase({"empty_tensor", ge::DT_FLOAT16, ge::DT_FLOAT16, {0}, 1.0F, 1.0F, 1.0F, false});
}