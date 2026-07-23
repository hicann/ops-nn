/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <string>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

#include "../../../op_kernel/selu_grad_tiling_data.h"
#include "../../../op_kernel/selu_grad_tiling_key.h"

using namespace ge;
using namespace ut_util;

namespace {
constexpr uint64_t EXPECTED_TOTAL_NUM = 120U;

struct SeluGradTilingCompileInfo {};

std::string GetCompileInfo()
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
void RunTilingCase(ge::DataType gradientsType, ge::DataType outputsType, uint32_t scheduleMode,
                   ge::graphStatus expectedStatus, bool emptyTensor = false, bool mismatchShape = false)
{
    gert::StorageShape gradientsShape = emptyTensor ? gert::StorageShape({0}, {0}) :
                                                      gert::StorageShape({2, 3, 4, 5}, {2, 3, 4, 5});
    gert::StorageShape outputsShape = mismatchShape ? gert::StorageShape({2, 3, 4, 6}, {2, 3, 4, 6}) : gradientsShape;
    gert::StorageShape yShape = gradientsShape;

    std::string compileInfoString = GetCompileInfo();
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    SeluGradTilingCompileInfo compileInfo;

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SeluGrad");
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
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(1);
    auto* workspaceSizes = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(tilingData, nullptr);
    ASSERT_NE(workspaceSizes, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("SeluGrad")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradientsShape, &outputsShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, gradientsType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, outputsType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, gradientsType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspaceSizes)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tilingFunc(tilingContext), expectedStatus);
    if (expectedStatus != ge::GRAPH_SUCCESS) {
        return;
    }

    EXPECT_EQ(tilingContext->GetTilingKey(), GET_TPL_TILING_KEY(scheduleMode));
    EXPECT_GE(tilingContext->GetBlockDim(), 1U);

    auto* rawTilingData = tilingContext->GetRawTilingData();
    ASSERT_NE(rawTilingData, nullptr);
    auto* seluGradTilingData = reinterpret_cast<const SeluGradTilingData*>(rawTilingData->GetData());
    ASSERT_NE(seluGradTilingData, nullptr);
    if (emptyTensor) {
        EXPECT_EQ(seluGradTilingData->totalNum, 0U);
        EXPECT_EQ(seluGradTilingData->blockFactor, 0U);
        EXPECT_EQ(seluGradTilingData->ubFactor, 0U);
        EXPECT_EQ(tilingContext->GetBlockDim(), 1U);
    } else {
        EXPECT_EQ(seluGradTilingData->totalNum, EXPECTED_TOTAL_NUM);
        EXPECT_GT(seluGradTilingData->blockFactor, 0U);
        EXPECT_GT(seluGradTilingData->ubFactor, 0U);
    }
}
} // namespace

class SeluGradTilingTest : public testing::Test {};

TEST_F(SeluGradTilingTest, float16_success)
{
    RunTilingCase(ge::DT_FLOAT16, ge::DT_FLOAT16, SELUGRAD_TPL_SCH_MODE_FP16, ge::GRAPH_SUCCESS);
}

TEST_F(SeluGradTilingTest, float32_success)
{
    RunTilingCase(ge::DT_FLOAT, ge::DT_FLOAT, SELUGRAD_TPL_SCH_MODE_FP32, ge::GRAPH_SUCCESS);
}

TEST_F(SeluGradTilingTest, bfloat16_success)
{
    RunTilingCase(ge::DT_BF16, ge::DT_BF16, SELUGRAD_TPL_SCH_MODE_BF16, ge::GRAPH_SUCCESS);
}

TEST_F(SeluGradTilingTest, int32_success)
{
    RunTilingCase(ge::DT_INT32, ge::DT_INT32, SELUGRAD_TPL_SCH_MODE_INT32, ge::GRAPH_SUCCESS);
}

TEST_F(SeluGradTilingTest, int8_success)
{
    RunTilingCase(ge::DT_INT8, ge::DT_INT8, SELUGRAD_TPL_SCH_MODE_INT8, ge::GRAPH_SUCCESS);
}

TEST_F(SeluGradTilingTest, uint8_success)
{
    RunTilingCase(ge::DT_UINT8, ge::DT_UINT8, SELUGRAD_TPL_SCH_MODE_UINT8, ge::GRAPH_SUCCESS);
}

TEST_F(SeluGradTilingTest, empty_tensor_success)
{
    RunTilingCase(ge::DT_FLOAT16, ge::DT_FLOAT16, SELUGRAD_TPL_SCH_MODE_FP16, ge::GRAPH_SUCCESS, true, false);
}

TEST_F(SeluGradTilingTest, dtype_mismatch_failed)
{
    RunTilingCase(ge::DT_FLOAT, ge::DT_FLOAT16, SELUGRAD_TPL_SCH_MODE_FP32, ge::GRAPH_FAILED);
}

TEST_F(SeluGradTilingTest, shape_mismatch_failed)
{
    RunTilingCase(ge::DT_FLOAT, ge::DT_FLOAT, SELUGRAD_TPL_SCH_MODE_FP32, ge::GRAPH_FAILED, false, true);
}
