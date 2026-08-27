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
 * \file test_apply_came_part4_tiling.cpp
 * \brief ApplyCamePart4 arch35 (Ascend950) tiling UT.
 *
 * 覆盖:fp32/fp16/bf16 正常 tiling(blockDim 与 per-core/per-loop 字段)、尾块 shape(n=65)、
 * 空 tensor(n=0 时 SetBlockDim(1))、param 3D 非法 shape 拒绝、非法 dtype 拒绝。
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_kernel/arch35/apply_came_part4_tiling_data.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "array_ops.h"
#include "tests/ut/common/ut_op_util.h"
#include "tests/ut/common/any_value.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace optiling {
// 与被测 tiling 实现中的空 CompileInfo 一致(tiling 不读取 CompileInfo)
struct ApplyCamePart4CompileInfo {};
} // namespace optiling

class ApplyCamePart4Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyCamePart4Tiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ApplyCamePart4Tiling TearDown" << std::endl; }
};

struct CamePart4TilingExpect {
    int64_t blockDim = 0;
    int64_t rNumPerCore = 0;
    int64_t rCoreNumToUse = 0;
    int64_t cNumPerCore = 0;
    int64_t cCoreNumToUse = 0;
    int64_t rRcNumPerCore = 0;
    int64_t rRcCoreNumToUse = 0;
    int64_t rRcNumOnTailCore = 0;
    int64_t rRcNumPerLoop = 0;
    int64_t cRcNumPerLoop = 0;
    int64_t cRcLoopCount = 0;
    int64_t cRcNumTailLoop = 0;
};

static void CamePart4TilingTest(std::initializer_list<int64_t> paramShape, ge::DataType tensorDtype,
                                bool withOptionalInputs, const ge::graphStatus expectStatus,
                                const CamePart4TilingExpect& expect = {})
{
    std::string opType("ApplyCamePart4");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tiling_func, nullptr);

    // Ascend950 平台:CORE_NUM 64 / UB 245760,与 apply_adam_w_quant arch35 UT 一致
    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, socVersion);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::ApplyCamePart4CompileInfo compile_info;

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    int64_t n = *paramShape.begin();
    int64_t m = paramShape.size() > 1 ? *(paramShape.begin() + 1) : 0;

    gert::StorageShape paramInShape = {paramShape, paramShape};
    gert::StorageShape mInShape = {paramShape, paramShape};
    gert::StorageShape rInShape = {{n}, {n}};
    gert::StorageShape cInShape = {{m}, {m}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape globalShape = {{2}, {2}};
    gert::StorageShape paramOutShape = {paramShape, paramShape};
    gert::StorageShape rOutShape = {{n}, {n}};
    gert::StorageShape cOutShape = {{m}, {m}};

    std::vector<uint32_t> irInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    gert::TilingContextFaker faker;
    faker.SetOpType(opType)
        .NodeIoNum(12, 3)
        .IrInstanceNum(irInstanceNum)
        .InputShapes({&paramInShape, &mInShape, &rInShape, &cInShape, &scalarShape, &scalarShape, &scalarShape,
                      &scalarShape, &scalarShape, &scalarShape, withOptionalInputs ? &scalarShape : nullptr,
                      withOptionalInputs ? &globalShape : nullptr})
        .OutputShapes({&paramOutShape, &rOutShape, &cOutShape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, tensorDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, tensorDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, tensorDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, tensorDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, tensorDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, tensorDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, tensorDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    if (withOptionalInputs) {
        faker.NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        faker.NodeInputTd(11, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.TilingData(param.get()).Workspace(ws_size).Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    EXPECT_EQ(tiling_func(tiling_context), expectStatus);
    if (expectStatus != ge::GRAPH_SUCCESS) {
        return;
    }

    EXPECT_EQ(tiling_context->GetBlockDim(), expect.blockDim);
    if (n <= 0 || m <= 0) {
        // 空 tensor:仅要求 SetBlockDim(1),无有效 tiling data
        return;
    }
    auto rawTilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(rawTilingData, nullptr);
    ASSERT_GE(rawTilingData->GetDataSize(), sizeof(ApplyCamePart4TilingData));
    const auto* tilingData = reinterpret_cast<const ApplyCamePart4TilingData*>(rawTilingData->GetData());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->n, n);
    EXPECT_EQ(tilingData->m, m);
    EXPECT_EQ(tilingData->rNumPerCore, expect.rNumPerCore);
    EXPECT_EQ(tilingData->rCoreNumToUse, expect.rCoreNumToUse);
    EXPECT_EQ(tilingData->cNumPerCore, expect.cNumPerCore);
    EXPECT_EQ(tilingData->cCoreNumToUse, expect.cCoreNumToUse);
    EXPECT_EQ(tilingData->rRcNumPerCore, expect.rRcNumPerCore);
    EXPECT_EQ(tilingData->rRcCoreNumToUse, expect.rRcCoreNumToUse);
    EXPECT_EQ(tilingData->rRcNumOnTailCore, expect.rRcNumOnTailCore);
    EXPECT_EQ(tilingData->rRcNumPerLoop, expect.rRcNumPerLoop);
    EXPECT_EQ(tilingData->cRcNumPerLoop, expect.cRcNumPerLoop);
    EXPECT_EQ(tilingData->cRcLoopCount, expect.cRcLoopCount);
    EXPECT_EQ(tilingData->cRcNumTailLoop, expect.cRcNumTailLoop);
}

TEST_F(ApplyCamePart4Tiling, apply_came_part4_tiling_fp32)
{
    // param [64, 32] fp32:typeSize=4, numPerBlock=8, 期望分核 R=8 / C=4 / Param=8
    CamePart4TilingExpect expect;
    expect.blockDim = 8;
    expect.rNumPerCore = 8;
    expect.rCoreNumToUse = 8;
    expect.cNumPerCore = 8;
    expect.cCoreNumToUse = 4;
    expect.rRcNumPerCore = 8;
    expect.rRcCoreNumToUse = 8;
    expect.rRcNumOnTailCore = 8;
    expect.rRcNumPerLoop = 8;  // (256 / 4) / 8
    expect.cRcNumPerLoop = 64; // 256 / 4
    expect.cRcLoopCount = 1;
    expect.cRcNumTailLoop = 32;
    CamePart4TilingTest({64, 32}, ge::DT_FLOAT, true, ge::GRAPH_SUCCESS, expect);
}

TEST_F(ApplyCamePart4Tiling, apply_came_part4_tiling_fp16)
{
    // param [96, 256] fp16:typeSize=2, numPerBlock=16, 期望分核 R=6 / C=16 / Param=6
    CamePart4TilingExpect expect;
    expect.blockDim = 16;
    expect.rNumPerCore = 16;
    expect.rCoreNumToUse = 6;
    expect.cNumPerCore = 16;
    expect.cCoreNumToUse = 16;
    expect.rRcNumPerCore = 16;
    expect.rRcCoreNumToUse = 6;
    expect.rRcNumOnTailCore = 16;
    expect.rRcNumPerLoop = 16;  // (256 / 2) / 8
    expect.cRcNumPerLoop = 128; // 256 / 2
    expect.cRcLoopCount = 2;
    expect.cRcNumTailLoop = 128;
    CamePart4TilingTest({96, 256}, ge::DT_FLOAT16, true, ge::GRAPH_SUCCESS, expect);
}

TEST_F(ApplyCamePart4Tiling, apply_came_part4_tiling_bf16)
{
    // param [48, 128] bf16:typeSize=2, numPerBlock=16, 期望分核 R=3 / C=8 / Param=3
    CamePart4TilingExpect expect;
    expect.blockDim = 8;
    expect.rNumPerCore = 16;
    expect.rCoreNumToUse = 3;
    expect.cNumPerCore = 16;
    expect.cCoreNumToUse = 8;
    expect.rRcNumPerCore = 16;
    expect.rRcCoreNumToUse = 3;
    expect.rRcNumOnTailCore = 16;
    expect.rRcNumPerLoop = 16;
    expect.cRcNumPerLoop = 128;
    expect.cRcLoopCount = 1;
    expect.cRcNumTailLoop = 128;
    CamePart4TilingTest({48, 128}, ge::DT_BF16, true, ge::GRAPH_SUCCESS, expect);
}

TEST_F(ApplyCamePart4Tiling, apply_came_part4_tiling_tail_block)
{
    // 尾块 shape:param [65, 65] fp32,尾核仅 1 行/列,cRcNumTailLoop=1
    CamePart4TilingExpect expect;
    expect.blockDim = 9;
    expect.rNumPerCore = 8;
    expect.rCoreNumToUse = 9;
    expect.cNumPerCore = 8;
    expect.cCoreNumToUse = 9;
    expect.rRcNumPerCore = 8;
    expect.rRcCoreNumToUse = 9;
    expect.rRcNumOnTailCore = 1;
    expect.rRcNumPerLoop = 8;
    expect.cRcNumPerLoop = 64;
    expect.cRcLoopCount = 2;
    expect.cRcNumTailLoop = 1;
    CamePart4TilingTest({65, 65}, ge::DT_FLOAT, false, ge::GRAPH_SUCCESS, expect);
}

TEST_F(ApplyCamePart4Tiling, apply_came_part4_tiling_empty_tensor)
{
    // 空 tensor:n=0,tiling 直接 SetBlockDim(1) 成功返回
    CamePart4TilingExpect expect;
    expect.blockDim = 1;
    CamePart4TilingTest({0, 32}, ge::DT_FLOAT, false, ge::GRAPH_SUCCESS, expect);
}

TEST_F(ApplyCamePart4Tiling, apply_came_part4_tiling_param_3d_rejected)
{
    // param/m 必须为 2D,3D 输入应被拒绝
    CamePart4TilingTest({2, 3, 4}, ge::DT_FLOAT, false, ge::GRAPH_FAILED);
}

TEST_F(ApplyCamePart4Tiling, apply_came_part4_tiling_invalid_dtype_rejected)
{
    // dtype 不在 fp32/fp16/bf16 白名单内(GetSizeByDataType 返回 -1)应被拒绝
    CamePart4TilingTest({64, 32}, ge::DT_UNDEFINED, false, ge::GRAPH_FAILED);
}
