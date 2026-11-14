/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "tiling_context_faker.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/quant_max_v2_tiling_arch35.h"

namespace {

struct TilingCase {
    ge::DataType xDtype = ge::DT_BF16;
    ge::DataType yDtype = ge::DT_HIFLOAT8;
    gert::StorageShape xShape = {{8192, 2048}, {8192, 2048}};
    gert::StorageShape yShape = {{8192, 2048}, {8192, 2048}};
    gert::StorageShape amaxShape = {{1}, {1}};
    gert::StorageShape yScaleShape = {{1}, {1}};
    std::string roundMode = "round";
    int64_t dstType = 34;
    float dstTypeMax = 15.0f;
    bool hasScale = false;
    bool hasGroupList = false;
    ge::graphStatus status = ge::GRAPH_SUCCESS;
};

class QuantMaxV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

void ExecuteTilingCase(const TilingCase& tc)
{
    const std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::QuantMaxV2CompileInfo compileInfo;

    const std::string opType("QuantMaxV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    std::vector<char> compileInfoBuffer(compileInfoString.begin(), compileInfoString.end());
    compileInfoBuffer.push_back('\0');
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({compileInfoBuffer.data(), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    auto* parsePlatform = kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo();
    ASSERT_TRUE(parsePlatform->Init());
    parsePlatform->SetPlatformRes("SoCInfo", socInfos);
    parsePlatform->SetPlatformRes("AICoreSpec", aicoreSpec);
    parsePlatform->SetCoreNumByCoreType("AICore");
    parsePlatform->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parsePlatform->SetPlatformRes("version", socVersions);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* workspaceSizes = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(tilingData, nullptr);

    gert::StorageShape xShape = tc.xShape;
    const int64_t groups = tc.hasGroupList ? tc.amaxShape.GetStorageShape().GetDim(0) : 1;
    // scale 长度必须等于 num_groups（静态分组），单张量为 [1]。
    gert::StorageShape scaleShape({groups}, {groups});
    gert::StorageShape groupShape = tc.amaxShape;
    gert::StorageShape yShape = tc.yShape;
    gert::StorageShape amaxShape = tc.amaxShape;
    gert::StorageShape yScaleShape = tc.yScaleShape;

    std::vector<uint32_t> inputInstanceNum = {1, 1, 1};
    std::vector<gert::StorageShape*> inputShapes = {&xShape, tc.hasScale ? &scaleShape : nullptr,
                                                    tc.hasGroupList ? &groupShape : nullptr};

    gert::TilingContextFaker contextFaker;
    contextFaker.SetOpType(opType)
        .NodeIoNum(3, 3)
        .IrInstanceNum(inputInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes({&yShape, &amaxShape, &yScaleShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, tc.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, tc.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, tc.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>(tc.roundMode)},
                    {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(tc.dstType)},
                    {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(tc.dstTypeMax)}})
        .TilingData(tilingData.get())
        .Workspace(workspaceSizes);
    if (tc.hasScale) {
        contextFaker.NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (tc.hasGroupList) {
        contextFaker.NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = contextFaker.Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersions);

    EXPECT_EQ(tilingFunc(tilingContext), tc.status);
}

// 动态单张量（bf16 → HiF8）。
TEST_F(QuantMaxV2TilingTest, tiling_dynamic_tensor_bf16)
{
    TilingCase tc;
    ExecuteTilingCase(tc);
}

// 动态分组 G=48（客户主场景：bf16 → HiF8）。
TEST_F(QuantMaxV2TilingTest, tiling_dynamic_group48_bf16)
{
    TilingCase tc;
    tc.hasGroupList = true;
    tc.amaxShape = {{48}, {48}};
    tc.yScaleShape = {{48}, {48}};
    ExecuteTilingCase(tc);
}

// 静态分组 G=4（fp32 → E5M2）：y_scale 静态必须缺省（nullptr 语义）。
TEST_F(QuantMaxV2TilingTest, tiling_static_group4_fp32)
{
    TilingCase tc;
    tc.xDtype = ge::DT_FLOAT;
    tc.yDtype = ge::DT_FLOAT8_E5M2;
    tc.roundMode = "rint";
    tc.dstType = 35;
    tc.dstTypeMax = 448.0f;
    tc.hasScale = true;
    tc.hasGroupList = true;
    tc.amaxShape = {{4}, {4}};
    tc.yScaleShape = {{0}, {0}}; // 静态不输出
    ExecuteTilingCase(tc);
}

// roundMode 与 dstType 不匹配（E5M2 只允许 rint）。
TEST_F(QuantMaxV2TilingTest, tiling_roundmode_mismatch_rejected)
{
    TilingCase tc;
    tc.yDtype = ge::DT_FLOAT8_E5M2;
    tc.roundMode = "round";
    tc.dstType = 35;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

// dstType 非法。
TEST_F(QuantMaxV2TilingTest, tiling_invalid_dst_type_rejected)
{
    TilingCase tc;
    tc.dstType = 33;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

// 静态分组 + 非空 y_scale（dim0 != 0）应拒绝：与 aclnn/InferShape 的「静态只接受 nullptr/[0]」契约一致。
TEST_F(QuantMaxV2TilingTest, tiling_static_nonnull_yscale_rejected)
{
    TilingCase tc;
    tc.hasScale = true;
    tc.hasGroupList = true;
    tc.amaxShape = {{4}, {4}};
    tc.yScaleShape = {{4}, {4}}; // 静态非空 y_scale（dim0=4 != 0）
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

} // namespace
