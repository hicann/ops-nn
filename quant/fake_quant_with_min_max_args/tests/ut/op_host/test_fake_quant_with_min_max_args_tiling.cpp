/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>

#include <gtest/gtest.h>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class FakeQuantWithMinMaxArgsTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FakeQuantWithMinMaxArgsTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FakeQuantWithMinMaxArgsTiling TearDown" << std::endl;
    }
};

struct FakeQuantWithMinMaxArgsCompileInfo {
    int32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
};

// Mirror of FakeQuantWithMinMaxArgsTilingData (op_kernel/arch35) for raw-buffer
// deserialization. Layout must stay byte-for-byte identical to the kernel struct.
struct FakeQuantWithMinMaxArgsTilingDataMirror {
    int64_t totalLen;
    int64_t numCore;
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t baseLen;
    float   nudgedMin;
    float   nudgedMax;
    float   scale;
    float   scaleInv;
    float   quantZero;
};

static void InitPlatForm(
    fe::PlatFormInfos& platform_info, map<string, string>& soc_infos, map<string, string>& aicore_spec,
    map<string, string>& intrinsics)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    platform_info.Init();
}

 static void DoTilingCase(
    float minVal, float maxVal, int64_t numBits, bool narrowRange,
    const gert::StorageShape& xShape, ge::graphStatus expected,
    bool checkTilingData = false, ge::DataType xDtype = ge::DT_FLOAT)
{
    fe::PlatFormInfos platform_info;
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    InitPlatForm(platform_info, soc_infos, aicore_spec, intrinsics);
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

    gert::StorageShape outShape = xShape;

    FakeQuantWithMinMaxArgsCompileInfo compile_info;
    std::string op_type("FakeQuantWithMinMaxArgs");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 1024 * 1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({const_cast<gert::StorageShape*>(&xShape)})
                      .OutputShapes({const_cast<gert::StorageShape*>(&outShape)})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(minVal)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(maxVal)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(numBits)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(narrowRange)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tiling_func(tiling_context), expected);

    // Deserialize raw TilingData and assert invariants on success (non-empty shape).
    if (checkTilingData && expected == ge::GRAPH_SUCCESS) {
        int64_t totalLen = 1;
        for (size_t i = 0; i < xShape.GetStorageShape().GetDimNum(); ++i) {
            totalLen *= xShape.GetStorageShape().GetDim(i);
        }
        auto* rawTd = reinterpret_cast<gert::TilingData*>(param.get());
        ASSERT_GE(rawTd->GetDataSize(), sizeof(FakeQuantWithMinMaxArgsTilingDataMirror));
        FakeQuantWithMinMaxArgsTilingDataMirror td;
        std::memcpy(&td, rawTd->GetData(), sizeof(td));

        EXPECT_EQ(td.totalLen, totalLen);
        EXPECT_GE(td.numCore, 1);
        EXPECT_GT(td.blockFactor, 0);
        EXPECT_GT(td.blockTailFactor, 0);
        EXPECT_LE(td.blockTailFactor, td.blockFactor);
        EXPECT_GT(td.baseLen, 0);
        // 多核切分守恒：blockFactor*(numCore-1) + blockTailFactor == totalLen
        EXPECT_EQ(td.blockFactor * (td.numCore - 1) + td.blockTailFactor, td.totalLen);
        // nudge 边界：nudgedMin < nudgedMax
        EXPECT_LT(td.nudgedMin, td.nudgedMax);
        // forward 额外断言：scale>0、scaleInv>0、quantZero 合理（落在 [0, qMax] 内）
        EXPECT_GT(td.scale, 0.0f);
        EXPECT_GT(td.scaleInv, 0.0f);
        float qMax = static_cast<float>((1ULL << static_cast<uint32_t>(numBits)) - 1ULL);
        EXPECT_GE(td.quantZero, 0.0f);
        EXPECT_LE(td.quantZero, qMax);
    }
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_default_8bits)
{
    gert::StorageShape xShape = {{4, 2, 10240}, {4, 2, 10240}};
    DoTilingCase(-6.0f, 6.0f, 8, false, xShape, ge::GRAPH_SUCCESS, true);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_narrow_range)
{
    gert::StorageShape xShape = {{4, 2, 10240}, {4, 2, 10240}};
    DoTilingCase(-6.0f, 6.0f, 8, true, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_4bits)
{
    gert::StorageShape xShape = {{4, 2, 10240}, {4, 2, 10240}};
    DoTilingCase(-1.0f, 1.0f, 4, false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_2bits)
{
    gert::StorageShape xShape = {{2048}, {2048}};
    DoTilingCase(-2.0f, 2.0f, 2, false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_small_shape)
{
    gert::StorageShape xShape = {{128}, {128}};
    DoTilingCase(-6.0f, 6.0f, 8, false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_16bits)
{
    gert::StorageShape xShape = {{4, 1024}, {4, 1024}};
    DoTilingCase(-1.0f, 1.0f, 16, false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_min_ge_max_fail)
{
    gert::StorageShape xShape = {{4, 1024}, {4, 1024}};
    DoTilingCase(6.0f, 6.0f, 8, false, xShape, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_num_bits_out_of_range_fail)
{
    gert::StorageShape xShape = {{4, 1024}, {4, 1024}};
    DoTilingCase(-6.0f, 6.0f, 1, false, xShape, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_num_bits_17_fail)
{
    gert::StorageShape xShape = {{4, 1024}, {4, 1024}};
    DoTilingCase(-6.0f, 6.0f, 17, false, xShape, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_empty_shape)
{
    gert::StorageShape xShape = {{0}, {0}};
    DoTilingCase(-6.0f, 6.0f, 8, false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_narrow_range_2bits)
{
    gert::StorageShape xShape = {{1024}, {1024}};
    DoTilingCase(-1.0f, 1.0f, 2, true, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_narrow_range_4bits)
{
    gert::StorageShape xShape = {{4, 1024}, {4, 1024}};
    DoTilingCase(-1.0f, 1.0f, 4, true, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_narrow_range_16bits)
{
    gert::StorageShape xShape = {{1024}, {1024}};
    DoTilingCase(-1.0f, 1.0f, 16, true, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_min_gt_max_fail)
{
    gert::StorageShape xShape = {{1024}, {1024}};
    DoTilingCase(6.0f, -6.0f, 8, false, xShape, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_3d)
{
    gert::StorageShape xShape = {{2, 3, 1024}, {2, 3, 1024}};
    DoTilingCase(-6.0f, 6.0f, 8, false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_5d)
{
    gert::StorageShape xShape = {{1, 2, 3, 4, 8}, {1, 2, 3, 4, 8}};
    DoTilingCase(-6.0f, 6.0f, 8, false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_single_element)
{
    gert::StorageShape xShape = {{1}, {1}};
    DoTilingCase(-6.0f, 6.0f, 8, false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_large_shape)
{
    gert::StorageShape xShape = {{48, 2048}, {48, 2048}};
    DoTilingCase(-6.0f, 6.0f, 8, false, xShape, ge::GRAPH_SUCCESS, true);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_fp32_asymmetric_positive_range)
{
    gert::StorageShape xShape = {{1024}, {1024}};
    DoTilingCase(0.0f, 6.0f, 8, false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_invalid_dtype_fail)
{
    gert::StorageShape xShape = {{4, 1024}, {4, 1024}};
    DoTilingCase(-6.0f, 6.0f, 8, false, xShape, ge::GRAPH_FAILED, false, ge::DT_FLOAT16);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_min_nan_fail)
{
    gert::StorageShape xShape = {{1024}, {1024}};
    DoTilingCase(std::nanf(""), 6.0f, 8, false, xShape, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_max_nan_fail)
{
    gert::StorageShape xShape = {{1024}, {1024}};
    DoTilingCase(-6.0f, std::nanf(""), 8, false, xShape, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_min_inf_fail)
{
    gert::StorageShape xShape = {{1024}, {1024}};
    DoTilingCase(-std::numeric_limits<float>::infinity(), 6.0f, 8, false, xShape, ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxArgsTiling, tiling_max_inf_fail)
{
    gert::StorageShape xShape = {{1024}, {1024}};
    DoTilingCase(-6.0f, std::numeric_limits<float>::infinity(), 8, false, xShape, ge::GRAPH_FAILED);
}