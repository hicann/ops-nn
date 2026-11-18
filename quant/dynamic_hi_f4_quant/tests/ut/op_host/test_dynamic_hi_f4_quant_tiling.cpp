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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "quant/dynamic_hi_f4_quant/op_host/arch35/dynamic_hi_f4_quant_tiling_arch35.h"
#include "any_value.h"

using namespace std;

class DynamicHiF4QuantTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DynamicHiF4QuantTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DynamicHiF4QuantTiling TearDown" << std::endl; }
};

static string TilingData2Str(const gert::TilingData* tilingData)
{
    auto data = tilingData->GetData();
    string result;
    for (size_t i = 0; i < tilingData->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tilingData->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

static void ExecuteTestCase(ge::DataType inDtype, ge::DataType outDtype, ge::DataType out2Dtype,
                            gert::StorageShape shape, gert::StorageShape outShape, gert::StorageShape scaleShape,
                            int64_t axis, string expectTilingData, int64_t scaleAlg = 0,
                            ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64}
                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend960DT"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::DynamicHiF4QuantCompileInfo compile_info;

    std::string op_type("DynamicHiF4Quant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&shape})
                      .OutputShapes({&outShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, out2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom(axis)},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom(scaleAlg)}})
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

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_ascendc_bfloat16_not_tail)
{
    // 横竖自适应(mock UB 253952: B_eff=min(253952/52480,4)=4): dimNeg1=256 → W=256, G=4/2=2,
    // splitBlockH=128; dimNeg2=128 精确 1 行带 → totalTask=1, 单核
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{2, 256, 1}, {2, 256, 1}};
    string expectTilingData = "64 1 1 1 128 256 256 128 128 256 1 1 1 0 1 0 0 0 1 4 2 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_ascendc_bfloat16_not_tail_scale_alg_1)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{2, 256, 1}, {2, 256, 1}};
    string expectTilingData = "64 1 1 1 128 256 256 128 128 256 1 1 1 0 1 0 0 0 1 4 2 0 0 1 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData, 1);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_scale_alg_invalid)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{2, 256, 1}, {2, 256, 1}};
    string expectTilingData = "";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData, 2, ge::GRAPH_FAILED);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_narrow_g4)
{
    // 窄形竖切档: dimNeg1=128 → W=128, G=4, H=256; dimNeg2=128 < H → 单行带行尾 128
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scaleShape = {{2, 128, 1}, {2, 128, 1}};
    string expectTilingData = "64 1 1 1 128 128 128 256 128 128 1 1 1 0 1 0 0 0 1 2 2 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_narrow_multitask)
{
    // 窄形多任务: 7168=256×28 精确整除 → 28 行带无尾, 28 任务 28 核
    gert::StorageShape shape = {{7168, 128}, {7168, 128}};
    gert::StorageShape scaleShape = {{112, 128, 1}, {112, 128, 1}};
    string expectTilingData = "64 28 1 1 7168 128 128 256 256 128 28 1 1 0 28 0 0 0 28 2 112 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_ceil_192)
{
    // ceil 上取整档: dimNeg1=192 → W=256(stale 64 列), G=2, H=128 精确整除无行尾
    gert::StorageShape shape = {{128, 192}, {128, 192}};
    gert::StorageShape scaleShape = {{2, 192, 1}, {2, 192, 1}};
    string expectTilingData = "64 1 1 1 128 192 256 128 128 192 1 1 1 0 1 0 1 0 1 3 2 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_ceil_320)
{
    // ceil 上取整档: dimNeg1=320 → W=384(stale 64 列), G=floor(4/3)=1(3/4 预算), H=64
    gert::StorageShape shape = {{64, 320}, {64, 320}};
    gert::StorageShape scaleShape = {{1, 320, 1}, {1, 320, 1}};
    string expectTilingData = "64 1 1 1 64 320 384 64 64 320 1 1 1 0 1 1 1 0 1 5 1 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_ceil_448)
{
    // ceil 上取整档: dimNeg1=448 → W=512(stale 64 列), G=1, H=64, 单 tile 满突发
    gert::StorageShape shape = {{64, 448}, {64, 448}};
    gert::StorageShape scaleShape = {{1, 448, 1}, {1, 448, 1}};
    string expectTilingData = "64 1 1 1 64 448 512 64 64 448 1 1 1 0 1 1 1 0 1 7 1 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_wide_multi_colblock)
{
    // 宽形多列块: dimNeg1=1024 → W=512, 2 列块精确整除, 2 任务 2 核
    gert::StorageShape shape = {{64, 1024}, {64, 1024}};
    gert::StorageShape scaleShape = {{1, 1024, 1}, {1, 1024, 1}};
    string expectTilingData = "64 2 1 1 64 1024 512 64 64 512 1 2 1 0 2 1 0 0 2 16 1 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_wide_coltail)
{
    // 宽形列尾: 8256=16×512+64 → 17 列块, 尾块 dataLen=64; 17 任务 17 核
    gert::StorageShape shape = {{64, 8256}, {64, 8256}};
    gert::StorageShape scaleShape = {{1, 8256, 1}, {1, 8256, 1}};
    string expectTilingData = "64 17 1 1 64 8256 512 64 64 64 1 17 1 0 17 1 1 0 17 129 1 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_rowtail_multichunk)
{
    // 行尾×多 chunk: dimNeg1=256 → W=256(chunkLoop=2), G=2, H=128; 320=2×128+64 → 3 行带尾 64
    gert::StorageShape shape = {{320, 256}, {320, 256}};
    gert::StorageShape scaleShape = {{5, 256, 1}, {5, 256, 1}};
    string expectTilingData = "64 3 1 1 320 256 256 128 64 256 3 1 1 0 3 1 0 0 3 4 5 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_rowtail_g4)
{
    // 窄形行尾: W=128, G=4, H=256; dimNeg2=192 < H → 单行带行尾 192
    gert::StorageShape shape = {{192, 128}, {192, 128}};
    gert::StorageShape scaleShape = {{3, 128, 1}, {3, 128, 1}};
    string expectTilingData = "64 1 1 1 192 128 128 256 192 128 1 1 1 0 1 1 0 0 1 2 3 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_rowtail_g4_multitask)
{
    // 窄形行尾多任务: 7232=28×256+64 → 29 行带尾 64, 29 任务 29 核
    gert::StorageShape shape = {{7232, 128}, {7232, 128}};
    gert::StorageShape scaleShape = {{113, 128, 1}, {113, 128, 1}};
    string expectTilingData = "64 29 1 1 7232 128 128 256 64 128 29 1 1 0 29 1 0 0 29 2 113 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_min_shape)
{
    // 最小 shape: W=128, G=4, H=256; dimNeg2=64 单行带行尾 64(循环数按 H' 切)
    gert::StorageShape shape = {{64, 64}, {64, 64}};
    gert::StorageShape scaleShape = {{1, 64, 1}, {1, 64, 1}};
    string expectTilingData = "64 1 1 1 64 64 128 256 64 64 1 1 1 0 1 1 1 0 1 1 1 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_not_tail_multibatch)
{
    // 多 batch: dim0=2; W=128, G=4, H=256; dimNeg2=192 单行带行尾 192 → totalTask=2
    gert::StorageShape shape = {{2, 192, 128}, {2, 192, 128}};
    gert::StorageShape scaleShape = {{2, 3, 128, 1}, {2, 3, 128, 1}};
    string expectTilingData = "64 2 1 2 192 128 128 256 192 128 1 1 1 0 2 1 0 0 1 2 3 0 0 0 ";
    int64_t axis = -2;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_ascendc_bfloat16_tail)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{128, 4, 1}, {128, 4, 1}};
    // CCE 移植后 -1 轴: tile=256(mock UB 253952 下 (253952-4096)/700=356, 256 对齐取 256),
    // groups=512 -> tiles=2, usedCore=min(64,2)=2, 头尾核各 1 tile; 仅传 7 个字段, 其余恒 0
    string expectTilingData = "64 2 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 512 256 0 ";
    int64_t axis = -1;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_ascendc_bfloat16_tail_last_tile_partial)
{
    // 尾 tile 非整块: groups=258 -> tiles=2, 全局尾 tile 有效 2 group(由 kernel 派生, tiling 不体现)
    gert::StorageShape shape = {{2, 8256}, {2, 8256}};
    gert::StorageShape scaleShape = {{2, 129, 1}, {2, 129, 1}};
    string expectTilingData = "64 2 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 258 256 0 ";
    int64_t axis = -1;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_ascendc_bfloat16_tail_multi_core_uneven)
{
    // tile 数 > 核数的不均分: groups=16640 -> tiles=65, usedCore=64,
    // 头部 1 核 2 tile(blockPerHeadCore=2), 其余 63 核 1 tile(blockPerTailCore=1)
    gert::StorageShape shape = {{130, 8192}, {130, 8192}};
    gert::StorageShape scaleShape = {{130, 128, 1}, {130, 128, 1}};
    string expectTilingData = "64 64 0 0 0 0 0 0 0 0 0 0 2 1 1 0 0 0 0 0 0 16640 256 0 ";
    int64_t axis = -1;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}

TEST_F(DynamicHiF4QuantTiling, DynamicHiF4Quant_tiling_ascendc_bfloat16_tail_single_group)
{
    // 最小 shape: groups=1 -> tiles=1, usedCore=1, 全局尾 tile 有效 1 group
    gert::StorageShape shape = {{1, 64}, {1, 64}};
    gert::StorageShape scaleShape = {{1, 1, 1}, {1, 1, 1}};
    string expectTilingData = "64 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 256 0 ";
    int64_t axis = -1;

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, shape, scaleShape, axis,
                    expectTilingData);
}
