/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_celu_tiling.cpp
 * \brief Celu tiling UT (arch35, Ascend950)
 *
 * 断言 tiling 输出与拒绝行为, 期望值按 celu_tiling_arch35.cpp 切分公式手工推导:
 *   coreNum=64, ubSize=245760 (platform faker: CORE_NUM=64, UB_SIZE=245760)
 *   bytesPerElem = 2*typeSize + 6*4 (IO 2 份 + 计算缓冲 5 份 fp32 + 保守余量 1 份)
 *   ubFactor    = FloorAlign(FloorDiv(ubSize, bytesPerElem), 32)   (fp32=7680, fp16=8768)
 *   blockFactor = CeilDiv(totalNum, 64); usedCoreNum = CeilDiv(totalNum, blockFactor)
 * 负向: GetShapeSize 维度乘积溢出 int64 时返回溢出哨兵 (kInvalidDimValue=INT64_MIN),
 *       totalNum<0 统一拒绝 (哨兵值直构与正维乘积溢出两形态); alpha2<=0/NaN 拒绝;
 *       rank-9 接受（elewise 按 flat totalNum 切分, rank 无关）。
 * GetShapeSize 语义 (exe_graph/runtime/shape.h): 先全维零扫描 (任一维==0 -> 0,
 * 先于乘法溢出判定), 再 MulOverflow 逐维累乘 (溢出 -> INT64_MIN)。
 */

#include <iostream>
#include <map>
#include <string>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "../../../../op_kernel/arch35/celu_tiling_data.h"

using namespace std;
using namespace ge;

class CeluTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CeluTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CeluTilingTest TearDown" << std::endl; }
};

static const char* kCeluUtPlatformJson = R"({
    "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
    }
})";

// 微型 UB 平台: UB_SIZE=64 < bytesPerElem*ubBlockSize(fp32: 32*32), ubFactor 对齐后归零,
// 触发 ReportTilingFailure(EZ1008) 合法输入下 Tiling 失败首报路径。
static const char* kCeluUtTinyUbPlatformJson = R"({
    "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 64, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
    }
})";

static void DoCeluTilingTest(gert::StorageShape& shape, ge::DataType dtype, ge::graphStatus expectStatus,
                             int64_t expectTotalNum, int64_t expectBlockFactor, int64_t expectUbFactor,
                             uint32_t expectBlockDim, float alpha1 = 1.0f, float alpha2 = 1.0f, float alpha3 = 1.0f,
                             bool expectAlphaWritten = true, bool withAttrs = true,
                             const char* platformJson = kCeluUtPlatformJson, gert::StorageShape* outShape = nullptr)
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(platformJson, soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    string op_type("Celu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_NE(tiling_func, nullptr);

    // tiling_parse: CeluCompileInfo 为空结构, parse 恒 SUCCESS
    int64_t dummy_compile_info = 0;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(1, 1)
                             .Inputs({const_cast<char*>(platformJson), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&dummy_compile_info})
                             .Build();
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    gert::TilingContextFaker faker;
    faker.SetOpType(op_type)
        .NodeIoNum(1, 1)
        .IrInstanceNum({1})
        .InputShapes({&shape})
        .OutputShapes({outShape != nullptr ? outShape : &shape})
        .CompileInfo(&dummy_compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    if (withAttrs) {
        faker.NodeAttrs({{"alpha1", Ops::NN::AnyValue::CreateFrom<float>(alpha1)},
                         {"alpha2", Ops::NN::AnyValue::CreateFrom<float>(alpha2)},
                         {"alpha3", Ops::NN::AnyValue::CreateFrom<float>(alpha3)}});
    }
    auto holder = faker.TilingData(param.get()).Workspace(ws_size).Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    EXPECT_EQ(tiling_func(tiling_context), expectStatus);
    if (expectStatus != ge::GRAPH_SUCCESS) {
        return;
    }

    EXPECT_EQ(tiling_context->GetBlockDim(), expectBlockDim);

    // workspace: WS_SYS_SIZE = 0
    auto* workspace_sizes = tiling_context->GetWorkspaceSizes(1);
    ASSERT_NE(workspace_sizes, nullptr);
    EXPECT_EQ(workspace_sizes[0], 0UL);

    // tilingData 字段
    auto* raw_tiling = reinterpret_cast<const CeluTilingData*>(tiling_context->GetRawTilingData()->GetData());
    EXPECT_EQ(raw_tiling->totalNum, expectTotalNum);
    EXPECT_EQ(raw_tiling->blockFactor, expectBlockFactor);
    EXPECT_EQ(raw_tiling->ubFactor, expectUbFactor);
    // 空 tensor 分支(totalNum==0)提前返回不写 alpha(kernel 侧不消费), 非空分支写入 attrs 值
    float expectAlpha1 = expectAlphaWritten ? alpha1 : 0.0f;
    float expectAlpha2 = expectAlphaWritten ? alpha2 : 0.0f;
    float expectAlpha3 = expectAlphaWritten ? alpha3 : 0.0f;
    EXPECT_EQ(raw_tiling->alpha1, expectAlpha1);
    EXPECT_EQ(raw_tiling->alpha2, expectAlpha2);
    EXPECT_EQ(raw_tiling->alpha3, expectAlpha3);
}

// fp32 {1,1024}: bytesPerElem=32, ubFactor=FloorAlign(245760/32,32)=7680
// blockFactor=CeilDiv(1024,64)=16, blockDim=CeilDiv(1024,16)=64
TEST_F(CeluTilingTest, ascend950_test_tiling_fp32_001)
{
    gert::StorageShape shape = {{1, 1024}, {1, 1024}};
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1024, 16, 7680, 64);
}

// fp16 {4,3,4}: bytesPerElem=28, ubFactor=FloorAlign(8777,32)=8768
// blockFactor=CeilDiv(48,64)=1, blockDim=CeilDiv(48,1)=48
TEST_F(CeluTilingTest, ascend950_test_tiling_fp16_002)
{
    gert::StorageShape shape = {{4, 3, 4}, {4, 3, 4}};
    DoCeluTilingTest(shape, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 48, 1, 8768, 48);
}

// 非默认 alpha 透传: totalNum/blockFactor/ubFactor 与维度无关, 仅 alpha 字段随 attrs
TEST_F(CeluTilingTest, ascend950_test_tiling_fp32_attrs_003)
{
    gert::StorageShape shape = {{2, 128}, {2, 128}};
    // blockFactor=CeilDiv(256,64)=4, blockDim=CeilDiv(256,4)=64
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 256, 4, 7680, 64, 1.5f, 2.5f, 0.5f);
}

// 空 tensor {0}: totalNum=0 走空分支, blockDim=1(该分支提前返回不写 alpha)
TEST_F(CeluTilingTest, ascend950_test_tiling_empty_004)
{
    gert::StorageShape shape = {{0}, {0}};
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 0, 0, 0, 1, 1.0f, 1.0f, 1.0f, false);
}

// GetShapeSize 溢出哨兵 INT64_MIN (kInvalidDimValue): totalNum<0 统一拒绝
TEST_F(CeluTilingTest, ascend950_test_tiling_shapesize_overflow_sentinel_007)
{
    gert::StorageShape shape = {{INT64_MIN}, {INT64_MIN}};
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_FAILED, 0, 0, 0, 0);
}

// alpha2 = 0: 既有拦截行为
TEST_F(CeluTilingTest, ascend950_test_tiling_alpha2_zero_008)
{
    gert::StorageShape shape = {{64}, {64}};
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_FAILED, 0, 0, 0, 0, 1.0f, 0.0f, 1.0f);
}

// alpha2 < 0: 既有拦截行为
TEST_F(CeluTilingTest, ascend950_test_tiling_alpha2_negative_009)
{
    gert::StorageShape shape = {{64}, {64}};
    DoCeluTilingTest(shape, ge::DT_FLOAT16, ge::GRAPH_FAILED, 0, 0, 0, 0, 1.0f, -1.5f, 1.0f);
}

// rank-9 (9 维 shape, totalNum=362880): rank≤8 限制已放开, 9 维输入正常切分
// blockFactor=CeilDiv(362880,64)=5670, blockDim=CeilDiv(362880,5670)=64, ubFactor=7680(fp32)
TEST_F(CeluTilingTest, ascend950_test_tiling_rank9_accepted_010)
{
    gert::StorageShape shape = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}};
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 362880, 5670, 7680, 64);
}

// 微型 UB(64B): fp32 bytesPerElem=32, ubFactor=FloorAlign(64/32,32)=0,
// 合法输入下 Tiling 失败走 ReportTilingFailure(EZ1008) 正式首报
TEST_F(CeluTilingTest, ascend950_test_tiling_ub_factor_zero_011)
{
    gert::StorageShape shape = {{64}, {64}};
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_FAILED, 0, 0, 0, 0, 1.0f, 1.0f, 1.0f, true, true,
                     kCeluUtTinyUbPlatformJson);
}

// rank-0 标量形态: EnsureNotScalar 归一为 {1}, totalNum=1 正常切分
// blockFactor=CeilDiv(1,64)=1, blockDim=CeilDiv(1,1)=1, ubFactor=7680
TEST_F(CeluTilingTest, ascend950_test_tiling_scalar_shape_012)
{
    gert::StorageShape shape = {{}, {}};
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1, 1, 7680, 1);
}

// dtype 越界(int32): 注册面外 dtype 在 tiling 层被拒(EZ0019)
TEST_F(CeluTilingTest, ascend950_test_tiling_dtype_rejected_013)
{
    gert::StorageShape shape = {{64}, {64}};
    DoCeluTilingTest(shape, ge::DT_INT32, ge::GRAPH_FAILED, 0, 0, 0, 0);
}

// 属性缺失(未物化): GetAttrPointer 返回 null 拒绝
TEST_F(CeluTilingTest, ascend950_test_tiling_attrs_missing_014)
{
    gert::StorageShape shape = {{64}, {64}};
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_FAILED, 0, 0, 0, 0, 1.0f, 1.0f, 1.0f, true, false);
}

// 输入输出 shape size 失配(6 != 4): tiling 层拒绝(EZ0010)
TEST_F(CeluTilingTest, ascend950_test_tiling_shape_mismatch_015)
{
    gert::StorageShape inShape = {{2, 3}, {2, 3}};
    gert::StorageShape outShape = {{4}, {4}};
    DoCeluTilingTest(inShape, ge::DT_FLOAT, ge::GRAPH_FAILED, 0, 0, 0, 0, 1.0f, 1.0f, 1.0f, true, true,
                     kCeluUtPlatformJson, &outShape);
}

// corner: 单维 INT64_MAX, 乘积不溢出 -> totalNum=INT64_MAX 合法放行
// blockFactor=CeilDiv(INT64_MAX,64)=144115188075855872, blockDim=CeilDiv(INT64_MAX,blockFactor)=64
TEST_F(CeluTilingTest, ascend950_test_tiling_max_dim_no_overflow_016)
{
    gert::StorageShape shape = {{INT64_MAX}, {INT64_MAX}};
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, INT64_MAX, 144115188075855872LL, 7680, 64);
}

// corner: 正维乘积溢出 (INT64_MAX*2) -> GetShapeSize 返回溢出哨兵 INT64_MIN -> 拒绝
TEST_F(CeluTilingTest, ascend950_test_tiling_dims_product_overflow_017)
{
    gert::StorageShape shape = {{INT64_MAX, 2}, {INT64_MAX, 2}};
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_FAILED, 0, 0, 0, 0);
}

// corner: 前缀维乘积溢出 x 后继 0 维 -> GetShapeSize 全维零扫描先行 -> totalNum=0
// 空 tensor fast-path 放行 (blockDim=1, 提前返回不写 alpha)
TEST_F(CeluTilingTest, ascend950_test_tiling_prefix_overflow_trailing_zero_018)
{
    gert::StorageShape shape = {{INT64_MAX, 2, 0}, {INT64_MAX, 2, 0}};
    DoCeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 0, 0, 0, 1, 1.0f, 1.0f, 1.0f, false);
}
