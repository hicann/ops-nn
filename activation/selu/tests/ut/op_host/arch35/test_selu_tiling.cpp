/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_selu_tiling.cpp
 * \brief Selu tiling UT (arch35, Ascend950)
 *
 * 断言 tiling 输出合法性: totalElements/blockFactor/ubFactor/blockDim/tilingKey/workspace,
 * 期望值按 selu_tiling_arch35.cpp 切分公式手工推导:
 *   coreNum=64, ubSize=245760 (platform faker: CORE_NUM=64, UB_SIZE=245760)
 *   blockFactor = alignUp(CeilDiv(total, coreNum), 32/typeSize)
 *   ubDivisor   = (2*typeSize + tmpBufNum*4) / typeSize  (tmpBufNum: int8=3, 其余=5)
 *   ubFactor    = FloorAlign(FloorDiv(ubSize/typeSize, ubDivisor), 32/typeSize)
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
#include "../../../../op_kernel/arch35/selu_tiling_data.h"
#include "../../../../op_kernel/arch35/selu_tiling_key.h"

using namespace std;
using namespace ge;

class SeluTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SeluTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SeluTilingTest TearDown" << std::endl; }
};

static const char* kSeluUtPlatformJson = R"({
    "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
    }
})";

static void DoSeluTilingTest(gert::StorageShape& shape, ge::DataType dtype, ge::graphStatus expectStatus,
                             int64_t expectTotalElements, int64_t expectBlockFactor, int64_t expectUbFactor,
                             uint32_t expectBlockDim, const char* platformJson = kSeluUtPlatformJson)
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(platformJson, soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    string op_type("Selu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_NE(tiling_func, nullptr);

    // tiling_parse: SeluCompileInfo 为空结构, parse 恒 SUCCESS
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

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&shape})
                      .OutputShapes({&shape})
                      .CompileInfo(&dummy_compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
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

    // TilingKey 仅编码占位调度维度 schMode(SELU_SCH_MODE_0), def 驱动 dtype 不入 key
    EXPECT_EQ(tiling_context->GetTilingKey(), GET_TPL_TILING_KEY(SELU_SCH_MODE_0));
    EXPECT_EQ(tiling_context->GetBlockDim(), expectBlockDim);

    // workspace: WS_SYS_SIZE = 0
    auto* workspace_sizes = tiling_context->GetWorkspaceSizes(1);
    ASSERT_NE(workspace_sizes, nullptr);
    EXPECT_EQ(workspace_sizes[0], 0UL);

    // tilingData 字段
    auto* raw_tiling = reinterpret_cast<const SeluTilingData*>(tiling_context->GetRawTilingData()->GetData());
    EXPECT_EQ(raw_tiling->totalElements, expectTotalElements);
    EXPECT_EQ(raw_tiling->blockFactor, expectBlockFactor);
    EXPECT_EQ(raw_tiling->ubFactor, expectUbFactor);
}

// fp32 {1,1024}: typeSize=4, ubDivisor=(8+20)/4=7, ubFactor=FloorAlign(245760/4/7,8)=8776
// blockFactor=CeilDiv(1024,64)=16, blockDim=CeilDiv(1024,16)=64
TEST_F(SeluTilingTest, ascend950_test_tiling_fp32_001)
{
    gert::StorageShape shape = {{1, 1024}, {1, 1024}};
    DoSeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1024, 16, 8776, 64);
}

// fp16 {1,64,2,64}: typeSize=2, ubDivisor=(4+20)/2=12, ubFactor=FloorAlign(245760/2/12,16)=10240
// blockFactor=CeilDiv(8192,64)=128, blockDim=64
TEST_F(SeluTilingTest, ascend950_test_tiling_fp16_002)
{
    gert::StorageShape shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    DoSeluTilingTest(shape, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 8192, 128, 10240, 64);
}

// bf16 {4,5,6}: typeSize=2, ubFactor=10240, blockFactor=alignUp(CeilDiv(120,64)=2,16)=16, blockDim=CeilDiv(120,16)=8
TEST_F(SeluTilingTest, ascend950_test_tiling_bf16_003)
{
    gert::StorageShape shape = {{4, 5, 6}, {4, 5, 6}};
    DoSeluTilingTest(shape, ge::DT_BF16, ge::GRAPH_SUCCESS, 120, 16, 10240, 8);
}

// int32 {1,1024}: 同 fp32 布局(typeSize=4, tmpBuf=5), ubFactor=8776, blockFactor=16, blockDim=64
TEST_F(SeluTilingTest, ascend950_test_tiling_int32_004)
{
    gert::StorageShape shape = {{1, 1024}, {1, 1024}};
    DoSeluTilingTest(shape, ge::DT_INT32, ge::GRAPH_SUCCESS, 1024, 16, 8776, 64);
}

// int8 {2,32}: typeSize=1, tmpBuf=3, ubDivisor=(2+12)/1=14, ubFactor=FloorAlign(245760/14,32)=17536
// blockFactor=alignUp(CeilDiv(64,64)=1,32)=32, blockDim=CeilDiv(64,32)=2
TEST_F(SeluTilingTest, ascend950_test_tiling_int8_005)
{
    gert::StorageShape shape = {{2, 32}, {2, 32}};
    DoSeluTilingTest(shape, ge::DT_INT8, ge::GRAPH_SUCCESS, 64, 32, 17536, 2);
}

// fp32 大 shape 多核均衡: {1024,1024}, blockFactor=CeilDiv(1048576,64)=16384, blockDim=64
TEST_F(SeluTilingTest, ascend950_test_tiling_fp32_large_006)
{
    gert::StorageShape shape = {{1024, 1024}, {1024, 1024}};
    DoSeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1048576, 16384, 8776, 64);
}

// 标量 rank-0: EnsureNotScalar 规整为 {1}, blockFactor=alignUp(1,8)=8, blockDim=1
TEST_F(SeluTilingTest, ascend950_test_tiling_scalar_007)
{
    gert::StorageShape shape = {{}, {}};
    DoSeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 1, 8, 8776, 1);
}

// 空 tensor: SetEmptyTensorTiling 提前返回, blockDim=1, 三因子全 0
TEST_F(SeluTilingTest, ascend950_test_tiling_empty_008)
{
    gert::StorageShape shape = {{0}, {0}};
    DoSeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 0, 0, 0, 1);
}

// 不支持 dtype(int64): GetShapeInfo 校验失败
TEST_F(SeluTilingTest, ascend950_test_tiling_unsupported_dtype_009)
{
    gert::StorageShape shape = {{1, 1024}, {1, 1024}};
    DoSeluTilingTest(shape, ge::DT_INT64, ge::GRAPH_FAILED, 0, 0, 0, 0);
}

// UB 过小: UB_SIZE=8 → fp32 ubFactor=FloorAlign(FloorDiv(2,7),8)=0 → ComputeTiling 失败
TEST_F(SeluTilingTest, ascend950_test_tiling_ub_too_small_010)
{
    static const char* tinyUbJson = R"({
        "hardware_info": {
            "BT_SIZE": 0, "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 8, "L2_SIZE": 33554432, "L1_SIZE": 524288,
            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
        }
    })";
    gert::StorageShape shape = {{1, 1024}, {1, 1024}};
    DoSeluTilingTest(shape, ge::DT_FLOAT, ge::GRAPH_FAILED, 0, 0, 0, 0, tinyUbJson);
}

// PlatformInfo 无资源(空 PlatFormInfos): GetAivCoreNum 读不到 ai_core_cnt 返回 0
// → SeluTilingFunc coreNum==0 防御分支失败
// (faker builder 强制要求 PlatformInfo slot, 传 null 会 Build 失败, 故以空资源 platform 模拟)
TEST_F(SeluTilingTest, ascend950_test_tiling_empty_platform_011)
{
    string op_type("Selu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    fe::PlatFormInfos platform_info; // 不调用 Init/SetPlatformRes, 无任何平台资源

    gert::StorageShape shape = {{1, 1024}, {1, 1024}};
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    int64_t dummy_compile_info = 0;
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&shape})
                      .OutputShapes({&shape})
                      .CompileInfo(&dummy_compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
