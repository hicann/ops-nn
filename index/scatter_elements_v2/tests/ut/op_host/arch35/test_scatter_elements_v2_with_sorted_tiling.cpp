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
 * \file test_scatter_elements_v2_with_sorted_tiling.cpp
 * \brief
 *        WithSorted（确定性 add 方案A）tiling 的单测：验证 2xxxxxx tilingKey 前缀提升、
 *        方案A（索引总数切核）触发条件、shapeMode / keySize / countMode / workspace 布局。
 *        通过 gert::OpImplRegistry 驱动完整 TilingRegistry -> ScatterElementsV2AscTiling 链路，
 *        平台信息对齐 Ascend950（Short_SoC_version=Ascend950, NpuArch=3510）。
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/scatter_elements_v2_base_tiling.h"
#include "../../../../op_host/arch35/scatter_elements_v2_asc_tiling.h"

using namespace std;
using namespace ge;

class ScatterElementsV2WithSortedTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScatterElementsV2WithSortedTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ScatterElementsV2WithSortedTiling TearDown" << std::endl; }
};

struct WithSortedTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    uint64_t indicesTotalNum = 0;
    uint64_t keySize = 0;
    uint64_t permSize = 0;
    int32_t shapeMode = -1;
    int32_t dimNormalized = -1;
    uint32_t sortUsedCoreNum = 0;
    uint32_t numTileData = 0;
    uint32_t tileCount = 0;
    uint32_t activeCores = 0;
    uint32_t tmpUbSize = 0;
    uint32_t isSingleCore = 0;
    uint64_t wsLinearIdxOff = 0;
    uint64_t wsSortedOff = 0;
    uint64_t wsPermOff = 0;
    uint64_t wsSrcPosOff = 0;
};

static void RunWithSortedTilingCase(ge::DataType inputDtype, ge::DataType indicesDtype, ge::DataType updatesDtype,
                                    gert::StorageShape& inputShape, gert::StorageShape& indicesShape,
                                    gert::StorageShape& updatesShape, int64_t axis, const std::string& reduction,
                                    int32_t deterministic, WithSortedTilingResult& result)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::ScatterElementsV2CompileInfoArch35 compile_info;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto ws_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(ws_holder.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape output_shape = inputShape;
    // DeterministicInfo 用值重载直接传入 0/1；传指针版本会取栈地址截断为 int32，导致
    // GetDeterministic() 得到任意值，确定性分支被绕过，key 回落 1xxxxxx。

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&inputShape, &indicesShape, &updatesShape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, updatesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .DeterministicInfo(deterministic)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<string>(reduction)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    result.status = tiling_func(tiling_context);
    if (result.status != ge::GRAPH_SUCCESS) {
        return;
    }
    result.tilingKey = tiling_context->GetTilingKey();
    result.blockDim = tiling_context->GetBlockDim();
    result.workspaceSize = *tiling_context->GetWorkspaceSizes(0);

    auto raw = tiling_context->GetRawTilingData();
    if (raw != nullptr && raw->GetData() != nullptr) {
        const int64_t* p64 = reinterpret_cast<const int64_t*>(raw->GetData());
        const uint32_t* p32 = reinterpret_cast<const uint32_t*>(raw->GetData());
        // TilingData 布局见 ScatterElementsV2AscTilingData（build/.../op_kernel/scatter_elements_v2_tiling_data.h），
        // 全字段天然对齐，按 8 字节槽（slot）排列：
        // [0-6] dataStride, [7-13] indicesStride, [14-20] updatesStride (各 7 x uint64)
        // [21] loopLength, [22] allAxis, [23] dataAxis, [24] updatesAxis
        // [25] preAxis, [26] midAxis, [27] afterAxis, [28] indicesUsedCoreNum
        // [29] indicesNormBlockData, [30] indicesTailBlockData, [31] baseS, [32] baseA
        // [33] isDeterministic
        // [34] rank(int16)|dim(int16)|sortSharedBufSize(uint32)
        // --- 以下为嵌套 struct sortTiling（ScatterElementsV2SortTilingData，起始 byte 280 已 8 字节对齐）---
        // [35] indicesTotalNum, [36] keySize, [37] permSize
        // [38] countMode(low32)|shapeMode(high32)
        // [39] dimNormalized(low32)|sortUsedCoreNum(high32)
        // [40] numTileData|tileCount, [41] activeCores|tmpUbSize, [42] isSingleCore|padding(4B)
        // [43] wsLinearIdxOff, [44] wsSortedOff, [45] wsPermOff, [46] wsSrcPosOff
        // 注：int16/uint32 字段与共享 8 字节字的相邻字段按小端字节序打包，读取时以 8 字节槽为定位单位。
        result.indicesTotalNum = static_cast<uint64_t>(p64[35]);
        result.keySize = static_cast<uint64_t>(p64[36]);
        result.permSize = static_cast<uint64_t>(p64[37]);
        result.wsLinearIdxOff = static_cast<uint64_t>(p64[43]);
        result.wsSortedOff = static_cast<uint64_t>(p64[44]);
        result.wsPermOff = static_cast<uint64_t>(p64[45]);
        result.wsSrcPosOff = static_cast<uint64_t>(p64[46]);

        // int32 字段：位于 8 字节槽的高/低 32 位。
        // slot38(bytes 304-311)=p32[76](low=countMode)|p32[77](high=shapeMode)
        // slot39(bytes 312-319)=p32[78](low=dimNormalized)|p32[79](high=sortUsedCoreNum)
        result.shapeMode = p32[77];
        result.dimNormalized = static_cast<int32_t>(p32[78]);
        result.sortUsedCoreNum = p32[79];
        // uint32 字段（numTileData/tileCount/activeCores/tmpUbSize/isSingleCore）字节偏移 = 40*8 起。
        const uint32_t* p32at40 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(raw->GetData()) +
                                                                    40 * 8);
        result.numTileData = p32at40[0];
        result.tileCount = p32at40[1];
        result.activeCores = p32at40[2];
        result.tmpUbSize = p32at40[3];
        result.isSingleCore = p32at40[4];
    }
}

// ============================================================
// 1D 大 N SUBSET（data 轴维 > indices 轴维）：方案A稳定触发 -> 2xxxxxx 前缀
// ============================================================
TEST_F(ScatterElementsV2WithSortedTiling, test_with_sorted_fp32_dim1_subset)
{
    gert::StorageShape data_shape = {{130000}, {130000}};
    gert::StorageShape indices_shape = {{100000}, {100000}};
    gert::StorageShape updates_shape = {{100000}, {100000}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, data_shape, indices_shape, updates_shape, 0,
                            "add", 1, r);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 2000100UL); // UINT32/IDX32/ADD/FP32 + WS 前缀
    EXPECT_GT(r.blockDim, 1U);
    EXPECT_GT(r.workspaceSize, 0UL);
    EXPECT_EQ(r.indicesTotalNum, 100000UL);
    EXPECT_EQ(r.keySize, 4UL); // 100000 <= INT32_MAX -> keySize=4
    EXPECT_EQ(r.shapeMode, 1); // 130000 != 100000 -> SUBSET
    EXPECT_EQ(r.dimNormalized, 0);
    EXPECT_EQ(r.permSize, 4UL);       // N=100000 int32 安全 -> countMode=0 -> permSize=4
    EXPECT_GT(r.sortUsedCoreNum, 0U); // 排序模板用核数由 SortLib coreNumNeed 给出
    EXPECT_GT(r.wsSrcPosOff, 0UL);    // shapeMode=1(SUBSET) 时有 srcPos 段
}

TEST_F(ScatterElementsV2WithSortedTiling, test_with_sorted_fp16_dim1_subset)
{
    gert::StorageShape data_shape = {{130000}, {130000}};
    gert::StorageShape indices_shape = {{100000}, {100000}};
    gert::StorageShape updates_shape = {{100000}, {100000}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, data_shape, indices_shape, updates_shape, 0,
                            "add", 1, r);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 2000101UL); // FP16
    EXPECT_EQ(r.shapeMode, 1);
}

TEST_F(ScatterElementsV2WithSortedTiling, test_with_sorted_bf16_dim1_subset)
{
    gert::StorageShape data_shape = {{130000}, {130000}};
    gert::StorageShape indices_shape = {{100000}, {100000}};
    gert::StorageShape updates_shape = {{100000}, {100000}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, data_shape, indices_shape, updates_shape, 0, "add",
                            1, r);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 2000127UL); // BF16
    EXPECT_EQ(r.shapeMode, 1);
}

TEST_F(ScatterElementsV2WithSortedTiling, test_with_sorted_fp32_dim1_same)
{
    gert::StorageShape data_shape = {{100000}, {100000}};
    gert::StorageShape indices_shape = {{100000}, {100000}};
    gert::StorageShape updates_shape = {{100000}, {100000}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, data_shape, indices_shape, updates_shape, 0,
                            "add", 1, r);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 2000100UL);
    EXPECT_EQ(r.shapeMode, 0); // 各维相等 -> SAME -> 无 srcPos 段
    EXPECT_EQ(r.wsSrcPosOff, 0UL);
}

TEST_F(ScatterElementsV2WithSortedTiling, test_with_sorted_fp32_dim1_index64)
{
    gert::StorageShape data_shape = {{130000}, {130000}};
    gert::StorageShape indices_shape = {{100000}, {100000}};
    gert::StorageShape updates_shape = {{100000}, {100000}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, data_shape, indices_shape, updates_shape, 0,
                            "add", 1, r);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 2001100UL); // IDX64 -> 千位+1
    EXPECT_EQ(r.shapeMode, 1);
}

// ============================================================
// 回退用例：非确定性 / 非 add / 小数据（方案A不触发）-> 保持 1xxxxxx
// ============================================================
// ============================================================
// 8 维输入走排序模板（uint8 + none，axis=7）：验证 2xxxxxx 前缀与 rank=8
// data=(1,1,1,1,1,1,1,16384) indices=updates=(1,1,1,1,1,1,1,8192)
//   索引轴主导：preAxis_=1、midAxis_=8192（末维/索引轴）、afterAxis_=1，
//   ratio = midAxis_/50 = 163 > outerAxisNum = 1，命中 50x 索引轴主导门槛。
//   - ComputeShape：axisSame=0（末维 data(16384)!=indices(8192)），combAxis=max(0, 8)=8
//     >= rank 而不触达合并分支 -> rank 保持 8（不再是 rank=1 的扁化 1D）
//   - CombineIndicesAxis：preAxis_=1，midAxis_=8192，afterAxis_=1
//     aSplitDim = max(pre, after) = 1
//     indicesTypeSize_=4（int32），baseS_ = min(mid=8192, BASE_S_MAX/indicesTypeSize_=256/4=64) = 64，
//     isPatternASA=false -> tmpSize=baseS_=64
//     -> indicesNormBlockData_ = max(CeilDiv(1, usedCoreNum_=64)=1, UB_MIN_FACTOR/indicesTypeSize_/tmpSize=1024/4/64=4)
//     = 4
//     -> indicesUsedCoreNum_ = CeilDiv(1, 4) = 1
//   - 准入：uint8 且 rank_=8 <= 8，none+uint8 在 isDetermType 白名单 -> isDeterministic_=1
//   - 按 index-count 切核：normBlockDataNew = max(CeilDiv(8192, totalCoreNum_=64)=128, 1024) = 1024
//     idxNumCoreNum = CeilDiv(8192, 1024) = 8 > aAxisCoreNum(1) -> isSortDeterm_ = true
//     -> key 前缀 +1000000 = 2xxxxxx
//   - 注意：2xxxxxx 会路由到 KernelScatterElementsWithSorted，即排序模板真正接管
TEST_F(ScatterElementsV2WithSortedTiling, test_with_sorted_uint8_dim8_none)
{
    // 8D 索引轴主导：前 7 维全 1（scatter 轴前无非索引维），末维为大索引轴。
    // axis=7（末维），preAxis_=1、afterAxis_=1，索引轴/非索引轴 = 8192/1 >> 50，命中
    // 排序模板 50x 索引轴主导门槛；合并后无尾维并入（data==indices==updates 位仅末维一致），
    // rank_ 保持 8，验证 8 维 WithSortedProcess 的 stride 缓冲（8 维 + 末列 stride=1）。
    gert::StorageShape data_shape = {{1, 1, 1, 1, 1, 1, 1, 16384}, {1, 1, 1, 1, 1, 1, 1, 16384}};
    gert::StorageShape indices_shape = {{1, 1, 1, 1, 1, 1, 1, 8192}, {1, 1, 1, 1, 1, 1, 1, 8192}};
    gert::StorageShape updates_shape = {{1, 1, 1, 1, 1, 1, 1, 8192}, {1, 1, 1, 1, 1, 1, 1, 8192}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_UINT8, ge::DT_INT32, ge::DT_UINT8, data_shape, indices_shape, updates_shape, 7,
                            "none", 1, r);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 2000001UL); // UINT8 + NONE + IDX32 + WS 前缀（rank=8）
    EXPECT_GT(r.indicesTotalNum, 0UL);
}

TEST_F(ScatterElementsV2WithSortedTiling, test_fallback_not_deterministic)
{
    gert::StorageShape data_shape = {{130000}, {130000}};
    gert::StorageShape indices_shape = {{100000}, {100000}};
    gert::StorageShape updates_shape = {{100000}, {100000}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, data_shape, indices_shape, updates_shape, 0,
                            "add", 0, r);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 1000100UL); // 未提升前缀
}

TEST_F(ScatterElementsV2WithSortedTiling, test_fallback_reduction_mul)
{
    gert::StorageShape data_shape = {{130000}, {130000}};
    gert::StorageShape indices_shape = {{100000}, {100000}};
    gert::StorageShape updates_shape = {{100000}, {100000}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, data_shape, indices_shape, updates_shape, 0,
                            "mul", 1, r);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 1000200UL); // mul + 确定性，非 add -> 不提升
}

TEST_F(ScatterElementsV2WithSortedTiling, test_fallback_small_index_num)
{
    // N 小（2048）：1D 时 aAxisCoreNum=indicesUsedCoreNum_=1（indicesNormBlockData_=max(1,1024/4/baseS_=4)=4，
    // CeilDiv(1,4)=1），normBlockDataNew=max(CeilDiv(2048,64)=32,1024)=1024，
    // idxNumCoreNum=CeilDiv(2048,1024)=2 > 1 -> isSortDeterm_=true -> 前缀 2（2000100）。
    // 注释原写 aAxisCoreNum=64 是误判（64 是 totalCoreNum_，非 A 轴切核核数）。
    gert::StorageShape data_shape = {{2048}, {2048}};
    gert::StorageShape indices_shape = {{2048}, {2048}};
    gert::StorageShape updates_shape = {{2048}, {2048}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, data_shape, indices_shape, updates_shape, 0,
                            "add", 1, r);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 2000100UL); // 1D 小 N：aAxisCoreNum=1，仍触发方案A -> 前缀 2
}
