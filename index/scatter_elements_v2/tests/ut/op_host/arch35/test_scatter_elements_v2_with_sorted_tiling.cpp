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
#include <cstring>
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

// Serialized wire layout, checked independently of the host tiling class.
struct ScatterElementsV2SmallShapeTiling {
    uint32_t allAxis;
    uint32_t rank;
    uint32_t dim;
    uint32_t threads;
    // Keep the low-rank scalar arguments together in the first 32 bytes.
    uint32_t dataStride;
    uint32_t indicesStride;
    uint32_t updatesStride;
    uint32_t axisStride;
    uint32_t targetStrides[8];
    uint32_t indicesStrides[7];
    uint32_t updatesStrides[7];
    uint32_t reserved[2];
};
static_assert(sizeof(ScatterElementsV2SmallShapeTiling) == 128);

struct WithSortedTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    size_t rawTilingSize = 0;
    ScatterElementsV2SmallShapeTiling smallShape{};
    uint64_t indicesTotalNum = 0;
    int64_t baseS = 0;
    int64_t baseA = 0;
    int64_t indicesUsedCoreNum = 0;
    int64_t indicesNormBlockData = 0;
    int64_t indicesTailBlockData = 0;
    int64_t indicesUpdatesSameShape = -1;
    int64_t adaptiveReduce = -1;
    int64_t scalarUpdates = -1;
    int64_t physicalUpdates = -1;
    int64_t deterministic = -1;
    int16_t rank = -1;
    uint64_t dataStride[2] = {};
    uint64_t indicesStride[2] = {};
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
                                    int32_t deterministic, WithSortedTilingResult& result, int32_t coreLimit = 0)
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
    if (coreLimit > 0) {
        compile_info.totalCoreNum = coreLimit;
    }

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
        result.rawTilingSize = raw->GetDataSize();
        if (result.tilingKey / 1000000 == 3) {
            ASSERT_EQ(result.rawTilingSize, sizeof(result.smallShape));
            std::memcpy(&result.smallShape, raw->GetData(), sizeof(result.smallShape));
            return;
        }
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
        // [47] indicesUpdatesSameShape (outside the nested sortTiling)
        // [48] enableAdaptiveReduce (appended, all earlier offsets unchanged)
        // 注：int16/uint32 字段与共享 8 字节字的相邻字段按小端字节序打包，读取时以 8 字节槽为定位单位。
        result.baseS = p64[31];
        result.physicalUpdates = p64[24];
        result.deterministic = p64[33];
        ASSERT_GE(raw->GetDataSize(), 50 * sizeof(int64_t));
        result.scalarUpdates = p64[49];
        result.rank = static_cast<int16_t>(p64[34]);
        for (size_t d = 0; d < 2; ++d) {
            result.dataStride[d] = p64[d];
            result.indicesStride[d] = p64[7 + d];
        }
        result.baseA = p64[32];
        result.indicesUsedCoreNum = p64[28];
        result.indicesNormBlockData = p64[29];
        result.indicesTailBlockData = p64[30];
        ASSERT_GE(raw->GetDataSize(), 48 * sizeof(int64_t));
        result.indicesUpdatesSameShape = p64[47];
        if (raw->GetDataSize() >= 49 * sizeof(int64_t)) {
            result.adaptiveReduce = p64[48];
        }
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

TEST_F(ScatterElementsV2WithSortedTiling, test_indices_updates_same_shape)
{
    gert::StorageShape dataShape = {{3, 8, 17}, {3, 8, 17}};
    gert::StorageShape indicesShape = {{2, 67, 13}, {2, 67, 13}};
    gert::StorageShape updatesShape = {{2, 67, 13}, {2, 67, 13}};
    for (const std::string reduction : {"add", "none"}) {
        WithSortedTilingResult r;
        RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, dataShape, indicesShape, updatesShape, 1,
                                reduction, 1, r);
        ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(r.indicesUpdatesSameShape, 1);
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, test_indices_updates_different_inner_shape)
{
    gert::StorageShape dataShape = {{3, 8, 17}, {3, 8, 17}};
    gert::StorageShape indicesShape = {{2, 67, 13}, {2, 67, 13}};
    gert::StorageShape updatesShape = {{2, 67, 15}, {2, 67, 15}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, dataShape, indicesShape, updatesShape, 1, "add",
                            1, r);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.indicesUpdatesSameShape, 0);
}

TEST_F(ScatterElementsV2WithSortedTiling, test_indices_updates_same_stride_different_shape)
{
    // Equal strides alone must not be reported as equal shapes.
    gert::StorageShape dataShape = {{8, 17}, {8, 17}};
    gert::StorageShape indicesShape = {{67, 13}, {67, 13}};
    gert::StorageShape updatesShape = {{69, 13}, {69, 13}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, dataShape, indicesShape, updatesShape, 0, "add",
                            1, r);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.indicesUpdatesSameShape, 0);
}

TEST_F(ScatterElementsV2WithSortedTiling, test_indices_updates_same_shape_folded_axes)
{
    gert::StorageShape dataShape = {{8, 3, 5}, {8, 3, 5}};
    gert::StorageShape indicesShape = {{67, 3, 5}, {67, 3, 5}};
    gert::StorageShape updatesShape = {{67, 3, 5}, {67, 3, 5}};
    WithSortedTilingResult r;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, dataShape, indicesShape, updatesShape, 0, "add",
                            1, r);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.indicesUpdatesSameShape, 1);
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
//     aSplitDim = max(pre, after) = 1；局部 baseS/baseA 由实际 UB 容量计算。
//   - 准入：uint8 且 rank_=8 <= 8，none 支持局部确定性模板。
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

TEST_F(ScatterElementsV2WithSortedTiling, test_with_sorted_none_all_legal_dtypes)
{
    const std::vector<std::pair<ge::DataType, uint64_t>> cases = {
        {ge::DT_FLOAT, 2000004UL},  {ge::DT_FLOAT16, 2000002UL}, {ge::DT_BF16, 2000002UL}, {ge::DT_INT64, 2000008UL},
        {ge::DT_INT32, 2000004UL},  {ge::DT_INT16, 2000002UL},   {ge::DT_INT8, 2000001UL}, {ge::DT_UINT8, 2000001UL},
        {ge::DT_DOUBLE, 2000008UL}, {ge::DT_BOOL, 2000001UL},
    };
    for (const auto& [dtype, expectedKey] : cases) {
        SCOPED_TRACE(static_cast<int32_t>(dtype));
        gert::StorageShape dataShape = {{16384}, {16384}};
        gert::StorageShape indicesShape = {{8192}, {8192}};
        gert::StorageShape updatesShape = {{8192}, {8192}};
        WithSortedTilingResult result;
        RunWithSortedTilingCase(dtype, ge::DT_INT32, dtype, dataShape, indicesShape, updatesShape, 0, "none", 1,
                                result);
        EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(result.tilingKey, expectedKey);
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, test_local_deterministic_base_a_is_at_least_32_bytes)
{
    gert::StorageShape dataShape = {{10}, {10}};
    gert::StorageShape indicesShape = {{10}, {10}};
    gert::StorageShape updatesShape = {{10}, {10}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_INT8, ge::DT_INT64, ge::DT_INT8, dataShape, indicesShape, updatesShape, 0, "none", 1,
                            result);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1001001UL);
    EXPECT_EQ(result.baseS, 10);
    EXPECT_GE(result.baseA * static_cast<int64_t>(sizeof(int64_t)), 32);
}

TEST_F(ScatterElementsV2WithSortedTiling, test_keep_local_when_base_s_cannot_cover_and_sort_admission_fails)
{
    // outerAxis=200 makes the 50x production admission fail, while midAxis=8192 cannot fit in one local tile
    // when baseA must occupy at least 32 bytes. baseS coverage alone must not force the global sort template.
    gert::StorageShape dataShape = {{200, 8192}, {200, 8192}};
    gert::StorageShape indicesShape = {{200, 8192}, {200, 8192}};
    gert::StorageShape updatesShape = {{200, 8192}, {200, 8192}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, dataShape, indicesShape, updatesShape, 1, "none",
                            1, result);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1000004UL);
    EXPECT_LT(result.baseS, 8192);
    EXPECT_GE(result.baseA * static_cast<int64_t>(sizeof(int32_t)), 32);
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

TEST_F(ScatterElementsV2WithSortedTiling, adaptive_reduce_capacity_limited_add)
{
    constexpr int64_t scatterAxis = 8192;
    gert::StorageShape input({2087, scatterAxis + 8}, {2087, scatterAxis + 8});
    gert::StorageShape index({2079, scatterAxis}, {2079, scatterAxis});
    gert::StorageShape updates({2086, scatterAxis}, {2086, scatterAxis});
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_FLOAT16, DT_INT32, DT_FLOAT16, input, index, updates, 1, "add", 1, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1000101UL);
    // SortLib's temporary-buffer estimate varies between CANN versions. Verify
    // the capacity-limited behavior without freezing the exact tile sizes.
    EXPECT_GT(result.baseS, 0);
    EXPECT_LT(result.baseS, scatterAxis);
    EXPECT_GE(result.baseA * static_cast<int64_t>(sizeof(int32_t)), 32);
    EXPECT_EQ(result.adaptiveReduce, 1);
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
    EXPECT_EQ(r.adaptiveReduce, 0);
}

TEST_F(ScatterElementsV2WithSortedTiling, minimum_valid_a_does_not_split_small_input)
{
    // Seven int32 A positions cannot form even one 32-byte A block.
    // Keep them on one core instead of making four undersized A tails.
    gert::StorageShape input({7, 19}, {7, 19});
    gert::StorageShape index({7, 257}, {7, 257});
    gert::StorageShape updates({7, 257}, {7, 257});
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_FLOAT16, DT_INT32, DT_FLOAT16, input, index, updates, 1, "add", 1, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(result.tilingKey, 1000101UL);
    EXPECT_EQ(result.indicesUsedCoreNum, 1);
    EXPECT_EQ(result.indicesTailBlockData, 7);
    EXPECT_GE(result.baseA, 8);
    // A shape-limited input may still contain a long duplicate-index chain.
    EXPECT_EQ(result.adaptiveReduce, 1);
}

TEST_F(ScatterElementsV2WithSortedTiling, minimum_valid_a_boundaries)
{
    struct Fixture {
        ge::DataType indexType;
        int64_t a;
        int64_t cores;
        int64_t normal;
        int64_t tail;
    };
    const Fixture fixtures[] = {{DT_INT32, 8, 1, 8, 8},  {DT_INT32, 9, 2, 8, 1}, {DT_INT32, 16, 2, 8, 8},
                                {DT_INT32, 17, 3, 8, 1}, {DT_INT64, 3, 1, 4, 3}, {DT_INT64, 4, 1, 4, 4},
                                {DT_INT64, 5, 2, 4, 1},  {DT_INT64, 9, 3, 4, 1}};
    for (const auto& f : fixtures) {
        for (const std::string reduction : {"add", "none"}) {
            for (int64_t axis : {0, 1}) {
                SCOPED_TRACE(std::to_string(f.a) + "/" + std::to_string(f.indexType) + "/" + reduction + "/" +
                             std::to_string(axis));
                gert::StorageShape input = axis == 1 ? gert::StorageShape({f.a, 19}, {f.a, 19}) :
                                                       gert::StorageShape({19, f.a}, {19, f.a});
                gert::StorageShape index = axis == 1 ? gert::StorageShape({f.a, 257}, {f.a, 257}) :
                                                       gert::StorageShape({257, f.a}, {257, f.a});
                gert::StorageShape updates = index;
                WithSortedTilingResult result;
                RunWithSortedTilingCase(DT_FLOAT16, f.indexType, DT_FLOAT16, input, index, updates, axis, reduction, 1,
                                        result);
                ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
                EXPECT_EQ(result.indicesUsedCoreNum, f.cores);
                EXPECT_EQ(result.indicesNormBlockData, f.normal);
                EXPECT_EQ(result.indicesTailBlockData, f.tail);
            }
        }
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, minimum_valid_a_asa_counts_both_outer_axes)
{
    for (bool splitPre : {false, true}) {
        gert::StorageShape input = splitPre ? gert::StorageShape({9, 19, 2}, {9, 19, 2}) :
                                              gert::StorageShape({2, 19, 9}, {2, 19, 9});
        gert::StorageShape index = splitPre ? gert::StorageShape({9, 257, 2}, {9, 257, 2}) :
                                              gert::StorageShape({2, 257, 9}, {2, 257, 9});
        gert::StorageShape updates = index;
        WithSortedTilingResult result;
        RunWithSortedTilingCase(DT_FLOAT16, DT_INT32, DT_FLOAT16, input, index, updates, 1, "add", 1, result);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        // Four positions on the split axis times two on the other axis give
        // eight valid flattened A positions; the last core has one times two.
        EXPECT_EQ(result.indicesUsedCoreNum, 3);
        EXPECT_EQ(result.indicesNormBlockData, 4);
        EXPECT_EQ(result.indicesTailBlockData, 1);
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, adaptive_reduce_shape_limited_input)
{
    gert::StorageShape input({8, 3}, {8, 3});
    gert::StorageShape index({67, 2}, {67, 2});
    gert::StorageShape updates({69, 4}, {69, 4});
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, input, index, updates, 0, "add", 1, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1000100UL);
    EXPECT_EQ(result.baseS, 67);
    EXPECT_EQ(result.baseA, 8);
    EXPECT_EQ(result.adaptiveReduce, 1);
}

TEST_F(ScatterElementsV2WithSortedTiling, adaptive_reduce_widened_int64_small_a)
{
    // 512 outer positions split evenly across 64 cores, so local A is eight
    // elements regardless of the SortLib temporary-buffer estimate.
    gert::StorageShape input({512, 1816}, {512, 1816});
    gert::StorageShape index({512, 600}, {512, 600});
    gert::StorageShape updates({512, 916}, {512, 916});
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_FLOAT16, DT_INT64, DT_FLOAT16, input, index, updates, 1, "add", 1, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1001101UL);
    EXPECT_EQ(result.baseS, 600);
    EXPECT_GT(result.baseA, 4);
    EXPECT_LE(result.baseA, 8);
    EXPECT_EQ(result.adaptiveReduce, 1);
}

TEST_F(ScatterElementsV2WithSortedTiling, adaptive_reduce_minimum_a_has_valid_work)
{
    gert::StorageShape input({64, 2049}, {64, 2049});
    gert::StorageShape index({64, 2049}, {64, 2049});
    gert::StorageShape updates({64, 2049}, {64, 2049});
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_FLOAT16, DT_INT32, DT_FLOAT16, input, index, updates, 1, "add", 1, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1000101UL);
    EXPECT_GT(result.baseS, 0);
    EXPECT_LT(result.baseS, 2049);
    EXPECT_EQ(result.baseA, 8);
    // Core splitting now supplies eight real A positions, not one real A
    // padded to eight. Full S no longer fits for this valid per-core workload.
    EXPECT_EQ(result.indicesUsedCoreNum, 8);
    EXPECT_EQ(result.indicesNormBlockData, 8);
    EXPECT_EQ(result.indicesTailBlockData, 8);
    EXPECT_EQ(result.adaptiveReduce, 1);
}

TEST_F(ScatterElementsV2WithSortedTiling, adaptive_reduce_none_and_nondeterministic_unchanged)
{
    gert::StorageShape input({2087, 1253}, {2087, 1253});
    gert::StorageShape index({2079, 1249}, {2079, 1249});
    gert::StorageShape updates({2086, 1253}, {2086, 1253});
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_FLOAT16, DT_INT32, DT_FLOAT16, input, index, updates, 1, "none", 1, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.adaptiveReduce, 0);
    RunWithSortedTilingCase(DT_FLOAT16, DT_INT32, DT_FLOAT16, input, index, updates, 1, "add", 0, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.adaptiveReduce, 0);
}

TEST_F(ScatterElementsV2WithSortedTiling, scalar_updates_keep_only_self_conversion_workspace)
{
    constexpr size_t systemWorkspace = 16777216;
    for (auto dtype : {DT_INT8, DT_UINT8, DT_INT16, DT_BOOL}) {
        for (int deterministic : {0, 1}) {
            gert::StorageShape input({17, 67}, {17, 67});
            gert::StorageShape index({15, 2049}, {15, 2049});
            gert::StorageShape updates({1}, {1});
            WithSortedTilingResult result;
            RunWithSortedTilingCase(dtype, DT_INT64, dtype, input, index, updates, 1, "add", deterministic, result);
            ASSERT_EQ(result.status, GRAPH_SUCCESS);
            EXPECT_EQ(result.scalarUpdates, 1);
            EXPECT_EQ(result.physicalUpdates, 1);
            EXPECT_EQ(result.deterministic, 0);
            EXPECT_EQ(result.adaptiveReduce, 0);
            EXPECT_EQ(result.indicesUpdatesSameShape, 0);
            EXPECT_LT(result.tilingKey, 2000000); // No sorted kernel, even with determinism enabled.
            size_t castBytes = dtype == DT_BOOL ? 2 : 4;
            EXPECT_EQ(result.workspaceSize, systemWorkspace + ((17 * 67 * castBytes + 511) / 512) * 512);
            EXPECT_EQ(result.blockDim, 61); // ceil(indices / 512), independent of updates storage.
        }
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, scalar_updates_accept_all_ranks_and_axes)
{
    for (int64_t rank = 1; rank <= 8; ++rank) {
        for (int64_t axis = 0; axis < rank; ++axis) {
            std::vector<int64_t> dataDims(rank, 3);
            std::vector<int64_t> indexDims(rank, 2);
            indexDims[axis] = 5;
            gert::StorageShape input;
            gert::StorageShape index;
            for (int64_t d = 0; d < rank; ++d) {
                input.MutableOriginShape().AppendDim(dataDims[d]);
                input.MutableStorageShape().AppendDim(dataDims[d]);
                index.MutableOriginShape().AppendDim(indexDims[d]);
                index.MutableStorageShape().AppendDim(indexDims[d]);
            }
            gert::StorageShape updates({1}, {1});
            WithSortedTilingResult result;
            RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, input, index, updates, axis, "none", 1, result);
            ASSERT_EQ(result.status, GRAPH_SUCCESS) << "rank=" << rank << ", axis=" << axis;
            EXPECT_EQ(result.scalarUpdates, 1);
            EXPECT_EQ(result.physicalUpdates, 1);
            EXPECT_EQ(result.deterministic, 0);
            EXPECT_EQ(result.workspaceSize, 16777216);
        }
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, scalar_updates_retain_self_and_index_shape_validation)
{
    gert::StorageShape input({3, 7}, {3, 7});
    gert::StorageShape index({4, 5}, {4, 5});
    gert::StorageShape updates({1}, {1});
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, input, index, updates, 1, "none", 0, result);
    EXPECT_EQ(result.status, GRAPH_FAILED);

    gert::StorageShape validIndex({2, 5}, {2, 5});
    gert::StorageShape nonScalar({2}, {2});
    RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, input, validIndex, nonScalar, 1, "none", 0, result);
    EXPECT_EQ(result.status, GRAPH_FAILED);
}

TEST_F(ScatterElementsV2WithSortedTiling, scalar_updates_coalesce_the_logical_shape)
{
    gert::StorageShape input({2, 3, 5, 7}, {2, 3, 5, 7});
    gert::StorageShape index({2, 4, 5, 7}, {2, 4, 5, 7});
    gert::StorageShape updates({1}, {1});
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_FLOAT, DT_INT64, DT_FLOAT, input, index, updates, 1, "add", 1, result);
    ASSERT_EQ(result.status, GRAPH_SUCCESS);
    EXPECT_EQ(result.scalarUpdates, 1);
    EXPECT_EQ(result.physicalUpdates, 1);
    EXPECT_EQ(result.rank, 3);
    EXPECT_EQ(result.dataStride[0], 105);
    EXPECT_EQ(result.dataStride[1], 35);
    EXPECT_EQ(result.indicesStride[0], 140);
    EXPECT_EQ(result.indicesStride[1], 35);
}

TEST_F(ScatterElementsV2WithSortedTiling, small_none_last_axis_preserves_source_stride)
{
    gert::StorageShape data = {{2946, 1718}, {2946, 1718}};
    gert::StorageShape index = {{1, 1713}, {1, 1713}};
    gert::StorageShape updates = {{2945, 1723}, {2945, 1723}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, data, index, updates, -1, "none", 0, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 3000004UL);
    EXPECT_EQ(result.rawTilingSize, 128UL);
    EXPECT_EQ(result.blockDim, 6U);
    EXPECT_EQ(result.smallShape.allAxis, 1713U);
    EXPECT_EQ(result.smallShape.dim, 1U);
    EXPECT_EQ(result.smallShape.targetStrides[0], 1718U);
    EXPECT_EQ(result.smallShape.indicesStrides[0], 1713U);
    EXPECT_EQ(result.smallShape.updatesStrides[0], 1723U);
    EXPECT_EQ(result.smallShape.threads, 320U);
}

TEST_F(ScatterElementsV2WithSortedTiling, small_none_first_axis_preserves_source_stride)
{
    gert::StorageShape data = {{2028, 2760}, {2028, 2760}};
    gert::StorageShape index = {{1, 2023}, {1, 2023}};
    gert::StorageShape updates = {{2028, 2757}, {2028, 2757}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_INT64, DT_INT64, DT_INT64, data, index, updates, -2, "none", 0, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 3001008UL);
    EXPECT_EQ(result.blockDim, 32U);
    EXPECT_EQ(result.smallShape.dim, 0U);
    EXPECT_EQ(result.smallShape.updatesStrides[0], 2757U);
    EXPECT_EQ(result.smallShape.threads, 64U);
}

TEST_F(ScatterElementsV2WithSortedTiling, small_none_launch_adapts_to_update_count)
{
    gert::StorageShape data = {{65536}, {65536}};
    gert::StorageShape updates = {{32768}, {32768}};
    struct LaunchCase {
        int64_t count;
        uint32_t cores;
        uint32_t threads;
    };
    for (const auto& test : std::vector<LaunchCase>{{1, 1, 64},
                                                    {64, 1, 64},
                                                    {65, 2, 64},
                                                    {256, 2, 128},
                                                    {512, 3, 192},
                                                    {1024, 4, 256},
                                                    {2048, 6, 384},
                                                    {3072, 7, 448},
                                                    {4096, 8, 512},
                                                    {8192, 12, 704}}) {
        SCOPED_TRACE(test.count);
        gert::StorageShape index = {{test.count}, {test.count}};
        WithSortedTilingResult result;
        RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, data, index, updates, 0, "none", 0, result);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.tilingKey, 3000004UL);
        EXPECT_EQ(result.blockDim, test.cores);
        EXPECT_EQ(result.smallShape.threads, test.threads);
        EXPECT_GE(result.blockDim * result.smallShape.threads, test.count);
        EXPECT_LT((result.blockDim - 1) * result.smallShape.threads, test.count);
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, small_none_multiple_rows_balance_duplicate_writers)
{
    gert::StorageShape data = {{1024, 257}, {1024, 257}};
    gert::StorageShape index = {{64, 64}, {64, 64}};
    gert::StorageShape updates = {{65, 71}, {65, 71}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_FLOAT16, DT_INT32, DT_FLOAT16, data, index, updates, 0, "none", 0, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 3000002UL);
    EXPECT_EQ(result.blockDim, 8U);
    EXPECT_EQ(result.smallShape.threads, 512U);
}

TEST_F(ScatterElementsV2WithSortedTiling, small_none_launch_respects_core_and_thread_limits)
{
    gert::StorageShape data = {{2028, 5000}, {2028, 5000}};
    gert::StorageShape index = {{1, 4096}, {1, 4096}};
    gert::StorageShape updates = {{2, 4103}, {2, 4103}};
    struct LaunchCase {
        int32_t coreLimit;
        uint32_t cores;
        uint32_t threads;
    };
    for (const auto& test :
         std::vector<LaunchCase>{{1, 1, 1024}, {3, 3, 1024}, {7, 7, 640}, {56, 32, 128}, {64, 64, 64}}) {
        SCOPED_TRACE(test.coreLimit);
        WithSortedTilingResult result;
        RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, data, index, updates, 0, "none", 0, result,
                                test.coreLimit);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(result.tilingKey, 3000004UL);
        EXPECT_EQ(result.blockDim, test.cores);
        EXPECT_EQ(result.smallShape.threads, test.threads);
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, small_none_limit_and_deterministic_fallback)
{
    gert::StorageShape data = {{32768}, {32768}};
    gert::StorageShape updates = {{32768}, {32768}};
    for (int64_t count : {8192, 8193, 16384}) {
        gert::StorageShape index = {{count}, {count}};
        WithSortedTilingResult result;
        RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, data, index, updates, 0, "none", 0, result);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(result.tilingKey / 1000000, count == 8192 ? 3UL : 1UL);
        if (count == 8192) {
            RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, data, index, updates, 0, "none", 1, result);
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
            EXPECT_NE(result.tilingKey / 1000000, 3UL);
        }
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, small_none_scalar_updates_use_original_path)
{
    gert::StorageShape data = {{16}, {16}};
    gert::StorageShape index = {{8}, {8}};
    gert::StorageShape updates = {{1}, {1}};
    for (int deterministic : {0, 1}) {
        WithSortedTilingResult result;
        RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, data, index, updates, 0, "none", deterministic, result);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(result.tilingKey, 1000004UL);
        EXPECT_EQ(result.scalarUpdates, 1);
        EXPECT_EQ(result.physicalUpdates, 1);
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, small_none_general_3d_all_axes_preserve_independent_strides)
{
    gert::StorageShape data = {{5, 7, 11}, {5, 7, 11}};
    gert::StorageShape index = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape updates = {{4, 5, 6}, {4, 5, 6}};
    const uint32_t dataStrides[] = {77, 11, 1};
    for (int64_t axis = -3; axis < 3; ++axis) {
        SCOPED_TRACE(axis);
        const auto dim = static_cast<uint32_t>((axis + 3) % 3);
        WithSortedTilingResult result;
        RunWithSortedTilingCase(DT_FLOAT, DT_INT64, DT_FLOAT, data, index, updates, axis, "none", 0, result);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.tilingKey, 3001004UL);
        EXPECT_EQ(result.rawTilingSize, 128UL);
        EXPECT_EQ(result.smallShape.rank, 3U);
        EXPECT_EQ(result.smallShape.dim, dim);
        EXPECT_EQ(result.smallShape.allAxis, 24U);
        EXPECT_EQ(result.smallShape.axisStride, dataStrides[dim]);
        for (uint32_t d = 0; d < 3; ++d) {
            EXPECT_EQ(result.smallShape.targetStrides[d], d == dim ? 0U : dataStrides[d]);
        }
        EXPECT_EQ(result.smallShape.indicesStrides[0], 12U);
        EXPECT_EQ(result.smallShape.indicesStrides[1], 4U);
        EXPECT_EQ(result.smallShape.updatesStrides[0], 30U);
        EXPECT_EQ(result.smallShape.updatesStrides[1], 6U);
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, small_none_rank8_limits_threads_and_keeps_last_axis_stride)
{
    gert::StorageShape data = {{3, 3, 3, 3, 3, 3, 5, 5}, {3, 3, 3, 3, 3, 3, 5, 5}};
    gert::StorageShape index = {{2, 2, 2, 2, 2, 2, 4, 4}, {2, 2, 2, 2, 2, 2, 4, 4}};
    gert::StorageShape updates = {{4, 4, 4, 4, 4, 4, 6, 6}, {4, 4, 4, 4, 4, 4, 6, 6}};
    for (int64_t axis = 0; axis < 8; ++axis) {
        WithSortedTilingResult result;
        RunWithSortedTilingCase(DT_INT64, DT_INT32, DT_INT64, data, index, updates, axis, "none", 0, result, 1);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.tilingKey, 3000008UL);
        EXPECT_EQ(result.smallShape.rank, 8U);
        EXPECT_EQ(result.smallShape.threads, 512U);
        EXPECT_EQ(result.blockDim, 1U);
        EXPECT_EQ(result.smallShape.targetStrides[7], axis == 7 ? 0U : 1U);
        EXPECT_EQ(result.smallShape.indicesStrides[6], 4U);
        EXPECT_EQ(result.smallShape.updatesStrides[6], 6U);
    }
}

TEST_F(ScatterElementsV2WithSortedTiling, small_none_nd_limit_and_fallbacks)
{
    gert::StorageShape data = {{7, 65, 2733}, {7, 65, 2733}};
    gert::StorageShape updates = {{6, 64, 2732}, {6, 64, 2732}};
    gert::StorageShape admitted = {{4, 32, 64}, {4, 32, 64}};
    gert::StorageShape aboveLimit = {{3, 1, 2731}, {3, 1, 2731}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, data, admitted, updates, 1, "none", 0, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 3000004UL);
    EXPECT_EQ(result.smallShape.allAxis, 8192U);
    EXPECT_LE(result.smallShape.threads, 512U);
    RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, data, aboveLimit, updates, 1, "none", 0, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey / 1000000, 1UL);
    for (const std::string& reduction : {"none", "add", "mul"}) {
        for (int deterministic : {0, 1}) {
            if (reduction == "none" && deterministic == 0) {
                continue;
            }
            RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, data, admitted, updates, 1, reduction, deterministic,
                                    result);
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
            EXPECT_LT(result.tilingKey / 1000000, 3UL);
        }
    }
    gert::StorageShape largeData = {{65536, 65536, 2}, {65536, 65536, 2}};
    gert::StorageShape index = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape src = {{3, 4, 5}, {3, 4, 5}};
    RunWithSortedTilingCase(DT_FLOAT, DT_INT32, DT_FLOAT, largeData, index, src, -1, "none", 0, result);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey / 1000000, 1UL);
}

TEST_F(ScatterElementsV2WithSortedTiling, small_none_registered_tiling_layout)
{
    const uint64_t keys[] = {3000001, 3000002, 3000004, 3000008, 3001001, 3001002, 3001004, 3001008};
    for (const auto key : keys) {
        const auto name = std::string("ScatterElementsV2_") + std::to_string(key);
        auto data = optiling::CTilingDataClassFactory::GetInstance().CreateTilingDataInstance(name.c_str());
        ASSERT_NE(data, nullptr);
        EXPECT_STREQ(data->GetTilingClassName(), "ScatterElementsV2SmallShapeTilingData");
        EXPECT_EQ(data->GetDataSize(), 128U);
    }
}
