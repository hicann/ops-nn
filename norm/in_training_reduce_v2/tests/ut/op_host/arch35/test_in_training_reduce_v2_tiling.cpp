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
 * \file test_in_training_reduce_v2_tiling.cpp
 * \brief arch35 核心路径 UT —— Tiling（AR_FULL_REDUCE / R 全载）
 *   契约（spec.yaml / DESIGN §5.1/§6.3）：
 *     - TilingKey == 200000（AR_FULL_REDUCE）
 *     - IsCapable 对 NCHW / ND / NCDHW 返回 true（一期支持）
 *     - 输入 1（x）/ 输出 2（sum, square_sum），无 attr
 *     - 典型 shape [4,16,32,32] fp32 tiling 成功，产出非空 TilingData
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "test_in_training_reduce_v2_tiling.h"
#include "../../../../op_kernel/arch35/in_training_reduce_v2_tiling_data.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class INTrainingReduceV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "INTrainingReduceV2TilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "INTrainingReduceV2TilingTest TearDown" << std::endl; }
};

namespace {
constexpr uint64_t EXPECTED_ARA_TILING_KEY = 400000UL;
constexpr uint64_t EXPECTED_AR_TILING_KEY = 200000UL;
constexpr uint64_t MOCK_UB_SIZE = 245760;

// 公共 compile_info（Ascend950，CORE_NUM=64，UB=245760）
static const char* COMPILE_INFO_STR = R"({
   "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                     "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                     "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                     "CORE_NUM": 64, "socVersion": "Ascend950"}
            })";

// 运行一次 in_training_reduce_v2 tiling，返回 tiling_key（失败返回 UINT64_MAX）。
// x_shape：输入 shape；out_shape：sum/square_sum shape（[N,C,1,1] 或 5D [N,C,1,1,1]）；
// fmt：输入 format（NCHW/ND/NCDHW）。
// 注意：shape 以非 const 引用传入——gert::TilingContextFaker 的 InputShapes/OutputShapes 需要
// 非 const 的 StorageShape*，若用 const& 则 &x_shape 得 const 指针，导致 const void*→void* 转换失败。
// td_out 非空时，额外把下发的 TilingData 拷回给调用方，供 sub-R 用例校验 rFactor / numChunks。
static uint64_t RunTiling(gert::StorageShape& xShape, gert::StorageShape& outShape, ge::DataType dtype, ge::Format fmt,
                          INTrainingReduceV2ARFullReduceTilingData* tdOut = nullptr,
                          ge::DataType sumDtype = ge::DT_FLOAT, ge::DataType squareSumDtype = ge::DT_FLOAT,
                          uint64_t ubSize = 245760, INTrainingReduceV2ARAReduceTilingData* araOut = nullptr,
                          gert::StorageShape* squareSumShape = nullptr, ge::graphStatus* tilingStatus = nullptr)
{
    std::map<std::string, std::string> socVersionInfos = {{"NpuArch", "3510"}};
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    std::string compileInfoString(COMPILE_INFO_STR);
    const std::string ubMarker = "\"UB_SIZE\": 245760";
    const auto ubPos = compileInfoString.find(ubMarker);
    if (ubPos == std::string::npos) {
        return UINT64_MAX;
    }
    compileInfoString.replace(ubPos, ubMarker.size(), "\"UB_SIZE\": " + std::to_string(ubSize));
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::INTrainingReduceV2CompileInfo compileInfo;

    std::string opType("INTrainingReduceV2");
    if (gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()) == nullptr) {
        return UINT64_MAX;
    }
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tiling parse
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(3, 3)
                            .Inputs({compileInfoString.data(), &platformInfo})
                            .Outputs({&compileInfo})
                            .Build();
    auto parseCtx = kernelHolder.GetContext<gert::TilingParseContext>();
    if (parseCtx == nullptr || parseCtx->GetPlatformInfo() == nullptr || !parseCtx->GetPlatformInfo()->Init()) {
        return UINT64_MAX;
    }
    parseCtx->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseCtx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parseCtx->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
    if (tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return UINT64_MAX;
    }

    // tiling：1 输入 x / 2 输出 sum,square_sum，无 attr
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    if (tilingData == nullptr || workspaceSize == nullptr) {
        return UINT64_MAX;
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outShape, squareSumShape == nullptr ? &outShape : squareSumShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(&platformInfo)
                      .NodeInputTd(0, dtype, fmt, fmt)
                      .NodeOutputTd(0, sumDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, squareSumDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        return UINT64_MAX;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    const auto status = tilingFunc(tilingContext);
    if (tilingStatus != nullptr) {
        *tilingStatus = status;
    }
    if (status != ge::GRAPH_SUCCESS) {
        return UINT64_MAX;
    }
    if (tdOut != nullptr) {
        auto raw = tilingContext->GetRawTilingData();
        if (raw == nullptr || raw->GetDataSize() < sizeof(*tdOut)) {
            return UINT64_MAX;
        }
        *tdOut = *static_cast<const INTrainingReduceV2ARFullReduceTilingData*>(raw->GetData());
    }
    if (araOut != nullptr) {
        auto raw = tilingContext->GetRawTilingData();
        if (raw == nullptr || raw->GetDataSize() < sizeof(*araOut)) {
            return UINT64_MAX;
        }
        *araOut = *static_cast<const INTrainingReduceV2ARAReduceTilingData*>(raw->GetData());
    }
    return tilingContext->GetTilingKey();
}

// 复算 Kernel InitSubR() 的 UB 占用，用于在 UT 里守住「Tiling 下发的参数确实塞得进 UB」。
// 公式与 op_host 的 CalcSubRUbBytes / op_kernel 的 InitSubR 三方一致。
static uint64_t SubRUbBytes(const INTrainingReduceV2ARFullReduceTilingData& td, uint64_t elemSize)
{
    constexpr uint64_t kBlockSize = 32;   // platform::GetUbBlockSize()
    constexpr uint64_t kVlFp32 = 64;      // 256B vector length / sizeof(float)
    constexpr uint64_t kDoubleBuffer = 2; // DOUBLE_BUFFER_NUM
    auto ceilAlign = [](uint64_t v, uint64_t a) { return (v + a - 1) / a * a; };
    uint64_t slots = td.chunksPerGroup + (td.numGroups > 1 ? 1U : 0U);
    uint64_t inBytes = ceilAlign(td.rFactor * elemSize, kBlockSize) * kDoubleBuffer;
    uint64_t partialBytes = ceilAlign(slots, kVlFp32) * sizeof(float) * 2 * 2; // outputs x components
    uint64_t outBytes = ceilAlign(sizeof(float), kBlockSize) * kDoubleBuffer * 2;
    return inBytes + partialBytes + outBytes;
}

constexpr uint64_t USABLE_UB = 245760 - 512; // UB_SIZE - RESERVE_FOR_ALIGN

// sub-R 一组自洽性断言：分块 / 分组参数彼此对得上，且 Kernel 按这组参数申请的 UB 不越界。
// 任何 R（含超 uint32）都必须满足，用它取代逐用例手写断言。
static void ExpectSubRSelfConsistent(const INTrainingReduceV2ARFullReduceTilingData& td, uint64_t elemSize)
{
    constexpr uint64_t kVlFp32 = 64;
    ASSERT_EQ(td.isSubRTiling, 1U);
    ASSERT_EQ(td.rFactor % kVlFp32, 0U);
    ASSERT_GT(td.chunksPerGroup, 0U);
    ASSERT_GT(td.numGroups, 0U);
    ASSERT_EQ(td.numChunks, td.numR / td.rFactor + ((td.numR % td.rFactor) != 0U ? 1U : 0U));
    ASSERT_EQ(td.tailLen, td.numR - (td.numChunks - 1) * td.rFactor);
    ASSERT_EQ(td.numGroups, td.numChunks / td.chunksPerGroup + ((td.numChunks % td.chunksPerGroup) != 0U ? 1U : 0U));
    ASSERT_EQ(td.tailChunks, td.numChunks - (td.numGroups - 1) * td.chunksPerGroup);
    ASSERT_LE(td.tailChunks, td.chunksPerGroup);
    ASSERT_LE(SubRUbBytes(td, elemSize), USABLE_UB);
    ASSERT_EQ(td.totalRows, td.numN * td.numC);
    ASSERT_EQ(td.totalElements, td.totalRows * td.numR);
    ASSERT_EQ(td.totalTiles, td.totalRows);
    ASSERT_EQ(td.inputBufferBytes, (td.rFactor * elemSize + 31U) / 32U * 32U);
    ASSERT_EQ(td.outputBufferBytes, 32U);
    ASSERT_EQ(td.scratchBufferBytes, 0U);
    uint64_t slots = td.chunksPerGroup + (td.numGroups > 1 ? 1U : 0U);
    ASSERT_EQ(td.partialBufferBytes, (slots + kVlFp32 - 1U) / kVlFp32 * kVlFp32 * sizeof(float) * 2);
    ASSERT_EQ(SubRUbBytes(td, elemSize),
              td.inputBufferBytes * 2U + td.partialBufferBytes * 2U + td.outputBufferBytes * 4U);
}
} // namespace

// ---------------------------------------------------------------------------
// AR_FULL_REDUCE：4D NCHW 典型 shape [4,16,32,32] → TilingKey 200000
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nchw_200000_001)
{
    gert::StorageShape x_shape = {{4, 16, 32, 32}, {4, 16, 32, 32}};
    gert::StorageShape out_shape = {{4, 16, 1, 1}, {4, 16, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW, &td);
    ASSERT_EQ(key, 200000U);
    EXPECT_EQ(td.isSubRTiling, 0U);
    EXPECT_EQ(td.totalRows, 4LL * 16LL);
    EXPECT_EQ(td.totalElements, 4LL * 16LL * 32LL * 32LL);
    EXPECT_EQ(td.totalTiles, td.numN * td.cOuter);
    EXPECT_GT(td.inputBufferBytes, 0U);
    EXPECT_GT(td.outputBufferBytes, 0U);
    EXPECT_GT(td.scratchBufferBytes, 0U);
    EXPECT_EQ(td.partialBufferBytes, 0U);
}

// ---------------------------------------------------------------------------
// AR_FULL_REDUCE：fp16 输入同样走 200000（dtype 由编译期宏分派，不进 key）
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nchw_fp16_200000_002)
{
    gert::StorageShape x_shape = {{2, 8, 16, 16}, {2, 8, 16, 16}};
    gert::StorageShape out_shape = {{2, 8, 1, 1}, {2, 8, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// AR_FULL_REDUCE：5D NCDHW [2,3,4,5,6] → [2,3,1,1,1]，IsCapable true，200000
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_ncdhw_5d_200000_004)
{
    gert::StorageShape x_shape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape out_shape = {{2, 3, 1, 1, 1}, {2, 3, 1, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCDHW);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// AR_FULL_REDUCE：ND format 也被 IsCapable 接受，200000
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nd_200000_005)
{
    gert::StorageShape x_shape = {{4, 16, 32, 32}, {4, 16, 32, 32}};
    gert::StorageShape out_shape = {{4, 16, 1, 1}, {4, 16, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// AR_FULL_REDUCE：退化保留维 N=1/C=1 + 大 R（[1,1,64,64]），tiling 成功
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_keepdim_200000_006)
{
    gert::StorageShape x_shape = {{1, 1, 64, 64}, {1, 1, 64, 64}};
    gert::StorageShape out_shape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// 除零守卫（代码检视 HIGH）：ND format 下 r = GetShapeSize()/a1/a0 在 a1/a0
// 后置校验之前执行，故 N=0（a1=0）必须被前置守卫拦截，返回 GRAPH_FAILED（而非
// 触发整数除零崩溃）。RunTiling 失败时返回 UINT64_MAX。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nd_n0_guard_graph_failed_007)
{
    gert::StorageShape x_shape = {{0, 16, 32, 32}, {0, 16, 32, 32}};
    gert::StorageShape out_shape = {{0, 16, 1, 1}, {0, 16, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_EQ(key, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 除零守卫（代码检视 HIGH）：ND format 下 C=0（a0=0）同样在除法前被守卫拦截，
// 返回 GRAPH_FAILED。此分支仅由新增前置守卫覆盖（后置 a0<=0 检查在除法之后）。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nd_c0_guard_graph_failed_008)
{
    gert::StorageShape x_shape = {{4, 0, 32, 32}, {4, 0, 32, 32}};
    gert::StorageShape out_shape = {{4, 0, 1, 1}, {4, 0, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_EQ(key, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// sub-R 分块路径：R 超大（超单次 UB 容量）时触发 DoSubRTiling，
// TilingKey 仍为 200000（同 key + isSubRTiling 标志区分）。
// fp32 [1,1,316,316] → R=99856，单行全载超 UB → sub-R 分块。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_nchw_fp32_200000_009)
{
    gert::StorageShape x_shape = {{1, 1, 316, 316}, {1, 1, 316, 316}};
    gert::StorageShape out_shape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// sub-R 分块路径：fp16 输入，5D NCDHW，R 超大 → sub-R 分块。
// [1,1,100,100,100] → R=1,000,000，fp16 elemSize=2 → sub-R。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_ncdhw_fp16_200000_010)
{
    gert::StorageShape x_shape = {{1, 1, 100, 100, 100}, {1, 1, 100, 100, 100}};
    gert::StorageShape out_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT16, ge::FORMAT_NCDHW);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// sub-R 分块路径：ND format，3D [1,1,1000000] → R=1,000,000，fp32 → sub-R。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_nd_fp32_200000_011)
{
    gert::StorageShape x_shape = {{1, 1, 1000000}, {1, 1, 1000000}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND);
    ASSERT_EQ(key, 200000U);
}

// ---------------------------------------------------------------------------
// 空 tensor：规约（空间）轴为 0 时 Tiling 明确失败。
// 本迭代不含 REDUCE_EMPTY 模板，图原型 / README 均已声明不支持空 tensor；
// 这条用例把「不支持」钉成可回归的行为，防止后续默默变成越界下发。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nchw_r0_empty_rejected_012)
{
    gert::StorageShape x_shape = {{2, 3, 0, 4}, {2, 3, 0, 4}};
    gert::StorageShape out_shape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW);
    ASSERT_EQ(key, UINT64_MAX);
}

// Empty tensors are unsupported.  Enumerate every semantic axis, multiple-zero
// combinations, and the rank boundaries so the rejection contract is not
// represented by only one spatial-zero example.
TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_empty_tensor_full_axis_space_013)
{
    struct EmptyCase {
        const char* desc;
        std::vector<int64_t> xDims;
        std::vector<int64_t> outputDims;
        ge::Format format;
    };
    const std::vector<EmptyCase> cases = {
        {"rank1_zero_invalid_rank", {0}, {0}, ge::FORMAT_ND},
        {"rank2_N0", {0, 3}, {0, 3}, ge::FORMAT_ND},
        {"rank2_C0", {2, 0}, {2, 0}, ge::FORMAT_ND},
        {"rank2_NC0", {0, 0}, {0, 0}, ge::FORMAT_ND},
        {"nchw_N0", {0, 3, 4, 5}, {0, 3, 1, 1}, ge::FORMAT_NCHW},
        {"nchw_C0", {2, 0, 4, 5}, {2, 0, 1, 1}, ge::FORMAT_NCHW},
        {"nchw_H0", {2, 3, 0, 5}, {2, 3, 1, 1}, ge::FORMAT_NCHW},
        {"nchw_W0", {2, 3, 4, 0}, {2, 3, 1, 1}, ge::FORMAT_NCHW},
        {"nchw_multi_zero", {0, 3, 0, 5}, {0, 3, 1, 1}, ge::FORMAT_NCHW},
        {"ncdhw_D0", {2, 3, 0, 4, 5}, {2, 3, 1, 1, 1}, ge::FORMAT_NCDHW},
        {"ncdhw_H0", {2, 3, 4, 0, 5}, {2, 3, 1, 1, 1}, ge::FORMAT_NCDHW},
        {"ncdhw_W0", {2, 3, 4, 5, 0}, {2, 3, 1, 1, 1}, ge::FORMAT_NCDHW},
        {"ndhwc_spatial_zero", {2, 0, 0, 0, 7}, {2, 1, 1, 1, 7}, ge::FORMAT_NDHWC},
        {"ndhwc_NC_zero", {0, 3, 5, 7, 0}, {0, 1, 1, 1, 0}, ge::FORMAT_NDHWC},
        {"ndhwc_all_zero", {0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, ge::FORMAT_NDHWC},
        {"rank8_last_axis0", {2, 3, 1, 1, 1, 1, 1, 0}, {2, 3, 1, 1, 1, 1, 1, 1}, ge::FORMAT_ND},
    };

    for (const auto& emptyCase : cases) {
        gert::StorageShape xShape;
        gert::StorageShape outputShape;
        for (int64_t dim : emptyCase.xDims) {
            xShape.MutableOriginShape().AppendDim(dim);
            xShape.MutableStorageShape().AppendDim(dim);
        }
        for (int64_t dim : emptyCase.outputDims) {
            outputShape.MutableOriginShape().AppendDim(dim);
            outputShape.MutableStorageShape().AppendDim(dim);
        }
        EXPECT_EQ(RunTiling(xShape, outputShape, ge::DT_FLOAT, emptyCase.format), UINT64_MAX) << emptyCase.desc;
    }
}

// Unsupported input/output contracts are rejected by Host rather than reaching the kernel.
TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_unsupported_input_dtype_020)
{
    gert::StorageShape x_shape = {{1, 1, 8, 8}, {1, 1, 8, 8}};
    gert::StorageShape out_shape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    EXPECT_EQ(RunTiling(x_shape, out_shape, ge::DT_DOUBLE, ge::FORMAT_NCHW), UINT64_MAX);
}

TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_non_fp32_sum_output_021)
{
    gert::StorageShape x_shape = {{1, 1, 8, 8}, {1, 1, 8, 8}};
    gert::StorageShape out_shape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    EXPECT_EQ(RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW, nullptr, ge::DT_FLOAT16, ge::DT_FLOAT),
              UINT64_MAX);
}

TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_non_fp32_square_sum_output_022)
{
    gert::StorageShape x_shape = {{1, 1, 8, 8}, {1, 1, 8, 8}};
    gert::StorageShape out_shape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    EXPECT_EQ(RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW, nullptr, ge::DT_FLOAT, ge::DT_FLOAT16),
              UINT64_MAX);
}

TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_rank4_ndhwc_023)
{
    gert::StorageShape x_shape = {{1, 8, 8, 1}, {1, 8, 8, 1}};
    gert::StorageShape out_shape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    EXPECT_EQ(RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NDHWC), UINT64_MAX);
}

TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_invalid_rank_for_format_024)
{
    gert::StorageShape x_shape = {{1, 1, 8}, {1, 1, 8}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    EXPECT_EQ(RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NCHW), UINT64_MAX);
}

TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_negative_spatial_dim_025)
{
    gert::StorageShape x_shape = {{1, 1, -1}, {1, 1, -1}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    EXPECT_EQ(RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND), UINT64_MAX);
}

TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_spatial_product_overflow_026)
{
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    gert::StorageShape x_shape = {{1, 1, kMax, 2}, {1, 1, kMax, 2}};
    gert::StorageShape out_shape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    EXPECT_EQ(RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND), UINT64_MAX);
}

TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_nc_product_overflow_027)
{
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    gert::StorageShape x_shape = {{kMax, 2, 1}, {kMax, 2, 1}};
    gert::StorageShape out_shape = {{kMax, 2, 1}, {kMax, 2, 1}};
    EXPECT_EQ(RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND), UINT64_MAX);
}

TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_total_element_overflow_028)
{
    constexpr int64_t kHalfMax = std::numeric_limits<int64_t>::max() / 2;
    gert::StorageShape x_shape = {{kHalfMax, 1, 3}, {kHalfMax, 1, 3}};
    gert::StorageShape out_shape = {{kHalfMax, 1, 1}, {kHalfMax, 1, 1}};
    EXPECT_EQ(RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND), UINT64_MAX);
}

TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_ub_without_sub_r_input_budget_029)
{
    // usable=640-512=128B，恰好只够 sum/square_sum 的四个 32B queue buffer。
    // Host 必须在做 usable-outReserve-partialReserve 前拒收，不得无符号下溢。
    gert::StorageShape x_shape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    EXPECT_EQ(RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, 640),
              UINT64_MAX);
}

// ---------------------------------------------------------------------------
// sub-R 联合求解：R 大到部分和缓存撑破 usable/8 的初始预留时，rFactor 必须回缩，
// 使 Kernel 侧实际 UB 申请仍落在 UB 内。
// fp32 ND [1,1,104439809]：按老实现 rFactor=26752 / numChunks=3905 → 申请 245888B，
// 超 245760B 的 UB 128B；联合求解后 rFactor 回缩、总占用回到 usable 以内。
//
// ⚠ 这里的 245760 是本 UT compile_info 里那份 UB_SIZE（仓内 arch35 UT 沿用的旧模板，
//   activation/ norm/ 下几十个算子都是这个值），**不是 Ascend950 的真实 UB**——
//   平台 ini 里 ub_size=253952。所以本用例验的是"给定 UB 下联合求解会回缩"这个逻辑，
//   而不是真实芯片上的越界起点（真实起点：fp32 R=109707265、fp16 R=219668481）。
//   UB_SIZE 是全仓级别的清理项，不在本算子范围内改。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_partial_buf_fits_ub_013)
{
    gert::StorageShape x_shape = {{1, 1, 104439809}, {1, 1, 104439809}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U);
    ASSERT_EQ(td.numR, 104439809LL);
    ExpectSubRSelfConsistent(td, sizeof(float));
    // 快路径：单次全折叠塞得进 UB，不分组 —— 参数与分组改造前逐位一致
    ASSERT_EQ(td.numGroups, 1U);
    ASSERT_EQ(td.chunksPerGroup, td.numChunks);
}

// ---------------------------------------------------------------------------
// 分组折叠：R 大到单组放不下时，改由多组折叠承接，而不是拒绝。
// fp32 R=2e9：快路径无解（最优点 rFactor≈sqrt(R) 时总占用远超 UB），落到分组路径。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_grouped_fold_2e9_014)
{
    gert::StorageShape x_shape = {{1, 1, 2000000000}, {1, 1, 2000000000}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U);
    ASSERT_EQ(td.numR, 2000000000LL);
    ExpectSubRSelfConsistent(td, sizeof(float));
    ASSERT_GT(td.numGroups, 1U); // 必须真的走了分组
}

// ---------------------------------------------------------------------------
// 保留旧长链压力 shape：双分量部分和扩大 UB 需求后，这个 R 应走分组路径。
// 不能仍按单分量预算下发原来的单组参数。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_solve_needs_many_iters_016)
{
    gert::StorageShape x_shape = {{1, 1, 233500000}, {1, 1, 233500000}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U); // 不是 UINT64_MAX —— 有解就不能拒绝
    ASSERT_EQ(td.numR, 233500000LL);
    ExpectSubRSelfConsistent(td, sizeof(float));
    ASSERT_GT(td.numGroups, 1U); // 高低位双分量使此 R 必须分组，容量检查仍须成立
}

// ---------------------------------------------------------------------------
// R 超 UINT32_MAX：分组折叠落地后 numR 全程 int64_t，不再有 uint32 上限。
// 这条正是上游反馈（changwei#26）里 InTrainingReduceV2_L1_upboundary_highpre_047
// 用的 R=2^32；改造前被 CheckSubRNarrowable() 拒绝，现在必须能出 tiling。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_r_over_uint32_015)
{
    gert::StorageShape x_shape = {{1, 1, 4294967296L}, {1, 1, 4294967296L}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U);
    ASSERT_EQ(td.numR, 4294967296LL);
    ExpectSubRSelfConsistent(td, sizeof(float));
    ASSERT_GT(td.numGroups, 1U);
}

// ---------------------------------------------------------------------------
// R 远超 UINT32_MAX（2^34，fp16）：验证上限确实解除而不是抬高了一档。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_r_far_over_uint32_fp16_017)
{
    gert::StorageShape x_shape = {{1, 1, 17179869184L}, {1, 1, 17179869184L}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT16, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U);
    ASSERT_EQ(td.numR, 17179869184LL);
    ExpectSubRSelfConsistent(td, sizeof(uint16_t));
    ASSERT_GT(td.numGroups, 1U);
}

// ---------------------------------------------------------------------------
// sub-R 路径的 GM 行号、N/C/R 都是 int64_t。N*C=2^32 时也必须正常生成 tiling，
// 不得因为物理数据规模很大而额外收窄 A2 已有的 shape 契约。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_nc_over_uint32_accepted_018)
{
    gert::StorageShape x_shape = {{65536, 65536, 100000}, {65536, 65536, 100000}};
    gert::StorageShape out_shape = {{65536, 65536, 1}, {65536, 65536, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U);
    ASSERT_EQ(td.isSubRTiling, 1U);
    ASSERT_EQ(td.numN, 65536LL);
    ASSERT_EQ(td.numC, 65536LL);
    ASSERT_EQ(td.numR, 100000LL);
    ExpectSubRSelfConsistent(td, sizeof(float));
}

// ---------------------------------------------------------------------------
// N*C 溢出 uint32 且 R 很小时，full-load 路径同样必须被接受。
// 本例 N=65536, C=65536（N*C=2^32 > UINT32_MAX）、R=4：
//   rAlign = CeilAlign(4*4, 32)/4 = 8
//   cInner = (245760-512) / (8*4*2 + 4*2*2 + 32*2) = 245248/144 = 1703 >= 1
// 故走 full-load；tiling 全程 64 位，无截断。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_nc_over_uint32_small_r_accepted_019)
{
    gert::StorageShape x_shape = {{65536, 65536, 4}, {65536, 65536, 4}};
    gert::StorageShape out_shape = {{65536, 65536, 1}, {65536, 65536, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_NE(key, UINT64_MAX);
    // 走 full-load，不是 sub-R
    ASSERT_EQ(td.isSubRTiling, 0U);
    // N / C / R 原样下发，未被收窄
    ASSERT_EQ(td.numN, 65536LL);
    ASSERT_EQ(td.numC, 65536LL);
    ASSERT_EQ(td.numR, 4LL);
    // N*C 用 64 位乘出来仍然正确（32 位会回绕成 0）
    ASSERT_EQ(td.numN * td.numC, 4294967296LL);
    ASSERT_EQ(td.totalRows, 4294967296LL);
    ASSERT_EQ(td.totalElements, 17179869184LL);
    ASSERT_EQ(td.totalTiles, td.numN * td.cOuter);
}

// ---------------------------------------------------------------------------
// R 到达有符号 64 位 shape 契约边界时，对齐值的字节表示不应反过来
// 收窄逻辑 shape 范围。本用例只构造 Host 元数据，不分配 INT64_MAX 大小的真实 Tensor。
// ---------------------------------------------------------------------------
TEST_F(INTrainingReduceV2TilingTest, tiling_ar_full_reduce_sub_r_int64_r_boundary_030)
{
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    gert::StorageShape x_shape = {{1, 1, kMax}, {1, 1, kMax}};
    gert::StorageShape out_shape = {{1, 1, 1}, {1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    uint64_t key = RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_ND, &td);
    ASSERT_EQ(key, 200000U);
    ASSERT_EQ(td.numR, kMax);
    ASSERT_EQ(td.totalElements, kMax);
    ExpectSubRSelfConsistent(td, sizeof(float));
    ASSERT_GT(td.numGroups, 1U);
}

TEST_F(INTrainingReduceV2TilingTest, nhwc_channel_tails_and_checked_buffers)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        const uint64_t elemBytes = dtype == ge::DT_FLOAT16 ? 2 : 4;
        for (int64_t channels : {3, 63, 64, 65, 127, 128, 129, 257}) {
            gert::StorageShape xShape = {{3, 13, 17, channels}, {3, 13, 17, channels}};
            gert::StorageShape outShape = {{3, 1, 1, channels}, {3, 1, 1, channels}};
            INTrainingReduceV2ARAReduceTilingData td{};
            ASSERT_EQ(
                RunTiling(xShape, outShape, dtype, ge::FORMAT_NHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, 245760, &td),
                400000U);
            EXPECT_EQ(td.numN, 3U);
            EXPECT_EQ(td.numR, 221U);
            EXPECT_EQ(td.numC, channels);
            EXPECT_EQ(td.totalRows, 3U * channels);
            EXPECT_EQ(td.totalElements, 3U * 221U * channels);
            EXPECT_EQ(td.tileCLen, 128U);
            EXPECT_EQ(td.cOuter, (channels + 127U) / 128U);
            EXPECT_EQ(td.tileCTail, channels - (td.cOuter - 1U) * td.tileCLen);
            EXPECT_EQ(td.totalTiles, 3U * td.cOuter);
            EXPECT_GT(td.tilesPerCore, 0U);
            const uint64_t maxChannels = std::min(static_cast<uint64_t>(channels), td.tileCLen);
            const uint64_t rowBytes = (maxChannels * elemBytes + 31U) / 32U * 32U;
            const uint64_t vectorBytes = (maxChannels + 63U) / 64U * 64U * elemBytes;
            EXPECT_GE(td.inputBufferBytes, (td.rFactor - 1U) * rowBytes + vectorBytes);
            EXPECT_EQ(td.outputBufferBytes, td.tileCLen * sizeof(float) * 2U);
            EXPECT_LE(td.inputBufferBytes * 2U + td.outputBufferBytes * 2U + 512U, 245760U);
        }
    }
}

TEST_F(INTrainingReduceV2TilingTest, NhwcSingleChannelReusesContiguousArTiling)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (int64_t rows : {221, 131073}) {
            gert::StorageShape nhwc = {{3, 1, rows, 1}, {3, 1, rows, 1}};
            gert::StorageShape nchw = {{3, 1, 1, rows}, {3, 1, 1, rows}};
            gert::StorageShape output = {{3, 1, 1, 1}, {3, 1, 1, 1}};
            INTrainingReduceV2ARFullReduceTilingData nhwcData{};
            INTrainingReduceV2ARFullReduceTilingData nchwData{};
            ASSERT_EQ(RunTiling(nhwc, output, dtype, ge::FORMAT_NHWC, &nhwcData), 200000U);
            ASSERT_EQ(RunTiling(nchw, output, dtype, ge::FORMAT_NCHW, &nchwData), 200000U);
            EXPECT_EQ(std::memcmp(&nhwcData, &nchwData, sizeof(nhwcData)), 0);
        }
    }
}

TEST_F(INTrainingReduceV2TilingTest, NhwcSmallChannelsUseAvailableInputBudget)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        gert::StorageShape input = {{1, 1, 131073, 7}, {1, 1, 131073, 7}};
        gert::StorageShape output = {{1, 1, 1, 7}, {1, 1, 1, 7}};
        INTrainingReduceV2ARAReduceTilingData td{};
        ASSERT_EQ(RunTiling(input, output, dtype, ge::FORMAT_NHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, 245760, &td),
                  400000U);
        EXPECT_GT(td.rFactor, 3000U);
        EXPECT_LE(td.rFactor, 4095U);
        EXPECT_LE(td.numChunks, 44U);
        EXPECT_LE(td.inputBufferBytes * 2U + td.outputBufferBytes * 2U + 512U, 245760U);
    }
}

TEST_F(INTrainingReduceV2TilingTest, NhwcLastVectorLoadFitsInputAllocation)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        const uint64_t elemBytes = dtype == ge::DT_FLOAT16 ? 2U : 4U;
        for (int64_t channels : {3, 65}) {
            for (int64_t rows : {1, 4097}) {
                gert::StorageShape input = {{1, 1, rows, channels}, {1, 1, rows, channels}};
                gert::StorageShape output = {{1, 1, 1, channels}, {1, 1, 1, channels}};
                INTrainingReduceV2ARAReduceTilingData td{};
                ASSERT_EQ(
                    RunTiling(input, output, dtype, ge::FORMAT_NHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, 245760, &td),
                    400000U);
                const uint64_t currentChannels = std::min(static_cast<uint64_t>(channels), td.tileCLen);
                const uint64_t rowBytes = (currentChannels * elemBytes + 31U) / 32U * 32U;
                const uint64_t vectorBytes = (currentChannels + 63U) / 64U * 64U * elemBytes;
                EXPECT_GE(td.inputBufferBytes, (td.rFactor - 1U) * rowBytes + vectorBytes);
                EXPECT_EQ(td.inputBufferBytes % 32U, 0U);
                EXPECT_LE(td.inputBufferBytes * 2U + td.outputBufferBytes * 2U + 512U, 245760U);
            }
        }
    }
}

TEST_F(INTrainingReduceV2TilingTest, FullArPartitionsIndependentRowsAcrossAvailableCores)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (int64_t batches : {1, 3, 4, 65}) {
            gert::StorageShape input = {{batches, 256, 14, 14}, {batches, 256, 14, 14}};
            gert::StorageShape output = {{batches, 256, 1, 1}, {batches, 256, 1, 1}};
            INTrainingReduceV2ARFullReduceTilingData td{};
            ASSERT_EQ(RunTiling(input, output, dtype, ge::FORMAT_NCHW, &td), EXPECTED_AR_TILING_KEY);
            ASSERT_EQ(td.isSubRTiling, 0U);
            const uint64_t tilesPerInstance = std::max(uint64_t{1}, 64U / static_cast<uint64_t>(batches));
            const uint64_t balancedRows = (256U + tilesPerInstance - 1U) / tilesPerInstance;
            EXPECT_LE(static_cast<uint64_t>(td.cInner), balancedRows);
            EXPECT_EQ(td.totalTiles, batches * td.cOuter);
            EXPECT_EQ(td.cTail, 256 - (td.cOuter - 1) * td.cInner);
            EXPECT_LE(td.inputBufferBytes * 2U + td.outputBufferBytes * 4U + td.scratchBufferBytes * 2U, MOCK_UB_SIZE);
            if (batches == 4) {
                EXPECT_EQ(td.cInner, 16);
                EXPECT_EQ(td.totalTiles, 64);
                EXPECT_EQ(td.perCoreCnt, 1);
            }
        }
    }
}

TEST_F(INTrainingReduceV2TilingTest, LargeArRowsUseCompensatedSubRAtHorizontalFoldBoundary)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        const uint64_t elemBytes = dtype == ge::DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float);
        for (int64_t reduceSize : {16383, 16384, 16385, 32768, 32769}) {
            for (int64_t channels : {1, 128}) {
                gert::StorageShape input = {{1, channels, 1, reduceSize}, {1, channels, 1, reduceSize}};
                gert::StorageShape output = {{1, channels, 1, 1}, {1, channels, 1, 1}};
                INTrainingReduceV2ARFullReduceTilingData td{};
                ASSERT_EQ(RunTiling(input, output, dtype, ge::FORMAT_NCHW, &td), EXPECTED_AR_TILING_KEY);
                if (reduceSize <= 16384) {
                    EXPECT_EQ(td.isSubRTiling, 0U);
                } else {
                    ExpectSubRSelfConsistent(td, elemBytes);
                    EXPECT_EQ(td.numGroups, 1U);
                    EXPECT_EQ(td.totalRows, channels);
                    EXPECT_EQ(td.totalElements, channels * reduceSize);
                }
            }
        }
    }
}

TEST_F(INTrainingReduceV2TilingTest, AraParallelChannelTilesPreserveCorrectionPlanesAndReadWidth)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        const uint64_t elemBytes = dtype == ge::DT_FLOAT16 ? 2U : 4U;
        for (auto format : {ge::FORMAT_NHWC, ge::FORMAT_NDHWC}) {
            for (int64_t rows : {1023, 1024, 65537}) {
                for (int64_t channels : {3, 7, 16, 65, 129, 257}) {
                    gert::StorageShape input = format == ge::FORMAT_NHWC ?
                                                   gert::StorageShape({1, 1, rows, channels}, {1, 1, rows, channels}) :
                                                   gert::StorageShape({1, 1, 1, rows, channels},
                                                                      {1, 1, 1, rows, channels});
                    gert::StorageShape output = format == ge::FORMAT_NHWC ?
                                                    gert::StorageShape({1, 1, 1, channels}, {1, 1, 1, channels}) :
                                                    gert::StorageShape({1, 1, 1, 1, channels}, {1, 1, 1, 1, channels});
                    INTrainingReduceV2ARAReduceTilingData td{};
                    ASSERT_EQ(
                        RunTiling(input, output, dtype, format, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, MOCK_UB_SIZE, &td),
                        EXPECTED_ARA_TILING_KEY);
                    const uint64_t expectedTile = rows < 1024 ? 128U :
                                                                std::min(uint64_t{128},
                                                                         (static_cast<uint64_t>(channels) + 63U) / 64U);
                    EXPECT_EQ(td.tileCLen, expectedTile);
                    EXPECT_EQ(td.cOuter, (channels + td.tileCLen - 1U) / td.tileCLen);
                    EXPECT_EQ(td.tileCTail, channels - (td.cOuter - 1U) * td.tileCLen);
                    EXPECT_EQ(td.totalTiles, td.cOuter);
                    // Init/finalize use the fixed low-plane offset128 even for tileCLen=1.
                    EXPECT_EQ(td.outputBufferBytes, 128U * sizeof(float) * 2U);
                    const uint64_t maxChannels = std::min(static_cast<uint64_t>(channels), td.tileCLen);
                    const uint64_t rowBytes = (maxChannels * elemBytes + 31U) / 32U * 32U;
                    const uint64_t vectorBytes = (maxChannels + 63U) / 64U * 64U * elemBytes;
                    EXPECT_GE(td.inputBufferBytes, (td.rFactor - 1U) * rowBytes + vectorBytes);
                    EXPECT_LE(td.inputBufferBytes * 2U + td.outputBufferBytes * 2U + 512U, MOCK_UB_SIZE);
                }
            }
        }
    }
}

TEST_F(INTrainingReduceV2TilingTest, SmallChannelGatherReservesCompleteContiguousRows)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        const uint64_t elemBytes = dtype == ge::DT_FLOAT16 ? 2U : 4U;
        for (auto format : {ge::FORMAT_NHWC, ge::FORMAT_NDHWC}) {
            for (int64_t batches : {1, 28}) {
                for (int64_t channels : {7, 9, 15, 16, 17}) {
                    constexpr int64_t rows = 65537;
                    gert::StorageShape input = format == ge::FORMAT_NHWC ?
                                                   gert::StorageShape({batches, 1, rows, channels},
                                                                      {batches, 1, rows, channels}) :
                                                   gert::StorageShape({batches, 1, 1, rows, channels},
                                                                      {batches, 1, 1, rows, channels});
                    gert::StorageShape output = format == ge::FORMAT_NHWC ?
                                                    gert::StorageShape({batches, 1, 1, channels},
                                                                       {batches, 1, 1, channels}) :
                                                    gert::StorageShape({batches, 1, 1, 1, channels},
                                                                       {batches, 1, 1, 1, channels});
                    INTrainingReduceV2ARAReduceTilingData td{};
                    ASSERT_EQ(
                        RunTiling(input, output, dtype, format, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, MOCK_UB_SIZE, &td),
                        EXPECTED_ARA_TILING_KEY);
                    const uint64_t tileChannels = std::min(static_cast<uint64_t>(channels), td.tileCLen);
                    const uint64_t copiedChannels = channels <= 16 && tileChannels <= 8 ?
                                                        static_cast<uint64_t>(channels) :
                                                        tileChannels;
                    const uint64_t rowBytes = (copiedChannels * elemBytes + 31U) / 32U * 32U;
                    const uint64_t vectorBytes = (copiedChannels + 63U) / 64U * 64U * elemBytes;
                    EXPECT_EQ(td.inputBufferBytes, td.rFactor * rowBytes + vectorBytes - rowBytes);
                    EXPECT_GE(td.inputBufferBytes, td.rFactor * copiedChannels * elemBytes);
                    EXPECT_LE(td.rFactor, 4095U);
                    EXPECT_LE(td.inputBufferBytes * 2U + td.outputBufferBytes * 2U + 512U, MOCK_UB_SIZE);
                }
            }
        }
    }
}

TEST_F(INTrainingReduceV2TilingTest, nhwc_large_r_uses_bounded_chunk_buffers)
{
    gert::StorageShape xShape = {{2, 1, 262145, 129}, {2, 1, 262145, 129}};
    gert::StorageShape outShape = {{2, 1, 1, 129}, {2, 1, 1, 129}};
    INTrainingReduceV2ARAReduceTilingData td{};
    ASSERT_EQ(
        RunTiling(xShape, outShape, ge::DT_FLOAT, ge::FORMAT_NHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, 245760, &td),
        400000U);
    EXPECT_GT(td.numChunks, 1U);
    EXPECT_EQ(td.numChunks, (td.numR + td.rFactor - 1U) / td.rFactor);
    EXPECT_GT(td.numR - (td.numChunks - 1U) * td.rFactor, 0U);
    EXPECT_LE(td.rFactor, 4095U);
    EXPECT_LE(td.inputBufferBytes * 2U + td.outputBufferBytes * 2U + 512U, 245760U);
}

TEST_F(INTrainingReduceV2TilingTest, nhwc_r_one_and_multicore_tile_tail)
{
    gert::StorageShape xShape = {{65, 1, 1, 129}, {65, 1, 1, 129}};
    gert::StorageShape outShape = xShape;
    INTrainingReduceV2ARAReduceTilingData td{};
    ASSERT_EQ(
        RunTiling(xShape, outShape, ge::DT_FLOAT16, ge::FORMAT_NHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, 245760, &td),
        400000U);
    EXPECT_EQ(td.rFactor, 1U);
    EXPECT_EQ(td.numChunks, 1U);
    EXPECT_EQ(td.totalTiles, 130U);
    EXPECT_EQ(td.tilesPerCore, 3U);
}

TEST_F(INTrainingReduceV2TilingTest, nhwc_stride_retains_more_than_32_bits)
{
    constexpr int64_t channels = (1LL << 32) + 129;
    gert::StorageShape xShape = {{1, 1, 2, channels}, {1, 1, 2, channels}};
    gert::StorageShape outShape = {{1, 1, 1, channels}, {1, 1, 1, channels}};
    INTrainingReduceV2ARAReduceTilingData td{};
    ASSERT_EQ(
        RunTiling(xShape, outShape, ge::DT_FLOAT, ge::FORMAT_NHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, 245760, &td),
        400000U);
    EXPECT_GT(CalcINTrainingReduceV2ARASrcStrideBytes(channels, td.tileCLen, sizeof(float)),
              std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(CalcINTrainingReduceV2ARASrcStrideBytes(channels, td.tileCLen, sizeof(float)),
              (channels - static_cast<int64_t>(td.tileCLen)) * static_cast<int64_t>(sizeof(float)));
}

TEST_F(INTrainingReduceV2TilingTest, nhwc_rejects_stride_beyond_40_bits)
{
    constexpr int64_t channels = 1LL << 38;
    gert::StorageShape xShape = {{1, 1, 2, channels}, {1, 1, 2, channels}};
    gert::StorageShape outShape = {{1, 1, 1, channels}, {1, 1, 1, channels}};
    EXPECT_EQ(RunTiling(xShape, outShape, ge::DT_FLOAT, ge::FORMAT_NHWC), UINT64_MAX);
}

TEST_F(INTrainingReduceV2TilingTest, nhwc_rejects_wrong_rank_negative_and_empty_axes)
{
    std::vector<gert::StorageShape> invalidShapes = {
        {{1, 2, 3}, {1, 2, 3}},         {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, {{0, 3, 5, 7}, {0, 3, 5, 7}},
        {{1, 0, 5, 7}, {1, 0, 5, 7}},   {{1, 3, 0, 7}, {1, 3, 0, 7}},       {{1, 3, 5, 0}, {1, 3, 5, 0}},
        {{1, -1, 5, 7}, {1, -1, 5, 7}}, {{0, 0, 5, 7}, {0, 0, 5, 7}},       {{1, 0, 0, 7}, {1, 0, 0, 7}},
    };
    for (auto& xShape : invalidShapes) {
        gert::StorageShape outShape = {{1, 1, 1, 7}, {1, 1, 1, 7}};
        if (xShape.GetStorageShape().GetDimNum() == 4U) {
            const int64_t n = xShape.GetStorageShape().GetDim(0);
            const int64_t c = xShape.GetStorageShape().GetDim(3);
            outShape = {{n, 1, 1, c}, {n, 1, 1, c}};
        }
        ge::graphStatus status = ge::GRAPH_SUCCESS;
        EXPECT_EQ(RunTiling(xShape, outShape, ge::DT_FLOAT, ge::FORMAT_NHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                            245760, nullptr, nullptr, &status),
                  UINT64_MAX);
        EXPECT_EQ(status, ge::GRAPH_FAILED);
    }
}

TEST_F(INTrainingReduceV2TilingTest, nhwc_rejects_shape_products_overflow)
{
    constexpr int64_t max = std::numeric_limits<int64_t>::max();
    std::vector<gert::StorageShape> invalidShapes = {
        {{1, max, 2, 1}, {1, max, 2, 1}},
        {{max, 1, 1, 2}, {max, 1, 1, 2}},
        {{1, max / 2, 3, 1}, {1, max / 2, 3, 1}},
    };
    gert::StorageShape outShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    for (auto& xShape : invalidShapes) {
        EXPECT_EQ(RunTiling(xShape, outShape, ge::DT_FLOAT, ge::FORMAT_NHWC), UINT64_MAX);
    }
}

TEST_F(INTrainingReduceV2TilingTest, nhwc_rejects_insufficient_ub_and_output_dtype)
{
    gert::StorageShape xShape = {{1, 3, 5, 7}, {1, 3, 5, 7}};
    gert::StorageShape outShape = {{1, 1, 1, 7}, {1, 1, 1, 7}};
    EXPECT_EQ(RunTiling(xShape, outShape, ge::DT_FLOAT, ge::FORMAT_NHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, 1536),
              UINT64_MAX);
    EXPECT_EQ(RunTiling(xShape, outShape, ge::DT_FLOAT, ge::FORMAT_NHWC, nullptr, ge::DT_FLOAT16, ge::DT_FLOAT),
              UINT64_MAX);
}

TEST_F(INTrainingReduceV2TilingTest, nhwc_dma_block_count_is_capped_with_large_ub)
{
    gert::StorageShape xShape = {{1, 1, 65537, 128}, {1, 1, 65537, 128}};
    gert::StorageShape outShape = {{1, 1, 1, 128}, {1, 1, 1, 128}};
    INTrainingReduceV2ARAReduceTilingData td{};
    ASSERT_EQ(RunTiling(xShape, outShape, ge::DT_FLOAT16, ge::FORMAT_NHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        64U * 1024U * 1024U, &td),
              400000U);
    EXPECT_EQ(td.rFactor, 4095U);
    EXPECT_GT(td.numChunks, 1U);
}

TEST_F(INTrainingReduceV2TilingTest, NullTilingContextReturnsFailure)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingReduceV2");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    EXPECT_EQ(impl->tiling(nullptr), ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, NullTilingParseContextReturnsFailure)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingReduceV2");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling_parse, nullptr);
    EXPECT_EQ(impl->tiling_parse(nullptr), ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, NullPlatformReturnsFailureBeforeArchitectureQuery)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingReduceV2");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    gert::StorageShape input = {{1, 7, 1, 65}, {1, 7, 1, 65}};
    gert::StorageShape output = {{1, 7, 1, 1}, {1, 7, 1, 1}};
    fe::PlatFormInfos platformInfo;
    ASSERT_TRUE(platformInfo.Init());
    optiling::INTrainingReduceV2CompileInfo compileInfo;
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    ASSERT_NE(tilingData, nullptr);
    ASSERT_NE(workspace, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("INTrainingReduceV2")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&input})
                      .OutputShapes({&output, &output})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(&platformInfo)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    // The SDK builder requires a platform; inject its absence into the completed context.
    const auto* nodeInfo = context->GetComputeNodeInfo();
    ASSERT_NE(nodeInfo, nullptr);
    auto* kernelContext = holder.GetContext<gert::KernelContext>();
    ASSERT_NE(kernelContext, nullptr);
    auto* platformSlot = kernelContext->MutableInput(nodeInfo->GetInputsNum() + nodeInfo->GetOutputsNum() + 1U);
    ASSERT_NE(platformSlot, nullptr);
    platformSlot->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(impl->tiling(context), ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ArLastVectorLoadFitsInputAllocation)
{
    constexpr uint64_t kVectorElementsFp32 = 64;
    constexpr uint64_t kUbBlockBytes = 32;
    constexpr uint64_t kDoubleBufferCount = 2;
    constexpr uint64_t kOutputCount = 2;
    constexpr uint64_t kScratchCount = 2;
    constexpr uint64_t kPlatformUbBytes = 245760;
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        const uint64_t elemSize = dtype == ge::DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float);
        for (int64_t channels : {1, 7}) {
            for (int64_t reduceSize : {1, 7, 63, 64, 65, 127, 129}) {
                gert::StorageShape xShape = {{1, channels, 1, reduceSize}, {1, channels, 1, reduceSize}};
                gert::StorageShape outShape = {{1, channels, 1, 1}, {1, channels, 1, 1}};
                INTrainingReduceV2ARFullReduceTilingData td{};
                ASSERT_EQ(RunTiling(xShape, outShape, dtype, ge::FORMAT_NCHW, &td), 200000U);
                ASSERT_EQ(td.isSubRTiling, 0U);
                const uint64_t finalReadEnd = ((td.cInner - 1U) * td.rAlign +
                                               (static_cast<uint64_t>(reduceSize) + kVectorElementsFp32 - 1U) /
                                                   kVectorElementsFp32 * kVectorElementsFp32) *
                                              elemSize;
                EXPECT_GE(td.inputBufferBytes, finalReadEnd);
                EXPECT_EQ(td.inputBufferBytes % kUbBlockBytes, 0U);
                EXPECT_LE(td.inputBufferBytes * kDoubleBufferCount +
                              td.outputBufferBytes * kDoubleBufferCount * kOutputCount +
                              td.scratchBufferBytes * kScratchCount,
                          kPlatformUbBytes);
            }
        }
    }
}

TEST_F(INTrainingReduceV2TilingTest, RejectsOutputShapesThatDoNotMatchReduction)
{
    for (auto format : {ge::FORMAT_NCHW, ge::FORMAT_NHWC}) {
        const bool isNhwc = format == ge::FORMAT_NHWC;
        gert::StorageShape xShape = isNhwc ? gert::StorageShape({2, 7, 9, 3}, {2, 7, 9, 3}) :
                                             gert::StorageShape({2, 3, 7, 9}, {2, 3, 7, 9});
        gert::StorageShape validShape = isNhwc ? gert::StorageShape({2, 1, 1, 3}, {2, 1, 1, 3}) :
                                                 gert::StorageShape({2, 3, 1, 1}, {2, 3, 1, 1});
        for (const auto& dims : {std::vector<int64_t>{1, 1, 1, 1}, std::vector<int64_t>{2, 3},
                                 std::vector<int64_t>{2, 3, 7, 9}, std::vector<int64_t>{0, 0, 0, 0}}) {
            gert::StorageShape invalidShape;
            for (auto dim : dims) {
                invalidShape.MutableStorageShape().AppendDim(dim);
                invalidShape.MutableOriginShape().AppendDim(dim);
            }
            EXPECT_EQ(RunTiling(xShape, invalidShape, ge::DT_FLOAT, format), UINT64_MAX);
            EXPECT_EQ(RunTiling(xShape, validShape, ge::DT_FLOAT, format, nullptr, ge::DT_FLOAT, ge::DT_FLOAT, 245760U,
                                nullptr, &invalidShape),
                      UINT64_MAX);
        }
    }
}

TEST_F(INTrainingReduceV2TilingTest, RejectsZeroDimensionalInput)
{
    gert::StorageShape scalarShape = {{}, {}};
    gert::StorageShape outShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    EXPECT_EQ(RunTiling(scalarShape, outShape, ge::DT_FLOAT, ge::FORMAT_NCHW), UINT64_MAX);
    EXPECT_EQ(RunTiling(scalarShape, outShape, ge::DT_FLOAT, ge::FORMAT_NHWC), UINT64_MAX);
    EXPECT_EQ(RunTiling(scalarShape, outShape, ge::DT_FLOAT, ge::FORMAT_ND), UINT64_MAX);
}

// NDHWC supplementary contract: rank-5 layout, both outputs and host safety bounds.

TEST_F(INTrainingReduceV2TilingTest, ndhwc_depth_spatial_dt_float16)
{
    gert::StorageShape inputShape = {{2, 3, 5, 7, 129}, {2, 3, 5, 7, 129}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 129}, {2, 1, 1, 1, 129}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT16, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 2UL);
    EXPECT_EQ(td.numC, 129UL);
    EXPECT_EQ(td.numR, 105UL);
    EXPECT_EQ(td.totalElements, 27090UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_depth_spatial_dt_float)
{
    gert::StorageShape inputShape = {{2, 3, 5, 7, 129}, {2, 3, 5, 7, 129}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 129}, {2, 1, 1, 1, 129}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 2UL);
    EXPECT_EQ(td.numC, 129UL);
    EXPECT_EQ(td.numR, 105UL);
    EXPECT_EQ(td.totalElements, 27090UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_small_c_gather_dt_float16)
{
    gert::StorageShape inputShape = {{3, 1, 1, 65, 7}, {3, 1, 1, 65, 7}};
    gert::StorageShape outputShape = {{3, 1, 1, 1, 7}, {3, 1, 1, 1, 7}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT16, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 3UL);
    EXPECT_EQ(td.numC, 7UL);
    EXPECT_EQ(td.numR, 65UL);
    EXPECT_EQ(td.totalElements, 1365UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_small_c_gather_dt_float)
{
    gert::StorageShape inputShape = {{3, 1, 1, 65, 7}, {3, 1, 1, 65, 7}};
    gert::StorageShape outputShape = {{3, 1, 1, 1, 7}, {3, 1, 1, 1, 7}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 3UL);
    EXPECT_EQ(td.numC, 7UL);
    EXPECT_EQ(td.numR, 65UL);
    EXPECT_EQ(td.totalElements, 1365UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_c128_aligned_dt_float16)
{
    gert::StorageShape inputShape = {{2, 1, 31, 33, 128}, {2, 1, 31, 33, 128}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 128}, {2, 1, 1, 1, 128}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT16, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 2UL);
    EXPECT_EQ(td.numC, 128UL);
    EXPECT_EQ(td.numR, 1023UL);
    EXPECT_EQ(td.totalElements, 261888UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_c128_aligned_dt_float)
{
    gert::StorageShape inputShape = {{2, 1, 31, 33, 128}, {2, 1, 31, 33, 128}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 128}, {2, 1, 1, 1, 128}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 2UL);
    EXPECT_EQ(td.numC, 128UL);
    EXPECT_EQ(td.numR, 1023UL);
    EXPECT_EQ(td.totalElements, 261888UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_c129_tail_dt_float16)
{
    gert::StorageShape inputShape = {{2, 1, 31, 33, 129}, {2, 1, 31, 33, 129}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 129}, {2, 1, 1, 1, 129}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT16, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 2UL);
    EXPECT_EQ(td.numC, 129UL);
    EXPECT_EQ(td.numR, 1023UL);
    EXPECT_EQ(td.totalElements, 263934UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_c129_tail_dt_float)
{
    gert::StorageShape inputShape = {{2, 1, 31, 33, 129}, {2, 1, 31, 33, 129}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 129}, {2, 1, 1, 1, 129}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 2UL);
    EXPECT_EQ(td.numC, 129UL);
    EXPECT_EQ(td.numR, 1023UL);
    EXPECT_EQ(td.totalElements, 263934UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_multi_chunk_dt_float16)
{
    gert::StorageShape inputShape = {{1, 17, 31, 33, 9}, {1, 17, 31, 33, 9}};
    gert::StorageShape outputShape = {{1, 1, 1, 1, 9}, {1, 1, 1, 1, 9}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT16, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 1UL);
    EXPECT_EQ(td.numC, 9UL);
    EXPECT_EQ(td.numR, 17391UL);
    EXPECT_EQ(td.totalElements, 156519UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_multi_chunk_dt_float)
{
    gert::StorageShape inputShape = {{1, 17, 31, 33, 9}, {1, 17, 31, 33, 9}};
    gert::StorageShape outputShape = {{1, 1, 1, 1, 9}, {1, 1, 1, 1, 9}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 1UL);
    EXPECT_EQ(td.numC, 9UL);
    EXPECT_EQ(td.numR, 17391UL);
    EXPECT_EQ(td.totalElements, 156519UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_r1_identity_dt_float16)
{
    gert::StorageShape inputShape = {{3, 1, 1, 1, 3}, {3, 1, 1, 1, 3}};
    gert::StorageShape outputShape = {{3, 1, 1, 1, 3}, {3, 1, 1, 1, 3}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT16, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 3UL);
    EXPECT_EQ(td.numC, 3UL);
    EXPECT_EQ(td.numR, 1UL);
    EXPECT_EQ(td.totalElements, 9UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_r1_identity_dt_float)
{
    gert::StorageShape inputShape = {{3, 1, 1, 1, 3}, {3, 1, 1, 1, 3}};
    gert::StorageShape outputShape = {{3, 1, 1, 1, 3}, {3, 1, 1, 1, 3}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 3UL);
    EXPECT_EQ(td.numC, 3UL);
    EXPECT_EQ(td.numR, 1UL);
    EXPECT_EQ(td.totalElements, 9UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_c1_ar_dt_float16)
{
    gert::StorageShape inputShape = {{3, 2, 3, 5, 1}, {3, 2, 3, 5, 1}};
    gert::StorageShape outputShape = {{3, 1, 1, 1, 1}, {3, 1, 1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT16, ge::FORMAT_NDHWC, &td, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE);
    ASSERT_EQ(key, EXPECTED_AR_TILING_KEY);
    EXPECT_EQ(td.numN, 3UL);
    EXPECT_EQ(td.numC, 1UL);
    EXPECT_EQ(td.numR, 30UL);
    EXPECT_EQ(td.totalElements, 90UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_c1_ar_dt_float)
{
    gert::StorageShape inputShape = {{3, 2, 3, 5, 1}, {3, 2, 3, 5, 1}};
    gert::StorageShape outputShape = {{3, 1, 1, 1, 1}, {3, 1, 1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, &td, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE);
    ASSERT_EQ(key, EXPECTED_AR_TILING_KEY);
    EXPECT_EQ(td.numN, 3UL);
    EXPECT_EQ(td.numC, 1UL);
    EXPECT_EQ(td.numR, 30UL);
    EXPECT_EQ(td.totalElements, 90UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_c1_sub_r_dt_float16)
{
    gert::StorageShape inputShape = {{1, 1, 1, 262145, 1}, {1, 1, 1, 262145, 1}};
    gert::StorageShape outputShape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT16, ge::FORMAT_NDHWC, &td, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE);
    ASSERT_EQ(key, EXPECTED_AR_TILING_KEY);
    EXPECT_EQ(td.numN, 1UL);
    EXPECT_EQ(td.numC, 1UL);
    EXPECT_EQ(td.numR, 262145UL);
    EXPECT_EQ(td.totalElements, 262145UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_c1_sub_r_dt_float)
{
    gert::StorageShape inputShape = {{1, 1, 1, 262145, 1}, {1, 1, 1, 262145, 1}};
    gert::StorageShape outputShape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    INTrainingReduceV2ARFullReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, &td, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE);
    ASSERT_EQ(key, EXPECTED_AR_TILING_KEY);
    EXPECT_EQ(td.numN, 1UL);
    EXPECT_EQ(td.numC, 1UL);
    EXPECT_EQ(td.numR, 262145UL);
    EXPECT_EQ(td.totalElements, 262145UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_spatial_exceeds_uint32)
{
    gert::StorageShape inputShape = {{1, 1, 1, 4294967297, 3}, {1, 1, 1, 4294967297, 3}};
    gert::StorageShape outputShape = {{1, 1, 1, 1, 3}, {1, 1, 1, 1, 3}};
    INTrainingReduceV2ARAReduceTilingData td{};
    const uint64_t key = RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT,
                                   ge::DT_FLOAT, MOCK_UB_SIZE, &td);
    ASSERT_EQ(key, EXPECTED_ARA_TILING_KEY);
    EXPECT_EQ(td.numN, 1UL);
    EXPECT_EQ(td.numC, 3UL);
    EXPECT_EQ(td.numR, 4294967297UL);
    EXPECT_EQ(td.totalElements, 12884901891UL);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_zero_n)
{
    gert::StorageShape inputShape = {{0, 3, 5, 7, 9}, {0, 3, 5, 7, 9}};
    gert::StorageShape outputShape = {{0, 1, 1, 1, 9}, {0, 1, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_zero_d)
{
    gert::StorageShape inputShape = {{2, 0, 5, 7, 9}, {2, 0, 5, 7, 9}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 9}, {2, 1, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_zero_h)
{
    gert::StorageShape inputShape = {{2, 3, 0, 7, 9}, {2, 3, 0, 7, 9}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 9}, {2, 1, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_zero_w)
{
    gert::StorageShape inputShape = {{2, 3, 5, 0, 9}, {2, 3, 5, 0, 9}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 9}, {2, 1, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_zero_c)
{
    gert::StorageShape inputShape = {{2, 3, 5, 7, 0}, {2, 3, 5, 7, 0}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 0}, {2, 1, 1, 1, 0}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_negative_depth)
{
    gert::StorageShape inputShape = {{2, -1, 5, 7, 9}, {2, -1, 5, 7, 9}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 9}, {2, 1, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_rank6)
{
    gert::StorageShape inputShape = {{1, 2, 3, 5, 7, 9}, {1, 2, 3, 5, 7, 9}};
    gert::StorageShape outputShape = {{1, 1, 1, 1, 1, 9}, {1, 1, 1, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_sum_channel_axis)
{
    gert::StorageShape inputShape = {{2, 3, 5, 7, 9}, {2, 3, 5, 7, 9}};
    gert::StorageShape outputShape = {{2, 9, 1, 1, 1}, {2, 9, 1, 1, 1}};
    gert::StorageShape squareShape = {{2, 1, 1, 1, 9}, {2, 1, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, &squareShape, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_square_sum_channel_axis)
{
    gert::StorageShape inputShape = {{2, 3, 5, 7, 9}, {2, 3, 5, 7, 9}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 9}, {2, 1, 1, 1, 9}};
    gert::StorageShape squareShape = {{2, 9, 1, 1, 1}, {2, 9, 1, 1, 1}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, &squareShape, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_sum_wrong_depth)
{
    gert::StorageShape inputShape = {{2, 3, 5, 7, 9}, {2, 3, 5, 7, 9}};
    gert::StorageShape outputShape = {{2, 3, 1, 1, 9}, {2, 3, 1, 1, 9}};
    gert::StorageShape squareShape = {{2, 1, 1, 1, 9}, {2, 1, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, &squareShape, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_square_sum_wrong_rank)
{
    gert::StorageShape inputShape = {{2, 3, 5, 7, 9}, {2, 3, 5, 7, 9}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 9}, {2, 1, 1, 1, 9}};
    gert::StorageShape squareShape = {{2, 1, 1, 9}, {2, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, &squareShape, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_fp64_input)
{
    gert::StorageShape inputShape = {{2, 3, 5, 7, 9}, {2, 3, 5, 7, 9}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 9}, {2, 1, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_DOUBLE, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_fp16_sum)
{
    gert::StorageShape inputShape = {{2, 3, 5, 7, 9}, {2, 3, 5, 7, 9}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 9}, {2, 1, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT16, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_fp16_square_sum)
{
    gert::StorageShape inputShape = {{2, 3, 5, 7, 9}, {2, 3, 5, 7, 9}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 9}, {2, 1, 1, 1, 9}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT16,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_dhw_int64_overflow)
{
    gert::StorageShape inputShape = {{1, 4611686018427387904, 2, 1, 1}, {1, 4611686018427387904, 2, 1, 1}};
    gert::StorageShape outputShape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_total_int64_overflow)
{
    gert::StorageShape inputShape = {{2, 1, 1, 4611686018427387904, 1}, {2, 1, 1, 4611686018427387904, 1}};
    gert::StorageShape outputShape = {{2, 1, 1, 1, 1}, {2, 1, 1, 1, 1}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_nc_int64_overflow)
{
    gert::StorageShape inputShape = {{4611686018427387904, 1, 1, 1, 2}, {4611686018427387904, 1, 1, 1, 2}};
    gert::StorageShape outputShape = {{4611686018427387904, 1, 1, 1, 2}, {4611686018427387904, 1, 1, 1, 2}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_stride_overflow_dt_float16)
{
    gert::StorageShape inputShape = {{1, 1, 1, 2, 549755813888}, {1, 1, 1, 2, 549755813888}};
    gert::StorageShape outputShape = {{1, 1, 1, 1, 549755813888}, {1, 1, 1, 1, 549755813888}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT16, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(INTrainingReduceV2TilingTest, ndhwc_rejects_stride_overflow_dt_float)
{
    gert::StorageShape inputShape = {{1, 1, 1, 2, 274877906944}, {1, 1, 1, 2, 274877906944}};
    gert::StorageShape outputShape = {{1, 1, 1, 1, 274877906944}, {1, 1, 1, 1, 274877906944}};
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    EXPECT_EQ(RunTiling(inputShape, outputShape, ge::DT_FLOAT, ge::FORMAT_NDHWC, nullptr, ge::DT_FLOAT, ge::DT_FLOAT,
                        MOCK_UB_SIZE, nullptr, nullptr, &status),
              UINT64_MAX);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
