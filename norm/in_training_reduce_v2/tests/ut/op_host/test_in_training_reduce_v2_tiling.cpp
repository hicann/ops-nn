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
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "test_in_training_reduce_v2_tiling.h"
#include "../../../op_kernel/arch35/in_training_reduce_v2_tiling_data.h"
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
                          uint64_t ubSize = 245760)
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
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    if (!kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init()) {
        return UINT64_MAX;
    }
    auto parseCtx = kernelHolder.GetContext<gert::TilingParseContext>();
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
    if (tilingData == nullptr) {
        return UINT64_MAX;
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outShape, &outShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dtype, fmt, fmt)
                      .NodeOutputTd(0, sumDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, squareSumDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext->GetPlatformInfo() == nullptr) {
        return UINT64_MAX;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    if (tilingFunc(tilingContext) != ge::GRAPH_SUCCESS) {
        return UINT64_MAX;
    }
    if (tdOut != nullptr) {
        auto raw = tilingContext->GetRawTilingData();
        if (raw == nullptr || raw->GetDataSize() < sizeof(*tdOut)) {
            return UINT64_MAX;
        }
        *tdOut = *reinterpret_cast<const INTrainingReduceV2ARFullReduceTilingData*>(raw->GetData());
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
    uint64_t partialBytes = ceilAlign(slots, kVlFp32) * sizeof(float) * 2;
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
    ASSERT_EQ(td.partialBufferBytes, (slots + kVlFp32 - 1U) / kVlFp32 * kVlFp32 * sizeof(float));
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

TEST_F(INTrainingReduceV2TilingTest, tiling_rejects_unsupported_format_023)
{
    gert::StorageShape x_shape = {{1, 8, 8, 1}, {1, 8, 8, 1}};
    gert::StorageShape out_shape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    EXPECT_EQ(RunTiling(x_shape, out_shape, ge::DT_FLOAT, ge::FORMAT_NHWC), UINT64_MAX);
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
// sub-R 联合求解的**迭代轮数**：R 逼近 UB 容量上限时，rFactor 已被部分和缓存挤得很小，
// 每轮只能再缩一点点，收敛轮数急剧上升（这里需要 20 轮）。SUB_R_SOLVE_MAX_ITER 原为 8，
// 落在这一段的 R 明明有可行解却会被判成 "cannot fit UB" 并拒绝下发 —— 失败方向安全，
// 但仍是错判。本用例锁死"有解就必须求出来"，上限调回 8 时它会失败。
// fp32 ND [1,1,233500000]：收敛解 rFactor=15936 / numChunks=14653 → 244864B ≤ 245248B。
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
    ASSERT_EQ(td.numGroups, 1U); // 仍在快路径内，不应被分组抢走
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
