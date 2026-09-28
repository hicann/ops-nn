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
 * \file test_wts_arq_tiling.cpp
 * \brief WtsARQ tiling UT (arch35 / ascend950).
 *
 * Coverage (DESIGN.md §3.4):
 *   - template dispatch: shapeLen <= 4 -> RANK=4 (tilingKey 0), 5~8 -> RANK=8 (tilingKey 1)
 *   - OneDim special case (shapeLen == 1): scalar broadcast / no broadcast / rank0 scalar
 *   - brcMode decision chain: NONE(0) / NDDMA(1) / UB BRC via fp16-32B-align(2) / UB BRC via NLast-large(2)
 *   - schMode: WithoutLoop(1) / WithLoop(2) for the NDDMA path
 *   - empty tensor: fusedProduct == 0, blockNum == 1
 *   - multi-core split + core-feeding shrink loop invariants
 *   - error paths: num_bits != 8 / dtype unsupported / dtype mismatch / non-ND storage format /
 *     dynamic runtime shape / rank > 8 / rank mismatch / w_min != w_max / y != w /
 *     illegal broadcast / > 2^31 elements
 *
 * Expected values are derived from the tiling implementation with the fake platform
 * coreNum = 64, ubSize = 262144. The core-feeding shrink loop (blockNum < coreNum)
 * shrinks perBufElems by 32 elems/round down to the 1024 floor for shapes too small
 * to feed more cores, so small-shape cases expect perBufElems == 1024.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../../../tests/ut/common/any_value.h"
#include "quant/wts_arq/op_kernel/arch35/wts_arq_tiling_data.h"

namespace {

constexpr const char* kOpType = "WtsARQ";
constexpr uint64_t kCoreNum = 64;
constexpr uint64_t kUbSize = 262144;
// The core-feeding shrink loop keeps shrinking perBufElems while blockNum < coreNum;
// shapes too small to feed more cores land on the floor: max(MIN_TILE_ELEMS=1024, align).
constexpr int64_t kPerBufShrinkFloor = 1024;
constexpr float kEps = 1.1920929e-07f;

// Dummy compile info: the tiling context builder requires a non-null compile info
// pointer (nullptr breaks slot wiring and segfaults GetPlatformInfo in the faker).
struct WtsArqUtCompileInfo {
    int64_t placeholder = 0;
};
WtsArqUtCompileInfo g_compileInfo;

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (const int64_t d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

gert::TilingContextPara MakePara(const std::vector<int64_t>& w, const std::vector<int64_t>& wMin,
                                 const std::vector<int64_t>& wMax, ge::DataType dt, int64_t numBits = 8,
                                 bool offsetFlag = false)
{
    return gert::TilingContextPara(
        kOpType,
        {{MakeStorageShape(w), dt, ge::FORMAT_ND},
         {MakeStorageShape(wMin), dt, ge::FORMAT_ND},
         {MakeStorageShape(wMax), dt, ge::FORMAT_ND}},
        {{MakeStorageShape(w), dt, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(numBits)),
         gert::TilingContextPara::OpAttr("offset_flag", Ops::NN::AnyValue::CreateFrom<bool>(offsetFlag))},
        &g_compileInfo, kCoreNum, kUbSize);
}

// Variant with independent y shape (y dynamic / y != w error paths)
gert::TilingContextPara MakeParaYShape(const std::vector<int64_t>& w, const std::vector<int64_t>& rangeShape,
                                       const std::vector<int64_t>& yShape, ge::DataType dt)
{
    return gert::TilingContextPara(
        kOpType,
        {{MakeStorageShape(w), dt, ge::FORMAT_ND},
         {MakeStorageShape(rangeShape), dt, ge::FORMAT_ND},
         {MakeStorageShape(rangeShape), dt, ge::FORMAT_ND}},
        {{MakeStorageShape(yShape), dt, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)),
         gert::TilingContextPara::OpAttr("offset_flag", Ops::NN::AnyValue::CreateFrom<bool>(false))},
        &g_compileInfo, kCoreNum, kUbSize);
}

// Variant with independent y dtype (dtype mismatch error paths)
gert::TilingContextPara MakeParaYDtype(const std::vector<int64_t>& w, const std::vector<int64_t>& rangeShape,
                                       ge::DataType wDt, ge::DataType yDt)
{
    return gert::TilingContextPara(
        kOpType,
        {{MakeStorageShape(w), wDt, ge::FORMAT_ND},
         {MakeStorageShape(rangeShape), wDt, ge::FORMAT_ND},
         {MakeStorageShape(rangeShape), wDt, ge::FORMAT_ND}},
        {{MakeStorageShape(w), yDt, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)),
         gert::TilingContextPara::OpAttr("offset_flag", Ops::NN::AnyValue::CreateFrom<bool>(false))},
        &g_compileInfo, kCoreNum, kUbSize);
}

// Variant with independent w_min dtype (dtype mismatch error paths)
gert::TilingContextPara MakeParaWMinDtype(const std::vector<int64_t>& w, const std::vector<int64_t>& rangeShape,
                                          ge::DataType wDt, ge::DataType wMinDt)
{
    return gert::TilingContextPara(
        kOpType,
        {{MakeStorageShape(w), wDt, ge::FORMAT_ND},
         {MakeStorageShape(rangeShape), wMinDt, ge::FORMAT_ND},
         {MakeStorageShape(rangeShape), wDt, ge::FORMAT_ND}},
        {{MakeStorageShape(w), wDt, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)),
         gert::TilingContextPara::OpAttr("offset_flag", Ops::NN::AnyValue::CreateFrom<bool>(false))},
        &g_compileInfo, kCoreNum, kUbSize);
}

// Variant with explicit storage/origin formats (format validation paths)
gert::TilingContextPara MakeParaFormat(const std::vector<int64_t>& w, const std::vector<int64_t>& rangeShape,
                                       ge::DataType dt, ge::Format storageFormat, ge::Format originFormat)
{
    return gert::TilingContextPara(
        kOpType,
        {gert::TilingContextPara::TensorDescription(MakeStorageShape(w), dt, storageFormat, false, nullptr,
                                                    originFormat),
         gert::TilingContextPara::TensorDescription(MakeStorageShape(rangeShape), dt, storageFormat, false, nullptr,
                                                    originFormat),
         gert::TilingContextPara::TensorDescription(MakeStorageShape(rangeShape), dt, storageFormat, false, nullptr,
                                                    originFormat)},
        {gert::TilingContextPara::TensorDescription(MakeStorageShape(w), dt, storageFormat, false, nullptr,
                                                    originFormat)},
        {gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)),
         gert::TilingContextPara::OpAttr("offset_flag", Ops::NN::AnyValue::CreateFrom<bool>(false))},
        &g_compileInfo, kCoreNum, kUbSize);
}

void ExpectCommonWtsArqFields(const WtsArqTilingData<4>* td, uint32_t shapeLen, uint32_t brcMode)
{
    EXPECT_EQ(td->shapeLen, shapeLen);
    EXPECT_EQ(td->brcMode, brcMode);
    EXPECT_EQ(td->numBits, 8u);
    EXPECT_FLOAT_EQ(td->eps, kEps);
    EXPECT_EQ(td->coreNum, td->blockNum);
}

} // namespace

class WtsArqTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "WtsArqTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "WtsArqTilingTest TearDown" << std::endl; }
};

// OneDim scalar broadcast: w=[100], w_min=w_max=[1], fp32
TEST_F(WtsArqTilingTest, onedim_scalar_broadcast_fp32)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara({100}, {1}, {1}, ge::DT_FLOAT), info));
    EXPECT_EQ(info.tilingKey, 0u); // RANK=4
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_EQ(info.workspaceSizes[0], 0u);
    EXPECT_EQ(info.blockNum, 1u);

    auto* td = reinterpret_cast<WtsArqTilingData<4>*>(info.tilingData.get());
    ExpectCommonWtsArqFields(td, 1u, WTS_ARQ_BRC_NONE);
    EXPECT_EQ(td->dims[3], 100);
    EXPECT_EQ(td->minStrides[3], 0);
    EXPECT_EQ(td->maxStrides[3], 0);
    EXPECT_EQ(td->ubSplitAxis, 3);
    EXPECT_EQ(td->ubFormer, 100);
    EXPECT_EQ(td->ubOuter, 1);
    EXPECT_EQ(td->ubTail, 100);
    EXPECT_EQ(td->fusedProduct, 1);
    EXPECT_EQ(td->blockFormer, 1);
    EXPECT_EQ(td->blockNum, 1);
    EXPECT_EQ(td->blockTail, 1);
    EXPECT_EQ(td->perBufElems, kPerBufShrinkFloor);
    EXPECT_EQ(td->offsetFlag, 0u);
}

// OneDim after collapse (no broadcast merges to 1D): w=[2,3], w_min=w_max=[2,3], fp32
TEST_F(WtsArqTilingTest, onedim_no_broadcast_fp32)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara({2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT), info));
    EXPECT_EQ(info.tilingKey, 0u);

    auto* td = reinterpret_cast<WtsArqTilingData<4>*>(info.tilingData.get());
    ExpectCommonWtsArqFields(td, 1u, WTS_ARQ_BRC_NONE);
    EXPECT_EQ(td->dims[3], 6);
    EXPECT_EQ(td->minStrides[3], 1); // real per-element stride, not scalar broadcast
    EXPECT_EQ(td->ubFormer, 6);
    EXPECT_EQ(td->fusedProduct, 1);
    EXPECT_EQ(td->blockNum, 1);
}

// Multi-dim NDDMA broadcast (fp32 small last axis): w=[2,3], w_min=w_max=[1,3]
TEST_F(WtsArqTilingTest, multidim_nddma_fp32)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara({2, 3}, {1, 3}, {1, 3}, ge::DT_FLOAT), info));
    EXPECT_EQ(info.tilingKey, 0u);

    auto* td = reinterpret_cast<WtsArqTilingData<4>*>(info.tilingData.get());
    ExpectCommonWtsArqFields(td, 2u, WTS_ARQ_BRC_NDDMA);
    EXPECT_EQ(td->schMode, WTS_ARQ_SCH_WITHOUT_LOOP);
    EXPECT_EQ(td->dims[2], 2);
    EXPECT_EQ(td->dims[3], 3);
    EXPECT_EQ(td->minStrides[2], 0); // broadcast axis
    EXPECT_EQ(td->minStrides[3], 1);
    EXPECT_EQ(td->ubSplitAxis, 2);
    EXPECT_EQ(td->ubFormer, 2);
    EXPECT_EQ(td->ubOuter, 1);
    EXPECT_EQ(td->ubTail, 2);
    EXPECT_EQ(td->perBufElems, kPerBufShrinkFloor);
}

// UB BRC via fp16 32B-aligned last axis: w=[2,16], w_min=w_max=[1,16], fp16
TEST_F(WtsArqTilingTest, multidim_ub_brc_fp16)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara({2, 16}, {1, 16}, {1, 16}, ge::DT_FLOAT16), info));
    EXPECT_EQ(info.tilingKey, 0u);

    auto* td = reinterpret_cast<WtsArqTilingData<4>*>(info.tilingData.get());
    ExpectCommonWtsArqFields(td, 2u, WTS_ARQ_BRC_UB);
    EXPECT_EQ(td->schMode, 0u); // schMode only meaningful for NDDMA
    EXPECT_EQ(td->ubSplitAxis, 2);
    EXPECT_EQ(td->ubFormer, 2);
    EXPECT_EQ(td->perBufElems, kPerBufShrinkFloor);
}

// UB BRC via NLast + large last axis (fp32, last axis bytes >= dcache/2 = 4096):
// w=[2,4096], w_min=w_max=[1,4096]
TEST_F(WtsArqTilingTest, multidim_ub_brc_nlast_large_fp32)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara({2, 4096}, {1, 4096}, {1, 4096}, ge::DT_FLOAT), info));
    EXPECT_EQ(info.tilingKey, 0u);

    auto* td = reinterpret_cast<WtsArqTilingData<4>*>(info.tilingData.get());
    ExpectCommonWtsArqFields(td, 2u, WTS_ARQ_BRC_UB);
    // perBufElems shrinks to the 1024 floor (shape too small to feed 64 cores);
    // 4096 > 1024 moves the split axis to the inner axis (padded index 3)
    EXPECT_EQ(td->ubSplitAxis, 3);
    EXPECT_EQ(td->ubFormer, 1024);
    EXPECT_EQ(td->ubOuter, 4);
    EXPECT_EQ(td->ubTail, 1024);
}

// RANK=8 dispatch: collapsed shapeLen=5 (alternating broadcast flags prevent merge)
TEST_F(WtsArqTilingTest, rank5_dispatch_rank8_key)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara({2, 3, 2, 3, 2}, {1, 3, 1, 3, 1}, {1, 3, 1, 3, 1}, ge::DT_FLOAT), info));
    EXPECT_EQ(info.tilingKey, 1u); // RANK=8

    auto* td = reinterpret_cast<WtsArqTilingData<8>*>(info.tilingData.get());
    EXPECT_EQ(td->shapeLen, 5u);
    EXPECT_EQ(td->brcMode, WTS_ARQ_BRC_NDDMA);        // last axis broadcast -> not NLast; fp32 -> NDDMA
    EXPECT_EQ(td->schMode, WTS_ARQ_SCH_WITHOUT_LOOP); // 5 - 0 <= 5
    EXPECT_EQ(td->fusedProduct, 1);
    EXPECT_EQ(td->blockNum, 1);
}

// NDDMA WithLoop: collapsed shapeLen=6, axes after split axis > 5
TEST_F(WtsArqTilingTest, rank6_nddma_with_loop)
{
    TilingInfo info;
    ASSERT_TRUE(
        ExecuteTiling(MakePara({2, 3, 2, 3, 2, 3}, {1, 3, 1, 3, 1, 3}, {1, 3, 1, 3, 1, 3}, ge::DT_FLOAT), info));
    EXPECT_EQ(info.tilingKey, 1u); // RANK=8

    auto* td = reinterpret_cast<WtsArqTilingData<8>*>(info.tilingData.get());
    EXPECT_EQ(td->shapeLen, 6u);
    EXPECT_EQ(td->brcMode, WTS_ARQ_BRC_NDDMA);
    EXPECT_EQ(td->schMode, WTS_ARQ_SCH_WITH_LOOP); // 6 - 0 > 5
}

// Empty tensor: w=[0,3], w_min=w_max=[1,3]; kernel side exits on fusedProduct == 0
TEST_F(WtsArqTilingTest, empty_tensor)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara({0, 3}, {1, 3}, {1, 3}, ge::DT_FLOAT), info));
    EXPECT_EQ(info.tilingKey, 0u);
    EXPECT_EQ(info.blockNum, 1u);

    auto* td = reinterpret_cast<WtsArqTilingData<4>*>(info.tilingData.get());
    EXPECT_EQ(td->fusedProduct, 0);
    EXPECT_EQ(td->blockNum, 1);
    EXPECT_EQ(td->blockFormer, 0);
    EXPECT_EQ(td->blockTail, 0);
}

// rank=0 scalar: normalized to [1], OneDim single tile
TEST_F(WtsArqTilingTest, scalar_rank0)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara({}, {}, {}, ge::DT_FLOAT), info));
    EXPECT_EQ(info.tilingKey, 0u);

    auto* td = reinterpret_cast<WtsArqTilingData<4>*>(info.tilingData.get());
    ExpectCommonWtsArqFields(td, 1u, WTS_ARQ_BRC_NONE);
    EXPECT_EQ(td->dims[3], 1);
    EXPECT_EQ(td->minStrides[3], 1);
    EXPECT_EQ(td->ubFormer, 1);
    EXPECT_EQ(td->fusedProduct, 1);
    EXPECT_EQ(td->blockNum, 1);
}

// offset_flag runtime branch lands in tiling data
TEST_F(WtsArqTilingTest, offset_flag_true)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara({2, 3}, {1, 3}, {1, 3}, ge::DT_FLOAT, 8, true), info));
    auto* td = reinterpret_cast<WtsArqTilingData<4>*>(info.tilingData.get());
    EXPECT_EQ(td->offsetFlag, 1u);
}

// Multi-core split + core-feeding shrink loop: invariants must hold for any coreNum
TEST_F(WtsArqTilingTest, multicore_split_invariants)
{
    const int64_t total = 2000000;
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara({total}, {total}, {total}, ge::DT_FLOAT), info));
    EXPECT_EQ(info.tilingKey, 0u);

    auto* td = reinterpret_cast<WtsArqTilingData<4>*>(info.tilingData.get());
    ExpectCommonWtsArqFields(td, 1u, WTS_ARQ_BRC_NONE);
    // UB split covers the whole axis
    EXPECT_EQ(td->ubFormer * (td->ubOuter - 1) + td->ubTail, total);
    // block split covers all tiles
    EXPECT_EQ(td->fusedProduct, td->ubOuter);
    EXPECT_EQ(td->blockFormer * (td->blockNum - 1) + td->blockTail, td->fusedProduct);
    EXPECT_GE(td->blockNum, 1);
    EXPECT_LE(td->blockNum, static_cast<int64_t>(kCoreNum));
    EXPECT_EQ(info.blockNum, static_cast<size_t>(td->blockNum));
    // shrink loop keeps perBufElems VL-aligned (64 fp32 elements) and above the floor:
    // full-width register loads must never read past the tile buffer
    EXPECT_EQ(td->perBufElems % 64, 0);
    EXPECT_GE(td->perBufElems, kPerBufShrinkFloor);
}

// SHAPE_SIZE_LIMIT boundary: exactly 2^31 elements is allowed
TEST_F(WtsArqTilingTest, shape_size_at_limit_success)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara({65536, 32768}, {1, 32768}, {1, 32768}, ge::DT_FLOAT), info));
    auto* td = reinterpret_cast<WtsArqTilingData<4>*>(info.tilingData.get());
    EXPECT_EQ(td->shapeLen, 2u);
    // NLast with large last axis (32768 * 4B >= dcache/2) -> UB BRC
    EXPECT_EQ(td->brcMode, WTS_ARQ_BRC_UB);
}

// ------------------------- error paths -------------------------

TEST_F(WtsArqTilingTest, error_num_bits_not_8)
{
    ExecuteTestCase(MakePara({2, 3}, {1, 3}, {1, 3}, ge::DT_FLOAT, 4, false), ge::GRAPH_FAILED, 0,
                    std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, error_dtype_not_supported)
{
    ExecuteTestCase(MakePara({2, 3}, {1, 3}, {1, 3}, ge::DT_BF16), ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, error_wmin_dtype_mismatch)
{
    ExecuteTestCase(MakeParaWMinDtype({2, 3}, {1, 3}, ge::DT_FLOAT, ge::DT_FLOAT16), ge::GRAPH_FAILED, 0,
                    std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, error_y_dtype_mismatch)
{
    ExecuteTestCase(MakeParaYDtype({2, 3}, {1, 3}, ge::DT_FLOAT, ge::DT_FLOAT16), ge::GRAPH_FAILED, 0,
                    std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, error_dynamic_runtime_shape)
{
    ExecuteTestCase(MakePara({-1, 3}, {1, 3}, {1, 3}, ge::DT_FLOAT), ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, error_rank_over_limit)
{
    ExecuteTestCase(
        MakePara({2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}, ge::DT_FLOAT),
        ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, error_rank_mismatch)
{
    ExecuteTestCase(MakePara({2, 3}, {3}, {3}, ge::DT_FLOAT), ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, error_wmin_wmax_shape_mismatch)
{
    ExecuteTestCase(MakePara({2, 3}, {2, 1}, {1, 3}, ge::DT_FLOAT), ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, error_illegal_broadcast_dim)
{
    ExecuteTestCase(MakePara({2, 3}, {2, 2}, {2, 2}, ge::DT_FLOAT), ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, error_shape_size_over_limit)
{
    // 65536 * 32769 = 2^31 + 65536 > 2^31
    ExecuteTestCase(MakePara({65536, 32769}, {1, 32769}, {1, 32769}, ge::DT_FLOAT), ge::GRAPH_FAILED, 0,
                    std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, error_storage_format_not_nd)
{
    // FRACTAL_NZ storage reaching Optiling would be misread as contiguous ND
    ExecuteTestCase(MakeParaFormat({2, 3}, {1, 3}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ),
                    ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, origin_format_nz_normalized_success)
{
    // GE may normalize a declared non-ND origin format with a transdata; storage is
    // already ND at Optiling, so the operator must still tile.
    TilingInfo info;
    ASSERT_TRUE(
        ExecuteTiling(MakeParaFormat({2, 3}, {1, 3}, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ), info));
}

TEST_F(WtsArqTilingTest, error_dynamic_output_shape)
{
    ExecuteTestCase(MakeParaYShape({2, 3}, {1, 3}, {-1, 3}, ge::DT_FLOAT), ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

TEST_F(WtsArqTilingTest, error_y_shape_mismatch)
{
    ExecuteTestCase(MakeParaYShape({2, 3}, {1, 3}, {3, 2}, ge::DT_FLOAT), ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

// TilingParse is a trivial success stub; cover it via the registry entry
TEST_F(WtsArqTilingTest, tiling_parse_success)
{
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType)->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    std::string compileInfoStr = "{}";
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    int64_t compileInfoOut = 0;
    auto holder = gert::KernelRunContextFaker()
                      .KernelIONum(2, 1)
                      .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                      .Outputs({&compileInfoOut})
                      .Build();
    ASSERT_EQ(tilingParseFunc(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}
