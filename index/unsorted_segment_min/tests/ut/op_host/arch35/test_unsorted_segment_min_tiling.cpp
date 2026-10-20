/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_unsorted_segment_min_tiling.cpp
 * \brief the ut of unsorted_segment_min_tiling
 */

#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/expand_dims_type.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "index/unsorted_segment_common/op_kernel/arch35/unsorted_segment_max_tiling_key.h"
#include "index/unsorted_segment_common/op_kernel/arch35/unsorted_segment_struct.h"

using namespace std;
using namespace ge;
using namespace ut_util;
using namespace UnsortedSegmentMax;

namespace {
// 布局与op_host/arch35/unsorted_segment_min_tiling.h引用的
// optiling::UnsortedSegmentCompileInfo保持一致，供faker构造compile info使用
struct UnsortedSegmentCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    uint64_t maxThread = 0;
};

// compile info取值，与ExecuteTestCase中compileInfoString的SoCInfo(CORE_NUM/UB_SIZE)保持一致
constexpr uint64_t DEFAULT_CORE_NUM = 64;
constexpr uint64_t DEFAULT_UB_SIZE = 253952;
constexpr uint64_t DEFAULT_MAX_THREAD = 2048;
} // namespace

class UnsortedSegmentMinTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UnsortedSegmentMinTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "UnsortedSegmentMinTiling TearDown" << std::endl; }
};

struct UnsortedSegmentMinOpsParamInfos {
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType segIdsDtype = ge::DT_INT32;
    ge::DataType numSegmentsDtype = ge::DT_INT32;
    gert::StorageShape xShape;
    gert::StorageShape segIdsShape;
    gert::StorageShape numSegmentsShape;
    gert::StorageShape yShape;
    int64_t numSegmentsValue = 0;
};

struct UnsortedSegmentMinTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    std::vector<uint8_t> tilingData;
};

template <typename T>
static T ParseTilingData(const UnsortedSegmentMinTilingResult& result)
{
    T tilingData{};
    if (result.tilingData.size() >= sizeof(T)) {
        (void)std::memcpy(&tilingData, result.tilingData.data(), sizeof(T));
    }
    return tilingData;
}

static UnsortedSegmentMinTilingResult ExecuteTestCase(const UnsortedSegmentMinOpsParamInfos& param,
                                                      ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersions;

    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, socVersions);

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    // compile info
    UnsortedSegmentCompileInfo compileInfo;
    compileInfo.coreNum = DEFAULT_CORE_NUM;
    compileInfo.ubSizePlatForm = DEFAULT_UB_SIZE;
    compileInfo.maxThread = DEFAULT_MAX_THREAD;

    std::string opType("UnsortedSegmentMin");
    UnsortedSegmentMinTilingResult result;
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr) {
        return result;
    }
    auto tilingFunc = opImpl->tiling;
    EXPECT_NE(tilingFunc, nullptr);
    if (tilingFunc == nullptr) {
        return result;
    }

    // tilingFunc simulate
    auto tilingDataCap = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    EXPECT_NE(tilingDataCap, nullptr);
    if (tilingDataCap == nullptr) {
        return result;
    }
    gert::StorageShape xShape = param.xShape;
    gert::StorageShape segIdsShape = param.segIdsShape;
    gert::StorageShape numSegmentsShape = param.numSegmentsShape;
    gert::StorageShape yShape = param.yShape;

    // num_segments为常量输入，通过InputTensors携带常量值
    int32_t numSegmentsDataInt32 = static_cast<int32_t>(param.numSegmentsValue);
    int64_t numSegmentsDataInt64 = param.numSegmentsValue;
    void* numSegmentsData = (param.numSegmentsDtype == ge::DT_INT64) ? reinterpret_cast<void*>(&numSegmentsDataInt64) :
                                                                       reinterpret_cast<void*>(&numSegmentsDataInt32);
    gert::Tensor xTensor(xShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, param.xDtype, nullptr);
    gert::Tensor segIdsTensor(segIdsShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                              gert::TensorPlacement::kOnHost, param.segIdsDtype, nullptr);
    gert::Tensor numSegmentsTensor(numSegmentsShape,
                                   gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                   gert::TensorPlacement::kOnHost, param.numSegmentsDtype, numSegmentsData);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("UnsortedSegmentMin")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &segIdsShape, &numSegmentsShape})
                      .OutputShapes({&yShape})
                      .InputTensors({&xTensor, &segIdsTensor, &numSegmentsTensor})
                      .NodeInputTd(0, param.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.segIdsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, param.numSegmentsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .TilingData(tilingDataCap.get())
                      .Workspace(wsSize)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tilingContext, nullptr);
    if (tilingContext == nullptr) {
        return result;
    }
    auto infos = tilingContext->GetPlatformInfo();
    EXPECT_NE(infos, nullptr);
    if (infos == nullptr) {
        return result;
    }
    infos->SetPlatformRes("SoCInfo", socInfos);
    infos->SetPlatformRes("AICoreSpec", aicoreSpec);
    infos->SetCoreNumByCoreType("AICore");
    infos->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    infos->SetPlatformRes("version", socVersions);

    result.status = tilingFunc(tilingContext);
    EXPECT_EQ(result.status, status);
    if (result.status != ge::GRAPH_SUCCESS) {
        return result;
    }
    result.tilingKey = tilingContext->GetTilingKey();
    auto rawTilingData = tilingContext->GetRawTilingData();
    if (rawTilingData != nullptr && rawTilingData->GetData() != nullptr && rawTilingData->GetDataSize() > 0) {
        const auto* tilingDataBytes = reinterpret_cast<const uint8_t*>(rawTilingData->GetData());
        result.tilingData.assign(tilingDataBytes, tilingDataBytes + rawTilingData->GetDataSize());
    }
    return result;
}

// 常规场景：float32小shape，输出全载（OutFl）模板
TEST_F(UnsortedSegmentMinTiling, UnsortedSegmentMin_tiling_ascendc_fp32_out_fullload)
{
    constexpr int64_t numSegments = 4; // 输出segment数，与yShape[0]一致
    UnsortedSegmentMinOpsParamInfos param;
    param.xDtype = ge::DT_FLOAT;
    param.segIdsDtype = ge::DT_INT32;
    param.numSegmentsDtype = ge::DT_INT32;
    param.xShape = {{8, 16}, {8, 16}};
    param.segIdsShape = {{8}, {8}};
    param.numSegmentsShape = {{1}, {1}};
    param.yShape = {{4, 16}, {4, 16}};
    param.numSegmentsValue = numSegments;

    auto result = ExecuteTestCase(param);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, GET_TPL_TILING_KEY(USS_TEMPLATE_OUT_FL, USS_CAST_NONE));
    ASSERT_GE(result.tilingData.size(), sizeof(UnsortedSegment::UnsortedSegmentOutFlTilingData));
    auto tilingData = ParseTilingData<UnsortedSegment::UnsortedSegmentOutFlTilingData>(result);
    EXPECT_EQ(tilingData.inputOuterDim, 8);
    EXPECT_EQ(tilingData.outputOuterDim, 4);
    EXPECT_EQ(tilingData.innerDim, 16);
    EXPECT_EQ(tilingData.maxIndexNum, 1);
    EXPECT_EQ(tilingData.oneCoreUbLoopTimes, 1);
    EXPECT_EQ(tilingData.rowNumUb, 1);
    EXPECT_EQ(tilingData.parallelNum, 16);
    EXPECT_EQ(tilingData.oneRowOutNumAlign, 64);
    EXPECT_EQ(tilingData.accStride, 72);
}

// 大内轴float16场景，命中simd non sort模板
TEST_F(UnsortedSegmentMinTiling, UnsortedSegmentMin_tiling_ascendc_fp16_simd_non_sort)
{
    constexpr int64_t numSegments = 16; // 输出segment数，与yShape[0]一致
    UnsortedSegmentMinOpsParamInfos param;
    param.xDtype = ge::DT_FLOAT16;
    param.segIdsDtype = ge::DT_INT64;
    param.numSegmentsDtype = ge::DT_INT32;
    param.xShape = {{64, 2048}, {64, 2048}};
    param.segIdsShape = {{64}, {64}};
    param.numSegmentsShape = {{1}, {1}};
    param.yShape = {{16, 2048}, {16, 2048}};
    param.numSegmentsValue = numSegments;

    auto result = ExecuteTestCase(param);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, GET_TPL_TILING_KEY(USS_TEMPLATE_SIMD_NON_SORT, USS_CAST_NONE));
    ASSERT_GE(result.tilingData.size(), sizeof(UnsortedSegment::UnsortedSegmentSimdNonSortTilingData));
    auto tilingData = ParseTilingData<UnsortedSegment::UnsortedSegmentSimdNonSortTilingData>(result);
    EXPECT_EQ(tilingData.outputOuterDim, 16);
    EXPECT_EQ(tilingData.innerDim, 2048);
    EXPECT_EQ(tilingData.sTileNum, 64);
    EXPECT_EQ(tilingData.aTileNum, 1);
    EXPECT_EQ(tilingData.normBlockS, 1);
    EXPECT_EQ(tilingData.tailBlockS, 1);
    EXPECT_EQ(tilingData.normBlockA, 2048);
    EXPECT_EQ(tilingData.tailBlockA, 2048);
    EXPECT_EQ(tilingData.baseS, 30);
    EXPECT_EQ(tilingData.baseA, 2048);
    EXPECT_EQ(tilingData.usedCoreNum, 64);
}

// 输入输出行比较大（ratio不小于5）的float32场景，命中simd dyn sort模板
TEST_F(UnsortedSegmentMinTiling, UnsortedSegmentMin_tiling_ascendc_fp32_simd_dyn_sort)
{
    constexpr int64_t numSegments = 16; // 输出segment数，与yShape[0]一致
    UnsortedSegmentMinOpsParamInfos param;
    param.xDtype = ge::DT_FLOAT;
    param.segIdsDtype = ge::DT_INT32;
    param.numSegmentsDtype = ge::DT_INT32;
    param.xShape = {{128, 256}, {128, 256}};
    param.segIdsShape = {{128}, {128}};
    param.numSegmentsShape = {{1}, {1}};
    param.yShape = {{16, 256}, {16, 256}};
    param.numSegmentsValue = numSegments;

    auto result = ExecuteTestCase(param);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    // segment_ids为int32且输出segment数小于255，排序前id降精度为uint8
    EXPECT_EQ(result.tilingKey, GET_TPL_TILING_KEY(USS_TEMPLATE_SIMD_DYN_SORT, USS_CAST_INT32_TO_UINT8));
    ASSERT_GE(result.tilingData.size(), sizeof(UnsortedSegment::UnsortedSegmentSimdDynSortTilingData));
    auto tilingData = ParseTilingData<UnsortedSegment::UnsortedSegmentSimdDynSortTilingData>(result);
    EXPECT_EQ(tilingData.outputOuterDim, 16);
    EXPECT_EQ(tilingData.innerDim, 256);
    EXPECT_EQ(tilingData.sTileNum, 64);
    EXPECT_EQ(tilingData.aTileNum, 1);
    EXPECT_EQ(tilingData.normBlockS, 2);
    EXPECT_EQ(tilingData.tailBlockS, 2);
    EXPECT_EQ(tilingData.normBlockA, 256);
    EXPECT_EQ(tilingData.tailBlockA, 256);
    EXPECT_EQ(tilingData.baseS, 2);
    EXPECT_EQ(tilingData.idCastMode, USS_CAST_INT32_TO_UINT8);
}

// int64数据类型不支持原子操作且ratio不小于5，命中sort simt模板
TEST_F(UnsortedSegmentMinTiling, UnsortedSegmentMin_tiling_ascendc_int64_sort_simt)
{
    constexpr int64_t numSegments = 4; // 输出segment数，与yShape[0]一致
    UnsortedSegmentMinOpsParamInfos param;
    param.xDtype = ge::DT_INT64;
    param.segIdsDtype = ge::DT_INT32;
    param.numSegmentsDtype = ge::DT_INT32;
    param.xShape = {{80, 4}, {80, 4}};
    param.segIdsShape = {{80}, {80}};
    param.numSegmentsShape = {{1}, {1}};
    param.yShape = {{4, 4}, {4, 4}};
    param.numSegmentsValue = numSegments;

    auto result = ExecuteTestCase(param);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, GET_TPL_TILING_KEY(USS_TEMPLATE_SORT_SIMT, USS_CAST_NONE));
    ASSERT_GE(result.tilingData.size(), sizeof(UnsortedSegment::UnsortedSegmentSortSimtTilingData));
    auto tilingData = ParseTilingData<UnsortedSegment::UnsortedSegmentSortSimtTilingData>(result);
    EXPECT_EQ(tilingData.inputOuterDim, 80);
    EXPECT_EQ(tilingData.outputOuterDim, 4);
    EXPECT_EQ(tilingData.innerDim, 4);
    EXPECT_EQ(tilingData.maxIndexNum, 2);
    EXPECT_EQ(tilingData.oneCoreUbLoopTimes, 1);
    EXPECT_EQ(tilingData.tailCoreUbLoopTimes, 1);
    EXPECT_EQ(tilingData.maxThread, DEFAULT_MAX_THREAD);
    EXPECT_EQ(tilingData.usedCoreNum, 40);
    EXPECT_EQ(tilingData.tailIndexNum, 2);
    EXPECT_EQ(tilingData.narrowSortKey, 0);
}

// int64数据类型小shape场景，回退到simt模板
TEST_F(UnsortedSegmentMinTiling, UnsortedSegmentMin_tiling_ascendc_int64_simt)
{
    constexpr int64_t numSegments = 4; // 输出segment数，与yShape[0]一致
    UnsortedSegmentMinOpsParamInfos param;
    param.xDtype = ge::DT_INT64;
    param.segIdsDtype = ge::DT_INT32;
    param.numSegmentsDtype = ge::DT_INT32;
    param.xShape = {{8, 8}, {8, 8}};
    param.segIdsShape = {{8}, {8}};
    param.numSegmentsShape = {{1}, {1}};
    param.yShape = {{4, 8}, {4, 8}};
    param.numSegmentsValue = numSegments;

    auto result = ExecuteTestCase(param);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, GET_TPL_TILING_KEY(USS_TEMPLATE_SIMT, USS_CAST_NONE));
    ASSERT_GE(result.tilingData.size(), sizeof(UnsortedSegment::UnsortedSegmentSimtTilingData));
    auto tilingData = ParseTilingData<UnsortedSegment::UnsortedSegmentSimtTilingData>(result);
    EXPECT_EQ(tilingData.inputOuterDim, 8);
    EXPECT_EQ(tilingData.outputOuterDim, 4);
    EXPECT_EQ(tilingData.innerDim, 8);
    EXPECT_EQ(tilingData.maxThread, DEFAULT_MAX_THREAD);
}

// 多维segment_ids场景：segment_ids shape为x shape的前两维
TEST_F(UnsortedSegmentMinTiling, UnsortedSegmentMin_tiling_ascendc_fp32_multi_dim_segment_ids)
{
    constexpr int64_t numSegments = 6; // 输出segment数，与yShape[0]一致
    UnsortedSegmentMinOpsParamInfos param;
    param.xDtype = ge::DT_FLOAT;
    param.segIdsDtype = ge::DT_INT32;
    param.numSegmentsDtype = ge::DT_INT64;
    param.xShape = {{2, 4, 16}, {2, 4, 16}};
    param.segIdsShape = {{2, 4}, {2, 4}};
    param.numSegmentsShape = {{1}, {1}};
    param.yShape = {{6, 16}, {6, 16}};
    param.numSegmentsValue = numSegments;

    auto result = ExecuteTestCase(param);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, GET_TPL_TILING_KEY(USS_TEMPLATE_OUT_FL, USS_CAST_NONE));
    ASSERT_GE(result.tilingData.size(), sizeof(UnsortedSegment::UnsortedSegmentOutFlTilingData));
    auto tilingData = ParseTilingData<UnsortedSegment::UnsortedSegmentOutFlTilingData>(result);
    EXPECT_EQ(tilingData.inputOuterDim, 8);
    EXPECT_EQ(tilingData.outputOuterDim, 6);
    EXPECT_EQ(tilingData.innerDim, 16);
    EXPECT_EQ(tilingData.maxIndexNum, 1);
    EXPECT_EQ(tilingData.oneCoreUbLoopTimes, 1);
    EXPECT_EQ(tilingData.rowNumUb, 1);
    EXPECT_EQ(tilingData.parallelNum, 16);
    EXPECT_EQ(tilingData.oneRowOutNumAlign, 96);
    EXPECT_EQ(tilingData.accStride, 104);
}

// x数据类型不支持时，tiling失败
TEST_F(UnsortedSegmentMinTiling, UnsortedSegmentMin_tiling_invalid_x_dtype)
{
    constexpr int64_t numSegments = 4; // 输出segment数，与yShape[0]一致
    UnsortedSegmentMinOpsParamInfos param;
    param.xDtype = ge::DT_INT8;
    param.segIdsDtype = ge::DT_INT32;
    param.numSegmentsDtype = ge::DT_INT32;
    param.xShape = {{8, 16}, {8, 16}};
    param.segIdsShape = {{8}, {8}};
    param.numSegmentsShape = {{1}, {1}};
    param.yShape = {{4, 16}, {4, 16}};
    param.numSegmentsValue = numSegments;

    auto result = ExecuteTestCase(param, ge::GRAPH_FAILED);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// segment_ids数据类型不支持时，tiling失败
TEST_F(UnsortedSegmentMinTiling, UnsortedSegmentMin_tiling_invalid_segment_ids_dtype)
{
    constexpr int64_t numSegments = 4; // 输出segment数，与yShape[0]一致
    UnsortedSegmentMinOpsParamInfos param;
    param.xDtype = ge::DT_FLOAT;
    param.segIdsDtype = ge::DT_FLOAT;
    param.numSegmentsDtype = ge::DT_INT32;
    param.xShape = {{8, 16}, {8, 16}};
    param.segIdsShape = {{8}, {8}};
    param.numSegmentsShape = {{1}, {1}};
    param.yShape = {{4, 16}, {4, 16}};
    param.numSegmentsValue = numSegments;

    auto result = ExecuteTestCase(param, ge::GRAPH_FAILED);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// num_segments常量值与输出shape[0]不一致时，tiling失败
TEST_F(UnsortedSegmentMinTiling, UnsortedSegmentMin_tiling_num_segments_mismatch)
{
    constexpr int64_t numSegments = 4; // 与yShape[0]故意不一致，验证tiling校验失败
    UnsortedSegmentMinOpsParamInfos param;
    param.xDtype = ge::DT_FLOAT;
    param.segIdsDtype = ge::DT_INT32;
    param.numSegmentsDtype = ge::DT_INT32;
    param.xShape = {{8, 16}, {8, 16}};
    param.segIdsShape = {{8}, {8}};
    param.numSegmentsShape = {{1}, {1}};
    param.yShape = {{8, 16}, {8, 16}};
    param.numSegmentsValue = numSegments;

    auto result = ExecuteTestCase(param, ge::GRAPH_FAILED);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// segment_ids的shape不是x的shape前缀时，tiling失败
TEST_F(UnsortedSegmentMinTiling, UnsortedSegmentMin_tiling_segment_ids_not_prefix_of_x)
{
    constexpr int64_t numSegments = 4; // 输出segment数，与yShape[0]一致
    UnsortedSegmentMinOpsParamInfos param;
    param.xDtype = ge::DT_FLOAT;
    param.segIdsDtype = ge::DT_INT32;
    param.numSegmentsDtype = ge::DT_INT32;
    param.xShape = {{8, 16}, {8, 16}};
    param.segIdsShape = {{6}, {6}};
    param.numSegmentsShape = {{1}, {1}};
    param.yShape = {{4, 16}, {4, 16}};
    param.numSegmentsValue = numSegments;

    auto result = ExecuteTestCase(param, ge::GRAPH_FAILED);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// num_segments的shape不是1维时，tiling失败
TEST_F(UnsortedSegmentMinTiling, UnsortedSegmentMin_tiling_num_segments_shape_not_1d)
{
    constexpr int64_t numSegments = 4; // 输出segment数，与yShape[0]一致
    UnsortedSegmentMinOpsParamInfos param;
    param.xDtype = ge::DT_FLOAT;
    param.segIdsDtype = ge::DT_INT32;
    param.numSegmentsDtype = ge::DT_INT32;
    param.xShape = {{8, 16}, {8, 16}};
    param.segIdsShape = {{8}, {8}};
    param.numSegmentsShape = {{1, 1}, {1, 1}};
    param.yShape = {{4, 16}, {4, 16}};
    param.numSegmentsValue = numSegments;

    auto result = ExecuteTestCase(param, ge::GRAPH_FAILED);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}
