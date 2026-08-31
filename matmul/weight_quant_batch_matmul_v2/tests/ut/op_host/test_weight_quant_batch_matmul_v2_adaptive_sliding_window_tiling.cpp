/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <thread>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "../../../op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_adaptive_split_tiling.h"
#include "../../../op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.h"
#include "../../../op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_asw_cmct_tiling.h"
#include "test_cube_util.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"

using namespace std;

struct WeightQuantBatchMatmulV2TilingTestParam {
    string caseName;

    // output
    uint32_t numBlocks;
    uint64_t tilingKey;
};

class TestWeightQuantBatchMatmulV2AdaptiveSlidingWindowTiling
    : public testing::TestWithParam<WeightQuantBatchMatmulV2TilingTestParam> {};

using namespace ge;
using namespace optiling;

static void SplitStr2Vec(const string& input, const string& delimiter, vector<string>& output)
{
    auto delimiterLen = delimiter.size();
    std::string::size_type currPos = 0;
    std::string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != std::string::npos) {
        output.emplace_back(input.substr(currPos, nextPos - currPos));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }

    if (currPos < input.size()) {
        output.emplace_back(input.substr(currPos));
    }
}

void replacePlaceholder(std::string& str, const std::string& placeholder, const std::string& replacement)
{
    size_t pos = str.find(placeholder);
    if (pos != std::string::npos) {
        str.replace(pos, placeholder.length(), replacement);
    }
}

static void TestOneParamCase(const WeightQuantBatchMatmulV2TilingTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    std::vector<string> testParam;
    SplitStr2Vec(param.caseName.substr(param.caseName.find_first_of('_') + 1), "_", testParam);
    map<string, ge::DataType> dtypeMap = {{"FLOAT16", ge::DT_FLOAT16},
                                          {"FLOAT", ge::DT_FLOAT},
                                          {"BF16", ge::DT_BF16},
                                          {"INT8", ge::DT_INT8},
                                          {"INT4", ge::DT_INT4},
                                          {"UINT64", ge::DT_UINT64},
                                          {"FP8E4M3", ge::DT_FLOAT8_E4M3FN},
                                          {"HIF8", ge::DT_HIFLOAT8}};

    size_t idx = 0;
    int64_t m = stol(testParam[idx++]);
    int64_t k = stol(testParam[idx++]);
    int64_t n = stol(testParam[idx++]);
    int64_t antiQuantOffsetExistFlag = stol(testParam[idx++]);
    int64_t quantScaleExistFlag = stol(testParam[idx++]);
    int64_t quantOffsetExistFlag = stol(testParam[idx++]);
    int64_t biasFlag = stol(testParam[idx++]);
    int64_t transA = stol(testParam[idx++]);
    int64_t transB = stol(testParam[idx++]);
    int64_t group = stol(testParam[idx++]);
    ge::DataType xDtype = dtypeMap[testParam[idx++]];
    ge::DataType weightDtype = dtypeMap[testParam[idx++]];
    ge::DataType antiQuantScaleDtype = dtypeMap[testParam[idx++]];
    ge::DataType quantScaleDtype = dtypeMap[testParam[idx++]];
    ge::DataType yDtype = dtypeMap[testParam[idx++]];
    uint32_t aicNum = stoul(testParam[idx++]);
    uint32_t aivNum = stoul(testParam[idx++]);
    string compileInfoStr = R"({
         "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true,
                           "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l12bt": true,
                           "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                           "Intrinsic_fix_pipe_pre_conv_cast": true,
                           "Intrinsic_data_move_l12bt": true, "Intrinsic_mmad": true,
                           "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 1048576,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": aicNum,
                           "cube_core_cnt": aicNum, "vector_core_cnt": aivNum, "core_type_list": "CubeCore,VectorCore",
                           "lut_type": "MTE2_QTABLE"}
                            })";
    ge::Format bFormat = ge::FORMAT_ND;
    if (stol(testParam[idx++]) == 1) {
        bFormat = ge::FORMAT_FRACTAL_NZ;
    }
    uint32_t soc = stol(testParam[idx++]);
    int64_t enableUncache = 0;
    if (testParam.size() > idx) {
        enableUncache = stol(testParam[idx++]);
    }
    ge::DataType biasDtype = xDtype;
    if (xDtype == ge::DT_BF16) {
        biasDtype = ge::DT_FLOAT;
    }
    gert::StorageShape xShape;
    gert::StorageShape weigthShape;
    gert::StorageShape antiQuantScaleShape;
    gert::StorageShape antiQuantOffsetShape;
    gert::StorageShape quantScaleShape;
    gert::StorageShape quantOffsetShape;
    gert::StorageShape biasShape;
    gert::StorageShape outputShape({m, n}, {m, n});

    if (transA) {
        xShape.MutableStorageShape() = gert::Shape({k, m});
        xShape.MutableOriginShape() = gert::Shape({k, m});
    } else {
        xShape.MutableStorageShape() = gert::Shape({m, k});
        xShape.MutableOriginShape() = gert::Shape({m, k});
    }
    if (transB) {
        if (bFormat == ge::FORMAT_FRACTAL_NZ) {
            weigthShape.MutableStorageShape() = gert::Shape({(k + 16) / 16, (n + 16) / 16L, 16L, 16});
        } else {
            weigthShape.MutableStorageShape() = gert::Shape({n, k});
        }
        weigthShape.MutableOriginShape() = gert::Shape({n, k});
    } else {
        if (bFormat == ge::FORMAT_FRACTAL_NZ) {
            weigthShape.MutableStorageShape() = gert::Shape({(n + 16) / 16, (k + 16) / 16L, 16L, 16});
        } else {
            weigthShape.MutableStorageShape() = gert::Shape({k, n});
        }
        weigthShape.MutableOriginShape() = gert::Shape({k, n});
    }
    int64_t groupSize = 0;
    if (group > 0) {
        groupSize = group;
        int64_t groupNum = (k + group - 1) / group;
        if (transB) {
            antiQuantOffsetShape.MutableStorageShape() = gert::Shape({n, groupNum});
            antiQuantScaleShape.MutableStorageShape() = gert::Shape({n, groupNum});
        } else {
            antiQuantOffsetShape.MutableStorageShape() = gert::Shape({groupNum, n});
            antiQuantScaleShape.MutableStorageShape() = gert::Shape({groupNum, n});
        }
    } else if (group < 0) {
        antiQuantOffsetShape.MutableStorageShape() = gert::Shape({n});
        antiQuantScaleShape.MutableStorageShape() = gert::Shape({n});
        quantScaleShape.MutableStorageShape() = gert::Shape({n});
        quantOffsetShape.MutableStorageShape() = gert::Shape({n});
    } else {
        antiQuantOffsetShape.MutableStorageShape() = gert::Shape({1});
        antiQuantScaleShape.MutableStorageShape() = gert::Shape({1});
        quantScaleShape.MutableStorageShape() = gert::Shape({1});
        quantOffsetShape.MutableStorageShape() = gert::Shape({1});
    }

    biasShape.MutableStorageShape() = gert::Shape({n});
    xShape.MutableStorageShape() = xShape.MutableStorageShape();
    xShape.MutableOriginShape() = xShape.MutableOriginShape();
    weigthShape.MutableStorageShape() = weigthShape.MutableStorageShape();
    weigthShape.MutableOriginShape() = weigthShape.MutableOriginShape();
    antiQuantScaleShape.MutableStorageShape() = antiQuantScaleShape.MutableStorageShape();
    antiQuantOffsetShape.MutableStorageShape() = antiQuantOffsetShape.MutableStorageShape();
    quantScaleShape.MutableStorageShape() = quantScaleShape.MutableStorageShape();
    quantOffsetShape.MutableStorageShape() = quantOffsetShape.MutableStorageShape();
    biasShape.MutableStorageShape() = biasShape.MutableStorageShape();

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    replacePlaceholder(compileInfoStr, "aicNum", to_string(aicNum));
    replacePlaceholder(compileInfoStr, "aicNum", to_string(aicNum));
    replacePlaceholder(compileInfoStr, "aivNum", to_string(aivNum));
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1800";
    if (static_cast<int>(soc) == 1) {
        aicoreSpec["cube_freq"] = "1500";
    }

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    MatmulV3CompileInfo compileInfo;

    auto kernelHold = gert::KernelRunContextFaker()
                          .KernelIONum(2, 1)
                          .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                          .Outputs({&compileInfo})
                          .Build();

    std::string opType("WeightQuantBatchMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &weigthShape, &antiQuantScaleShape,
                                    antiQuantOffsetExistFlag ? &antiQuantOffsetShape : nullptr,
                                    quantScaleExistFlag ? &quantScaleShape : nullptr,
                                    quantOffsetExistFlag ? &quantOffsetShape : nullptr,
                                    biasFlag ? &biasShape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, weightDtype, ge::FORMAT_ND, bFormat)
                      .NodeInputTd(2, antiQuantScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, quantScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
                                  {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(transB)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"inner_precise", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"ascendc_op_para_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"enable_uncache", Ops::NN::AnyValue::CreateFrom<int64_t>(enableUncache)}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .SetOpType(opType)
                      .Build();

    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "RESERVED_VERSION"));

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    ASSERT_EQ(tilingContext->GetTilingKey(), param.tilingKey);
    ASSERT_EQ(tilingContext->GetBlockDim(), param.numBlocks);

    // Directly exercise ASW tiling template (priority 11) which is never selected by the framework
    // because ASW_CMCT (priority 10) always returns true for valid shapes. This covers ASW-specific
    // methods: DoOpTiling, AnalyseSlidingWinInfo, CalcBasicBlock, CalcTailBasicBlock, CalL1Tiling,
    // SetTilingData, GetTilingKey, etc.
    auto aswTiling = optiling::weight_quant_batch_matmul_v2::WeightQuantBatchMatmulV2TilingASW(tilingContext);
    ASSERT_EQ(aswTiling.DoTiling(), ge::GRAPH_SUCCESS);
}

// Test ASW DoTiling directly for INT4/NZ cases that the framework rejects.
// This covers 4-bit dtype branches (GetShapeWithDataType, GetSizeWithDataType)
// and NZ tail split branches (IsValidWeightNzTailSplit, CalUsedCoreNum).
TEST(TestWeightQuantBatchMatmulV2AdaptiveSlidingWindowTilingDirect, DirectDoTilingInt4Nz)
{
    int64_t m = 128;
    int64_t k = 128;
    int64_t n = 128;
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType weightDtype = ge::DT_INT4;
    ge::DataType antiQuantScaleDtype = ge::DT_FLOAT16;
    ge::DataType quantScaleDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_FLOAT16;
    ge::Format bFormat = ge::FORMAT_FRACTAL_NZ;

    gert::StorageShape xShape({m, k}, {m, k});
    gert::StorageShape weigthShape({(n + 16) / 16, (k + 16) / 16, 16, 16}, {(n + 16) / 16, (k + 16) / 16, 16, 16});
    weigthShape.MutableOriginShape() = gert::Shape({k, n});
    gert::StorageShape antiQuantScaleShape({1}, {1});
    gert::StorageShape quantScaleShape({1}, {1});
    gert::StorageShape outputShape({m, n}, {m, n});

    string compileInfoStr = R"({
         "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true,
                           "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l12bt": true,
                           "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                           "Intrinsic_fix_pipe_pre_conv_cast": true,
                           "Intrinsic_data_move_l12bt": true, "Intrinsic_mmad": true,
                           "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 1048576,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 16,
                           "cube_core_cnt": 16, "vector_core_cnt": 16, "core_type_list": "CubeCore,VectorCore",
                           "lut_type": "MTE2_QTABLE"}
                            })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    uint32_t aicNum = 16;
    uint32_t aivNum = 16;
    replacePlaceholder(compileInfoStr, "aicNum", to_string(aicNum));
    replacePlaceholder(compileInfoStr, "aivNum", to_string(aivNum));
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1800";

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    MatmulV3CompileInfo compileInfo;

    auto kernelHold = gert::KernelRunContextFaker()
                          .KernelIONum(2, 1)
                          .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                          .Outputs({&compileInfo})
                          .Build();

    std::string opType("WeightQuantBatchMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&xShape, &weigthShape, &antiQuantScaleShape, nullptr, &quantScaleShape, nullptr, nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, weightDtype, ge::FORMAT_ND, bFormat)
                      .NodeInputTd(2, antiQuantScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, quantScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"inner_precise", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"ascendc_op_para_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"enable_uncache", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .SetOpType(opType)
                      .Build();

    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "RESERVED_VERSION"));

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // Call ASW DoTiling directly (bypass framework template selection which rejects INT4+NZ on this platform)
    auto aswTiling = optiling::weight_quant_batch_matmul_v2::WeightQuantBatchMatmulV2TilingASW(tilingContext);
    ASSERT_EQ(aswTiling.DoTiling(), ge::GRAPH_SUCCESS);

    // Also call ASW_CMCT DoTiling directly to cover 4-bit dtype and NZ tail split branches
    auto aswCmctTiling = optiling::weight_quant_batch_matmul_v2::WeightQuantBatchMatmulV2TilingAswCmct(tilingContext);
    ASSERT_EQ(aswCmctTiling.DoTiling(), ge::GRAPH_SUCCESS);
}

TEST_P(TestWeightQuantBatchMatmulV2AdaptiveSlidingWindowTiling, generalTest)
{
    WeightQuantBatchMatmulV2TilingTestParam param = GetParam();
    TestOneParamCase(param);
}

// format: m k n antiQuantOffsetExistFlag quantScaleExistFlag quantOffsetExistFlag biasFlag transA transB group Xdtype
//         weigthDtype antiQuantScaleDtype quantScaleDtype yDtype aicNum aivNum weightFormat(0:ND,1:NZ)
// Note: group value
//       -1: per channel, 0: per tensor, > 0: per group
// Note: socversion
//        0: RESERVED
static WeightQuantBatchMatmulV2TilingTestParam casesParams[] = {
    {"Case_64_64_64_0_0_0_0_0_0_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 4, 285250051UL},
    {"Case_64_64_64_0_0_0_0_0_0_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 4, 302027267UL},
    {"Case_1_2048_4096_0_0_0_0_0_0_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_2_2_0_0", 2, 285250051UL},
    {"Case_1_2048_4096_0_0_0_0_0_0_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_2_2_0_0", 2, 302027267UL},
    {"Case_3072_2048_4096_0_0_0_0_0_0_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_14_14_0_0", 14, 285250051UL},
    {"Case_3072_2048_4096_0_0_0_0_0_0_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_14_14_0_0", 14, 302027267UL},
    {"Case_64_64_64_0_0_0_1_1_1_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 4, 8887767555UL},
    {"Case_64_64_64_0_0_0_1_0_0_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 4, 8875184643UL},
    {"Case_64_64_64_0_0_0_1_1_0_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 1, 8879378947UL},
    {"Case_64_64_64_0_0_0_1_0_1_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 16, 8883573251UL},
    {"Case_64_64_64_0_0_0_0_1_1_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 4, 297832963UL},
    {"Case_64_64_64_0_0_0_0_0_1_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 16, 293638659UL},
    {"Case_64_64_64_0_0_0_0_1_0_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 1, 289444355UL},
    {"Case_64_64_64_0_0_0_1_1_1_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 4, 8904544771UL},
    {"Case_64_64_64_0_0_0_1_0_0_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 4, 8891961859UL},
    {"Case_64_64_64_0_0_0_1_1_0_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 1, 8896156163UL},
    {"Case_64_64_64_0_0_0_1_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 16, 8900350467UL},
    {"Case_64_64_64_0_0_0_0_1_1_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 4, 314610179UL},
    {"Case_64_64_64_0_0_0_0_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 16, 310415875UL},
    {"Case_64_64_64_0_0_0_0_1_0_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0", 1, 306221571UL},
    // enable_uncache=1 to cover SetDisableL2cache L2 uncache logic in ASW
    {"Uncache_64_64_64_0_0_0_0_0_0_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0_1", 4, 285250051UL},
    {"Uncache_64_64_64_0_0_0_0_0_0_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0_1", 4, 302027267UL},
    // n=128 aligned to 128B for INT8 → triggers rightNotL2Cache=true in SetDisableL2cache
    {"Uncache_128_128_128_0_0_0_0_0_0_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0_1", 8, 285250051UL},
    {"Uncache_128_128_128_0_0_0_0_0_0_-1_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0_1", 8, 302027267UL},
    // transB + n=128 aligned to 128B → triggers flagB transB=true and rightNotL2Cache
    {"Uncache_128_128_128_0_0_0_0_0_1_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0_1", 16, 293638659UL},
    // transA + enable_uncache → covers flagA transA=true branch
    {"Uncache_128_128_128_0_0_0_0_1_0_0_FLOAT16_INT8_UINT64_FLOAT16_FLOAT16_16_16_0_0_1", 2, 289444355UL},
};

INSTANTIATE_TEST_CASE_P(MM, TestWeightQuantBatchMatmulV2AdaptiveSlidingWindowTiling, testing::ValuesIn(casesParams));
