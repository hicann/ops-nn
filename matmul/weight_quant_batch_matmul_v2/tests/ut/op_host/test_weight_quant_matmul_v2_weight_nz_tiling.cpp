/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <stdlib.h>

#include <iostream>
#include <thread>
#include <vector>

#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"

using namespace std;
using namespace ge;
using namespace optiling;

namespace {
struct WeightQuantBatchMatmulV2WeightNzTilingTestParam {
    string caseName;
    // output
    uint32_t blockDim;
    uint64_t tilingKey;
};

class TestWeightQuantBatchMatmulV2WeightNzTiling
    : public testing::TestWithParam<WeightQuantBatchMatmulV2WeightNzTilingTestParam> {
    virtual void SetUp()
    {}
};

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

static void TestOneParamCase(const WeightQuantBatchMatmulV2WeightNzTilingTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    std::vector<string> testParam;
    SplitStr2Vec(param.caseName.substr(param.caseName.find_first_of('_') + 1), "_", testParam);
    map<string, ge::DataType> dtypeMap = {{"FLOAT16", ge::DT_FLOAT16}, {"FLOAT", ge::DT_FLOAT},
                                          {"BF16", ge::DT_BF16},       {"INT8", ge::DT_INT8},
                                          {"INT4", ge::DT_INT4},       {"UINT64", ge::DT_UINT64}};
    size_t idx = 0;
    int64_t batch = stol(testParam[idx++]);
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
    ge::DataType quantScaleDtype = dtypeMap[testParam[idx++]];
    ge::DataType yDtype = dtypeMap[testParam[idx++]];
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
    gert::StorageShape outputShape({batch, m, n}, {batch, m, n});

    if (transA) {
        xShape.MutableStorageShape() = gert::Shape({batch, k, m});
        xShape.MutableOriginShape() = gert::Shape({batch, k, m});
    } else {
        xShape.MutableStorageShape() = gert::Shape({batch, m, k});
        xShape.MutableOriginShape() = gert::Shape({batch, m, k});
    }

    int64_t n1 = (n + 15) / 16;
    int64_t k1 = (k + 31) / 32;
    weigthShape.MutableStorageShape() = gert::Shape({batch, k1, n1, 16, 32});
    weigthShape.MutableOriginShape() = gert::Shape({batch, n, k});

    int64_t groupSize = 0;
    if (group > 0) {
        groupSize = group;
        int64_t groupNum = (k + group - 1) / group;
        antiQuantOffsetShape.MutableStorageShape() = gert::Shape({groupNum, n});
        antiQuantScaleShape.MutableStorageShape() = gert::Shape({groupNum, n});
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

    biasShape.MutableStorageShape() = gert::Shape({batch, 1, n});
    xShape.MutableStorageShape() = xShape.MutableStorageShape();
    weigthShape.MutableStorageShape() = weigthShape.MutableStorageShape();
    antiQuantScaleShape.MutableStorageShape() = antiQuantScaleShape.MutableStorageShape();
    antiQuantOffsetShape.MutableStorageShape() = antiQuantOffsetShape.MutableStorageShape();
    quantScaleShape.MutableStorageShape() = quantScaleShape.MutableStorageShape();
    quantOffsetShape.MutableStorageShape() = quantOffsetShape.MutableStorageShape();
    biasShape.MutableStorageShape() = biasShape.MutableStorageShape();

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0","Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 1048576,
                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8}
                          })";
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1800";

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
                      .InputShapes(
                          {&xShape, &weigthShape, &antiQuantScaleShape,
                           antiQuantOffsetExistFlag ? &antiQuantOffsetShape : nullptr, nullptr, nullptr,
                           biasFlag ? &biasShape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, weightDtype, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                      .NodeInputTd(2, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
                           {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(transB)},
                           {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .SetOpType(opType)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend310P"));
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), param.tilingKey);
    ASSERT_EQ(tilingContext->GetBlockDim(), param.blockDim);
}

TEST_P(TestWeightQuantBatchMatmulV2WeightNzTiling, generalTest)
{
    WeightQuantBatchMatmulV2WeightNzTilingTestParam param = GetParam();
    TestOneParamCase(param);
}

static WeightQuantBatchMatmulV2WeightNzTilingTestParam casesParams[] = {
    {"Key1_1_4_4096_2048_1_0_0_1_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16", 8, 13471165448961},
    {"Key1_1_4_4096_2048_1_0_0_0_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16", 8, 13471165448961},
    {"Key1_1_4_4096_2048_1_0_0_0_0_1_32_FLOAT16_INT8_UINT64_FLOAT16", 8, 13472239190785},
    {"BATCH_2_4_4096_2048_1_0_0_1_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16", 8, 1139371072291585},
    {"BATCH_2_4_4096_2048_1_0_0_0_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16", 8, 1139371072291585},
    {"BATCH_2_4_4096_2048_1_0_0_0_0_1_32_FLOAT16_INT8_UINT64_FLOAT16", 8, 1139372146033409},
    {"BATCH_2_4_4097_2048_1_0_0_0_0_1_32_FLOAT16_INT8_UINT64_FLOAT16", 8, 1139372146033409},
    {"TBETILINGP3_1_21_985_55681_0_0_0_0_0_1_32_FLOAT16_INT8_UINT64_FLOAT16", 8, 13197361283841}};

INSTANTIATE_TEST_CASE_P(
    WeightQuantBatchMatmulV2WeightNz, TestWeightQuantBatchMatmulV2WeightNzTiling, testing::ValuesIn(casesParams));

static void ThreadFunc(
    const WeightQuantBatchMatmulV2WeightNzTilingTestParam* params, size_t testcase_num, size_t thread_idx,
    size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(
    const WeightQuantBatchMatmulV2WeightNzTilingTestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_F(TestWeightQuantBatchMatmulV2WeightNzTiling, multi_thread)
{
    // 用3个线程测试
    TestMultiThread(casesParams, sizeof(casesParams) / sizeof(WeightQuantBatchMatmulV2WeightNzTilingTestParam), 3);
}
} // namespace