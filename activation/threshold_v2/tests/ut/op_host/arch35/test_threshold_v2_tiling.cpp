/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <stdlib.h>

#include <mutex>
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>

#include "log/log.h"
#include "nlohmann/json.hpp"

#define protected public
#define private public
#include "op_host/tiling_templates_registry.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/threshold_v2_tiling_arch35.h"

using namespace std;
using namespace ge;
using namespace ut_util;
using namespace optiling;

class ThresholdV2TilingTestParam {
public:
    void Prepare(ThresholdCompileInfo &compileInfo) const;
    void InvokeTilingFunc(ThresholdCompileInfo &compileInfo) const;
    void Test() const;
    std::string socVersion;
    std::string caseName;
    std::string kernelUtDir;
    std::string prefix;
    int64_t coreNum;
    int64_t dim;
    int64_t d1;
    int64_t d2;
    int64_t d3;
    int64_t d4;
    ge::DataType dtypeX;
    ge::DataType dtypeThreshold;
    ge::DataType dtypeValue;
    ge::DataType dtypeOutput;
    std::vector<int64_t> shapeXVec;
    std::vector<int64_t> shapeThresholdVec;
    std::vector<int64_t> shapeValueVec;
    std::vector<int64_t> shapeOutputVec;
    bool result;
    uint32_t numBlocks;
    uint64_t tilingKey;
    std::string comment;
};

static void SplitStr2Vec(const string &input, const string &delimiter, vector<string> &output)
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

static int64_t SafeStol(const string &str, int64_t defaultVal = 0)
{
    if (str.empty()) {
        return defaultVal;
    }
    try {
        return stol(str);
    } catch (...) {
        return defaultVal;
    }
}

static std::vector<int64_t> ParseShape(const std::string &shapeStr)
{
    std::vector<int64_t> shape;
    std::vector<std::string> dims;
    SplitStr2Vec(shapeStr, "_", dims);
    for (auto &dim : dims) {
        if (!dim.empty()) {
            shape.push_back(SafeStol(dim));
        }
    }
    return shape;
}

static void InitPlatformInfo(const std::string &socVersion, gert::TilingContext *tilingContext, string &compileInfoStr)
{
    map<string, string> soc_version_infos = {{"Short_SoC_version", socVersion}, {"NpuArch", "3510"}};
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "intrinsic_fix_pipe_l0c2out_f322bf16": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                          "Intrinsic_fix_pipe_pre_conv_cast": true,
                          "Intrinsic_data_move_l12bt": true,
                          "UB_SIZE": 245760, "L2_SIZE": 134217728, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 64,
                          "cube_core_cnt": 64, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
                          })";
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1800";

    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
}

static std::vector<ThresholdV2TilingTestParam> GetParams(const std::string &socVersion)
{
    std::vector<ThresholdV2TilingTestParam> params;
    std::string rootPath(GetExeDirPath() + "../../../../");
    std::string casePath(rootPath + "activation/threshold_v2/tests/ut/op_host/arch35/test_threshold_v2_tiling.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
        return params;
    }

    map<string, ge::DataType> dtypeMap = {{"FLOAT16", ge::DT_FLOAT16}, {"FLOAT", ge::DT_FLOAT},
                                          {"BF16", ge::DT_BF16},       {"INT8", ge::DT_INT8},
                                          {"UINT8", ge::DT_UINT8},     {"INT32", ge::DT_INT32},
                                          {"INT64", ge::DT_INT64},     {"DOUBLE", ge::DT_DOUBLE}};

    std::string line;
    while (std::getline(csvData, line)) {
        std::vector<std::string> testParam;
        SplitStr2Vec(line, ",", testParam);

        ThresholdV2TilingTestParam param;
        size_t idx = 0UL;
        param.socVersion = testParam[idx++];
        if (param.socVersion != socVersion) {
            continue;
        }

        param.caseName = testParam[idx++];
        param.kernelUtDir = testParam[idx++];
        param.prefix = testParam[idx++];
        param.coreNum = SafeStol(testParam[idx++], 64);
        param.dim = SafeStol(testParam[idx++]);
        param.d1 = SafeStol(testParam[idx++]);
        param.d2 = SafeStol(testParam[idx++]);
        param.d3 = SafeStol(testParam[idx++]);
        param.d4 = SafeStol(testParam[idx++]);
        param.dtypeX = dtypeMap[testParam[idx++]];
        param.dtypeThreshold = dtypeMap[testParam[idx++]];
        param.dtypeValue = dtypeMap[testParam[idx++]];
        param.dtypeOutput = dtypeMap[testParam[idx++]];
        param.shapeXVec = ParseShape(testParam[idx++]);
        param.shapeThresholdVec = ParseShape(testParam[idx++]);
        param.shapeValueVec = ParseShape(testParam[idx++]);
        param.shapeOutputVec = ParseShape(testParam[idx++]);
        param.result = (strcasecmp(testParam[idx++].c_str(), "true") == 0);
        param.numBlocks = SafeStol(testParam[idx++]);
        param.tilingKey = SafeStol(testParam[idx++]);
        if (idx < testParam.size()) {
            param.comment = testParam[idx++];
        }
        params.push_back(param);
    }

    return params;
}

static gert::Shape BuildShape(const std::vector<int64_t> &shapeVec)
{
    gert::Shape shape;
    for (auto dim : shapeVec) {
        shape.AppendDim(dim);
    }
    return shape;
}

void ThresholdV2TilingTestParam::Prepare(ThresholdCompileInfo &compileInfo) const
{
    compileInfo.coreNum = coreNum > 0 ? coreNum : 64;
    compileInfo.ubSize = 262144;

    gert::StorageShape xShape;
    gert::StorageShape thresholdShape;
    gert::StorageShape valueShape;
    gert::StorageShape outputShape;

    xShape.MutableOriginShape() = BuildShape(shapeXVec);
    thresholdShape.MutableOriginShape() = BuildShape(shapeThresholdVec);
    valueShape.MutableOriginShape() = BuildShape(shapeValueVec);
    outputShape.MutableOriginShape() = BuildShape(shapeOutputVec);

    xShape.MutableStorageShape() = xShape.MutableOriginShape();
    thresholdShape.MutableStorageShape() = thresholdShape.MutableOriginShape();
    valueShape.MutableStorageShape() = valueShape.MutableOriginShape();
    outputShape.MutableStorageShape() = outputShape.MutableOriginShape();

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    std::string opType("ThresholdV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());

    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(3, 1)
            .IrInstanceNum({1, 1, 1})
            .InputShapes({&xShape, &thresholdShape, &valueShape})
            .OutputShapes({&outputShape})
            .CompileInfo(&compileInfo)
            .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
            .NodeInputTd(0, dtypeX, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, dtypeThreshold, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, dtypeValue, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, dtypeOutput, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(rawTilingData.get())
            .Workspace(workspace)
            .SetOpType(opType)
            .Build();

    string compileInfoStr;
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();
    InitPlatformInfo(socVersion, tilingContext, compileInfoStr);
}

void ThresholdV2TilingTestParam::InvokeTilingFunc(ThresholdCompileInfo &compileInfo) const
{
    gert::StorageShape xShape;
    gert::StorageShape thresholdShape;
    gert::StorageShape valueShape;
    gert::StorageShape outputShape;

    cout << "run case " << prefix << std::endl;

    xShape.MutableOriginShape() = BuildShape(shapeXVec);
    thresholdShape.MutableOriginShape() = BuildShape(shapeThresholdVec);
    valueShape.MutableOriginShape() = BuildShape(shapeValueVec);
    outputShape.MutableOriginShape() = BuildShape(shapeOutputVec);

    xShape.MutableStorageShape() = xShape.MutableOriginShape();
    thresholdShape.MutableStorageShape() = thresholdShape.MutableOriginShape();
    valueShape.MutableStorageShape() = valueShape.MutableOriginShape();
    outputShape.MutableStorageShape() = outputShape.MutableOriginShape();

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    std::string opType("ThresholdV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());

    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(3, 1)
            .IrInstanceNum({1, 1, 1})
            .InputShapes({&xShape, &thresholdShape, &valueShape})
            .OutputShapes({&outputShape})
            .CompileInfo(&compileInfo)
            .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
            .NodeInputTd(0, dtypeX, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, dtypeThreshold, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, dtypeValue, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, dtypeOutput, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(rawTilingData.get())
            .Workspace(workspace)
            .SetOpType(opType)
            .Build();

    string compileInfoStr;
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();
    InitPlatformInfo(socVersion, tilingContext, compileInfoStr);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr) << "socVersion is: " << socVersion << ", caseName is: " << caseName
                                   << ", prefix is: " << prefix;
    if (result) {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS)
            << "socVersion is: " << socVersion << ", caseName is: " << caseName << ", prefix is: " << prefix;
        ASSERT_EQ(tilingContext->GetTilingKey(), tilingKey)
            << "socVersion is: " << socVersion << ", caseName is: " << caseName << ", prefix is: " << prefix;
        ASSERT_GT(tilingContext->GetBlockDim(), 0U)
            << "socVersion is: " << socVersion << ", caseName is: " << caseName << ", prefix is: " << prefix;
    } else {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED)
            << "socVersion is: " << socVersion << ", caseName is: " << caseName << ", prefix is: " << prefix;
    }
}

void ThresholdV2TilingTestParam::Test() const
{
    ThresholdCompileInfo compileInfo;
    Prepare(compileInfo);
    InvokeTilingFunc(compileInfo);
}

class TestThresholdV2Tiling : public testing::TestWithParam<ThresholdV2TilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ThresholdV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ThresholdV2Tiling TearDown" << std::endl;
    }
};

TEST_P(TestThresholdV2Tiling, generalTest)
{
    GetParam().Test();
}

INSTANTIATE_TEST_CASE_P(ThresholdV2Arch35, TestThresholdV2Tiling, testing::ValuesIn(GetParams("Ascend950")));

static void ThreadFunc(const ThresholdV2TilingTestParam *params, size_t testcaseNum, size_t threadIdx,
                       size_t threadNum)
{
    for (size_t idx = threadIdx; idx < testcaseNum; idx += threadNum) {
        params[idx].Test();
    }
}

static void TestMultiThread(const ThresholdV2TilingTestParam *params, size_t testcaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcaseNum, idx, threadNum);
    }

    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_F(TestThresholdV2Tiling, multiThread950)
{
    auto casesParams950 = GetParams("Ascend950");
    TestMultiThread(casesParams950.data(), casesParams950.size(), 3);
}