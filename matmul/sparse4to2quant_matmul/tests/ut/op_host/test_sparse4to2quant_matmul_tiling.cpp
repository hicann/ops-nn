/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "tiling_base/tiling_templates_registry.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../op_host/op_tiling/sparse4to2quant_matmul_tiling.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;
using namespace ut_util;
using namespace optiling;

class Sparse4to2QuantMatmulTilingTestParam {
public:
    void Prepare(Sparse4to2QuantMatmulCompileInfo &compileInfo) const;
    void InvokeTilingFunc(Sparse4to2QuantMatmulCompileInfo &compileInfo) const;
    void Test() const;
    std::string socVersion;
    std::string caseName;
    std::string kernelUtDir;
    std::string prefix;
    int64_t coreNum;
    int64_t x1Dim;
    int64_t x2Dim;
    int64_t yDim;
    int64_t m;
    int64_t k;
    int64_t n;
    bool biasFlag;
    ge::DataType xDtype;
    ge::DataType sparseWeightDtype;
    ge::DataType indexDtype;
    ge::DataType xScaleDtype;
    ge::DataType sparseWeightScaleDtype;
    ge::DataType biasDtype;
    ge::DataType yDtype;

    // output
    bool result; // false means tiling fail
    uint32_t blockDim;
    uint64_t tilingKey;
    std::string tilingData;
    bool tilingStub; // 是否tililg打桩，只给kernel的用例，此时tiling ut里不校验tiling出参
};

#define GET_API_TILING_VALUE(obj, fieldName) *(int32_t *)(obj.data_ptr_ + obj.fieldName##_offset_)

static string TilingData2Str(const void *tilingData, size_t tilingSize)
{
    string result;
    for (size_t i = 0; i < tilingSize; i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t *>(tilingData)[i / sizeof(int32_t)]));
        result += " ";
    }

    return result;
}

static gert::Shape TransNd2Nz(const gert::Shape &inShape) {
    gert::Shape outShape;
    for (size_t idx = 0; idx < inShape.GetDimNum() - 2; ++idx) {
        outShape.AppendDim(inShape.GetDim(idx));
    }

    int64_t m = inShape.GetDim(inShape.GetDimNum() - 2);
    int64_t n = inShape.GetDim(inShape.GetDimNum() - 1);
    outShape.AppendDim((n + 31) / 32);
    outShape.AppendDim((m + 15) / 16);
    outShape.AppendDim(16);
    outShape.AppendDim(32);
    return outShape;
}

class TestSparse4to2QuantMatmulTiling : public testing::TestWithParam<Sparse4to2QuantMatmulTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
    }

    static void TearDownTestCase()
    {
    }
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

static void InitPlatformInfo(const std::string &socVersion, gert::TilingContext *tilingContext, string &compileInfoStr,
                             int64_t coreNum = -1)
{
    map<string, string> soc_version_infos = {{"SoC_version", socVersion},};
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196352, "L2_SIZE": 201326592, "L1_SIZE": 524032,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
    if (socVersion.compare("Ascend310P3") == 0) {
        compileInfoStr = R"({
        "hardware_info": {"load3d_constraints": "1",
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "UB_SIZE": 262144, "L2_SIZE": 16777216, "L1_SIZE": 1048576,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8,
                          "ai_core_cnt": 8, "vector_core_cnt": 47, "core_type_list": "AiCore,VectorCore"}
                          })";
    } else if (socVersion.compare("Ascend910B4") == 0) {
        compileInfoStr = R"({
            "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                            "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196352, "L2_SIZE": 201326592, "L1_SIZE": 524032,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20,
                            "cube_core_cnt": 20, "vector_core_cnt": 40, "core_type_list": "CubeCore,VectorCore"}
                            })";
    }
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1800";
    if (socVersion.compare("Ascend310P3") == 0) {
        aicoreSpec["cube_freq"] = "1000";
    } else if (socVersion.compare("Ascend910B4") == 0) {
        aicoreSpec["cube_freq"] = "1650";
    }

    if (coreNum > 0) {
        socInfos["ai_core_cnt"] = std::to_string(coreNum);
        socInfos["cube_core_cnt"] = std::to_string(coreNum);
        socInfos["vector_core_cnt"] = std::to_string(coreNum * 2);
    }

    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
}

static std::vector<Sparse4to2QuantMatmulTilingTestParam> GetParams(const std::string &socVersion)
{
    std::vector<Sparse4to2QuantMatmulTilingTestParam> params;
    std::string rootPath(GetExeDirPath() + "../../../../");
    std::string casePath(rootPath + "matmul/sparse4to2quant_matmul/tests/ut/op_host/test_sparse4to2quant_matmul.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
        return params;
    }

    map<string, ge::DataType> dtypeMap = {{"FLOAT16", ge::DT_FLOAT16}, {"FLOAT", ge::DT_FLOAT},
                                          {"BF16", ge::DT_BF16},       {"INT8", ge::DT_INT8},
                                          {"INT4", ge::DT_INT4},       {"UINT64", ge::DT_UINT64},
                                          {"INT32", ge::DT_INT32},     {"INT64", ge::DT_INT64}, };

    std::string line;
    while (std::getline(csvData, line)) {
        std::vector<std::string> testParam;
        SplitStr2Vec(line, ",", testParam);

        Sparse4to2QuantMatmulTilingTestParam param;
        size_t idx = 0UL;
        param.socVersion = testParam[idx++];
        if (param.socVersion != socVersion) {
            continue;
        }

        param.caseName = testParam[idx++];
        param.kernelUtDir = testParam[idx++];
        param.prefix = testParam[idx++];
        auto coreNum = testParam[idx++];
        if (coreNum.empty()) {
            param.coreNum = -1;
        } else {
            param.coreNum = stol(coreNum);
        }
        param.x1Dim = stol(testParam[idx++]);
        param.x2Dim = stol(testParam[idx++]);
        param.yDim = stol(testParam[idx++]);
        param.m = stol(testParam[idx++]);
        param.k = stol(testParam[idx++]);
        param.n = stol(testParam[idx++]);
        param.biasFlag = stol(testParam[idx++]);

        param.xDtype = dtypeMap[testParam[idx++]];
        param.sparseWeightDtype = dtypeMap[testParam[idx++]];
        param.indexDtype = dtypeMap[testParam[idx++]];
        param.xScaleDtype = dtypeMap[testParam[idx++]];
        param.sparseWeightScaleDtype = dtypeMap[testParam[idx++]];
        param.biasDtype = dtypeMap[testParam[idx++]];
        param.yDtype = dtypeMap[testParam[idx++]];

        param.result = (strcasecmp(testParam[idx++].c_str(), "true") == 0);
        param.blockDim = stol(testParam[idx++]);
        param.tilingKey = stol(testParam[idx++]);
        param.tilingData = testParam[idx++];
        param.tilingStub = (strcasecmp(testParam[idx++].c_str(), "true") == 0);
        params.push_back(param);
    }

    return params;
}

void Sparse4to2QuantMatmulTilingTestParam::Prepare(Sparse4to2QuantMatmulCompileInfo &compileInfo) const
{
    gert::StorageShape xShape;
    gert::StorageShape sparseWeightShape;
    gert::StorageShape indexShape;
    gert::StorageShape xScaleShape;
    gert::StorageShape sparseWeightscaleShape;
    gert::StorageShape biasShape;
    gert::StorageShape outputShape;

    xShape.MutableOriginShape() = gert::Shape({m, k});
    sparseWeightShape.MutableOriginShape() = gert::Shape({n, k / 2});
    indexShape.MutableOriginShape() = gert::Shape({(n + 63) / 64 * 64, (k + 15) / 16 * 16, 16, 8});
    xScaleShape.MutableStorageShape() = gert::Shape({m});
    sparseWeightscaleShape.MutableStorageShape() = gert::Shape({n});
    biasShape.MutableStorageShape() = gert::Shape({n});
    outputShape.MutableOriginShape() = gert::Shape({m, n});

    sparseWeightscaleShape.MutableOriginShape() = sparseWeightscaleShape.MutableStorageShape();
    biasShape.MutableOriginShape() = biasShape.MutableStorageShape();
    xScaleShape.MutableOriginShape() = xScaleShape.MutableStorageShape();
    sparseWeightscaleShape.MutableOriginShape() = sparseWeightscaleShape.MutableStorageShape();
    biasShape.MutableOriginShape() = biasShape.MutableStorageShape();
    outputShape.MutableStorageShape() = outputShape.MutableOriginShape();

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    std::string opType("Sparse4to2QuantMatmul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());
    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(6, 1)
            .IrInstanceNum({1, 1, 1, 1, 1, 1})
            .InputShapes({&xShape, &sparseWeightShape, &indexShape, &xScaleShape, &sparseWeightscaleShape, &biasShape})
            .OutputShapes({&outputShape})
            .CompileInfo(&compileInfo)
            .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
            .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, sparseWeightDtype, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
            .NodeInputTd(2, indexDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, xScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, sparseWeightScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(yDtype)}})
            .TilingData(rawTilingData.get())
            .Workspace(workspace)
            .SetOpType(opType)
            .Build();

    string compileInfoStr;
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();
    InitPlatformInfo(socVersion, tilingContext, compileInfoStr, coreNum);

    auto kernelHold = gert::KernelRunContextFaker()
                          .KernelIONum(2, 1)
                          .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                          .Outputs({&compileInfo})
                          .Build();

    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

void Sparse4to2QuantMatmulTilingTestParam::InvokeTilingFunc(Sparse4to2QuantMatmulCompileInfo &compileInfo) const
{   gert::StorageShape xShape;
    gert::StorageShape sparseWeightShape;
    gert::StorageShape indexShape;
    gert::StorageShape xScaleShape;
    gert::StorageShape sparseWeightscaleShape;
    gert::StorageShape biasShape;
    gert::StorageShape outputShape;

    cout<<"run case "<<prefix<<std::endl;

    xShape.MutableOriginShape() = gert::Shape({m, k});
    sparseWeightShape.MutableOriginShape() = gert::Shape({n, k / 2});
    indexShape.MutableOriginShape() = gert::Shape({(n + 63) / 64 * 64, (k + 15) / 16 * 16, 16, 8});
    xScaleShape.MutableStorageShape() = gert::Shape({m});
    sparseWeightscaleShape.MutableStorageShape() = gert::Shape({n});
    biasShape.MutableStorageShape() = gert::Shape({n});
    outputShape.MutableOriginShape() = gert::Shape({m, n});

    sparseWeightscaleShape.MutableOriginShape() = sparseWeightscaleShape.MutableStorageShape();
    biasShape.MutableOriginShape() = biasShape.MutableStorageShape();
    xScaleShape.MutableOriginShape() = xScaleShape.MutableStorageShape();
    sparseWeightscaleShape.MutableOriginShape() = sparseWeightscaleShape.MutableStorageShape();
    biasShape.MutableOriginShape() = biasShape.MutableStorageShape();
    outputShape.MutableStorageShape() = outputShape.MutableOriginShape();

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    std::string opType("Sparse4to2QuantMatmul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());
    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(6, 1)
            .IrInstanceNum({1, 1, 1, 1, 1, 1})
            .InputShapes({&xShape, &sparseWeightShape, &indexShape, &xScaleShape, &sparseWeightscaleShape, &biasShape})
            .CompileInfo(&compileInfo)
            .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
            .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, sparseWeightDtype, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
            .NodeInputTd(2, indexDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, xScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, sparseWeightScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(yDtype)}})
            .TilingData(rawTilingData.get())
            .Workspace(workspace)
            .SetOpType(opType)
            .Build();

    string compileInfoStr;
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    if (result) {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
        if (tilingStub) {
            return;
        }
        ASSERT_EQ(tilingContext->GetTilingKey(), tilingKey);
        ASSERT_EQ(tilingContext->GetBlockDim(), blockDim);
    } else {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
    }
}

void Sparse4to2QuantMatmulTilingTestParam::Test() const
{
    Sparse4to2QuantMatmulCompileInfo compileInfo;
    Prepare(compileInfo);
    InvokeTilingFunc(compileInfo);
}


TEST_P(TestSparse4to2QuantMatmulTiling, generalTest)
{
    GetParam().Test();
}

INSTANTIATE_TEST_CASE_P(QUANTMM910B, TestSparse4to2QuantMatmulTiling, testing::ValuesIn(GetParams("Ascend910B2")));
INSTANTIATE_TEST_CASE_P(QUANTMM910B4, TestSparse4to2QuantMatmulTiling, testing::ValuesIn(GetParams("Ascend910B4")));

static void ThreadFunc(const Sparse4to2QuantMatmulTilingTestParam *params, size_t testcaseNum, size_t threadIdx,
                       size_t threadNum)
{
    int32_t logLevel = 0;
    int32_t enableEvent = 0;
    for (size_t idx = threadIdx; idx < testcaseNum; idx += threadNum) {
        params[idx].Test();
    }
}

namespace {
mutex sparseQuantMatmulcompileMutex;
}

static void ThreadFuncPrepare(const Sparse4to2QuantMatmulTilingTestParam *params, size_t testcaseNum, size_t threadIdx,
                       size_t threadNum, map<size_t, Sparse4to2QuantMatmulCompileInfo> &compileInfos)
{
    if (threadIdx >= testcaseNum)
        return;
    int32_t logLevel = 0;
    int32_t enableEvent = 0;
    Sparse4to2QuantMatmulCompileInfo compileInfo;
    params[threadIdx].Prepare(compileInfo);

    {
        lock_guard<mutex> lock(sparseQuantMatmulcompileMutex);
        compileInfos[threadIdx] = compileInfo;
    }
}

static void ThreadFuncInvokeTilingFunc(const Sparse4to2QuantMatmulTilingTestParam *params, size_t testcaseNum, size_t threadIdx,
                       size_t threadNum, Sparse4to2QuantMatmulCompileInfo &compileInfo)
{
    if (threadIdx >= testcaseNum)
        return;
    int32_t logLevel = 0;
    int32_t enableEvent = 0;
    params[threadIdx].InvokeTilingFunc(compileInfo);
}

static void TestMultiThread(const Sparse4to2QuantMatmulTilingTestParam *params, size_t testcaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcaseNum, idx, threadNum);
    }

    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

static void TestMultiThreadSeparate(const Sparse4to2QuantMatmulTilingTestParam *params, size_t testcaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    map<size_t, Sparse4to2QuantMatmulCompileInfo> compileInfos;
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFuncPrepare, params, testcaseNum, idx, threadNum, std::ref(compileInfos));
    }

    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }

    std::thread threadsInvoke[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threadsInvoke[idx] = std::thread(ThreadFuncInvokeTilingFunc, params, testcaseNum, idx, threadNum, std::ref(compileInfos[idx]));
    }

    for (size_t idx = 0; idx < threadNum; ++idx) {
        threadsInvoke[idx].join();
    }
}


