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

#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "log/log.h"

#include "op_host/tiling_templates_registry.h"
#include "../../../../common/op_host/op_tiling/tiling_type.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "../../../op_host/op_tiling/arch35/quant_batch_matmul_v4_tiling.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "../../../../common/op_host/math_util.h"
#include "ut_string_utils.h"

using namespace ut_str;

using namespace std;

struct QuantBatchMatmulV4TilingTestParam {
    string caseName;
    // output
    uint32_t numBlocks;
    ge::graphStatus tilingResult;
    uint64_t tilingKey;
};

class TestQuantBatchMatmulV4Tiling : public testing::TestWithParam<QuantBatchMatmulV4TilingTestParam> {};
class TestQuantBatchMatmulV4TilingRegbase
    : public testing::TestWithParam<QuantBatchMatmulV4TilingTestParam> {};

using namespace ge;
using namespace ut_util;
using namespace optiling;

static std::vector<QuantBatchMatmulV4TilingTestParam> GetParams()
{
    std::vector<QuantBatchMatmulV4TilingTestParam> params;
    const std::string rootPath(ut_str::GetExeDirPath() + "../../../../");
    const std::string casePath(
        rootPath + "matmul/quant_batch_matmul_v4/tests/ut/op_host/test_quant_batch_matmul_v4_tiling.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
        return params;
    }

    std::string line;
    bool headerSkipped = false;
    while (std::getline(csvData, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }

        std::vector<std::string> testParam;
        SplitStr2Vec(line, ",", testParam);
        if (testParam.size() < 4) {
            continue;
        }

        QuantBatchMatmulV4TilingTestParam param;
        size_t idx = 0;
        param.caseName = Trim(testParam[idx++]);
        param.numBlocks = static_cast<uint32_t>(std::stoul(Trim(testParam[idx++])));
        param.tilingResult = ParseGraphStatus(Trim(testParam[idx++]));
        param.tilingKey = std::stoull(Trim(testParam[idx++]));
        params.push_back(param);
    }
    return params;
}

static void TestOneParamCase(const QuantBatchMatmulV4TilingTestParam &param)
{
    std::cout << "run case " << param.caseName << std::endl;
    std::vector<string> testParam;
    SplitStr2Vec(param.caseName.substr(param.caseName.find_first_of('_') + 1), "_", testParam);
    ASSERT_GE(testParam.size(), 19U) << "invalid caseName in csv: " << param.caseName;

    size_t idx = 0;
    string socVersion = testParam[idx++];
    int64_t m = stol(testParam[idx++]);
    int64_t k = stol(testParam[idx++]);
    int64_t k0 = 16;
    int64_t k1 = ops::CeilDiv(k, k0);
    int64_t n = stol(testParam[idx++]);
    int64_t n0 = 32;
    int64_t n1 = ops::CeilDiv(n, n0);
    int64_t transA = stol(testParam[idx++]);
    int64_t transB = stol(testParam[idx++]);
    int64_t group = stol(testParam[idx++]);
    ge::Format x1Format = ParseFormat(testParam[idx++]);
    ge::Format x2Format = ParseFormat(testParam[idx++]);
    ge::DataType x1Dtype = ParseDtype(testParam[idx++]);
    ge::DataType x2Dtype = ParseDtype(testParam[idx++]);
    if (transB) {
        k0 = 32;
        k1 = ops::CeilDiv(k, k0);
        n0 = 16;
        n1 = ops::CeilDiv(n, n0);
    }
    bool hasBias = true;
    ge::DataType biasDtype = ge::DT_FLOAT;
    string biasDtypeStr = testParam[idx++];
    if (biasDtypeStr == "NULL") {
        hasBias = false;
    } else {
        biasDtype = ParseDtype(biasDtypeStr);
    }

    bool hasX1Scale = true;
    ge::DataType x1ScaleDtype = ge::DT_FLOAT;
    string x1ScaleDtypeStr = testParam[idx++];
    if (x1ScaleDtypeStr == "NULL") {
        hasX1Scale = false;
    } else {
        x1ScaleDtype = ParseDtype(x1ScaleDtypeStr);
    }

    bool hasX2Scale = true;
    ge::DataType x2ScaleDtype = ge::DT_FLOAT;
    string x2ScaleDtypeStr = testParam[idx++];
    if (x2ScaleDtypeStr == "NULL") {
        hasX2Scale = false;
    } else {
        x2ScaleDtype = ParseDtype(x2ScaleDtypeStr);
    }

    bool hasYScale = true;
    ge::DataType yScaleDtype = ge::DT_FLOAT;
    string yScaleDtypeStr = testParam[idx++];
    if (yScaleDtypeStr == "NULL") {
        hasYScale = false;
    } else {
        yScaleDtype = ParseDtype(yScaleDtypeStr);
    }

    ge::DataType x1OffsetDtype = ge::DT_FLOAT;
    ge::DataType x2OffsetDtype = ge::DT_FLOAT;
    ge::DataType yOffsetDtype = ge::DT_FLOAT;
    bool hasX2Table = true;
    ge::DataType x2TableDtype = ge::DT_FLOAT;
    string x2TableDtypeStr = testParam[idx++];
    if (x2TableDtypeStr == "NULL") {
        hasX2Table = false;
    } else {
        x2TableDtype = ParseDtype(x2TableDtypeStr);
    }

    ge::DataType yDtype = ParseDtype(testParam[idx++]);
    uint32_t aicNum = stoul(testParam[idx++]);
    uint32_t aivNum = stoul(testParam[idx++]);
    string compileInfoStr = R"({
         "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                           "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l12bt": true,
                           "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 196352, "L2_SIZE": 33554432, "L1_SIZE": 524032,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": aicNum,
                           "cube_core_cnt": aicNum, "vector_core_cnt": aivNum, "core_type_list": "CubeCore,VectorCore"}
                            })";
    if (socVersion.find("RESERVED") != string::npos) {
        compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l12bt": true,
                           "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 262144, "L2_SIZE": 16777216, "L1_SIZE": 1048576,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": aicNum,
                           "cube_core_cnt": aicNum, "vector_core_cnt": aivNum, "core_type_list": "CubeCore,VectorCore",
                           "Intrinsic_mmad": true, "lut_type": "MTE2_QTABLE"}
                            })";
    }

    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape biasShape;
    gert::StorageShape x1ScaleShape;
    gert::StorageShape x2ScaleShape;
    gert::StorageShape yScaleShape;
    gert::StorageShape x2TableShape;
    gert::StorageShape outputShape({m, n}, {m, n});

    if (transA) {
        x1Shape.MutableStorageShape() = gert::Shape({k, m});
    } else {
        x1Shape.MutableStorageShape() = gert::Shape({m, k});
    }
    x1Shape.MutableOriginShape() = x1Shape.MutableStorageShape();
    if (x2Format == ge::FORMAT_ND) {
        if (transB) {
            x2Shape.MutableStorageShape() = gert::Shape({n, k});
        } else {
            x2Shape.MutableStorageShape() = gert::Shape({k, n});
        }
        x2Shape.MutableOriginShape() = x2Shape.MutableStorageShape();
    } else if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        if (transB) {
            x2Shape.MutableOriginShape() = gert::Shape({n, k});
            x2Shape.MutableStorageShape() = gert::Shape({k1, n1, n0, k0});
        } else {
            x2Shape.MutableOriginShape() = gert::Shape({k, n});
            x2Shape.MutableStorageShape() = gert::Shape({n1, k1, k0, n0});
        }
    }
    biasShape.MutableOriginShape() = gert::Shape({1, n});
    biasShape.MutableStorageShape() = gert::Shape({1, n});
    int64_t groupSize = 0;
    int64_t groupK = 0;
    int64_t groupN = 0;
    int64_t groupM = 0;
    if (group > 0) {
        groupSize = group;
        groupK = group & 0xFFFF; // 0-15bit group_k
        groupN = static_cast<int64_t>((group & 0xFFFF0000) >> 16); // 16-31bit group_n
        groupM = static_cast<int64_t>((group & 0xFFFF00000000) >> 32); // 32-47bit group_m
        int64_t groupNum = (k + group - 1) / group;
        if (!hasX2Table) {
            x1ScaleShape.MutableStorageShape() =
                x1ScaleDtype == ge::DT_FLOAT8_E8M0 ? gert::Shape({m, groupNum / 2, 2}) : gert::Shape({m, groupNum});
            if (transB) {
                x2ScaleShape.MutableStorageShape() =
                    x2ScaleDtype == ge::DT_FLOAT8_E8M0 ? gert::Shape({n, groupNum / 2, 2}) : gert::Shape({n, groupNum});
            } else {
                x2ScaleShape.MutableStorageShape() = gert::Shape({groupNum, n});
            }
        } else {
            x2ScaleShape.MutableStorageShape() = gert::Shape({n});
        }
    } else if (group < 0) {
        x2ScaleShape.MutableStorageShape() = gert::Shape({n});
    } else {
        x2ScaleShape.MutableStorageShape() = gert::Shape({1});
    }
    yScaleShape.MutableStorageShape() = gert::Shape({1, n});
    if (hasX2Table && groupK > 0 && groupN > 0 &&
        (x2Dtype == ge::DT_INT2 || x2Dtype == ge::DT_INT4 || x2Dtype == ge::DT_UINT1)) {
        x2TableShape.MutableStorageShape() = gert::Shape({(k + groupK - 1) / groupK, (n + groupN - 1) / groupN * 16});
    }

    x2ScaleShape.MutableOriginShape() = x2ScaleShape.MutableStorageShape();
    x2TableShape.MutableOriginShape() = x2TableShape.MutableStorageShape();

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> version;
    // 6为替换原aicNum字符串的长度，配置CORE_NUM
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(aicNum));
    // 6为替换原aicNum字符串的长度，配置cube_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(aicNum));
    // 6为替换原aivNum字符串的长度，配置vector_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aivNum"), 6, to_string(aivNum));
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics, version);
    aicoreSpec["cube_freq"] = "1800";

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    QuantBatchMatmulV4CompileInfo compileInfo;

    auto kernelHold = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    std::string opType("QuantBatchMatmulV4");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&x1Shape, &x2Shape, hasBias ? &biasShape : nullptr,
                                    hasX1Scale ? &x1ScaleShape : nullptr, hasX2Scale ? &x2ScaleShape : nullptr,
                                    hasYScale ? &yScaleShape : nullptr, nullptr, nullptr, nullptr, hasX2Table ?
                                    &x2TableShape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, x1Dtype, x1Format, x1Format)
                      .NodeInputTd(1, x2Dtype, x2Format, x2Format)
                      .NodeInputTd(2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, x1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, x2ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, yScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, x1OffsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, x2OffsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, yOffsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, x2TableDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)},
                                  {"compute_type", Ops::NN::AnyValue::CreateFrom<bool>(groupSize)},
                                  {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(transB)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .SetOpType(opType)
                      .Build();

    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", version);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS)
        << "caseName: " << param.caseName;

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    ge::graphStatus ret = tilingFunc(tilingContext);
    ASSERT_EQ(ret, param.tilingResult)
        << "caseName: " << param.caseName;
    if (ret == ge::GRAPH_SUCCESS) {
        ASSERT_EQ(tilingContext->GetTilingKey(), param.tilingKey)
            << "caseName: " << param.caseName << ", actual tilingKey: " << tilingContext->GetTilingKey()
            << ", expected tilingKey: " << param.tilingKey;
        ASSERT_EQ(tilingContext->GetBlockDim(), param.numBlocks)
            << "caseName: " << param.caseName << ", actual blockDim: " << tilingContext->GetBlockDim()
            << ", expected blockDim: " << param.numBlocks;
    }
}

TEST_P(TestQuantBatchMatmulV4Tiling, generalTest)
{
    QuantBatchMatmulV4TilingTestParam param = GetParam();
    TestOneParamCase(param);
}

static const std::vector<QuantBatchMatmulV4TilingTestParam> kCaseParams = GetParams();

INSTANTIATE_TEST_CASE_P(MM, TestQuantBatchMatmulV4Tiling, testing::ValuesIn(kCaseParams));
