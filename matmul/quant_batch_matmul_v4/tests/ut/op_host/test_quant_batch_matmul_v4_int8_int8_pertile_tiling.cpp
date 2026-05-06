/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
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
    uint32_t blockDim;
    ge::graphStatus tilingResult;
    uint64_t tilingKey;
};

class TestQuantBatchMatmulV4PerTileTiling : public testing::TestWithParam<QuantBatchMatmulV4TilingTestParam> {};
class TestQuantBatchMatmulV4TilingRegbase : public testing::TestWithParam<QuantBatchMatmulV4TilingTestParam> {};

using namespace ge;
using namespace ut_util;
using namespace optiling;

static std::vector<QuantBatchMatmulV4TilingTestParam> GetParams()
{
    std::vector<QuantBatchMatmulV4TilingTestParam> params;
    const std::string rootPath(ut_str::GetExeDirPath() + "../../../../");
    const std::string casePath(
        rootPath + "matmul/quant_batch_matmul_v4/tests/ut/op_host/test_quant_batch_matmul_v4_int8_int8_pertile_tiling.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
        return params;
    }

    std::string line;
    bool skipHeader = true;
    while (std::getline(csvData, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (skipHeader) {
            skipHeader = false;
            continue;
        }
        std::vector<std::string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() < 4UL) {
            continue;
        }

        QuantBatchMatmulV4TilingTestParam param;
        param.caseName = Trim(cols[0]);
        param.blockDim = static_cast<uint32_t>(std::stoul(Trim(cols[1])));
        param.tilingResult = ParseGraphStatus(cols[2]);
        param.tilingKey = std::stoull(Trim(cols[3]));
        params.push_back(param);
    }
    return params;
}

static void TestOneParamCase(const QuantBatchMatmulV4TilingTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    std::vector<string> testParam;
    SplitStr2Vec(param.caseName.substr(param.caseName.find_first_of('_') + 1), "_", testParam);

    size_t idx = 0;
    string socversion = testParam[idx++];
    int64_t m = stol(testParam[idx++]);
    int64_t k = stol(testParam[idx++]);
    int64_t k0 = 16;
    int64_t k1 = ops::CeilDiv(k, k0);
    int64_t n = stol(testParam[idx++]);
    int64_t n0 = 32;
    int64_t n1 = ops::CeilDiv(n, n0);
    int64_t transA = stol(testParam[idx++]);
    int64_t transB = stol(testParam[idx++]);
    int64_t groupSize = stol(testParam[idx++]);
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

    bool hasX2Table = false;
    ge::DataType x2TableDtype = ge::DT_FLOAT;

    ge::DataType yDtype = ParseDtype(testParam[idx++]);
    uint32_t aicNum = stoul(testParam[idx++]);
    uint32_t aivNum = stoul(testParam[idx++]);

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

    if (transB) {
        x2Shape.MutableStorageShape() = gert::Shape({n, k});
    } else {
        x2Shape.MutableStorageShape() = gert::Shape({k, n});
    }
    x2Shape.MutableOriginShape() = x2Shape.MutableStorageShape();

    biasShape.MutableOriginShape() = gert::Shape({n});
    biasShape.MutableStorageShape() = gert::Shape({n});
    int64_t groupK = 0;
    int64_t groupN = 0;
    int64_t groupM = 0;

    int64_t scaleM = m;
    int64_t scaleK = (k + 127) / 128;
    int64_t scaleN = (n + 127) / 128;
    if (transA) {
        x1ScaleShape.MutableStorageShape() = gert::Shape({scaleK, scaleM});
    } else {
        x1ScaleShape.MutableStorageShape() = gert::Shape({scaleM, scaleK});
    }
    if (transB) {
        x2ScaleShape.MutableStorageShape() = gert::Shape({scaleN, scaleK});
    } else {
        x2ScaleShape.MutableStorageShape() = gert::Shape({scaleK, scaleN});
    }

    yScaleShape.MutableStorageShape() = gert::Shape({1, n});
    if (hasX2Table && groupK > 0 && groupN > 0 &&
        (x2Dtype == ge::DT_INT2 || x2Dtype == ge::DT_INT4 || x2Dtype == ge::DT_UINT1)) {
        x2TableShape.MutableStorageShape() = gert::Shape({(k + groupK - 1) / groupK, (n + groupN - 1) / groupN * 16});
    }
    x1ScaleShape.MutableOriginShape() = x1ScaleShape.MutableStorageShape();
    x2ScaleShape.MutableOriginShape() = x2ScaleShape.MutableStorageShape();
    string compileInfoStr = R"({
                            "hardware_info" : {
                                "BT_SIZE" : 4096,
                                "load3d_constraints" : "unknown",
                                "Intrinsic_fix_pipe_l0c2out" : true,
                                "Intrinsic_data_move_l12ub" : false,
                                "Intrinsic_data_move_l0c2ub" : false,
                                "Intrinsic_data_move_l12bt" : true,
                                "Intrinsic_data_move_out2l1_nd2nz" : true,
                                "UB_SIZE" : 253952,
                                "L2_SIZE" : 134217728,
                                "L1_SIZE" : 524288,
                                "L0A_SIZE" : 65536,
                                "L0B_SIZE" : 65536,
                                "L0C_SIZE" : 262144,
                                "CORE_NUM" : aicNum,
                                "cube_core_cnt": aicNum,
                                "vector_core_cnt": aivNum,
                                "core_type_list": "CubeCore,VectorCore",
                                "socVersion" : "Ascend950",
                                "NpuArch" : "3510"
                            }})";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion;
    // 6为替换原aicNum字符串的长度，配置CORE_NUM
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(aicNum));
    // 6为替换原aicNum字符串的长度，配置cube_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(aicNum));
    // 6为替换原aivNum字符串的长度，配置vector_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aivNum"), 6, to_string(aivNum));
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    aicoreSpec["cube_freq"] = "1650";

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
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&x1Shape, &x2Shape, hasBias ? &biasShape : nullptr, &x1ScaleShape, &x2ScaleShape, nullptr,
                           nullptr, nullptr, nullptr, nullptr})
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
                      .NodeAttrs(
                          {{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)},
                           {"compute_type", Ops::NN::AnyValue::CreateFrom<bool>(groupSize)},
                           {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
                           {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(transB)},
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
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersion);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    ge::graphStatus ret = tilingFunc(tilingContext);
    ASSERT_EQ(ret, param.tilingResult);
    if (ret == ge::GRAPH_SUCCESS) {
        ASSERT_EQ(tilingContext->GetTilingKey(), param.tilingKey);
        ASSERT_EQ(tilingContext->GetBlockDim(), param.blockDim);
    }
}

TEST_P(TestQuantBatchMatmulV4PerTileTiling, generalTest)
{
    QuantBatchMatmulV4TilingTestParam param = GetParam();
    TestOneParamCase(param);
}

// format: caseName m k n transA transB groupSize x1Format x2Format x1Dtype x2Dtype biasDtype x1ScaleDtype
//         x2ScaleDtype yScaleDtype x2TableDtype yDtype aicNum aivNum platform weightFormat
INSTANTIATE_TEST_CASE_P(MM, TestQuantBatchMatmulV4PerTileTiling, testing::ValuesIn(GetParams()));
