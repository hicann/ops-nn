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
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "log/log.h"
#include "nlohmann/json.hpp"

#define protected public
#define private public
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "matmul/fused_quant_mat_mul/op_host/op_tiling/arch35/fused_quant_matmul_asw_tiling.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "ut_op_util.h"

using namespace std;
using namespace ge;
using namespace ut_util;
using namespace optiling;


#define GET_TILING_VALUE(obj, fieldName) *(int32_t *)(obj.data_ptr_ + obj.fieldName##_offset_)

class FusedQuantMatMulTilingTestParam {
public:
    void Prepare(FusedQuantMatMulCompileInfo &compileInfo) const;
    void InvokeTilingFunc(FusedQuantMatMulCompileInfo &compileInfo) const;
    void Test() const;
    std::string socVersion;
    std::string caseName;
    int64_t coreNum;
    int64_t x1Dim;
    int64_t x2Dim;
    int64_t yDim;
    int64_t batchA;
    int64_t batchB;
    int64_t batchC;
    int64_t m;
    int64_t k;
    int64_t n;
    bool offsetFlag;
    bool biasFlag;
    bool transA;
    bool transB;
    ge::DataType x1Dtype;
    ge::DataType x2Dtype;
    ge::DataType scaleDtype;
    ge::DataType biasDtype;
    ge::DataType yDtype;
    int64_t quantMode = 0;
    bool x2Nz;
    std::string fusedTypeOp;
    // output
    bool tilingResult;
    uint64_t tilingKey;
};

class TestFusedQuantMatMulTiling : public testing::TestWithParam<FusedQuantMatMulTilingTestParam> {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

namespace fqmm_tiling_ut {

static string TilingData2Str(const void *tilingData, size_t tilingSize) {
    string result;
    for (size_t i = 0; i < tilingSize; i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t *>(tilingData)[i / sizeof(int32_t)]));
        result += " ";
    }

    return result;
}

map<ge::DataType, int64_t> dtypeC0SizeMap = {
    {ge::DT_FLOAT16, 16}, {ge::DT_FLOAT, 8}, {ge::DT_INT8, 32}, {ge::DT_INT4, 64}, {ge::DT_INT2, 64},
};

static gert::Shape TransNd2Nz(const gert::Shape &inShape, const ge::DataType &dtype) {
    gert::Shape outShape;
    for (size_t idx = 0; idx < inShape.GetDimNum() - 2; ++idx) {
        outShape.AppendDim(inShape.GetDim(idx));
    }

    int64_t m = inShape.GetDim(inShape.GetDimNum() - 2);
    int64_t n = inShape.GetDim(inShape.GetDimNum() - 1);
    int64_t c0 = dtypeC0SizeMap[dtype];
    outShape.AppendDim((n + c0 - 1) / c0);
    outShape.AppendDim((m + 15) / 16);
    outShape.AppendDim(16);
    outShape.AppendDim(c0);
    return outShape;
}

static void SplitStr2Vec(const string &input, const string &delimiter, vector<string> &output) {
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

static void InitPlatformInfo(const std::string &socVersion, string &compileInfoStr) {
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
    if (socVersion.compare("Ascend910_95") == 0) {
        compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "intrinsic_fix_pipe_l0c2out_f322bf16": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                          "Intrinsic_fix_pipe_pre_conv_cast": true,
                          "Intrinsic_data_move_l12bt": true,
                          "UB_SIZE": 245760, "L2_SIZE": 134217728, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
                          "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
                          })";
    } else if (socVersion.compare("RESERVED") == 0) {
        compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
                        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                        "intrinsic_fix_pipe_l0c2out_f322bf16": true,
                        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                        "Intrinsic_fix_pipe_pre_conv_cast": true,
                        "Intrinsic_data_move_l12bt": true,
                        "Intrinsic_mmad": true,
                        "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 1048576,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": aicNum,
                        "cube_core_cnt": aicNum, "vector_core_cnt": aivNum, "core_type_list": "CubeCore,VectorCore",
                        "lut_type": "MTE2_QTABLE"}
                        })";
    }
}

static std::vector<FusedQuantMatMulTilingTestParam> GetParams() {
    std::vector<FusedQuantMatMulTilingTestParam> params;
    std::string rootPath(GetExeDirPath() + "../../../../");
    std::string casePath(rootPath + "matmul/fused_quant_mat_mul/tests/ut/op_host/test_fused_quant_mat_mul.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
        return params;
    }

    map<string, ge::DataType> dtypeMap = {{"FLOAT16", ge::DT_FLOAT16},
                                          {"FLOAT", ge::DT_FLOAT},
                                          {"BF16", ge::DT_BF16},
                                          {"INT8", ge::DT_INT8},
                                          {"INT4", ge::DT_INT4},
                                          {"UINT64", ge::DT_UINT64},
                                          {"INT32", ge::DT_INT32},
                                          {"INT64", ge::DT_INT64},
                                          {"FLOAT8-E8M0", ge::DT_FLOAT8_E8M0},
                                          {"HIFLOAT8", ge::DT_HIFLOAT8},
                                          {"FLOAT8-E5M2", ge::DT_FLOAT8_E5M2},
                                          {"FLOAT8-E4M3", ge::DT_FLOAT8_E4M3FN},
                                          {"FLOAT4-E2M1", ge::DT_FLOAT4_E2M1},
                                          {"FLOAT4-E1M2", ge::DT_FLOAT4_E1M2}};

    std::string line;
    while (std::getline(csvData, line)) {
        std::vector<std::string> testParam;
        SplitStr2Vec(line, ",", testParam);

        FusedQuantMatMulTilingTestParam param;
        size_t idx = 0UL;
        param.socVersion = testParam[idx++];
        if (param.socVersion == "socVersion") {
            // skip title line
            continue;
        }
        param.caseName = testParam[idx++];
        param.coreNum = stol(testParam[idx++]);
        param.x1Dim = stol(testParam[idx++]);
        param.x2Dim = stol(testParam[idx++]);
        param.yDim = stol(testParam[idx++]);
        param.batchA = stol(testParam[idx++]);
        param.batchB = stol(testParam[idx++]);
        param.batchC = stol(testParam[idx++]);
        param.m = stol(testParam[idx++]);
        param.k = stol(testParam[idx++]);
        param.n = stol(testParam[idx++]);
        param.offsetFlag = stol(testParam[idx++]);
        param.biasFlag = stol(testParam[idx++]);
        param.transA = stol(testParam[idx++]);
        param.transB = stol(testParam[idx++]);
        param.quantMode = stol(testParam[idx++]);
        param.x1Dtype = dtypeMap[testParam[idx++]];
        param.x2Dtype = dtypeMap[testParam[idx++]];
        param.scaleDtype = dtypeMap[testParam[idx++]];
        param.biasDtype = dtypeMap[testParam[idx++]];
        param.yDtype = dtypeMap[testParam[idx++]];
        param.x2Nz =  testParam[idx++] == "NZ";
        param.fusedTypeOp = testParam[idx++];
        param.tilingResult = (strcasecmp(testParam[idx++].c_str(), "true") == 0) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
        param.tilingKey = stol(testParam[idx++]);
        params.push_back(param);
    }

    return params;
}

static void TestOneParamCase(const FusedQuantMatMulTilingTestParam &param) {
    std::cout << "run case " << param.caseName << std::endl;

    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape x3Shape;
    gert::StorageShape biasShape;
    gert::StorageShape x2ScaleShape;
    auto m = param.m;
    auto k = param.k;
    auto n = param.n;

    gert::StorageShape outputShape({m, n}, {m, n});

    if (param.transA) {
        x1Shape.MutableStorageShape() = gert::Shape({k, m});
    } else {
        x1Shape.MutableStorageShape() = gert::Shape({m, k});
    }
    x1Shape.MutableOriginShape() = x1Shape.MutableStorageShape();

    if (param.transB) {
        if (param.fusedTypeOp == "swiglu") {
            x2Shape.MutableOriginShape() = gert::Shape({2, n, k});
        } else {
            x2Shape.MutableOriginShape() = gert::Shape({n, k});
        }
    } else {
        if (param.fusedTypeOp == "swiglu") {
            x2Shape.MutableOriginShape() = gert::Shape({2, k, n});
        } else {
            x2Shape.MutableOriginShape() = gert::Shape({k, n});
        }
    }

    if (param.x2Nz) {
        x2Shape.MutableStorageShape() = TransNd2Nz(x2Shape.MutableOriginShape(), param.x2Dtype);
    } else {
        x2Shape.MutableStorageShape() = x2Shape.MutableOriginShape();
    }

    if (param.biasFlag) {
        biasShape.MutableStorageShape() = gert::Shape({n});
        biasShape.MutableOriginShape() = biasShape.MutableStorageShape();
    }

    if (param.quantMode == 0) {
        x2ScaleShape.MutableStorageShape() = gert::Shape({1});
        x3Shape.MutableStorageShape() = gert::Shape({1});
    } else if (param.quantMode == 1) {
        x2ScaleShape.MutableStorageShape() = gert::Shape({n});
        x3Shape.MutableStorageShape() = gert::Shape({n});
    }

    x2ScaleShape.MutableOriginShape() = x2ScaleShape.MutableStorageShape();
    x3Shape.MutableOriginShape() = x3Shape.MutableStorageShape();

    string compileInfoStr;
    InitPlatformInfo(param.socVersion, compileInfoStr);
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    // 6为替换原aicNum字符串的长度，配置CORE_NUM
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(param.coreNum));
    // 6为替换原aicNum字符串的长度，配置cube_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(param.coreNum));
    // 6为替换原aivNum字符串的长度，配置vector_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aivNum"), 6, to_string(param.coreNum * 2));
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1800";

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    QuantBatchMatmulV3CompileInfo compileInfo;

    auto kernelHold = gert::KernelRunContextFaker()
                          .KernelIONum(2, 1)
                          .Inputs({const_cast<char *>(compileInfoStr.c_str()), reinterpret_cast<void *>(&platformInfo)})
                          .Outputs({&compileInfo})
                          .Build();

    std::string opType("FusedQuantMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(11, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&x1Shape, &x2Shape, param.biasFlag ? &biasShape : nullptr, nullptr, &x2ScaleShape,
                                    nullptr, nullptr, nullptr, nullptr, nullptr, param.yDtype == ge::DT_INT8 ? &x3Shape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                      .NodeInputTd(0, param.x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.x2Dtype, ge::FORMAT_ND, param.x2Nz ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND)
                      .NodeInputTd(2, param.biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, param.scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"compute_type", Ops::NN::AnyValue::CreateFrom<bool>(0)},
                                  {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.transA)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.transB)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"fused_op_type", Ops::NN::AnyValue::CreateFrom<string>(param.fusedTypeOp)}})
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
    map<string, string> soc_version_infos;
    if (param.socVersion.compare("Ascend910B4") == 0) {
        soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend910B"));
    } else if (param.socVersion.compare("RESERVED") == 0) {
        soc_version_infos.insert(make_pair("Short_SoC_version", "RESERVED"));
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    ge::graphStatus ret = tilingFunc(tilingContext);
    ASSERT_EQ(ret, param.tilingResult);
    if (ret == ge::GRAPH_SUCCESS) {
        ASSERT_EQ(tilingContext->GetTilingKey(), param.tilingKey);
    }
}

} // namespace fqmm_tiling_ut

TEST_P(TestFusedQuantMatMulTiling, generalTest) {
    for (const auto &param : fqmm_tiling_ut::GetParams()) {
        fqmm_tiling_ut::TestOneParamCase(param);
    }
}
INSTANTIATE_TEST_CASE_P(FQMM, TestFusedQuantMatMulTiling, testing::ValuesIn(fqmm_tiling_ut::GetParams()));
