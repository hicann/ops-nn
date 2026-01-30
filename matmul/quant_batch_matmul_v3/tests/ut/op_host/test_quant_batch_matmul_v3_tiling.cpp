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
#include "../../../op_host/op_tiling/quant_batch_matmul_v3_basic_tiling.h"
#include "../../../op_host/op_tiling/quant_batch_matmul_v3_tiling.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;
using namespace ut_util;
using namespace optiling;

class QuantBatchMatmulV3TilingTestParam {
public:
    void Prepare(QuantBatchMatmulV3CompileInfo &compileInfo) const;
    void InvokeTilingFunc(QuantBatchMatmulV3CompileInfo &compileInfo) const;
    void Test() const;
    std::string socVersion;
    std::string caseName;
    std::string kernelUtDir;
    std::string prefix;
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
    bool pertokenFlag;
    bool biasFlag;
    bool transA;
    bool transB;
    size_t quantMode;
    ge::DataType x1Dtype;
    ge::DataType x2Dtype;
    ge::DataType scaleDtype;
    ge::DataType perTokenScaleDtype;
    ge::DataType biasDtype;
    ge::DataType yDtype;
    bool fmapNz;
    bool weightNz;

    // output
    bool result; // false means tiling fail
    uint32_t numBlocks;
    uint64_t tilingKey;
    std::string tilingData;
    bool tilingStub; // 是否tililg打桩，只给kernel的用例，此时tiling ut里不校验tiling出参
};


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

class TestQuantBatchMatmulV3Tiling : public testing::TestWithParam<QuantBatchMatmulV3TilingTestParam> {
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
    map<string, string> soc2Arch = {
        {"Ascend910B2", "2201"},
        {"Ascend910B4", "2201"},
        {"Ascend310P3", "2002"},
        {"Ascend950", "3510"},
        {"MC62CM12AA", "5102"},
    };
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
    } else if (socVersion.compare("Ascend950") == 0) {
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
    } else if (socVersion.compare("MC62CM12AA") == 0) {
        compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
                        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                        "intrinsic_fix_pipe_l0c2out_f322bf16": true,
                        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                        "Intrinsic_fix_pipe_pre_conv_cast": true,
                        "Intrinsic_data_move_l12bt": true,
                        "Intrinsic_mmad": true,
                        "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 1048576,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 16,
                        "cube_core_cnt": 16, "vector_core_cnt": 16, "core_type_list": "CubeCore,VectorCore",
                        "lut_type": "MTE2_QTABLE"}
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

static std::vector<QuantBatchMatmulV3TilingTestParam> GetParams(const std::string &socVersion)
{
    std::vector<QuantBatchMatmulV3TilingTestParam> params;
    std::string rootPath(GetExeDirPath() + "../../../../");
    std::string casePath(rootPath + "matmul/quant_batch_matmul_v3/tests/ut/op_host/test_quant_batch_matmul_v3.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
        return params;
    }

    map<string, ge::DataType> dtypeMap = {{"FLOAT16", ge::DT_FLOAT16}, {"FLOAT", ge::DT_FLOAT},
                                          {"BF16", ge::DT_BF16},       {"INT8", ge::DT_INT8},
                                          {"INT4", ge::DT_INT4},       {"UINT64", ge::DT_UINT64},
                                          {"INT32", ge::DT_INT32},     {"INT64", ge::DT_INT64}, {"FLOAT8-E8M0", ge::DT_FLOAT8_E8M0},
                                          {"HIFLOAT8", ge::DT_HIFLOAT8},{"FLOAT8-E5M2", ge::DT_FLOAT8_E5M2},
                                          {"FLOAT8-E4M3", ge::DT_FLOAT8_E4M3FN}, {"FLOAT4-E2M1", ge::DT_FLOAT4_E2M1},
                                          {"FLOAT4-E1M2", ge::DT_FLOAT4_E1M2}};

    std::string line;
    while (std::getline(csvData, line)) {
        std::vector<std::string> testParam;
        SplitStr2Vec(line, ",", testParam);

        QuantBatchMatmulV3TilingTestParam param;
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
        param.batchA = stol(testParam[idx++]);
        param.batchB = stol(testParam[idx++]);
        param.batchC = stol(testParam[idx++]);
        param.m = stol(testParam[idx++]);
        param.k = stol(testParam[idx++]);
        param.n = stol(testParam[idx++]);
        param.offsetFlag = stol(testParam[idx++]);
        param.pertokenFlag = stol(testParam[idx++]);
        param.biasFlag = stol(testParam[idx++]);
        param.transA = stol(testParam[idx++]);
        param.transB = stol(testParam[idx++]);
        param.quantMode = stol(testParam[idx++]);
        param.x1Dtype = dtypeMap[testParam[idx++]];
        param.x2Dtype = dtypeMap[testParam[idx++]];
        param.scaleDtype = dtypeMap[testParam[idx++]];
        param.perTokenScaleDtype = dtypeMap[testParam[idx++]];
        param.biasDtype = dtypeMap[testParam[idx++]];
        param.yDtype = dtypeMap[testParam[idx++]];
        param.fmapNz = testParam[idx++] == "NZ";
        param.weightNz = testParam[idx++] == "NZ";
        param.result = (strcasecmp(testParam[idx++].c_str(), "true") == 0);
        param.numBlocks = stol(testParam[idx++]);
        param.tilingKey = stol(testParam[idx++]);
        param.tilingData = testParam[idx++];
        param.tilingStub = (strcasecmp(testParam[idx++].c_str(), "true") == 0);
        params.push_back(param);
    }

    return params;
}

void QuantBatchMatmulV3TilingTestParam::Prepare(QuantBatchMatmulV3CompileInfo &compileInfo) const
{
    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape scaleShape;
    gert::StorageShape pertokenShape;
    gert::StorageShape biasShape;
    gert::StorageShape outputShape;

    if (yDim == 6) {
        outputShape.MutableOriginShape() = gert::Shape({batchC, batchC, batchC, batchC, m, n});
    } else if (yDim == 3) {
        outputShape.MutableOriginShape() = gert::Shape({batchC, m, n});
    } else if (yDim == 2) {
        outputShape.MutableOriginShape() = gert::Shape({m, n});
    } else {
        outputShape.MutableOriginShape() = gert::Shape({m});
    }

    if (transA) {
        if (x1Dim == 6) {
            x1Shape.MutableOriginShape() = gert::Shape({batchA, batchA, batchA, batchA, k, m});
        } else if (x1Dim == 3) {
            x1Shape.MutableOriginShape() = gert::Shape({batchA, k, m});
        } else if (x1Dim == 2) {
            x1Shape.MutableOriginShape() = gert::Shape({k, m});
        } else {
            x1Shape.MutableOriginShape() = gert::Shape({m});
        }
    } else {
        if (x1Dim == 6) {
            x1Shape.MutableOriginShape() = gert::Shape({batchA, batchA, batchA, batchA, m, k});
        } else if (x1Dim == 3) {
            x1Shape.MutableOriginShape() = gert::Shape({batchA, m, k});
        } else if (x1Dim == 2) {
            x1Shape.MutableOriginShape() = gert::Shape({m, k});
        } else {
            x1Shape.MutableOriginShape() = gert::Shape({m});
        }
    }

    if (transB) {
        if (x2Dim == 6) {
            x2Shape.MutableOriginShape() = gert::Shape({batchB, batchB, batchB, batchB, n, k});
        } else if (x2Dim == 3) {
            x2Shape.MutableOriginShape() = gert::Shape({batchB, n, k});
        } else if (x2Dim == 2) {
            x2Shape.MutableOriginShape() = gert::Shape({n, k});
        } else {
            x2Shape.MutableOriginShape() = gert::Shape({n});
        }
    } else {
        if (x2Dim == 6) {
            x2Shape.MutableOriginShape() = gert::Shape({batchB, batchB, batchB, batchB, k, n});
        } else if (x2Dim == 3) {
            x2Shape.MutableOriginShape() = gert::Shape({batchB, k, n});
        } else if (x2Dim == 2) {
            x2Shape.MutableOriginShape() = gert::Shape({k, n});
        } else {
            x2Shape.MutableOriginShape() = gert::Shape({n});
        }
    }

    pertokenShape.MutableStorageShape() = gert::Shape({m});
    if (quantMode == 0) {  // per_tensor
        scaleShape.MutableStorageShape() = gert::Shape({1});
    } else if (quantMode == 1) {  // per_channel
        scaleShape.MutableStorageShape() = gert::Shape({n});
    } else if (quantMode == 2) {
        int64_t scaleK = (k + 63) / 64 * 2;
        scaleShape.MutableStorageShape() = gert::Shape({n, scaleK});
        pertokenShape.MutableStorageShape() = gert::Shape({m, scaleK});
    } else if (quantMode == 3 || quantMode == 4) {
        int64_t scaleM = (m + 127) / 128;
        if (quantMode == 4) {
            scaleM = m;
        }
        int64_t scaleK = (k + 127) / 128;
        int64_t scaleN = (n + 127) / 128;
        if (transA) {
            if (x1Dim == 6) {
                pertokenShape.MutableStorageShape() = gert::Shape({batchA, batchA, batchA, batchA, scaleK, scaleM});
            } else if (x1Dim == 3) {
                pertokenShape.MutableStorageShape() = gert::Shape({batchA, scaleK, scaleM});
            } else if (x1Dim == 2) {
                pertokenShape.MutableStorageShape() = gert::Shape({scaleK, scaleM});
            }
        } else {
            if (x1Dim == 6) {
                pertokenShape.MutableStorageShape() = gert::Shape({batchA, batchA, batchA, batchA, scaleM, scaleK});
            } else if (x1Dim == 3) {
                pertokenShape.MutableStorageShape() = gert::Shape({batchA, scaleM, scaleK});
            } else if (x1Dim == 2) {
                pertokenShape.MutableStorageShape() = gert::Shape({scaleM, scaleK});
            }
        }
        if (transB) {
            if (x2Dim == 6) {
                scaleShape.MutableStorageShape() = gert::Shape({batchB, batchB, batchB, batchB, scaleN, scaleK});
            } else if (x2Dim == 3) {
                scaleShape.MutableStorageShape() = gert::Shape({batchB, scaleN, scaleK});
            } else if (x2Dim == 2) {
                scaleShape.MutableStorageShape() = gert::Shape({scaleN, scaleK});
            }
        } else {
            if (x2Dim == 6) {
                scaleShape.MutableStorageShape() = gert::Shape({batchB, batchB, batchB, batchB, scaleK, scaleN});
            } else if (x2Dim == 3) {
                scaleShape.MutableStorageShape() = gert::Shape({batchB, scaleK, scaleN});
            } else if (x2Dim == 2) {
                scaleShape.MutableStorageShape() = gert::Shape({scaleK, scaleN});
            }
        }
    }

    biasShape.MutableStorageShape() = gert::Shape({n});
    scaleShape.MutableOriginShape() = scaleShape.MutableStorageShape();
    biasShape.MutableOriginShape() = biasShape.MutableStorageShape();
    pertokenShape.MutableOriginShape() = pertokenShape.MutableStorageShape();
    if (fmapNz) {
        x1Shape.MutableStorageShape() = TransNd2Nz(x1Shape.MutableOriginShape());
    } else {
        x1Shape.MutableStorageShape() = x1Shape.MutableOriginShape();
    }

    if (weightNz) {
        x2Shape.MutableStorageShape() = TransNd2Nz(x2Shape.MutableOriginShape());
    } else {
        x2Shape.MutableStorageShape() = x2Shape.MutableOriginShape();
    }
    scaleShape.MutableOriginShape() = scaleShape.MutableStorageShape();
    biasShape.MutableOriginShape() = biasShape.MutableStorageShape();
    outputShape.MutableStorageShape() = outputShape.MutableOriginShape();

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    std::string opType("QuantBatchMatmulV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());
    int64_t groupSize = 0;
    if (quantMode == 2) {
        groupSize = 4295032864;
    } else if (quantMode == 3) {
        groupSize = 549764202624;
    } else if (quantMode == 4) {
        groupSize = 4303356032;
    }
    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(6, 1)
            .IrInstanceNum({1, 1, 1, 1, 1, 1})
            .InputShapes({&x1Shape, &x2Shape, &scaleShape, offsetFlag ? &scaleShape : nullptr, biasFlag ? &biasShape : nullptr, pertokenFlag ? &pertokenShape : nullptr})
            .OutputShapes({&outputShape})
            .CompileInfo(&compileInfo)
            .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
            .NodeInputTd(0, x1Dtype, ge::FORMAT_ND, fmapNz ? ge::FORMAT_FRACTAL_NZ: ge::FORMAT_ND)
            .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, weightNz ? ge::FORMAT_FRACTAL_NZ: ge::FORMAT_ND)
            .NodeInputTd(2, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, perTokenScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(yDtype)},
                        {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
                        {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(transB)},
                        {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)}})
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

void QuantBatchMatmulV3TilingTestParam::InvokeTilingFunc(QuantBatchMatmulV3CompileInfo &compileInfo) const
{
    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape scaleShape;
    gert::StorageShape pertokenShape;
    gert::StorageShape biasShape;
    gert::StorageShape outputShape;
    cout << "run case " << prefix << std::endl;
    if (yDim == 6) {
        outputShape.MutableOriginShape() = gert::Shape({batchC, batchC, batchC, batchC, m, n});
    } else if (yDim == 3) {
        outputShape.MutableOriginShape() = gert::Shape({batchC, m, n});
    } else if (yDim == 2) {
        outputShape.MutableOriginShape() = gert::Shape({m, n});
    } else {
        outputShape.MutableOriginShape() = gert::Shape({m});
    }

    if (transA) {
        if (x1Dim == 6) {
            x1Shape.MutableOriginShape() = gert::Shape({batchA, batchA, batchA, batchA, k, m});
        } else if (x1Dim == 3) {
            x1Shape.MutableOriginShape() = gert::Shape({batchA, k, m});
        } else if (x1Dim == 2) {
            x1Shape.MutableOriginShape() = gert::Shape({k, m});
        } else {
            x1Shape.MutableOriginShape() = gert::Shape({m});
        }
    } else {
        if (x1Dim == 6) {
            x1Shape.MutableOriginShape() = gert::Shape({batchA, batchA, batchA, batchA, m, k});
        } else if (x1Dim == 3) {
            x1Shape.MutableOriginShape() = gert::Shape({batchA, m, k});
        } else if (x1Dim == 2) {
            x1Shape.MutableOriginShape() = gert::Shape({m, k});
        } else {
            x1Shape.MutableOriginShape() = gert::Shape({m});
        }
    }

    if (transB) {
        if (x2Dim == 6) {
            x2Shape.MutableOriginShape() = gert::Shape({batchB, batchB, batchB, batchB, n, k});
        } else if (x2Dim == 3) {
            x2Shape.MutableOriginShape() = gert::Shape({batchB, n, k});
        } else if (x2Dim == 2) {
            x2Shape.MutableOriginShape() = gert::Shape({n, k});
        } else {
            x2Shape.MutableOriginShape() = gert::Shape({n});
        }
    } else {
        if (x2Dim == 6) {
            x2Shape.MutableOriginShape() = gert::Shape({batchB, batchB, batchB, batchB, k, n});
        } else if (x2Dim == 3) {
            x2Shape.MutableOriginShape() = gert::Shape({batchB, k, n});
        } else if (x2Dim == 2) {
            x2Shape.MutableOriginShape() = gert::Shape({k, n});
        } else {
            x2Shape.MutableOriginShape() = gert::Shape({n});
        }
    }

    pertokenShape.MutableStorageShape() = gert::Shape({m});
    if (quantMode == 0) {  // per_tensor
        scaleShape.MutableStorageShape() = gert::Shape({1});
    } else if (quantMode == 1) {  // per_channel
        scaleShape.MutableStorageShape() = gert::Shape({n});
    } else if (quantMode == 2) {
        int64_t scaleK = (k + 63) / 64;
        if (transA) {
            pertokenShape.MutableStorageShape() = gert::Shape({scaleK, m, 2});
        } else {
            pertokenShape.MutableStorageShape() = gert::Shape({m, scaleK, 2});
        }
        if (transB) {
            scaleShape.MutableStorageShape() = gert::Shape({n, scaleK, 2});
        } else{
            scaleShape.MutableStorageShape() = gert::Shape({scaleK, n, 2});
        }
    } else if (quantMode == 3 || quantMode == 4) {
        int64_t scaleM = (m + 127) / 128;
        if (quantMode == 4) {
            scaleM = m;
        }
        int64_t scaleK = (k + 127) / 128;
        int64_t scaleN = (n + 127) / 128;
        if (transA) {
            if (x1Dim == 6) {
                pertokenShape.MutableStorageShape() = gert::Shape({batchA, batchA, batchA, batchA, scaleK, scaleM});
            } else if (x1Dim == 3) {
                pertokenShape.MutableStorageShape() = gert::Shape({batchA, scaleK, scaleM});
            } else if (x1Dim == 2) {
                pertokenShape.MutableStorageShape() = gert::Shape({scaleK, scaleM});
            }
        } else {
            if (x1Dim == 6) {
                pertokenShape.MutableStorageShape() = gert::Shape({batchA, batchA, batchA, batchA, scaleM, scaleK});
            } else if (x1Dim == 3) {
                pertokenShape.MutableStorageShape() = gert::Shape({batchA, scaleM, scaleK});
            } else if (x1Dim == 2) {
                pertokenShape.MutableStorageShape() = gert::Shape({scaleM, scaleK});
            }
        }
        if (transB) {
            if (x2Dim == 6) {
                scaleShape.MutableStorageShape() = gert::Shape({batchB, batchB, batchB, batchB, scaleN, scaleK});
            } else if (x2Dim == 3) {
                scaleShape.MutableStorageShape() = gert::Shape({batchB, scaleN, scaleK});
            } else if (x2Dim == 2) {
                scaleShape.MutableStorageShape() = gert::Shape({scaleN, scaleK});
            }
        } else {
            if (x2Dim == 6) {
                scaleShape.MutableStorageShape() = gert::Shape({batchB, batchB, batchB, batchB, scaleK, scaleN});
            } else if (x2Dim == 3) {
                scaleShape.MutableStorageShape() = gert::Shape({batchB, scaleK, scaleN});
            } else if (x2Dim == 2) {
                scaleShape.MutableStorageShape() = gert::Shape({scaleK, scaleN});
            }
        }
    }

    biasShape.MutableStorageShape() = gert::Shape({n});
    scaleShape.MutableOriginShape() = scaleShape.MutableStorageShape();
    biasShape.MutableOriginShape() = biasShape.MutableStorageShape();
    pertokenShape.MutableOriginShape() = pertokenShape.MutableStorageShape();
    if (fmapNz) {
        x1Shape.MutableStorageShape() = TransNd2Nz(x1Shape.MutableOriginShape());
    } else {
        x1Shape.MutableStorageShape() = x1Shape.MutableOriginShape();
    }

    if (weightNz) {
        x2Shape.MutableStorageShape() = TransNd2Nz(x2Shape.MutableOriginShape());
    } else {
        x2Shape.MutableStorageShape() = x2Shape.MutableOriginShape();
    }
    scaleShape.MutableOriginShape() = scaleShape.MutableStorageShape();
    biasShape.MutableOriginShape() = biasShape.MutableStorageShape();
    outputShape.MutableStorageShape() = outputShape.MutableOriginShape();

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    std::string opType("QuantBatchMatmulV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());
    int64_t groupSize = 0;
    if (quantMode == 2) {
        groupSize = 4295032864;
    } else if (quantMode == 3) {
        groupSize = 549764202624;
    } else if (quantMode == 4) {
        groupSize = 4303356032;
    }
    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(6, 1)
            .IrInstanceNum({1, 1, 1, 1, 1, 1})
            .InputShapes({&x1Shape, &x2Shape, &scaleShape, offsetFlag ? &scaleShape : nullptr, biasFlag ? &biasShape : nullptr, pertokenFlag ? &pertokenShape : nullptr})
            .OutputShapes({&outputShape})
            .CompileInfo(&compileInfo)
            .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
            .NodeInputTd(0, x1Dtype, ge::FORMAT_ND, fmapNz ? ge::FORMAT_FRACTAL_NZ: ge::FORMAT_ND)
            .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, weightNz ? ge::FORMAT_FRACTAL_NZ: ge::FORMAT_ND)
            .NodeInputTd(2, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, perTokenScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(yDtype)},
                        {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
                        {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(transB)},
                        {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)}})
            .TilingData(rawTilingData.get())
            .Workspace(workspace)
            .SetOpType(opType)
            .Build();

    string compileInfoStr;
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr) << "socVersion is: " << socVersion << ", caseName is: " << caseName
                                   << ", prefix is: " << prefix;
    if (result) {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS)
            << "socVersion is: " << socVersion << ", caseName is: " << caseName << ", prefix is: " << prefix;
        if (tilingStub) {
            return;
        }
        ASSERT_EQ(tilingContext->GetTilingKey(), tilingKey)
            << "socVersion is: " << socVersion << ", caseName is: " << caseName << ", prefix is: " << prefix;
        ASSERT_EQ(tilingContext->GetBlockDim(), numBlocks)
            << "socVersion is: " << socVersion << ", caseName is: " << caseName << ", prefix is: " << prefix;

        std::vector<std::string> tilingDataStrs;
        SplitStr2Vec(tilingData, " ", tilingDataStrs);
        std::vector<int32_t> tilingDataInt;
        tilingDataInt.reserve(tilingDataStrs.size());
        for (auto &tilingValue : tilingDataStrs) {
            tilingDataInt.push_back(atoi(tilingValue.c_str()));
        }

        bool isMxfp8 = quantMode == 2 && (x1Dtype == ge::DT_FLOAT8_E4M3FN || x1Dtype == ge::DT_FLOAT8_E5M2) && (x2Dtype == ge::DT_FLOAT8_E4M3FN || x2Dtype == ge::DT_FLOAT8_E5M2);
        if ((quantMode == 3 || quantMode == 4 || isMxfp8 || (quantMode == 0 && !pertokenFlag) || (quantMode == 1 && (scaleDtype == ge::DT_UINT64 || yDtype == ge::DT_INT32))) && !weightNz) {
            DequantBmm::QuantBatchMatmulV3BasicAPITilingData& actualTilingData = *reinterpret_cast<DequantBmm::QuantBatchMatmulV3BasicAPITilingData*>(tilingContext->GetRawTilingData()->GetData());
            DequantBmm::QuantBatchMatmulV3BasicAPITilingData& expectTilingData = *reinterpret_cast<DequantBmm::QuantBatchMatmulV3BasicAPITilingData*>(tilingDataInt.data());
            // biasFlag 为0时，biasDtype在kernel侧不使用，忽略校验
            if (biasFlag == false) {
                expectTilingData.params.biasDtype = actualTilingData.params.biasDtype;
            }
            string actualTilingDataStr = TilingData2Str(tilingContext->GetRawTilingData()->GetData(),
                                                        tilingContext->GetRawTilingData()->GetDataSize());
            string expectTilingDataStr = TilingData2Str(tilingDataInt.data(), tilingDataInt.size() * sizeof(int32_t));
            ASSERT_EQ(actualTilingDataStr, expectTilingDataStr)
                << "socVersion is: " << socVersion << ", caseName is: " << caseName << ", prefix is: " << prefix;
        } else {
            QuantBatchMatmulV3TilingData& actualTilingData = *reinterpret_cast<QuantBatchMatmulV3TilingData*>(tilingContext->GetRawTilingData()->GetData());
            QuantBatchMatmulV3TilingData& expectTilingData = *reinterpret_cast<QuantBatchMatmulV3TilingData*>(tilingDataInt.data());
            // 这里通过重置预期结果里的部分字段来忽略不关心的tiling字段，后续有新增的话可以仿照这个方法来忽略其他字段
            expectTilingData.matmulTiling.shareL1Size = actualTilingData.matmulTiling.shareL1Size;
            // biasFlag 为0时，biasDtype在kernel侧不使用，忽略校验
            if (biasFlag == false) {
                expectTilingData.params.biasDtype = actualTilingData.params.biasDtype;
            }
            string actualTilingDataStr = TilingData2Str(tilingContext->GetRawTilingData()->GetData(),
                                                        tilingContext->GetRawTilingData()->GetDataSize());
            string expectTilingDataStr = TilingData2Str(tilingDataInt.data(), tilingDataInt.size() * sizeof(int32_t));
            ASSERT_EQ(actualTilingDataStr, expectTilingDataStr)
                << "socVersion is: " << socVersion << ", caseName is: " << caseName << ", prefix is: " << prefix;
        }
    } else {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED)
            << "socVersion is: " << socVersion << ", caseName is: " << caseName << ", prefix is: " << prefix;
    }
}

void QuantBatchMatmulV3TilingTestParam::Test() const
{
    QuantBatchMatmulV3CompileInfo compileInfo;
    Prepare(compileInfo);
    InvokeTilingFunc(compileInfo);
}


TEST_P(TestQuantBatchMatmulV3Tiling, generalTest)
{
    GetParam().Test();
}

INSTANTIATE_TEST_CASE_P(QUANTMM910B, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(GetParams("Ascend910B2")));
INSTANTIATE_TEST_CASE_P(QUANTMM910B4, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(GetParams("Ascend910B4")));
INSTANTIATE_TEST_CASE_P(QUANTMM310P, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(GetParams("Ascend310P3")));
INSTANTIATE_TEST_CASE_P(QUANTMM950, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(GetParams("Ascend950")));
INSTANTIATE_TEST_CASE_P(QUANTMMMC62CM12AA, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(GetParams("MC62CM12AA")));

static void ThreadFunc(const QuantBatchMatmulV3TilingTestParam *params, size_t testcaseNum, size_t threadIdx,
                       size_t threadNum)
{
    int32_t logLevel = 0;
    int32_t enableEvent = 0;
    for (size_t idx = threadIdx; idx < testcaseNum; idx += threadNum) {
        params[idx].Test();
    }
}

static mutex compileMutex;

static void ThreadFuncPrepare(const QuantBatchMatmulV3TilingTestParam *params, size_t testcaseNum, size_t threadIdx,
                       size_t threadNum, map<size_t, QuantBatchMatmulV3CompileInfo> &compileInfos)
{
    if (threadIdx >= testcaseNum)
        return;
    int32_t logLevel = 0;
    int32_t enableEvent = 0;
    QuantBatchMatmulV3CompileInfo compileInfo;
    params[threadIdx].Prepare(compileInfo);

    {
        lock_guard<mutex> lock(compileMutex);
        compileInfos[threadIdx] = compileInfo;
    }
}

static void ThreadFuncInvokeTilingFunc(const QuantBatchMatmulV3TilingTestParam *params, size_t testcaseNum, size_t threadIdx,
                       size_t threadNum, QuantBatchMatmulV3CompileInfo &compileInfo)
{
    if (threadIdx >= testcaseNum)
        return;
    int32_t logLevel = 0;
    int32_t enableEvent = 0;
    params[threadIdx].InvokeTilingFunc(compileInfo);
}

static void TestMultiThread(const QuantBatchMatmulV3TilingTestParam *params, size_t testcaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcaseNum, idx, threadNum);
    }

    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

static void TestMultiThreadSeparate(const QuantBatchMatmulV3TilingTestParam *params, size_t testcaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    map<size_t, QuantBatchMatmulV3CompileInfo> compileInfos;
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

TEST_F(TestQuantBatchMatmulV3Tiling, multiThread310P3)
{
    auto casesParams310P3 = GetParams("Ascend310P3");
    TestMultiThread(casesParams310P3.data(), casesParams310P3.size(), 3);
}

TEST_F(TestQuantBatchMatmulV3Tiling, multiThread950)
{
    auto casesParams950 = GetParams("Ascend950");
    TestMultiThread(casesParams950.data(), casesParams950.size(), 3);
}
