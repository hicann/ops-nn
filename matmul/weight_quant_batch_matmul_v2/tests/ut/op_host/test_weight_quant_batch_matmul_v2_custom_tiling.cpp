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

#include <iostream>
#include <thread>
#include <vector>

#include "log/log.h"

#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"

using namespace std;

struct WeightQuantBatchMatmulV2TilingCustomTestParam {
    string caseName;

    // output
    uint32_t blockDim;
    uint64_t tilingKey;
};

class TestWeightQuantBatchMatmulV2TilingCustom
    : public testing::TestWithParam<WeightQuantBatchMatmulV2TilingCustomTestParam>
{
};

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

static void TestOneParamCase(const WeightQuantBatchMatmulV2TilingCustomTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    std::vector<string> testParam;
    SplitStr2Vec(param.caseName.substr(param.caseName.find_first_of('_') + 1), "_", testParam);
    map<string, ge::DataType> dtypeMap = {
        {"FLOAT16", ge::DT_FLOAT16}, {"FLOAT", ge::DT_FLOAT},   {"BF16", ge::DT_BF16},  {"INT8", ge::DT_INT8},
        {"INT4", ge::DT_INT4},       {"UINT64", ge::DT_UINT64}, {"INT32", ge::DT_INT32}};

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
    ge::DataType quantScaleDtype = dtypeMap[testParam[idx++]];
    ge::DataType yDtype = dtypeMap[testParam[idx++]];
    uint32_t aicNum = 24;
    uint32_t aivNum = 48;
    if (testParam.size() >= 15) { // 第15、16个参数为核数
        aicNum = stoul(testParam[idx++]);
        aivNum = stoul(testParam[idx++]);
    }
    string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196352, "L2_SIZE": 33554432, "L1_SIZE": 524032,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": aicNum,
                          "cube_core_cnt": aicNum, "vector_core_cnt": aivNum, "core_type_list": "CubeCore,VectorCore"}
                          })";
    bool hasTilingCompileInfo = true;
    if (testParam.size() >= 17) { // 第17个参数为编译参数
        string compileParam = testParam[idx++];
        if (compileParam.find("supportL12BtBf16") != string::npos) {
            compileInfoStr = R"({
                "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                                "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                                "Intrinsic_data_move_l12bt": true,
                                "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                                "UB_SIZE": 196352, "L2_SIZE": 33554432, "L1_SIZE": 524032,
                                "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": aicNum,
                                "cube_core_cnt": aicNum, "vector_core_cnt": aivNum, "core_type_list": "CubeCore,VectorCore"}
                                })";
        } else if (compileParam.find("nullTilingCompileInfo") != string::npos) {
            hasTilingCompileInfo = false;
        }
    }
    ge::Format bFormat = ge::FORMAT_ND;
    if (testParam.size() >= 18 && stol(testParam[idx++]) == 1) { // 第18个参数为b矩阵数据排布，1表示Nz
        bFormat = ge::FORMAT_FRACTAL_NZ;
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
            if (weightDtype == ge::DT_INT4) {
                weigthShape.MutableStorageShape() = gert::Shape({(k + 63L) / 64L, (n + 15L) / 16L, 16L, 64L});
            } else {
                weigthShape.MutableStorageShape() = gert::Shape({(k + 31L) / 32L, (n + 15L) / 16L, 16L, 32L});
            }
        } else {
            weigthShape.MutableStorageShape() = gert::Shape({n, k});
        }
        if (weightDtype == ge::DT_INT32) {
            weigthShape.MutableOriginShape() = gert::Shape({n, k / 8});
        } else {
            weigthShape.MutableOriginShape() = gert::Shape({n, k});
        }

    } else {
        if (bFormat == ge::FORMAT_FRACTAL_NZ) {
            if (weightDtype == ge::DT_INT4) {
                weigthShape.MutableStorageShape() = gert::Shape({(n + 63L) / 64L, (k + 15L) / 16L, 16L, 64L});
            } else {
                weigthShape.MutableStorageShape() = gert::Shape({(n + 31L) / 32L, (k + 15L) / 16L, 16L, 32L});
            }
        } else {
            weigthShape.MutableStorageShape() = gert::Shape({k, n});
        }
        if (weightDtype == ge::DT_INT32) {
            weigthShape.MutableOriginShape() = gert::Shape({k, n / 8});
        } else {
            weigthShape.MutableOriginShape() = gert::Shape({k, n});
        }
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
    // 6为替换原aicNum字符串的长度，配置CORE_NUM
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(aicNum));
    // 6为替换原aicNum字符串的长度，配置cube_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(aicNum));
    // 6为替换原aivNum字符串的长度，配置vector_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aivNum"), 6, to_string(aivNum));
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
                           antiQuantOffsetExistFlag ? &antiQuantOffsetShape : nullptr,
                           quantScaleExistFlag ? &quantScaleShape : nullptr,
                           quantOffsetExistFlag ? &quantOffsetShape : nullptr, biasFlag ? &biasShape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(hasTilingCompileInfo ? &compileInfo : nullptr)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, weightDtype, ge::FORMAT_ND, bFormat)
                      .NodeInputTd(2, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, quantScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
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
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend910B"));
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

TEST_P(TestWeightQuantBatchMatmulV2TilingCustom, generalTest)
{
    WeightQuantBatchMatmulV2TilingCustomTestParam param = GetParam();
    TestOneParamCase(param);
}

// format: m k n antiQuantOffsetExistFlag quantScaleExistFlag quantOffsetExistFlag biasFlag transA transB group Xdtype
//         weigthDtype quantScaleDtype yDtype aicNum aivNum platform weightFormat
// Note: group value
//       -1: per channel, 1: per tensor, > 1: per group

static WeightQuantBatchMatmulV2TilingCustomTestParam casesParams2448[] = {
    // custom
    {"fuzzcase_255_255_255_0_1_0_1_0_0_0_BF16_INT8_UINT64_INT8", 24, 365056114230017}, //310100ULL
    {"int4fuzzcase_10112_8280_190_1_0_0_0_0_0_0_FLOAT16_INT4_UINT64_FLOAT16", 20, 365330992136961}, //311100UL
    {"Key_96_11264_6912_1_0_0_0_0_1_0_BF16_INT8_UINT64_BF16", 24, 365331529007873}, //311110ULL
    {"Key_96_6912_11264_1_0_0_0_0_1_0_BF16_INT8_UINT64_BF16", 24, 365331529007873}, //311110ULL
    {"Key_1024_11264_1664_1_0_0_0_0_1_0_BF16_INT8_UINT64_BF16", 24, 365331529007873}, //311110ULL
    {"Key_1024_11264_6912_1_0_0_0_0_1_0_BF16_INT8_UINT64_BF16", 24, 365331529007873}, //311110ULL
    {"Key_1024_1408_11264_1_0_0_0_0_1_0_BF16_INT8_UINT64_BF16", 24, 365331529007873}, //311110ULL
    {"Key_1024_6912_11264_1_0_0_0_0_1_0_BF16_INT8_UINT64_BF16", 24, 365331529007873}, //311110ULL
    {"Int4fuzzCase_64_65504_10208_0_0_0_0_0_0_-1_FLOAT16_INT4_UINT64_FLOAT16_24_48", 24, 365057187971841}, //310200UL
    {"fuzzcase_14783_3004_214_0_0_0_0_0_0_-1_BF16_INT8_UINT64_BF16", 24, 365057187971841}, //310200UL
    {"fuzzcase_10112_8280_190_1_0_0_0_0_0_-1_FLOAT16_INT8_UINT64_FLOAT16", 22, 365332065878785}, //311200ULL
    {"fuzzcase_3_57713_1077_1_0_0_0_0_0_-1_BF16_INT8_UINT64_BF16", 24, 365332065878785}, //311200ULL
    {"fuzzcase_103_2659_2169_1_0_0_0_0_0_-1_BF16_INT8_UINT64_BF16", 24, 365332065878785}, //311200ULL
    {"int4fuzzCase_51_800_64_0_0_0_0_0_1_-1_FLOAT16_INT4_UINT64_FLOAT16_24_48", 16, 365057724842753}, //310210UL
    {"Int4fuzzCase_64_65504_32_0_0_0_0_0_1_-1_FLOAT16_INT4_UINT64_FLOAT16_24_48", 24, 365057724842753},
    {"networkCase_96_11264_6912_1_0_0_0_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"networkCase_96_6912_11264_1_0_0_0_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"networkCase_1024_11264_1664_1_0_0_0_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"networkCase_1024_11264_6912_1_0_0_0_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"networkCase_1024_1408_11264_1_0_0_0_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"networkCase_1024_6912_11264_1_0_0_0_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"fuzzcase_1024_6912_11264_1_0_0_0_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"fuzzcase_96_11263_1663_1_0_0_0_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"fuzzcase_3486_355_25600_1_0_0_1_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"fuzzcase_8480_336_14128_1_0_0_0_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16", 17, 365332602749697}, //311210ULL
    {"fuzzcase_732_42131_73_1_0_0_1_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"fuzzcase_800_17136_912_1_0_0_0_1_0_-1_BF16_INT8_UINT64_BF16", 24, 365332334314241}, //311201ULL
    {"fuzzcase_3248_480_42976_1_0_0_1_1_0_-1_BF16_INT8_UINT64_BF16", 22, 365332334314241}, //311201ULL
    {"fuzzcase_177_3311_3553_1_0_0_0_1_0_-1_BF16_INT8_UINT64_BF16", 23, 365332334314241}, //311201ULL
    {"fuzzcase_37204_2339_198_1_0_0_1_1_0_-1_FLOAT16_INT8_UINT64_FLOAT16", 24, 365332334314241}, //311201ULL
    {"fuzzcase_2768_592_58416_1_0_0_1_1_1_-1_BF16_INT8_UINT64_BF16", 22, 365332871185153}, //311211ULL
    {"fuzzcase_22_5698_73_1_0_0_0_1_1_-1_BF16_INT8_UINT64_BF16", 18, 365332871185153}, //311211ULL
    {"fuzzcase_127_2049_1023_1_0_0_0_0_0_-1_FLOAT16_INT8_UINT64_FLOAT16_24_48_supportL0C2Out_1", 24, 365332066206465}, //811200UL
    {"fuzzcase_6472_3_9313_1_0_0_1_0_0_-1_BF16_INT8_UINT64_BF16_24_48_supportL0C2Out_1", 23, 365332066206465},//811200UL
    {"fuzzcase_108_15980_13_0_0_0_1_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16_24_48_supportL0C2Out_1", 21, 365057725170433}, //810210
    {"fuzzcase_4096_8192_3584_1_0_0_0_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16_24_48_supportL0C2Out_1", 24, 365332603077377},
    {"fuzzcase_4096_8192_32000_1_0_0_0_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16_24_48_supportL0C2Out_1", 24, 365332603077377},
    {"fuzzcase_3712_176_2592_1_0_0_1_0_1_-1_BF16_INT8_UINT64_BF16_24_48_supportL0C2Out_1", 24, 365332603077377}, //811210UL
    {"fuzzcase_127_2049_1023_0_0_0_1_1_1_-1_FLOAT16_INT8_UINT64_FLOAT16_24_48_supportL0C2Out_1", 24, 365057993605889}, //810211UL
    {"fuzzcase_127_2049_1025_1_0_0_1_1_1_-1_BF16_INT8_UINT64_BF16_24_48_supportL0C2Out_1", 22, 365332871512833}, //811211UL
    {"fuzzcase_127_2049_1023_1_0_0_1_1_1_-1_BF16_INT8_UINT64_BF16_24_48_supportL0C2Out_1", 24, 365332871512833},
    {"fuzzcase_3631_3902_11074_1_0_0_0_0_1_-1_FLOAT16_INT4_UINT64_FLOAT16_24_48_supportL0C2Out_1", 24, 365332603077377},
    {"fuzzcase_3631_3902_11074_1_0_0_0_0_0_-1_FLOAT16_INT4_UINT64_FLOAT16_24_48_supportL0C2Out_1", 24, 365332066206465},//811200UL
    {"fuzzcase_3631_3902_11074_1_0_0_0_0_1_-1_BF16_INT4_UINT64_BF16_24_48_supportL0C2Out_1", 24, 365332603077377},
    {"fuzzcase_3631_3902_11074_1_0_0_0_0_0_-1_BF16_INT4_UINT64_BF16_24_48_supportL0C2Out_1", 24, 365332066206465},//811200UL
    {"Key_96_8192_1024_1_0_0_1_0_1_128_FLOAT16_INT32_UINT64_FLOAT16", 24, 365333676491521}, //311310ULL
    {"Key_96_8192_1024_1_0_0_1_0_1_128_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333676491521},
    {"Key_96_5120_640_1_0_0_1_0_1_128_FLOAT16_INT4_UINT64_FLOAT16", 23, 365333676491521},
    {"Key_451_8192_1024_1_0_0_1_0_1_128_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333676491521},
    {"Key_2048_8192_1024_1_0_0_1_0_1_128_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333676491521},
    {"Key_2977_8192_1024_1_0_0_1_0_1_128_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333676491521},
    {"Key_4096_8192_1024_1_0_0_1_0_1_128_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333676491521},
    {"Key_340_5120_640_1_0_0_1_0_1_128_FLOAT16_INT4_UINT64_FLOAT16", 23, 365333676491521},
    {"Key_2048_5120_640_1_0_0_1_0_1_128_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333676491521},
    {"Key_2142_5120_640_1_0_0_1_0_1_128_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333676491521},
    {"Key_4096_5120_640_1_0_0_1_0_1_128_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333676491521},
    {"Key_96_8192_1024_1_1_0_1_0_1_128_FLOAT16_INT4_UINT64_INT8", 24, 365333676491521},
    {"Key_96_5120_640_1_1_0_1_0_1_128_FLOAT16_INT4_UINT64_INT8", 23, 365333676491521},
    {"Key_451_8192_1024_1_1_0_1_0_1_128_FLOAT16_INT4_UINT64_INT8", 24, 365333676491521},
    {"Key_2048_8192_1024_1_1_0_1_0_1_128_FLOAT16_INT4_UINT64_INT8", 24, 365333676491521},
    {"Key_2977_8192_1024_1_1_0_1_0_1_128_FLOAT16_INT4_UINT64_INT8", 24, 365333676491521},
    {"Key_4096_8192_1024_1_1_0_1_0_1_128_FLOAT16_INT4_UINT64_INT8", 24, 365333676491521},
    {"Key_340_5120_640_1_1_0_1_0_1_128_FLOAT16_INT4_UINT64_INT8", 23, 365333676491521},
    {"Key_2048_5120_640_1_1_0_1_0_1_128_FLOAT16_INT4_UINT64_INT8", 24, 365333676491521},
    {"Key_2142_5120_640_1_1_0_1_0_1_128_FLOAT16_INT4_UINT64_INT8", 24, 365333676491521},
    {"Key_4096_5120_640_1_1_0_1_0_1_128_FLOAT16_INT4_UINT64_INT8", 24, 365333676491521},
    {"Key_1_4096_8192_1_0_0_0_0_1_128_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333676491521}, //311310UL
    {"int4fuzzcase_255_23000_255_1_0_0_0_0_1_32_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333676491521},
    {"fuzzcase_1024_6912_11264_1_0_0_0_0_1_128_BF16_INT4_UINT64_BF16", 24, 365333676491521},
    {"Key_1_4096_8192_1_0_0_0_1_0_128_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333408056065}, //311301UL
    {"int4fuzzcase_573_4850_5040_1_0_0_0_1_1_3648_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333944926977}, //311311UL
    {"int4fuzzcase_255_23000_256_1_0_0_0_0_0_32_FLOAT16_INT4_UINT64_FLOAT16", 24, 365333139620609}, //311300UL
    {"fuzzcase_1024_6912_11264_1_0_0_0_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"fuzzcase_96_11263_1663_1_0_0_0_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"fuzzcase_3486_355_25600_1_0_0_1_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"fuzzcase_8480_336_14128_1_0_0_0_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16", 17, 365332602749697}, //311210ULL
    {"fuzzcase_732_42131_73_1_0_0_1_0_1_-1_BF16_INT8_UINT64_BF16", 24, 365332602749697}, //311210ULL
    {"fuzzcase_2997_3072_32004_1_1_0_1_1_1_-1_BF16_INT8_UINT64_INT8", 24, 365350051054337}, //321211ULL
};

INSTANTIATE_TEST_CASE_P(MM2448, TestWeightQuantBatchMatmulV2TilingCustom, testing::ValuesIn(casesParams2448));

static void ThreadFunc(
    const WeightQuantBatchMatmulV2TilingCustomTestParam* params, size_t testcase_num, size_t thread_idx,
    size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(
    const WeightQuantBatchMatmulV2TilingCustomTestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_F(TestWeightQuantBatchMatmulV2TilingCustom, multi_thread_2448)
{
    // 用3个线程测试
    TestMultiThread(
        casesParams2448, sizeof(casesParams2448) / sizeof(WeightQuantBatchMatmulV2TilingCustomTestParam), 3);
}

