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
 
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>

#include "log/log.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "quant/ifmr/op_kernel/ifmr_tiling_data.h"

struct IFMRCompileInfo {
    int32_t dataNum;
};

class IFMRTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "IFMRTilingTest SetUp" << std::endl;
    }
 
    static void TearDownTestCase() {
        std::cout << "IFMRTilingTest TearDown" << std::endl;
    }
};
 
struct IFMRAttrs {
    float minPercentile;
    float maxPercentile;
    std::vector<float> searchRange;
    float searchStep;
    bool withOffset;
    uint32_t quantBits;
};
 
ge::graphStatus IFMRTestCase(vector<vector<int64_t>> input_shapes, vector<vector<int64_t>> output_shapes,
                  IFMRAttrs attrs, ge::DataType dataType, IfmrTilingData &tilingParam) {
    std::string opType("IFMR");
    if (gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()) == nullptr) {
        std::cout << "gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()) is nullptr" << std::endl;
        return ge::GRAPH_FAILED;
    }
 
    gert::StorageShape data = {{input_shapes[0][0]}, {input_shapes[0][0]}};
    gert::StorageShape dataMin = {{input_shapes[1][0]}, {input_shapes[1][0]}};
    gert::StorageShape dataMax = {{input_shapes[2][0]}, {input_shapes[2][0]}};
    gert::StorageShape cumsum = {{input_shapes[3][0]}, {input_shapes[3][0]}};
    gert::StorageShape scale = {{output_shapes[0][0]}, {output_shapes[0][0]}};
    gert::StorageShape offset = {{output_shapes[0][0]}, {output_shapes[0][0]}};
 
    if (gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()) == nullptr) {
        std::cout << "gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()) is nullptr" << std::endl;
        return ge::GRAPH_FAILED;
    }
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    uint64_t L1_SIZE = 524288;
    uint64_t L0a_SIZE = 65536;
    uint64_t L0b_SIZE = 65536;
    uint64_t L0c_SIZE = 262144;
    uint64_t aicoreNum = 32;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", 
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, 
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288, 
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, 
                            "CORE_NUM": 32}
                            })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), socInfos, aicoreSpec, intrinsics);
    map<string, string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}};
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto tilingDataPtr = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector *>(workspaceSizeHoler.get());
    if (tilingDataPtr == nullptr) {
        std::cout << "tilingDataPtr is nullptr" << std::endl;
        return ge::GRAPH_FAILED;
    }
 
    std::vector<void*> inputShapeRef = {&data, &dataMin, &dataMax, &cumsum};
    std::vector<void*> outputShapesRef = {&scale, &offset};
 
    // compile info
    IFMRCompileInfo compileInfo;
    // tilingParseFunc simulate
    auto kernelHolder = gert::KernelRunContextFaker()
                      .KernelIONum(2, 1)
                      .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platformInfo)})
                      .Outputs({&compileInfo})
                      .Build();
    if (kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init() == false) {
        std::cout << "kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init() is nullptr" << std::endl;
        return ge::GRAPH_FAILED;
    }
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto holder = gert::TilingContextFaker().SetOpType(opType)
                              .NodeIoNum(4, 2)
                              .IrInstanceNum({1, 1, 1, 1})
                              .InputShapes(inputShapeRef)
                              .OutputShapes(outputShapesRef)
                              .CompileInfo(&compileInfo)
                              .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                              .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({
                                {"min_percentile", Ops::NN::AnyValue::CreateFrom<float>(attrs.minPercentile)},
                                {"max_percentile", Ops::NN::AnyValue::CreateFrom<float>(attrs.maxPercentile)},
                                {"search_range", Ops::NN::AnyValue::CreateFrom<std::vector<float>>(attrs.searchRange)},
                                {"search_step", Ops::NN::AnyValue::CreateFrom<float>(attrs.searchStep)},
                                {"with_offset", Ops::NN::AnyValue::CreateFrom<bool>(attrs.withOffset)},
                                {"quant_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(attrs.quantBits)}
                                })
                              .TilingData(tilingDataPtr.get())
                              .Workspace(wsSize)
                              .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext->GetPlatformInfo() == nullptr) {
        std::cout << "tilingContext->GetPlatformInfo() is nullptr" << std::endl;
        return ge::GRAPH_FAILED;
    }
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    auto ret = tilingFunc(tilingContext);
    auto buf = (IfmrTilingData*)tilingContext->GetTilingData<IfmrTilingData>();
    tilingParam = *buf;
    return ret;
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_0) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.9;
    attrs.maxPercentile = 0.9;
    attrs.searchRange = {0.7, 1.3};
    attrs.searchStep = 0.1;
    attrs.withOffset = true;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    
    EXPECT_EQ(attrs.minPercentile, tilingdata.minPercentile);
    EXPECT_EQ(attrs.maxPercentile, tilingdata.maxPercentile);
    EXPECT_EQ(attrs.searchRange[0], tilingdata.searchRange[0]);
    EXPECT_EQ(attrs.searchRange[1], tilingdata.searchRange[1]);
    EXPECT_EQ(attrs.searchStep, tilingdata.searchStep);
    EXPECT_EQ((uint32_t)attrs.withOffset, tilingdata.withOffset);
    EXPECT_EQ(attrs.quantBits, tilingdata.quantBits);
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_1) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.9;
    attrs.maxPercentile = 0.9;
    attrs.searchRange = {0.7, 1.3};
    attrs.searchStep = 0.1;
    attrs.withOffset = true;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT16;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    
    EXPECT_EQ(attrs.minPercentile, tilingdata.minPercentile);
    EXPECT_EQ(attrs.maxPercentile, tilingdata.maxPercentile);
    EXPECT_EQ(attrs.searchRange[0], tilingdata.searchRange[0]);
    EXPECT_EQ(attrs.searchRange[1], tilingdata.searchRange[1]);
    EXPECT_EQ(attrs.searchStep, tilingdata.searchStep);
    EXPECT_EQ((uint32_t)attrs.withOffset, tilingdata.withOffset);
    EXPECT_EQ(attrs.quantBits, tilingdata.quantBits);
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_2) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.9;
    attrs.maxPercentile = 0.9;
    attrs.searchRange = {0.7, 1.3};
    attrs.searchStep = 0.1;
    attrs.withOffset = false;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT16;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    
    EXPECT_EQ(attrs.minPercentile, tilingdata.minPercentile);
    EXPECT_EQ(attrs.maxPercentile, tilingdata.maxPercentile);
    EXPECT_EQ(attrs.searchRange[0], tilingdata.searchRange[0]);
    EXPECT_EQ(attrs.searchRange[1], tilingdata.searchRange[1]);
    EXPECT_EQ(attrs.searchStep, tilingdata.searchStep);
    EXPECT_EQ((uint32_t)attrs.withOffset, tilingdata.withOffset);
    EXPECT_EQ(attrs.quantBits, tilingdata.quantBits);
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_3) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.9;
    attrs.maxPercentile = 0.9;
    attrs.searchRange = {0.7, 1.3};
    attrs.searchStep = 0.1;
    attrs.withOffset = false;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    
    EXPECT_EQ(attrs.minPercentile, tilingdata.minPercentile);
    EXPECT_EQ(attrs.maxPercentile, tilingdata.maxPercentile);
    EXPECT_EQ(attrs.searchRange[0], tilingdata.searchRange[0]);
    EXPECT_EQ(attrs.searchRange[1], tilingdata.searchRange[1]);
    EXPECT_EQ(attrs.searchStep, tilingdata.searchStep);
    EXPECT_EQ((uint32_t)attrs.withOffset, tilingdata.withOffset);
    EXPECT_EQ(attrs.quantBits, tilingdata.quantBits);
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_invalid_minpercentile_01) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.1;
    attrs.maxPercentile = 0.9;
    attrs.searchRange = {0.7, 1.3};
    attrs.searchStep = 0.1;
    attrs.withOffset = false;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_invalid_minpercentile_02) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 1.1;
    attrs.maxPercentile = 0.9;
    attrs.searchRange = {0.7, 1.3};
    attrs.searchStep = 0.1;
    attrs.withOffset = false;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_invalid_maxpercentile_01) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.9;
    attrs.maxPercentile = 0.1;
    attrs.searchRange = {0.7, 1.3};
    attrs.searchStep = 0.1;
    attrs.withOffset = false;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_invalid_maxpercentile_02) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.9;
    attrs.maxPercentile = 1.1;
    attrs.searchRange = {0.7, 1.3};
    attrs.searchStep = 0.1;
    attrs.withOffset = false;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_invalid_searchRange_01) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.9;
    attrs.maxPercentile = 0.9;
    attrs.searchRange = {0, 1};
    attrs.searchStep = 0.1;
    attrs.withOffset = false;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_invalid_searchRange_02) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.9;
    attrs.maxPercentile = 0.9;
    attrs.searchRange = {1.1, 1.1};
    attrs.searchStep = 0.1;
    attrs.withOffset = false;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_invalid_searchStep_01) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.9;
    attrs.maxPercentile = 0.9;
    attrs.searchRange = {1, 8193};
    attrs.searchStep = 1;
    attrs.withOffset = false;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
 
TEST_F(IFMRTilingTest, run_IFMR_case_invalid_quantBits_01) {
    vector<vector<int64_t>> input_shapes = {{1024}, {1}, {1}, {1024}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.9;
    attrs.maxPercentile = 0.9;
    attrs.searchRange = {0.7, 1.3};
    attrs.searchStep = 0.1;
    attrs.withOffset = false;
    attrs.quantBits = 9;
    ge::DataType dataType = ge::DT_FLOAT;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

 
TEST_F(IFMRTilingTest, run_IFMR_case_invalid_data_length) {
    vector<vector<int64_t>> input_shapes = {{0}, {0}, {0}, {0}};
    vector<vector<int64_t>> output_shapes = {{1}, {1}};
    IFMRAttrs attrs;
    attrs.minPercentile = 0.9;
    attrs.maxPercentile = 0.9;
    attrs.searchRange = {0.7, 1.3};
    attrs.searchStep = 0.1;
    attrs.withOffset = false;
    attrs.quantBits = 8;
    ge::DataType dataType = ge::DT_FLOAT;
    
    IfmrTilingData tilingdata;
    
    auto ret = IFMRTestCase(input_shapes, output_shapes, attrs, dataType, tilingdata);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
