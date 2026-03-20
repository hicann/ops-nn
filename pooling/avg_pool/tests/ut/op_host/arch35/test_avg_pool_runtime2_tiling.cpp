/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "array_ops.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "conv/common/op_host/op_tiling/cube_tiling.h"

using namespace std;
using namespace ge;
using namespace gert;
using namespace optiling;

struct AvgPoolTilingParseInfo: CubeTilingCommonParseInfo {
    int64_t stridesH;
    int64_t stridesW;
    int64_t kSizeH;
    int64_t kSizeW;
    int64_t filter;
    int64_t ub_ele;
    int64_t core_num;
    int64_t ksize_h;
    int64_t ksize_w;
    int64_t strides_h;
    int64_t strides_w;
    int64_t padding;
};

struct AvgPoolTilingRuntime2TestParam {
    string caseName;
    string opType;
    string compileInfo;

    // input of op tiling
    initializer_list<int64_t> xOriginShape;
    Format xOriginFormat;
    initializer_list<int64_t> xStorageShape;
    Format xStorageFormat;
    initializer_list<int64_t> yOriginShape;
    Format yOriginFormat;
    initializer_list<int64_t> yStorageShape;
    Format yStorageFormat;

    vector<int64_t> ksize;
    vector<int64_t> strides;
    string padding;
    string dataFormat;

    // output
    uint32_t blockDim = 0;
    uint64_t tilingKey = 0;
    string tilingData;

    bool result;
};

class AvgPoolTilingRunTime2 : public testing::TestWithParam<AvgPoolTilingRuntime2TestParam> {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPoolTilingRunTime2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPoolTilingRunTime2 TearDown" << std::endl;
  }
};

static string TilingData2Str(const gert::TilingData *tilingData) {
  auto data = tilingData->GetData();
  string result;
  for (size_t i = 0; i < tilingData->GetDataSize(); i += sizeof(int32_t)) {
    result += std::to_string((reinterpret_cast<const int32_t *>(tilingData->GetData())[i / sizeof(int32_t)]));
    result += " ";
  }

  return result;
}

TEST_P(AvgPoolTilingRunTime2, general_cases) {
    AvgPoolTilingRuntime2TestParam param = GetParam();
    cout << "run case " << param.caseName << endl;

    // platform setup
    string platform_info_str = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
        "CORE_NUM": 64}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(platform_info_str.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // get function
    ASSERT_NE(OpImplRegistry::GetInstance().GetOpImpl(param.opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(param.opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(param.opType.c_str())->tiling_parse;

    AvgPoolTilingParseInfo opInfo;
    auto kernelHolder = gert::KernelRunContextFaker()
        .KernelIONum(1, 1)
        .Inputs({const_cast<char *>(param.compileInfo.c_str())})
        .Outputs({&opInfo})
        .Build();
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), GRAPH_SUCCESS);

    StorageShape xShape = {param.xOriginShape, param.xStorageShape};
    StorageShape yShape = {param.yOriginShape, param.yStorageShape};
    vector<pair<string, Ops::NN::AnyValue>> attrsPairs = {
        make_pair("ksize", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(param.ksize)),
        make_pair("strides", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(param.strides)),
        make_pair("padding", Ops::NN::AnyValue::CreateFrom<string>(param.padding)),
        make_pair("data_format", Ops::NN::AnyValue::CreateFrom<string>(param.dataFormat)),
        };
    auto tilingData = gert::TilingData::CreateCap(2048);
    auto holder = gert::TilingContextFaker()
        .SetOpType(param.opType)
        .NodeIoNum(1, 1)
        .IrInstanceNum({1})
        .InputShapes({&xShape})
        .OutputShapes({&yShape})
        .NodeAttrs(attrsPairs)
        .NodeInputTd(0, DT_FLOAT16, param.xOriginFormat, param.xStorageFormat)
        .NodeOutputTd(0, DT_FLOAT16, param.yOriginFormat, param.yStorageFormat)
        .CompileInfo(&opInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(tilingData.get())
        .Build();

    TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    if (param.result) {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    } else {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
        return;
    }

    auto tilingKey = tilingContext->GetTilingKey();
    auto blockDim = tilingContext->GetBlockDim();
    string outputTilingData = TilingData2Str(tilingContext->GetRawTilingData());
    cout << "tilingKey=" << tilingKey << " blockDim=" << blockDim << " tilingData=" << outputTilingData << endl;

    EXPECT_EQ(tilingKey, param.tilingKey);
    EXPECT_EQ(blockDim, param.blockDim);
    EXPECT_EQ(outputTilingData, param.tilingData.c_str());
}

static AvgPoolTilingRuntime2TestParam general_cases_params[] = {
    {   "AvgPool_tiling_dynamic_invalidStride", "AvgPool",
        R"({"_pattern": "Convolution", "push_status": 0,)"\
        R"("tiling_type": "default_tiling", "default_range": {"10000": [1, 2147483647, 16, 16, 16, 16]},)"\
        R"("block_dim": {"10000": 1}, "strides_h": 64, "strides_w" : 64, "k_size_h": 2, "k_size_w": 2,)"\
        R"("filter" : 1, "_vars": {"10000": ["batch_n"]}})",
        {1, 32, 16, 16}, Format::FORMAT_NCHW,
        {1, 32, 16, 16}, Format::FORMAT_NCHW,
        {1, 64, 16, 16}, Format::FORMAT_NCHW,
        {1, 64, 16, 16}, Format::FORMAT_NCHW,
        {1, 1, 1, 1}, {1, 1, 1, 1}, "VALID", "NCHW",
        1, 10000, "1 ", false
    },
};

INSTANTIATE_TEST_CASE_P(AvgPool, AvgPoolTilingRunTime2, testing::ValuesIn(general_cases_params));
