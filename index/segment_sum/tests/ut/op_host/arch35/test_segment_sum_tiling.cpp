/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/segment_sum_tiling_base.h"

using namespace ut_util;
using namespace std;
using namespace ge;

template <typename T>
static string to_string(void *buf, size_t size)
{
    std::string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

class SegmentSumTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SegmentSumTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SegmentSumTiling TearDown" << std::endl;
    }
};

static void ExecuteTestCase(
    gert::StorageShape& dataStorageShape, gert::StorageShape& segmentIdsStorageShape, 
    gert::StorageShape& outputStorageShape, ge::DataType dtype, ge::DataType segmentIds_dtype,
    uint64_t except_tilingkey, std::string expect)
{
    dlog_setlevel(0, 0, 0);

    gert::Shape dataShape = dataStorageShape.GetStorageShape();
    gert::Shape segmentIdsShape = segmentIdsStorageShape.GetStorageShape();
    gert::Shape outputShape = outputStorageShape.GetStorageShape();
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::SegmentSumCompileInfo compile_info;

    std::string op_type("SegmentSum");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    std::vector<gert::Shape*> input_shape_ptrs = {&dataShape, &segmentIdsShape};
    std::vector<gert::Shape*> output_shape_ptrs = {&outputShape};
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .InputShapes(input_shape_ptrs)
                      .OutputShapes(output_shape_ptrs)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, segmentIds_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, except_tilingkey);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    dlog_setlevel(0, 3, 0);
    auto tiling_data_result = to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize());
    std::cout << tiling_data_result << std::endl;
    EXPECT_EQ(tiling_data_result, expect);
}

TEST_F(SegmentSumTiling, simt) {
    // input
    gert::StorageShape data_shape = {{4, 2}, {4, 2}};
    gert::StorageShape segment_ids_shape = {{4}, {4}};
    // output
    gert::StorageShape y_shape = {{4, 2}, {4, 2}};   // assume that max segment_id is 3
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType segment_ids_dtype = ge::DT_INT32;
    uint64_t except_tilingkey = 1000;
    std::string expect = "4 0 2 0 0 0 8 0 0 27640 0 1 0 4 0 4 ";
    ExecuteTestCase(
        data_shape, segment_ids_shape, y_shape, 
        dtype, segment_ids_dtype, except_tilingkey, expect);
}

TEST_F(SegmentSumTiling, simt_perf) {
    // input
    gert::StorageShape data_shape = {{2500, 1}, {2500, 1}};
    gert::StorageShape segment_ids_shape = {{2500}, {2500}};
    // output
    gert::StorageShape y_shape = {{2500, 1}, {2500, 1}};   // assume that max segment_id is 3
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType segment_ids_dtype = ge::DT_INT32;
    uint64_t except_tilingkey = 1000;
    std::string expect = "2500 0 1 0 39 0 43 0 0 27640 1 1 39 43 39 43 ";
    ExecuteTestCase(
        data_shape, segment_ids_shape, y_shape, 
        dtype, segment_ids_dtype, except_tilingkey, expect);
}
