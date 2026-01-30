/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_index_put_v2_tiling.cpp
 * \brief
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/index_put_v2_tiling.h"
#include "any_value.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class IndexPutV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "IndexPutV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IndexPutV2Tiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData *tiling_data) {
auto data = tiling_data->GetData();
string result;
for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
    result += std::to_string((reinterpret_cast<const int32_t *>(tiling_data->GetData())[i / sizeof(int32_t)]));
    result += " ";
}
return result;
}

template <typename T>
void SetConstInput(size_t const_index, ge::DataType dtype, T* const_data, int64_t data_size,
                std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> &const_tensors) {
std::unique_ptr<uint8_t[]> input_tensor_holder =
    std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(T) * data_size]);
auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
gert::Tensor tensor({{data_size}, {data_size}},               // shape
                    {ge::FORMAT_ND, ge::FORMAT_ND, {}},        // format
                    gert::kFollowing,                          // placement
                    dtype,                                     //dt
                    nullptr);
std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
auto tensor_data = reinterpret_cast<T *>(input_tensor + 1);
for(int64_t i =0; i < data_size; i++) {
    tensor_data[i] = const_data[i];
}
input_tensor->SetData(gert::TensorData{tensor_data});
auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
const_tensors.push_back(std::move(pair));
}

TEST_F(IndexPutV2Tiling, IndexPutV2_AC_tiling_fp16_continue_0) {
    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::IndexPutV2CompileInfo compile_info; 

    std::string op_type("IndexPutV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{4096,6400}, {4096,6400}};
    gert::StorageShape value = {{4096}, {4096}};
    gert::StorageShape indexedSizes = {{2}, {2}};
    gert::StorageShape indexedStrides = {{1}, {1}};
    gert::StorageShape indices = {{4096}, {4096}};
    gert::StorageShape y = {{4096}, {4096}};
    std::map<std::string, std::string> soc_version_infos = { { "Short_SoC_version", "Ascend950" } };

    int64_t mask[2] = {1,1};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(2, DT_INT64, mask, 2, const_tensors);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(5, 1)
        .IrInstanceNum({1,1,1,1,1})
        .InputShapes({&x, &value, &indexedSizes, &indexedStrides, &indices})
        .OutputShapes({&y})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"accumulate", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
        .ConstInput(const_tensors)
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 102);
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::string expect_tiling = "26214400 0 4096 0 4096 0 2 1 2 1 4096 0 6400 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    ASSERT_EQ(expect_tiling, tiling_data_result);
}