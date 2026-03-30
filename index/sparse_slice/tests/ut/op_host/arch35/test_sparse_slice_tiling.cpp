/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sparse_slice_tiling.cpp
 * \brief
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
#include "../../../../op_host/arch35/sparse_slice_tiling_arch35.h"

using namespace ge;
using namespace std;

class SparseSliceTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseSliceTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseSliceTilingTest TearDown" << std::endl;
    }
};

template <typename T>
void SetConstInput(
    size_t const_index, ge::DataType dtype, T* const_data, int64_t data_size,
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors)
{
    std::unique_ptr<uint8_t[]> input_tensor_holder =
        std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(T) * data_size]);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    gert::Tensor tensor(
        {{data_size}, {data_size}},         // shape
        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
        gert::kFollowing,                   // placement
        dtype,                              // dt
        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

TEST_F(SparseSliceTilingTest, SparseSliceTiling_10000)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, 
                            "Intrinsic_data_move_l12ub": true, 
                            "Intrinsic_data_move_l0c2ub": true, 
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, socversions);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    gert::StorageShape indicesShape = {{8, 2}, {8, 2}};
    gert::StorageShape valuesShape = {{8}, {8}};
    gert::StorageShape shapeShape = {{2}, {2}};
    gert::StorageShape startShape = {{2}, {2}};
    gert::StorageShape sizeShape = {{2}, {2}};

    gert::StorageShape yindicesShape = {{8, 2}, {8, 2}};
    gert::StorageShape yvaluesShape = {{8}, {8}};
    gert::StorageShape ysizeShape = {{2}, {2}};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> ConstTensors;
    int64_t shape[2] = {3, 15};
    SetConstInput(2, ge::DT_INT64, shape, 2, ConstTensors);
    int64_t start[2] = {0, 0};
    SetConstInput(3, ge::DT_INT64, start, 2, ConstTensors);
    int64_t size[2] = {3, 15};
    SetConstInput(4, ge::DT_INT64, size, 2, ConstTensors);

    // compile info
    optiling::SparseSliceCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    std::string opType("SparseSlice");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platformInfo)})
                             .Outputs({&compileInfo})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&indicesShape, &valuesShape, &shapeShape, &startShape, &sizeShape})
                      .OutputShapes({&yindicesShape, &yvaluesShape, &ysizeShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(ConstTensors)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    int tilingKeyValue = 10000;
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tiling_key = tilingContext->GetTilingKey();
    EXPECT_EQ(tiling_key, tilingKeyValue);
}

TEST_F(SparseSliceTilingTest, SparseSliceTiling_40000)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, 
                            "Intrinsic_data_move_l12ub": true, 
                            "Intrinsic_data_move_l0c2ub": true, 
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, socversions);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    gert::StorageShape indicesShape = {{8, 3}, {8, 3}};
    gert::StorageShape valuesShape = {{8}, {8}};
    gert::StorageShape shapeShape = {{3}, {3}};
    gert::StorageShape startShape = {{3}, {3}};
    gert::StorageShape sizeShape = {{3}, {3}};

    gert::StorageShape yindicesShape = {{8, 3}, {8, 3}};
    gert::StorageShape yvaluesShape = {{8}, {8}};
    gert::StorageShape ysizeShape = {{3}, {3}};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> ConstTensors;
    int64_t shape[3] = {3, 15, 4};
    SetConstInput(2, ge::DT_INT64, shape, 3, ConstTensors);
    int64_t start[3] = {0, 0, 0};
    SetConstInput(3, ge::DT_INT64, start, 3, ConstTensors);
    int64_t size[3] = {3, 15, 4};
    SetConstInput(4, ge::DT_INT64, size, 3, ConstTensors);

    // compile info
    optiling::SparseSliceCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    std::string opType("SparseSlice");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platformInfo)})
                             .Outputs({&compileInfo})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&indicesShape, &valuesShape, &shapeShape, &startShape, &sizeShape})
                      .OutputShapes({&yindicesShape, &yvaluesShape, &ysizeShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(ConstTensors)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    int tilingKeyValue = 40000;
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tiling_key = tilingContext->GetTilingKey();
    EXPECT_EQ(tiling_key, tilingKeyValue);
}

TEST_F(SparseSliceTilingTest, SparseSliceTiling_20000)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, 
                            "Intrinsic_data_move_l12ub": true, 
                            "Intrinsic_data_move_l0c2ub": true, 
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, socversions);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    gert::StorageShape indicesShape = {{8, 2}, {8, 2}};
    gert::StorageShape valuesShape = {{8}, {8}};
    gert::StorageShape shapeShape = {{2}, {2}};
    gert::StorageShape startShape = {{2}, {2}};
    gert::StorageShape sizeShape = {{2}, {2}};

    gert::StorageShape yindicesShape = {{8, 2}, {8, 2}};
    gert::StorageShape yvaluesShape = {{8}, {8}};
    gert::StorageShape ysizeShape = {{2}, {2}};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> ConstTensors;
    int64_t shape[2] = {3, 15};
    SetConstInput(2, ge::DT_INT64, shape, 2, ConstTensors);
    int64_t start[2] = {0, 0};
    SetConstInput(3, ge::DT_INT64, start, 2, ConstTensors);
    int64_t size[2] = {0, 0};
    SetConstInput(4, ge::DT_INT64, size, 2, ConstTensors);

    // compile info
    optiling::SparseSliceCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    std::string opType("SparseSlice");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platformInfo)})
                             .Outputs({&compileInfo})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&indicesShape, &valuesShape, &shapeShape, &startShape, &sizeShape})
                      .OutputShapes({&yindicesShape, &yvaluesShape, &ysizeShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(ConstTensors)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    int tilingKeyValue = 20000;
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tiling_key = tilingContext->GetTilingKey();
    EXPECT_EQ(tiling_key, tilingKeyValue);
}

TEST_F(SparseSliceTilingTest, SparseSliceTiling_20000_failed)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, 
                            "Intrinsic_data_move_l12ub": true, 
                            "Intrinsic_data_move_l0c2ub": true, 
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, socversions);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    gert::StorageShape indicesShape = {{8, 2}, {8, 2}};
    gert::StorageShape valuesShape = {{8}, {8}};
    gert::StorageShape shapeShape = {{1}, {1}};
    gert::StorageShape startShape = {{2}, {2}};
    gert::StorageShape sizeShape = {{2}, {2}};

    gert::StorageShape yindicesShape = {{8, 2}, {8, 2}};
    gert::StorageShape yvaluesShape = {{8}, {8}};
    gert::StorageShape ysizeShape = {{2}, {2}};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> ConstTensors;
    int64_t shape[1] = {3};
    SetConstInput(2, ge::DT_INT64, shape, 1, ConstTensors);
    int64_t start[2] = {0, 0};
    SetConstInput(3, ge::DT_INT64, start, 2, ConstTensors);
    int64_t size[2] = {0, 0};
    SetConstInput(4, ge::DT_INT64, size, 2, ConstTensors);

    // compile info
    optiling::SparseSliceCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    std::string opType("SparseSlice");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platformInfo)})
                             .Outputs({&compileInfo})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&indicesShape, &valuesShape, &shapeShape, &startShape, &sizeShape})
                      .OutputShapes({&yindicesShape, &yvaluesShape, &ysizeShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(ConstTensors)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tilingFunc(tilingContext), ge::FAILED);
}

TEST_F(SparseSliceTilingTest, SparseSliceTiling_20000_EMPTY)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, 
                            "Intrinsic_data_move_l12ub": true, 
                            "Intrinsic_data_move_l0c2ub": true, 
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, socversions);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    gert::StorageShape indicesShape = {{0, 2}, {0, 2}};
    gert::StorageShape valuesShape = {{0}, {0}};
    gert::StorageShape shapeShape = {{2}, {2}};
    gert::StorageShape startShape = {{2}, {2}};
    gert::StorageShape sizeShape = {{2}, {2}};

    gert::StorageShape yindicesShape = {{0, 2}, {0, 2}};
    gert::StorageShape yvaluesShape = {{0}, {0}};
    gert::StorageShape ysizeShape = {{2}, {2}};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> ConstTensors;
    int64_t shape[2] = {3, 15};
    SetConstInput(2, ge::DT_INT64, shape, 2, ConstTensors);
    int64_t start[2] = {0, 0};
    SetConstInput(3, ge::DT_INT64, start, 2, ConstTensors);
    int64_t size[2] = {3, 15};
    SetConstInput(4, ge::DT_INT64, size, 2, ConstTensors);

    // compile info
    optiling::SparseSliceCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    std::string opType("SparseSlice");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platformInfo)})
                             .Outputs({&compileInfo})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&indicesShape, &valuesShape, &shapeShape, &startShape, &sizeShape})
                      .OutputShapes({&yindicesShape, &yvaluesShape, &ysizeShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(ConstTensors)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    int tilingKeyValue = 20000;
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tiling_key = tilingContext->GetTilingKey();
    EXPECT_EQ(tiling_key, tilingKeyValue);
}
