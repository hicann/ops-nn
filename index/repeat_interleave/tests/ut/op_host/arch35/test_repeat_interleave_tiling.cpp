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
 * \file test_repeat_interleave_tiling.cpp
 * \brief the ut of repeat_interleave_tiling for ascendc
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/repeat_interleave_tiling_arch35.h"
#include "../../../../op_host/arch35/repeat_interleave_tiling_normal.h"

using namespace std;
class RepeatInterleaveTilingAscendC : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RepeatInterleaveTilingAscendC SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RepeatInterleaveTilingAscendC TearDown" << std::endl;
    }
};

template <typename T>
string ToString(void* buf, size_t size)
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

struct OpsParamInfosRepeatInterleave {
    ge::DataType xDtype;
    ge::DataType repeatsDtype;
    ge::DataType yDtype;
    gert::StorageShape xShape;
    gert::StorageShape repeatsShape;
    gert::StorageShape yShape;
    int64_t axis;
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
        {{data_size}, {data_size}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, dtype, nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

static void ExecuteTestCase(
    const OpsParamInfosRepeatInterleave& opsParamInfos, string& expectTilingData,
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors,
    ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;

    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    // compile info
    optiling::RepeatInterleaveCompileInfo compileInfo;

    compileInfo.coreNumAscendc = 64;
    compileInfo.ubSizeAscendc = 253952;

    std::string opType("RepeatInterleave");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    auto parseContext = kernelHolder.GetContext<gert::TilingParseContext>();
    ASSERT_NE(parseContext, nullptr);
    auto platformInfoPtr = parseContext->GetPlatformInfo();
    ASSERT_NE(platformInfoPtr, nullptr);
    ASSERT_TRUE(platformInfoPtr->Init());
    platformInfoPtr->SetPlatformRes("SoCInfo", socInfos);
    platformInfoPtr->SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfoPtr->SetCoreNumByCoreType("AICore");
    platformInfoPtr->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = opsParamInfos.xShape;
    gert::StorageShape repeatsShape = opsParamInfos.repeatsShape;
    gert::StorageShape yShape = opsParamInfos.yShape;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &repeatsShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, opsParamInfos.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, opsParamInfos.repeatsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, opsParamInfos.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(opsParamInfos.axis)}})
                      .ConstInput(const_tensors)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    auto infos = tilingContext->GetPlatformInfo();
    ASSERT_NE(infos, nullptr);
    infos->SetPlatformRes("SoCInfo", socInfos);
    infos->SetPlatformRes("AICoreSpec", aicoreSpec);
    infos->SetCoreNumByCoreType("AICore");
    infos->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    // todo check tiling result
    auto rawTilingData = tilingContext->GetRawTilingData();
    auto tilingDataResult = ToString<int64_t>(rawTilingData->GetData(), rawTilingData->GetDataSize());
    EXPECT_EQ(tilingDataResult, expectTilingData);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_0_test0)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{2, 3, 4}, {2, 3, 4}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{4, 3, 4}, {4, 3, 4}};
    opsParamInfos.axis = 0;
    int32_t repeatsValue[1] = {2};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 2 1 1 0 0 0 0 10560 2 4 2 1 12 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_1_test1)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{64, 3, 4}, {64, 3, 4}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{64, 30, 4}, {64, 30, 4}};
    opsParamInfos.axis = 1;
    int32_t repeatsValue[1] = {10};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 64 3 3 0 0 0 0 10560 10 30 192 1 4 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_1_test2)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{2, 3, 256}, {2, 3, 256}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{2, 30, 256}, {2, 30, 256}};
    opsParamInfos.axis = 1;
    int32_t repeatsValue[1] = {10};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 48 1 1 1 32 32 8 10560 10 30 6 1 256 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}
TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_1_test3)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{64, 3, 4}, {64, 3, 4}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{64, 15, 4}, {64, 15, 4}};
    opsParamInfos.axis = 1;
    int32_t repeatsValue[1] = {5};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 64 3 3 0 0 0 0 10560 5 15 192 1 4 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_2_test4)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{64, 3, 4}, {64, 3, 4}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{64, 3, 20}, {64, 3, 20}};
    opsParamInfos.axis = 2;
    int32_t repeatsValue[1] = {5};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 64 12 12 0 0 0 0 10560 5 20 768 1 1 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}
TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_None_test5)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{64, 3, 4}, {64, 3, 4}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{64, 3, 20}, {64, 3, 20}};
    opsParamInfos.axis = 1000;
    int32_t repeatsValue[1] = {5};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 64 12 12 0 0 0 0 10560 5 20 768 1 1 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_None_test6)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{64, 3, 4}, {64, 3, 4}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{3840}, {3840}};
    opsParamInfos.axis = 1000;
    int32_t repeatsValue[1] = {5};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 64 12 12 0 0 0 0 10560 0 0 768 1 1 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_1_test7)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{96, 3, 4}, {96, 3, 4}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{96, 12, 4}, {96, 12, 4}};
    opsParamInfos.axis = 1;
    int32_t repeatsValue[1] = {4};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 58 5 3 0 0 0 0 10560 4 12 288 1 4 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_1_test8)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{35, 3, 4}, {35, 3, 4}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{35, 12, 4}, {35, 12, 4}};
    opsParamInfos.axis = 1;
    int32_t repeatsValue[1] = {4};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 53 2 1 0 0 0 0 10560 4 12 105 1 4 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_1_test9)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{16, 3, 4}, {16, 3, 4}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{16, 12, 4}, {16, 12, 4}};
    opsParamInfos.axis = 1;
    int32_t repeatsValue[1] = {4};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 48 1 1 0 0 0 0 10560 4 12 48 1 4 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_0_test10)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{440}, {440}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{880}, {880}};
    opsParamInfos.axis = 0;
    int32_t repeatsValue[1] = {2};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 63 7 6 0 0 0 0 10560 2 880 440 1 1 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}
TEST_F(RepeatInterleaveTilingAscendC, Repeats_Scalar_Axis_0_test11)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{2, 3, 1024}, {2, 3, 1024}};
    opsParamInfos.repeatsShape = {{1}, {1}};
    opsParamInfos.yShape = {{4, 3, 1024}, {4, 3, 1024}};
    opsParamInfos.axis = 0;
    int32_t repeatsValue[1] = {2};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, 1, const_tensors);

    string expectTilingData = "64 64 1 1 1 96 96 32 10560 2 4 2 1 3072 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Tensor_Axis_0_test20)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{10, 3, 4}, {10, 3, 4}};
    opsParamInfos.repeatsShape = {{10}, {10}};
    opsParamInfos.yShape = {{55, 3, 4}, {55, 3, 4}};
    opsParamInfos.axis = 0;
    const int64_t dataSize = 10;
    int32_t repeatsValue[dataSize] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, dataSize, const_tensors);

    string expectTilingData = "64 1 1 1 0 0 0 0 10560 -1 55 1 10 12 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Tensor_Axis_1_test21)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{64, 10, 4}, {64, 10, 4}};
    opsParamInfos.repeatsShape = {{10}, {10}};
    opsParamInfos.yShape = {{64, 55, 4}, {64, 55, 4}};
    opsParamInfos.axis = 1;
    const int64_t dataSize = 10;
    int32_t repeatsValue[dataSize] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, dataSize, const_tensors);

    string expectTilingData = "64 64 1 1 0 0 0 0 10560 -1 55 64 10 4 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Tensor_Axis_1_test22)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{32, 10, 4}, {32, 10, 4}};
    opsParamInfos.repeatsShape = {{10}, {10}};
    opsParamInfos.yShape = {{32, 55, 4}, {32, 55, 4}};
    opsParamInfos.axis = 1;
    const int64_t dataSize = 10;
    int32_t repeatsValue[dataSize] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, dataSize, const_tensors);

    string expectTilingData = "64 32 1 1 0 0 0 0 10560 -1 55 32 10 4 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Tensor_Axis_1_test23)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{2, 10, 4}, {2, 10, 4}};
    opsParamInfos.repeatsShape = {{10}, {10}};
    opsParamInfos.yShape = {{2, 55, 4}, {2, 55, 4}};
    opsParamInfos.axis = 1;
    const int64_t dataSize = 10;
    int32_t repeatsValue[dataSize] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, dataSize, const_tensors);

    string expectTilingData = "64 2 1 1 0 0 0 0 10560 -1 55 2 10 4 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Tensor_Axis_1_test24)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{2, 10, 4}, {2, 10, 4}};
    opsParamInfos.repeatsShape = {{10}, {10}};
    opsParamInfos.yShape = {{2, 55, 4}, {2, 55, 4}};
    opsParamInfos.axis = 1;
    const int64_t dataSize = 10;
    int32_t repeatsValue[dataSize] = {10, 9, 3, 4, 5, 6, 7, 8, 2, 1};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, dataSize, const_tensors);

    string expectTilingData = "64 2 1 1 0 0 0 0 10560 -1 55 2 10 4 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}

TEST_F(RepeatInterleaveTilingAscendC, Repeats_Tensor_Axis_1_test25)
{
    OpsParamInfosRepeatInterleave opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.repeatsDtype = ge::DT_INT32;
    opsParamInfos.yDtype = opsParamInfos.xDtype;
    opsParamInfos.xShape = {{2, 70, 256}, {2, 70, 256}};
    opsParamInfos.repeatsShape = {{70}, {70}};
    opsParamInfos.yShape = {{2, 1380, 256}, {2, 1380, 256}};
    opsParamInfos.axis = 1;
    const int64_t dataSize = 70;
    int32_t repeatsValue[dataSize] = {110, 120, 130, 100, 90, 150, 160, 80, 200, 120, 2, 2, 2, 2, 2, 2, 2, 2,
                                      2,   2,   2,   2,   2,  2,   2,   2,  2,   2,   2, 2, 2, 2, 2, 2, 2, 2,
                                      2,   2,   2,   2,   2,  2,   2,   2,  2,   2,   2, 2, 2, 2, 2, 2, 2, 2,
                                      2,   2,   2,   2,   2,  2,   2,   2,  2,   2,   2, 2, 2, 2, 2, 2};

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput<int32_t>(1, ge::DT_INT32, repeatsValue, dataSize, const_tensors);

    string expectTilingData = "64 48 1 1 3 1 24 20800 1380 19 2 70 256 ";
    ExecuteTestCase(opsParamInfos, expectTilingData, const_tensors);
}