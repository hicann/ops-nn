/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_max_pool3d_extra_tiling.cpp
 * \brief extra arch35 tiling cases for MaxPool3D, target template priorities:
 *        0 OneKsize / 10 NcdhwSmallKernel / 11 NcdhwBigKernel / 13 BigKernelNDHWC / 19 Simt
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
#include "pooling/pool_3d_common/op_host/arch35/max_pool_3d_tiling_common.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
// ascend950: UB_SIZE 245760, CORE_NUM 64, NpuArch 3510 (regbase)
const char* const MAX_POOL_3D_ASCEND950_COMPILE_INFO = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                      "Intrinsic_fix_pipe_l0c2out": false,
                      "Intrinsic_data_move_l12ub": true,
                      "Intrinsic_data_move_l0c2ub": true,
                      "Intrinsic_data_move_out2l1_nd2nz": false,
                      "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                      "CORE_NUM": 64}
                      })";

template <typename T>
std::string TilingDataToString(void* buf, size_t size)
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

// expectTilingKey == 0 means tiling key is not asserted;
// expectBlockDim < 0 means only block_dim >= 1 is asserted;
// expectTilingData empty means raw tiling data is not asserted.
void ExecuteMaxPool3DCase(gert::StorageShape xShape, gert::StorageShape yShape, std::vector<int64_t> ksize,
                          std::vector<int64_t> strides, std::string paddingMode, std::vector<int64_t> pads,
                          std::vector<int64_t> dilation, std::string dataFormat, int64_t ceilMode, ge::DataType dtype,
                          ge::graphStatus expectStatus, uint64_t expectTilingKey, int64_t expectBlockDim,
                          const std::string& expectTilingData = "")
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(MAX_POOL_3D_ASCEND950_COMPILE_INFO, soc_infos, aicore_spec, intrinsics, soc_version_infos);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPool3DCompileInfo compile_info;

    std::string op_type("MaxPool3D");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(paddingMode)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(ceilMode)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)}})
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

    EXPECT_EQ(tiling_func(tiling_context), expectStatus);
    if (expectStatus != ge::GRAPH_SUCCESS) {
        return;
    }
    if (expectTilingKey != 0UL) {
        ASSERT_EQ(tiling_context->GetTilingKey(), expectTilingKey);
    }
    if (expectBlockDim >= 0) {
        ASSERT_EQ(tiling_context->GetBlockDim(), static_cast<uint32_t>(expectBlockDim));
    } else {
        ASSERT_GE(tiling_context->GetBlockDim(), 1U);
    }
    if (!expectTilingData.empty()) {
        auto tilingData = tiling_context->GetRawTilingData();
        ASSERT_NE(tilingData, nullptr);
        std::cout << TilingDataToString<int64_t>(tilingData->GetData(), tilingData->GetDataSize()) << std::endl;
        EXPECT_EQ(TilingDataToString<int64_t>(tilingData->GetData(), tilingData->GetDataSize()), expectTilingData);
    }
}
} // namespace

class MaxPool3DExtraTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPool3DExtraTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaxPool3DExtraTiling TearDown" << std::endl; }
};

// priority 0 OneKsize, NCDHW, kD/kH/kW all 1, no other template is reached
TEST_F(MaxPool3DExtraTiling, MaxPool3D_OneKsize_NCDHW)
{
    gert::StorageShape xShape = {{2, 8, 8, 8, 8}, {2, 8, 8, 8, 8}};
    gert::StorageShape yShape = {{2, 8, 4, 4, 4}, {2, 8, 4, 4, 4}};
    std::vector<int64_t> ksize = {1, 1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1, 1, 1};
    // availableUb = ((245760-1024)/2)/2/2-16 = 30576, halved down to 955 => ubFactorN=14, nLoop=2,
    // totalLoop=2 => blockTail=2, usedCoreNum=2
    std::string expectTilingData = "896 0 0 2 14 4 4 4 2 1 1 1 1 8 8 8 16 4 4 4 2 2 2 1 ";
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NCDHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 100001UL, 2, expectTilingData);
}

// priority 0 OneKsize, NDHWC branch (channels * dtypeSize <= MAX_CHANNEL), dataFormat=1 in tiling data
TEST_F(MaxPool3DExtraTiling, MaxPool3D_OneKsize_NDHWC)
{
    gert::StorageShape xShape = {{2, 8, 8, 8, 4}, {2, 8, 8, 8, 4}};
    gert::StorageShape yShape = {{2, 4, 4, 4, 4}, {2, 4, 4, 4, 4}};
    std::vector<int64_t> ksize = {1, 1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 2, 2, 2, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1, 1, 1};
    // batches=2, outD*outH*outW=64, ubFactorN stays 2 => totalLoop=1 => blockTail=1, usedCoreNum=1
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NDHWC", 0, ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 100001UL, 1);
}

// priority 10 NcdhwSmallKernel, NCDHW, no padding, kernel 3x3x3 stride 2, output count 10976 >= coreNum
TEST_F(MaxPool3DExtraTiling, MaxPool3D_NcdhwSmallKernel_NoPadding)
{
    gert::StorageShape xShape = {{2, 16, 16, 16, 16}, {2, 16, 16, 16, 16}};
    gert::StorageShape yShape = {{2, 16, 7, 7, 7}, {2, 16, 7, 7, 7}};
    std::vector<int64_t> ksize = {1, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1, 1, 1};
    // 0 rejected: kernel is not all 1; 10 accepted: NCDHW + output count 32*7*7*7 >= 64 + buffer capable
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NCDHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 300001UL, -1);
}

// priority 11 NcdhwBigKernel normal path: output count 500 >= 2*coreNum, kernel bytes 16384 >= 256,
// strong overlap (5 * stride 3 < ksize 16); priority 10 rejected by dilation&stride both >= 3
TEST_F(MaxPool3DExtraTiling, MaxPool3D_NcdhwBigKernel_StrongOverlap)
{
    gert::StorageShape xShape = {{1, 4, 60, 60, 60}, {1, 4, 60, 60, 60}};
    gert::StorageShape yShape = {{1, 4, 5, 5, 5}, {1, 4, 5, 5, 5}};
    std::vector<int64_t> ksize = {1, 1, 16, 16, 16};
    std::vector<int64_t> strides = {1, 1, 3, 3, 3};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 3, 3, 3};
    // totalIdx = 4*5*5*5 = 500 => blockFactor=7, blockTail=52, coreNums=64
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NCDHW", 0, ge::DT_FLOAT,
                         ge::GRAPH_SUCCESS, 511110UL, 64);
}

// priority 11 NcdhwBigKernel direct path: output count 27 < 2*coreNum(128);
// priority 10 rejected because batches*outD*outH*outW = 27 < coreNum(64)
TEST_F(MaxPool3DExtraTiling, MaxPool3D_NcdhwBigKernel_SmallOutput)
{
    gert::StorageShape xShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}};
    std::vector<int64_t> ksize = {1, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1, 1, 1};
    // totalIdx = 27 => blockFactor=0, coreNums=27
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NCDHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 511110UL, 27);
}

// priority 13 BigKernelNDHWC normal path: outNum 200 >= 2*coreNum, kernel*D*H*W*C*dtypeSize 512 >= 256;
// priority 10 rejected (NDHWC with pad), priority 12 rejected (stride 8 >= 2 * effKsize 8)
TEST_F(MaxPool3DExtraTiling, MaxPool3D_BigKernelNDHWC_BigKernel)
{
    gert::StorageShape xShape = {{2, 34, 34, 34, 4}, {2, 34, 34, 34, 4}};
    gert::StorageShape yShape = {{2, 4, 5, 5, 4}, {2, 4, 5, 5, 4}};
    std::vector<int64_t> ksize = {1, 4, 4, 4, 1};
    std::vector<int64_t> strides = {1, 8, 8, 8, 1};
    std::vector<int64_t> pads = {0, 0, 1, 1, 1, 1};
    std::vector<int64_t> dilation = {1, 1, 1, 1, 1};
    // outD = DivRtn(34-4,8)+1 = 4 (D pads are 0), outH = outW = DivRtn(34+2-4,8)+1 = 5;
    // totalIdx = 2*4*5*5 = 200 => blockFactor=3, blockTail=8, coreNums=64
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NDHWC", 0, ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 411110UL, 64);
}

// priority 13 BigKernelNDHWC direct path: outNum 27 < 2*coreNum(128);
// priority 10 rejected (outNum 27 < 64), priority 12 rejected (totalLoops 27*2 = 54 < coreNum)
TEST_F(MaxPool3DExtraTiling, MaxPool3D_BigKernelNDHWC_SmallOutput)
{
    gert::StorageShape xShape = {{1, 8, 8, 8, 2}, {1, 8, 8, 8, 2}};
    gert::StorageShape yShape = {{1, 3, 3, 3, 2}, {1, 3, 3, 3, 2}};
    std::vector<int64_t> ksize = {1, 3, 3, 3, 1};
    std::vector<int64_t> strides = {1, 2, 2, 2, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1, 1, 1};
    // totalIdx = 27 => blockFactor=0, coreNums=27
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NDHWC", 0, ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 411110UL, 27);
}

// priority 19 Simt fallback: priority 10 rejected (dilation 3 && stride 3 in H/D),
// priority 11 rejected (kernel 3*3*3*2 = 54 < 256)
TEST_F(MaxPool3DExtraTiling, MaxPool3D_Simt_DilationStrideReject)
{
    gert::StorageShape xShape = {{2, 8, 30, 30, 30}, {2, 8, 30, 30, 30}};
    gert::StorageShape yShape = {{2, 8, 8, 8, 8}, {2, 8, 8, 8, 8}};
    std::vector<int64_t> ksize = {1, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 3, 3, 3};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 3, 3, 3};
    // NCDHW, xSize = 432000 <= INT32_MAX => SIMT NCDHW INT32 key, block dim = coreNum
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NCDHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 911100UL, 64);
}

// priority 19 Simt fallback: priority 10 rejected (padding && stride 14 >= 2 * effKsize 14),
// priority 11 rejected (kernel bytes 686 >= 256 but no strong overlap: 5 * 14 >= 7)
TEST_F(MaxPool3DExtraTiling, MaxPool3D_Simt_NoStrongOverlap)
{
    gert::StorageShape xShape = {{2, 8, 40, 40, 40}, {2, 8, 40, 40, 40}};
    gert::StorageShape yShape = {{2, 8, 3, 3, 3}, {2, 8, 3, 3, 3}};
    std::vector<int64_t> ksize = {1, 1, 7, 7, 7};
    std::vector<int64_t> strides = {1, 1, 14, 14, 14};
    std::vector<int64_t> pads = {0, 0, 1, 1, 1, 1};
    std::vector<int64_t> dilation = {1, 1, 1, 1, 1};
    // NCDHW, xSize = 1024000 <= INT32_MAX => SIMT NCDHW INT32 key, block dim = coreNum
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NCDHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 911100UL, 64);
}

// error branch: 5D ksize whose N/C dim is not 1, GetKernelKsizeInfo fails
TEST_F(MaxPool3DExtraTiling, MaxPool3D_InvalidKsizeNC_Dim_Failed)
{
    gert::StorageShape xShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}};
    std::vector<int64_t> ksize = {2, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1, 1, 1};
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NCDHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_FAILED, 0UL, -1);
}

// error branch: CALCULATED mode requires pads with 6 elements
TEST_F(MaxPool3DExtraTiling, MaxPool3D_InvalidPadsSize_Failed)
{
    gert::StorageShape xShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}};
    std::vector<int64_t> ksize = {1, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1, 1, 1, 1};
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NCDHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_FAILED, 0UL, -1);
}

// error branch: data_format only supports NCDHW/NDHWC
TEST_F(MaxPool3DExtraTiling, MaxPool3D_InvalidDataFormat_Failed)
{
    gert::StorageShape xShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}};
    std::vector<int64_t> ksize = {1, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> dilation = {1, 1, 1, 1, 1};
    ExecuteMaxPool3DCase(xShape, yShape, ksize, strides, "CALCULATED", pads, dilation, "NCHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_FAILED, 0UL, -1);
}

// ===================== padding_mode SAME/VALID 校验(CheckOutPutShapeForSame/Valid) =====================

// VALID: 期望输出 = (in - (k-1)*dil - 1)/s + 1, 16/2-1+... D/H/W=(16-2-1)/2+1=7, 校验通过
TEST_F(MaxPool3DExtraTiling, MaxPool3D_Valid_PadMode_NCDHW_Success)
{
    ExecuteMaxPool3DCase({{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}}, {{1, 8, 7, 7, 7}, {1, 8, 7, 7, 7}}, {1, 1, 3, 3, 3},
                         {1, 1, 2, 2, 2}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, "NCDHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 0UL, -1);
}

// SAME: 期望输出 = ceil(in/s) = 8, 校验通过
TEST_F(MaxPool3DExtraTiling, MaxPool3D_Same_PadMode_NCDHW_Success)
{
    ExecuteMaxPool3DCase({{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}}, {{1, 8, 8, 8, 8}, {1, 8, 8, 8, 8}}, {1, 1, 3, 3, 3},
                         {1, 1, 2, 2, 2}, "SAME", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, "NCDHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 0UL, -1);
}

// VALID 输出 shape 不匹配 -> CheckOutPutShapeForValid 失败分支
TEST_F(MaxPool3DExtraTiling, MaxPool3D_Valid_PadMode_OutShapeMismatch_Failed)
{
    ExecuteMaxPool3DCase({{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}}, {{1, 8, 8, 8, 8}, {1, 8, 8, 8, 8}}, {1, 1, 3, 3, 3},
                         {1, 1, 2, 2, 2}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, "NCDHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_FAILED, 0UL, -1);
}

// SAME 输出 shape 不匹配 -> CheckOutPutShapeForSame 失败分支
TEST_F(MaxPool3DExtraTiling, MaxPool3D_Same_PadMode_OutShapeMismatch_Failed)
{
    ExecuteMaxPool3DCase({{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}}, {{1, 8, 7, 7, 7}, {1, 8, 7, 7, 7}}, {1, 1, 3, 3, 3},
                         {1, 1, 2, 2, 2}, "SAME", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, "NCDHW", 0, ge::DT_FLOAT16,
                         ge::GRAPH_FAILED, 0UL, -1);
}

// ===================== NcdhwSmallKernel: sparse 模式 + 大输入切分 =====================

// stride(8) >= 2*effKsize(3) 且无 pad -> isSparseD/H/W=true, 走 CalcInputSizeByOutput/
// CalcOutputSizeByInput/CalxMaxInputSize 的 sparse 分支; 256^3 大输入触发 CalcSplitMaxRows/
// CalcSplitMaxCols/CalcEnableSplit 切分链路
TEST_F(MaxPool3DExtraTiling, MaxPool3D_NcdhwSmallKernel_Sparse_LargeInput)
{
    ExecuteMaxPool3DCase({{1, 16, 256, 256, 256}, {1, 16, 256, 256, 256}}, {{1, 16, 32, 32, 32}, {1, 16, 32, 32, 32}},
                         {1, 1, 3, 3, 3}, {1, 1, 8, 8, 8}, "CALCULATED", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, "NCDHW",
                         0, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 0UL, -1);
}

// ===================== Simt: NDHWC 全拒场景(key 911101 变体) =====================

// NDHWC+pad 使 NcdhwSmallKernel(10) 拒; stride(8)>=MAX_STRIDE(2)*effK(2) 使 SmallKernelNDHWC(12) 拒;
// kernel 2*2*2*C(8)*dtypeSize(2)=128<MIN_KERNEL(256) 使 BigKernelNDHWC(13) 拒 -> Simt(19) 兜底
TEST_F(MaxPool3DExtraTiling, MaxPool3D_Ndhwc_Simt_Fallback)
{
    ExecuteMaxPool3DCase({{1, 32, 64, 64, 8}, {1, 32, 64, 64, 8}}, {{1, 5, 9, 9, 8}, {1, 5, 9, 9, 8}}, {1, 2, 2, 2, 1},
                         {1, 8, 8, 8, 1}, "CALCULATED", {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, "NDHWC", 0, ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 0UL, -1);
}
