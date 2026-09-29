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
 * \file test_avg_pool3d_extra_tiling.cpp
 * \brief extra arch35 tiling cases for AvgPool3D on ascend950(regbase), target template priorities:
 *        0 OneKsize / 10 NcdhwSmallKernel / 11 NcdhwBigKernel / 12 SmallKernelNDHWC / 13 BigKernelNDHWC / 19 Simt
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
#include "register/op_impl_registry.h"
#include "pooling/avg_pool3_d/op_host/avg_pool_cube_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
// ascend950: UB_SIZE 245760, CORE_NUM 64, NpuArch 3510 (regbase)
const char* const AVG_POOL_3D_ASCEND950_COMPILE_INFO = R"({
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
void ExecuteAvgPool3DCase(gert::StorageShape xShape, gert::StorageShape yShape, std::vector<int64_t> ksize,
                          std::vector<int64_t> strides, std::vector<int64_t> pads, bool ceilMode, bool countIncludePad,
                          int64_t divisorOverride, std::string dataFormat, ge::DataType dtype,
                          ge::graphStatus expectStatus, uint64_t expectTilingKey, int64_t expectBlockDim,
                          const std::string& expectTilingData = "")
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(AVG_POOL_3D_ASCEND950_COMPILE_INFO, soc_infos, aicore_spec, intrinsics, soc_version_infos);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info: tiling_parse fills is_regbase so that Tiling4AvgPool3D enters the template registry
    optiling::avgPool3DTilingCompileInfo::AvgPool3DCubeCompileInfo compile_info;

    std::string op_type("AvgPool3D");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate, empty json => ascend-c path, NpuArch 3510 => is_regbase = true
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

    // tilingFunc simulate, the same compile info object is passed to the tiling context
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
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceilMode)},
                                  {"count_include_pad", Ops::NN::AnyValue::CreateFrom<bool>(countIncludePad)},
                                  {"divisor_override", Ops::NN::AnyValue::CreateFrom<int64_t>(divisorOverride)},
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

/*
 * arch35 模板可用性探测: AvgPool3D 为多 SoC 算子(ascend350/910_93/910b/950/kirin 系),
 * pool_3d_common 的 arch35 tiling 模板仅在 ascend950 构建中注册; 在其他 SoC 的 UT
 * 步骤(如 ut_acc)中模板注册表为空, DoTilingImpl 必然失败。本套件为 arch35 专项用例,
 * 用最小合法输入(OneKsize 必收, ascend950 下必然成功)探测一次, 失败则跳过。
 */
bool AvgPool3DArch35TemplateAvailable()
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(AVG_POOL_3D_ASCEND950_COMPILE_INFO, soc_infos, aicore_spec, intrinsics, soc_version_infos);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::avgPool3DTilingCompileInfo::AvgPool3DCubeCompileInfo compile_info;

    std::string op_type("AvgPool3D");
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (op_impl == nullptr || op_impl->tiling == nullptr || op_impl->tiling_parse == nullptr) {
        return false;
    }

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    if (!kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init()) {
        return false;
    }
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    if (op_impl->tiling_parse(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return false;
    }

    gert::StorageShape xShape = {{1, 1, 4, 4, 4}, {1, 1, 4, 4, 4}};
    gert::StorageShape yShape = {{1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}};
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"count_include_pad", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"divisor_override", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context == nullptr || tiling_context->GetPlatformInfo() == nullptr) {
        return false;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    // ascend950 下 OneKsize(优先级 0) 必然成功; 非 950 构建模板注册表为空, 返回 FAILED
    return op_impl->tiling(tiling_context) == ge::GRAPH_SUCCESS;
}
} // namespace

class AvgPool3DExtraTiling : public testing::Test {
protected:
    // arch35 专项用例: 非 ascend950 构建(如 ut_acc 的 Atlas 训练系列 SoC)未链接
    // pool_3d_common 的 arch35 模板, 探测失败则逐用例跳过(结果静态缓存, 仅探测一次)
    void SetUp() override
    {
        static const bool arch35Available = AvgPool3DArch35TemplateAvailable();
        if (!arch35Available) {
            GTEST_SKIP() << "arch35 tiling templates not linked in this build, skip arch35-specific cases";
        }
    }

    static void TearDownTestCase() { std::cout << "AvgPool3DExtraTiling TearDown" << std::endl; }
};

// priority 0 OneKsize, NCDHW, kD/kH/kW all 1, no other template is reached
TEST_F(AvgPool3DExtraTiling, AvgPool3D_OneKsize_NCDHW)
{
    gert::StorageShape xShape = {{2, 8, 8, 8, 8}, {2, 8, 8, 8, 8}};
    gert::StorageShape yShape = {{2, 8, 4, 4, 4}, {2, 8, 4, 4, 4}};
    std::vector<int64_t> ksize = {1, 1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    // batches = N*C = 16, availableUb halved to 955 => ubFactorN=14, nLoop=2 => usedCoreNum=2
    std::string expectTilingData = "896 0 0 2 14 4 4 4 2 1 1 1 1 8 8 8 16 4 4 4 2 2 2 1 ";
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, true, 0, "NCDHW", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 100001UL, 2, expectTilingData);
}

// priority 0 OneKsize, NDHWC branch (channels * dtypeSize <= MAX_CHANNEL)
TEST_F(AvgPool3DExtraTiling, AvgPool3D_OneKsize_NDHWC)
{
    gert::StorageShape xShape = {{2, 8, 8, 8, 4}, {2, 8, 8, 8, 4}};
    gert::StorageShape yShape = {{2, 4, 4, 4, 4}, {2, 4, 4, 4, 4}};
    std::vector<int64_t> ksize = {1, 1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 2, 2, 2, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    // batches=2, ubFactorN stays 2 => totalLoop=1 => usedCoreNum=1
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, true, 0, "NDHWC", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 100001UL, 1);
}

// priority 0 OneKsize rejected (NDHWC channels * dtypeSize = 400 > 256), priority 10 rejected
// (NDHWC dtypeSize * channels >= VRegSize/4), priority 12 accepted with big channels (no gather) key
TEST_F(AvgPool3DExtraTiling, AvgPool3D_OneKsizeReject_BigChannel_SmallKernelNDHWC)
{
    gert::StorageShape xShape = {{1, 10, 10, 10, 200}, {1, 10, 10, 10, 200}};
    gert::StorageShape yShape = {{1, 10, 10, 10, 200}, {1, 10, 10, 10, 200}};
    std::vector<int64_t> ksize = {1, 1, 1, 1, 1};
    std::vector<int64_t> strides = {1, 1, 1, 1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    // channels * dtypeSize = 400 >= 64 => big channels tiling key, divisor = 1 (no padding)
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, true, 0, "NDHWC", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 222220UL, 64);
}

// priority 10 NcdhwSmallKernel, NCDHW, no padding, kernel 3x3x3 stride 2, output count 10976 >= coreNum
TEST_F(AvgPool3DExtraTiling, AvgPool3D_NcdhwSmallKernel_NoPadding)
{
    gert::StorageShape xShape = {{2, 16, 16, 16, 16}, {2, 16, 16, 16, 16}};
    gert::StorageShape yShape = {{2, 16, 7, 7, 7}, {2, 16, 7, 7, 7}};
    std::vector<int64_t> ksize = {1, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    // 0 rejected: kernel is not all 1; 10 accepted, no padding and no sparse => 300001
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, true, 0, "NCDHW", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 300001UL, -1);
}

// priority 10 NcdhwSmallKernel with padding and count_include_pad=false => divisor 0 => real div key 300003
TEST_F(AvgPool3DExtraTiling, AvgPool3D_NcdhwSmallKernel_Padding_RealDiv)
{
    gert::StorageShape xShape = {{2, 16, 16, 16, 16}, {2, 16, 16, 16, 16}};
    gert::StorageShape yShape = {{2, 16, 7, 8, 8}, {2, 16, 7, 8, 8}};
    std::vector<int64_t> ksize = {1, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0, 1, 1, 1, 1};
    // padding + count_include_pad=false => divisor_ = 0 => AVG_REAL_DIV key, divisor ub buffer reserved
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, false, 0, "NCDHW", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 300003UL, -1);
}

// priority 11 NcdhwBigKernel direct path: output count 27 < 2*coreNum(128);
// priority 10 rejected because batches*outD*outH*outW = 27 < coreNum(64)
TEST_F(AvgPool3DExtraTiling, AvgPool3D_NcdhwBigKernel_SmallOutput)
{
    gert::StorageShape xShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}};
    std::vector<int64_t> ksize = {1, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    // totalIdx = 27 => blockFactor=0, coreNums=27
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, true, 0, "NCDHW", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 511110UL, 27);
}

// priority 13 BigKernelNDHWC direct path: outNum 27 < 2*coreNum(128);
// priority 10 rejected (outNum 27 < 64), priority 12 rejected (totalLoops 27*2 = 54 < coreNum)
TEST_F(AvgPool3DExtraTiling, AvgPool3D_BigKernelNDHWC_SmallOutput)
{
    gert::StorageShape xShape = {{1, 8, 8, 8, 2}, {1, 8, 8, 8, 2}};
    gert::StorageShape yShape = {{1, 3, 3, 3, 2}, {1, 3, 3, 3, 2}};
    std::vector<int64_t> ksize = {1, 3, 3, 3, 1};
    std::vector<int64_t> strides = {1, 2, 2, 2, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    // totalIdx = 27 => blockFactor=0, coreNums=27
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, true, 0, "NDHWC", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 411110UL, 27);
}

// priority 12 SmallKernelNDHWC with padding: priority 10 rejected (NDHWC + padding),
// totalLoops 2*4*5*5*2 = 400 >= coreNum, divisor = 27 => padding key 211110
TEST_F(AvgPool3DExtraTiling, AvgPool3D_SmallKernelNDHWC_Padding)
{
    gert::StorageShape xShape = {{2, 10, 10, 10, 2}, {2, 10, 10, 10, 2}};
    gert::StorageShape yShape = {{2, 4, 5, 5, 2}, {2, 4, 5, 5, 2}};
    std::vector<int64_t> ksize = {1, 3, 3, 3, 1};
    std::vector<int64_t> strides = {1, 2, 2, 2, 1};
    std::vector<int64_t> pads = {0, 0, 1, 1, 1, 1};
    // padding and count_include_pad=true with all windows inside pad => divisor = 27 => 211110
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, true, 0, "NDHWC", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 211110UL, 64);
}

// priority 12 SmallKernelNDHWC with padding and count_include_pad=false => divisor 0 => pad div key 211111
TEST_F(AvgPool3DExtraTiling, AvgPool3D_SmallKernelNDHWC_Padding_RealDiv)
{
    gert::StorageShape xShape = {{2, 10, 10, 10, 2}, {2, 10, 10, 10, 2}};
    gert::StorageShape yShape = {{2, 4, 5, 5, 2}, {2, 4, 5, 5, 2}};
    std::vector<int64_t> ksize = {1, 3, 3, 3, 1};
    std::vector<int64_t> strides = {1, 2, 2, 2, 1};
    std::vector<int64_t> pads = {0, 0, 1, 1, 1, 1};
    // padding + count_include_pad=false => divisor_ = 0 => PAD_DIV key, divisor ub buffer reserved
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, false, 0, "NDHWC", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 211111UL, 64);
}

// priority 19 Simt fallback: priority 10 rejected (padding && stride 6 >= 2 * effKsize 6),
// priority 11 rejected (kernel 3*3*3*2 = 54 < 256)
TEST_F(AvgPool3DExtraTiling, AvgPool3D_Simt_Fallback)
{
    gert::StorageShape xShape = {{2, 8, 20, 20, 20}, {2, 8, 20, 20, 20}};
    gert::StorageShape yShape = {{2, 8, 3, 4, 4}, {2, 8, 3, 4, 4}};
    std::vector<int64_t> ksize = {1, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 6, 6, 6};
    std::vector<int64_t> pads = {0, 0, 1, 1, 1, 1};
    // NCDHW, xSize = 128000 <= INT32_MAX => SIMT NCDHW INT32 key, block dim = coreNum
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, true, 0, "NCDHW", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 911100UL, 64);
}

// error branch: ksize must have 1, 3 or 5 elements
TEST_F(AvgPool3DExtraTiling, AvgPool3D_InvalidKsizeSize_Failed)
{
    gert::StorageShape xShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}};
    std::vector<int64_t> ksize = {1, 2};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, true, 0, "NCDHW", ge::DT_FLOAT16,
                         ge::GRAPH_FAILED, 0UL, -1);
}

// error branch: data_format only supports NCDHW/NDHWC
TEST_F(AvgPool3DExtraTiling, AvgPool3D_InvalidDataFormat_Failed)
{
    gert::StorageShape xShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}};
    std::vector<int64_t> ksize = {1, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, true, 0, "NCHW", ge::DT_FLOAT16, ge::GRAPH_FAILED,
                         0UL, -1);
}

// error branch: negative pad is not allowed
TEST_F(AvgPool3DExtraTiling, AvgPool3D_NegativePad_Failed)
{
    gert::StorageShape xShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 1, 3, 3, 3}, {1, 1, 3, 3, 3}};
    std::vector<int64_t> ksize = {1, 1, 3, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2, 2};
    std::vector<int64_t> pads = {-1, 0, 0, 0, 0, 0};
    ExecuteAvgPool3DCase(xShape, yShape, ksize, strides, pads, false, true, 0, "NCDHW", ge::DT_FLOAT16,
                         ge::GRAPH_FAILED, 0UL, -1);
}

// ===================== NcdhwSmallKernel: sparse + 大输入切分(Avg 侧) =====================

// stride(8)>=2*effK(3) 无 pad -> sparse 分支; 256^3 大输入触发切分与 avg 的 divisor 计算路径
TEST_F(AvgPool3DExtraTiling, AvgPool3D_NcdhwSmallKernel_Sparse_LargeInput)
{
    ExecuteAvgPool3DCase({{1, 16, 256, 256, 256}, {1, 16, 256, 256, 256}}, {{1, 16, 32, 32, 32}, {1, 16, 32, 32, 32}},
                         {1, 1, 3, 3, 3}, {1, 1, 8, 8, 8}, {0, 0, 0, 0, 0, 0}, false, true, 0, "NCDHW", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 0UL, -1);
}

// ===================== OneKsize: divisor_override 与大输出循环 =====================

// divisor_override=3 + count_include_pad=false -> CalcDivisor 的 override 分支
TEST_F(AvgPool3DExtraTiling, AvgPool3D_OneKsize_DivisorOverride)
{
    ExecuteAvgPool3DCase({{1, 3, 8, 8, 8}, {1, 3, 8, 8, 8}}, {{1, 3, 4, 4, 4}, {1, 3, 4, 4, 4}}, {1, 1, 1, 1, 1},
                         {1, 1, 2, 2, 2}, {0, 0, 0, 0, 0, 0}, false, false, 3, "NCDHW", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 0UL, -1);
}

// 恒等 OneKsize + 大输出(outD*outH*outW=64000>30576) -> DoUBTilingSingle 的多轮装载循环
TEST_F(AvgPool3DExtraTiling, AvgPool3D_OneKsize_LargeOutput_MultiLoop)
{
    ExecuteAvgPool3DCase({{1, 1, 40, 40, 40}, {1, 1, 40, 40, 40}}, {{1, 1, 40, 40, 40}, {1, 1, 40, 40, 40}},
                         {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, false, true, 0, "NCDHW", ge::DT_FLOAT16,
                         ge::GRAPH_SUCCESS, 0UL, -1);
}

// OneKsize + ceil_mode=true: 输出上取整推导分支
TEST_F(AvgPool3DExtraTiling, AvgPool3D_OneKsize_CeilMode)
{
    ExecuteAvgPool3DCase({{1, 3, 7, 7, 7}, {1, 3, 7, 7, 7}}, {{1, 3, 4, 4, 4}, {1, 3, 4, 4, 4}}, {1, 1, 1, 1, 1},
                         {1, 1, 2, 2, 2}, {0, 0, 0, 0, 0, 0}, true, true, 0, "NCDHW", ge::DT_FLOAT16, ge::GRAPH_SUCCESS,
                         0UL, -1);
}
