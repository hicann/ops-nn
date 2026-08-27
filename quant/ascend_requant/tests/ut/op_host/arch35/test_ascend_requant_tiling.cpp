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
#include <memory>
#include <vector>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {

constexpr char kCompileInfo[] = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                    "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                    "CORE_NUM": 24}
})";

struct AscendRequantCompileInfoForTest {};

void SetupPlatform(fe::PlatFormInfos& platformInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                   map<string, string>& intrinsics)
{
    GetPlatFormInfos(kCompileInfo, socInfos, aicoreSpec, intrinsics);
    platformInfo.Init();
}

void SetupKernelHolder(fe::PlatFormInfos& platformInfo, AscendRequantCompileInfoForTest& compileInfo,
                       const map<string, string>& socInfos, const map<string, string>& aicoreSpec,
                       const map<string, string>& intrinsics, gert::KernelRunContextHolder& kernelHolder)
{
    std::string compileInfoStr(kCompileInfo);
    kernelHolder = gert::KernelRunContextFaker()
                       .KernelIONum(2, 1)
                       .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                       .Outputs({&compileInfo})
                       .Build();
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "SoCInfo", const_cast<map<string, string>&>(socInfos));
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreSpec", const_cast<map<string, string>&>(aicoreSpec));
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", const_cast<map<string, string>&>(intrinsics));
}

} // namespace

class AscendRequantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AscendRequantTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AscendRequantTilingTest TearDown" << std::endl; }
};

TEST_F(AscendRequantTilingTest, tiling_rank1_scalar_broadcast)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{16}, {16}};
    gert::StorageShape sShape = {{1}, {1}};
    gert::StorageShape yShape = {{16}, {16}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_rank2_per_channel)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{4, 8}, {4, 8}};
    gert::StorageShape sShape = {{8}, {8}};
    gert::StorageShape yShape = {{4, 8}, {4, 8}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_rank4_per_channel)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3, 5, 5}, {2, 3, 5, 5}};
    gert::StorageShape sShape = {{1, 3, 1, 1}, {1, 3, 1, 1}};
    gert::StorageShape yShape = {{2, 3, 5, 5}, {2, 3, 5, 5}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_relu_true)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 4}, {2, 4}};
    gert::StorageShape sShape = {{4}, {4}};
    gert::StorageShape yShape = {{2, 4}, {2, 4}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = true;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_empty_tensor)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{0, 3}, {0, 3}};
    gert::StorageShape sShape = {{3}, {3}};
    gert::StorageShape yShape = {{0, 3}, {0, 3}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_rank5_scalar_broadcast)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape sShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_rank8_per_channel)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3, 4, 5, 6, 7, 8, 9}, {2, 3, 4, 5, 6, 7, 8, 9}};
    gert::StorageShape sShape = {{1, 1, 1, 1, 1, 1, 1, 9}, {1, 1, 1, 1, 1, 1, 1, 9}};
    gert::StorageShape yShape = {{2, 3, 4, 5, 6, 7, 8, 9}, {2, 3, 4, 5, 6, 7, 8, 9}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_non_aligned_tail)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{4, 7, 3, 5}, {4, 7, 3, 5}};
    gert::StorageShape sShape = {{1, 7, 1, 1}, {1, 7, 1, 1}};
    gert::StorageShape yShape = {{4, 7, 3, 5}, {4, 7, 3, 5}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_full_shape_no_broadcast)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{3, 4}, {3, 4}};
    gert::StorageShape sShape = {{3, 4}, {3, 4}};
    gert::StorageShape yShape = {{3, 4}, {3, 4}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_rank3_per_channel)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape sShape = {{4}, {4}};
    gert::StorageShape yShape = {{2, 3, 4}, {2, 3, 4}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_dtype_not_supported)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{4}, {4}};
    gert::StorageShape sShape = {{1}, {1}};
    gert::StorageShape yShape = {{4}, {4}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_NE(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_shape_mismatch)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape sShape = {{2, 4}, {2, 4}};
    gert::StorageShape yShape = {{2, 3}, {2, 3}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_NE(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}
