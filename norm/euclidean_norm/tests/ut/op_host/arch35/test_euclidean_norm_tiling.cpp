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
#include <memory>
#include <vector>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
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
                    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                    "CORE_NUM": 24}
})";

// 构造 1-D int32 const tensor，shape=(N,)，内容 = axes
std::unique_ptr<uint8_t[]> MakeAxesConstTensor(const std::vector<int32_t>& axes)
{
    size_t totalSize = 0;
    auto holder = gert::Tensor::CreateFollowing(static_cast<int64_t>(axes.size()), ge::DT_INT32, totalSize);
    auto* t = reinterpret_cast<gert::Tensor*>(holder.get());
    t->MutableStorageShape().SetDimNum(1);
    t->MutableStorageShape().SetDim(0, static_cast<int64_t>(axes.size()));
    t->MutableOriginShape().SetDimNum(1);
    t->MutableOriginShape().SetDim(0, static_cast<int64_t>(axes.size()));
    auto* dataPtr = reinterpret_cast<int32_t*>(holder.get() + sizeof(gert::Tensor));
    for (size_t i = 0; i < axes.size(); ++i) {
        dataPtr[i] = axes[i];
    }
    return holder;
}

struct EuclideanNormCompileInfoForTest {};

// 共用 setup：构造 platform_info / kernel context faker
void SetupPlatform(fe::PlatFormInfos& platformInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                   map<string, string>& intrinsics)
{
    GetPlatFormInfos(kCompileInfo, socInfos, aicoreSpec, intrinsics);
    platformInfo.Init();
}

void SetupKernelHolder(fe::PlatFormInfos& platformInfo, EuclideanNormCompileInfoForTest& compileInfo,
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

class EuclideanNormTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "EuclideanNormTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "EuclideanNormTiling TearDown" << std::endl; }
};

// ────────────────────────────────────────────────────────────────────────────
// 用例 1：fp32 输入，shape=(4, 64)，axes=[1]，keep_dims=false → tiling success
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp32_2d_reduce_last)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{4, 64}, {4, 64}};
    gert::StorageShape yShape = {{4}, {4}};

    auto axesHolder = MakeAxesConstTensor({1});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    ASSERT_NE(param, nullptr);

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesTensor->MutableStorageShape()})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 2：fp16 输入，shape=(8, 1024)，axes=[1]，keep_dims=false
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp16_2d_reduce_last)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{8, 1024}, {8, 1024}};
    gert::StorageShape yShape = {{8}, {8}};

    auto axesHolder = MakeAxesConstTensor({1});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesTensor->MutableStorageShape()})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 3：int32 输入，shape=(16, 32)，axes=[0] → tail R 路径 (axisNum=2)
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_int32_2d_reduce_first)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    // reduce axis = 0：经合轴 / 补 leading A 后 pattern 仍是 AR（前置 A=1）
    gert::StorageShape xShape = {{16, 32}, {16, 32}};
    gert::StorageShape yShape = {{32}, {32}};

    auto axesHolder = MakeAxesConstTensor({0});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesTensor->MutableStorageShape()})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 5（v2 专属）：fp32 (4, 4, 8, 8) reduce axes=[1, 3] → ARAR pattern (tail R)
//   预期 climb：rSplit 从 LastR=3 climb 出去到 1，UB 内 R bundle 含 2 根 R 轴；
//   aSplit 可能也 climb。这是 v1 简化下做不到的多轴 bundle。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_v2_climb_arar_fp32)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    // shape (4, 4, 8, 8)，axes=[1, 3] → ARAR pattern：A=4, R=4, A=8, R=8 (4 axes，tail R)
    gert::StorageShape xShape = {{4, 4, 8, 8}, {4, 4, 8, 8}};
    gert::StorageShape yShape = {{4, 8}, {4, 8}};

    auto axesHolder = MakeAxesConstTensor({1, 3});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesTensor->MutableStorageShape()})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 6（v2 专属）：fp16 (2, 3, 4, 5, 6) reduce axes=[1, 3] → ARARA pattern (tail A)
//   预期 climb：aSplit 从 LastA=4 climb 出去；UB 内 A bundle 含多根 A 轴。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_v2_climb_arara_fp16)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape yShape = {{2, 4, 6}, {2, 4, 6}};

    auto axesHolder = MakeAxesConstTensor({1, 3});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesTensor->MutableStorageShape()})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 4：bf16 输入，shape=(2,4,8)，axes=[2] → ARA pattern (tail A)
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_bf16_3d_reduce_middle)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    // reduce axis = 1 → 合轴后变 A=2, R=4, A=8 = ARA tail A
    gert::StorageShape xShape = {{2, 4, 8}, {2, 4, 8}};
    gert::StorageShape yShape = {{2, 8}, {2, 8}};

    auto axesHolder = MakeAxesConstTensor({1});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesTensor->MutableStorageShape()})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 7：axes=[] (full reduce) — 对标 TF/PyTorch，归一化为 axes=[0..rank-1]
//   shape=(4, 64) fp32 axes=[] keep_dims=true → 等价 axes=[0,1] → 合轴后单 R 轴 → AR 路径
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp32_2d_empty_axes_full_reduce)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{4, 64}, {4, 64}};
    gert::StorageShape yShape = {{1, 1}, {1, 1}}; // full reduce + keep_dims=true → (1, 1)

    auto axesHolder = MakeAxesConstTensor({}); // 空 axes
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = true;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesTensor->MutableStorageShape()})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}
