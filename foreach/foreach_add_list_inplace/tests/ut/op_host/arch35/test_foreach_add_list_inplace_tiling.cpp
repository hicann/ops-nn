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
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"

// ForeachAddListInplace routes through the shared ForeachRegbaseTilingBinary template
// (foreach_utils/op_host/foreach_regbase_tiling.cpp) on arch35/Ascend950. These cases
// exercise the x1/x2 shape and dtype validation added to that template's GetShapeAttrsInfo.

class ForeachAddListInplaceTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachAddListInplaceTilingArch35 SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachAddListInplaceTilingArch35 TearDown" << std::endl; }
};

// Sink buffer for the TilingParse callback's optiling::ForeachCompileInfo output (9 x uint64_t).
// Its contents are unused on the tiling path: ForeachBaseClass::GetPlatformInfo() reads core/UB
// numbers from the (non-null) PlatformInfo below, so what matters is that the platform is wired
// with valid SoC/AICore info (mirrors the foreach_binary_op arch35 tiling-UT setup).
struct FakeForeachCompileInfo {
    uint64_t f[9] = {0};
};

static const char* kCompileInfoStr = R"({
    "hardware_info": {
        "BT_SIZE": 0,
        "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 262144,
        "L2_SIZE": 33554432,
        "L1_SIZE": 524288,
        "L0A_SIZE": 65536,
        "L0B_SIZE": 65536,
        "L0C_SIZE": 131072,
        "CORE_NUM": 64
    }
})";

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableStorageShape().AppendDim(d);
        s.MutableOriginShape().AppendDim(d);
    }
    return s;
}

// Builds a ForeachAddListInplace arch35 tiling context from arbitrary (possibly illegal)
// x1/x2 lists plus an alpha, runs tiling and returns its graphStatus. x1 also serves as the
// (in-place) output so CheckOutput on the success path stays satisfied.
static ge::graphStatus RunAddListInplaceTiling(const std::vector<std::vector<int64_t>>& x1Shapes,
                                               const std::vector<std::vector<int64_t>>& x2Shapes,
                                               const std::vector<ge::DataType>& x1Dtypes,
                                               const std::vector<ge::DataType>& x2Dtypes,
                                               const std::vector<int64_t>& alphaShape, ge::DataType alphaDtype)
{
    std::string op_type("ForeachAddListInplace");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (opImpl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = opImpl->tiling;
    auto tiling_parse_func = opImpl->tiling_parse;

    map<string, string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(kCompileInfoStr, soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    FakeForeachCompileInfo compile_info;

    std::string compile_info_string(kCompileInfoStr);
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    if (tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    size_t x1Num = x1Shapes.size();
    size_t x2Num = x2Shapes.size();
    // Flat input-shape storage (x1 tensors, then x2 tensors, then alpha). Reserve so pointers stay valid.
    std::vector<gert::StorageShape> inStore;
    inStore.reserve(x1Num + x2Num + 1);
    for (const auto& s : x1Shapes) {
        inStore.push_back(MakeStorageShape(s));
    }
    for (const auto& s : x2Shapes) {
        inStore.push_back(MakeStorageShape(s));
    }
    inStore.push_back(MakeStorageShape(alphaShape));
    std::vector<void*> inRefs;
    for (auto& ss : inStore) {
        inRefs.push_back(&ss);
    }

    // In-place: outputs mirror x1.
    std::vector<gert::StorageShape> outStore;
    outStore.reserve(x1Num);
    for (const auto& s : x1Shapes) {
        outStore.push_back(MakeStorageShape(s));
    }
    std::vector<void*> outRefs;
    for (auto& ss : outStore) {
        outRefs.push_back(&ss);
    }

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::TilingContextFaker faker;
    // ForeachAddListInplace tiling dispatches through TilingRegistry::DoTilingImpl, which selects the
    // registered template by context->GetNodeType(). Without SetOpType the faker node type defaults to
    // "fakeOp", no template matches, and tiling returns GRAPH_FAILED before any x1/x2 validation runs.
    faker.SetOpType(op_type)
        .NodeIoNum(3, 1)
        .IrInstanceNum({static_cast<uint32_t>(x1Num), static_cast<uint32_t>(x2Num), 1}, {static_cast<uint32_t>(x1Num)})
        .InputShapes(inRefs)
        .OutputShapes(outRefs)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info));
    // NodeInputTd is indexed by IR input (0=x1, 1=x2, 2=alpha); it sets the dtype for
    // every instance of that IR input, which suffices for these list-level cases.
    faker.NodeInputTd(0, x1Dtypes[0], ge::FORMAT_ND, ge::FORMAT_ND);
    faker.NodeInputTd(1, x2Dtypes[0], ge::FORMAT_ND, ge::FORMAT_ND);
    faker.NodeInputTd(2, alphaDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    faker.NodeOutputTd(0, x1Dtypes[0], ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = faker.TilingData(param.get()).Workspace(ws_size).Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    return tiling_func(tiling_context);
}

// Reachability + happy path: proves ForeachAddListInplace is registered in this UT binary
// (so the negative cases below fail for the right reason) and that equal x1/x2 shapes pass.
TEST_F(ForeachAddListInplaceTilingArch35, test_op_registered_and_positive_arch35)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachAddListInplace"), nullptr);
    auto ret = RunAddListInplaceTiling({{32, 4}}, {{32, 4}}, {ge::DT_FLOAT}, {ge::DT_FLOAT}, {1}, ge::DT_FLOAT);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// x1/x2 per-tensor shape not strictly equal -> the x2 shape check in the shared template fails.
TEST_F(ForeachAddListInplaceTilingArch35, test_neg_shape_mismatch_arch35)
{
    auto ret = RunAddListInplaceTiling({{32, 4}}, {{32, 8}}, {ge::DT_FLOAT}, {ge::DT_FLOAT}, {1}, ge::DT_FLOAT);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// x2 dtype != x1 dtype -> the x2 dtype check in the shared template fails.
TEST_F(ForeachAddListInplaceTilingArch35, test_neg_x2_dtype_mismatch_arch35)
{
    auto ret = RunAddListInplaceTiling({{32, 4}}, {{32, 4}}, {ge::DT_FLOAT}, {ge::DT_FLOAT16}, {1}, ge::DT_FLOAT);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// NOTE on the x1/x2 tensor-count mismatch check (x2 InstanceNum != x1 tensorNum): reproducing it
// needs a list with >=2 tensors (IrInstanceNum >= 2), but the tiling-context faker returns a nil
// PlatformInfo/CompileInfo for any IR input with instance num >= 2, so such a context cannot be
// driven through the tiling function at all. That validation is therefore not covered here; it is
// still enforced in the shared template's GetShapeAttrsInfo.
