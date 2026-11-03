/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../op_kernel/arch35/modulate_grad_struct.h"
#include "../../../op_kernel/arch35/modulate_grad_regbase_tiling_key.h"

using namespace ge;
using namespace std;
using namespace ut_util;
class ModulateGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ModulateGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ModulateGradTiling TearDown" << std::endl; }
};

TEST_F(ModulateGradTiling, test_modulate_grad_tiling_000)
{
    std::string opType("ModulateGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
    std::cout << "------test------1;" << std::endl;
    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // compile info
    struct ModulateGradCompileInfo {
        int32_t totalCoreNum = 48;
        uint64_t ubSizePlatForm = 0;
    } compileInfo;
    // tilingParseFunc simulate
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    std::cout << "------test------2;" << std::endl;
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    std::cout << "------test------3;" << std::endl;
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    std::cout << "------test------4;" << std::endl;
    ASSERT_NE(param, nullptr);
    std::cout << "------test------5;" << std::endl;
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape gradoutputShape = {{32, 8, 1024}, {32, 8, 1024}};
    gert::StorageShape inputShape = {{32, 8, 1024}, {32, 8, 1024}};
    gert::StorageShape scaleShape = {{32, 1024}, {32, 1024}};
    gert::StorageShape shiftShape = {{32, 1024}, {32, 1024}};
    gert::StorageShape gradinputShape = {{32, 8, 1024}, {32, 8, 1024}};
    gert::StorageShape gradscaleShape = {{32, 1024}, {32, 1024}};
    gert::StorageShape gradshiftShape = {{32, 1024}, {32, 1024}};
    std::cout << "------test------6;" << std::endl;
    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ModulateGrad")
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&gradoutputShape, &inputShape, &scaleShape, &shiftShape})
                      .OutputShapes({&gradinputShape, &gradscaleShape, &gradshiftShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    std::cout << "------test------7;" << std::endl;
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    std::cout << "------test------8;" << std::endl;
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

TEST_F(ModulateGradTiling, test_modulate_grad_tiling_zero_size_optional_is_absent_arch35)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ModulateGrad");
    ASSERT_NE(opImpl, nullptr);

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 72, "socVersion": "Ascend950"}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    struct ModulateGradCompileInfo {
        int32_t totalCoreNum = 72;
        uint64_t ubSizePlatForm = 0;
    } compileInfo;

    auto tilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(tilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape gradOutputShape = {{2, 0, 4}, {2, 0, 4}};
    gert::StorageShape inputShape = {{2, 0, 4}, {2, 0, 4}};
    gert::StorageShape scaleShape = {{2, 0}, {2, 0}};
    gert::StorageShape shiftShape = {{}, {}};
    gert::StorageShape gradInputShape = {{2, 0, 4}, {2, 0, 4}};
    gert::StorageShape gradScaleShape = {{2, 0}, {2, 0}};
    gert::StorageShape gradShiftShape = {{}, {}};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ModulateGrad")
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({&gradOutputShape, &inputShape, &scaleShape, &shiftShape})
                      .OutputShapes({&gradInputShape, &gradScaleShape, &gradShiftShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    auto tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    ASSERT_EQ(opImpl->tiling(tilingContext), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingContext->GetTilingKey(), GET_TPL_TILING_KEY(MODULATE_GRAD_TILING_D, false, false));
    ASSERT_GE(tilingContext->GetRawTilingData()->GetDataSize(), sizeof(ModulateGradRegbaseTilingData));
    const auto* result = reinterpret_cast<const ModulateGradRegbaseTilingData*>(
        tilingContext->GetRawTilingData()->GetData());
    EXPECT_EQ(result->has_scale, 0U);
    EXPECT_EQ(result->has_shift, 0U);
}

// TTK kernel 直连模式下可选输出未实例化时运行时输出按在场压缩：scale 缺席、shift 在场时
// 仅有 [grad_input, grad_shift] 两个输出（输出总数<3）。tiling 须按输出总数推导运行时索引
// 完成输出校验，不得误报 missing grad_shift（回归：118 例 INVALID_TILING）。
TEST_F(ModulateGradTiling, test_modulate_grad_tiling_scale_absent_shift_present_compressed)
{
    std::string opType("ModulateGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 72, "socVersion": "Ascend950"}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    struct ModulateGradCompileInfo {
        int32_t totalCoreNum = 72;
        uint64_t ubSizePlatForm = 0;
    } compileInfo;

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape gradoutputShape = {{128, 256, 512}, {128, 256, 512}};
    gert::StorageShape inputShape = {{128, 256, 512}, {128, 256, 512}};
    gert::StorageShape scaleShape = {{}, {}};
    gert::StorageShape shiftShape = {{128, 512}, {128, 512}};
    // 压缩后的运行时输出：仅 grad_input 与 grad_shift，grad_scale 未实例化。
    gert::StorageShape gradinputShape = {{128, 256, 512}, {128, 256, 512}};
    gert::StorageShape gradshiftShape = {{128, 512}, {128, 512}};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ModulateGrad")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 0, 1})
                      .InputShapes({&gradoutputShape, &inputShape, &scaleShape, &shiftShape})
                      .OutputShapes({&gradinputShape, &gradshiftShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

// 在场可选输入 dtype 与 input 不一致时必须拒绝（回归：L2 exc_dtype_008/011 complex128
// scale/shift 通过校验并执行）。仅必选输入做 dtype 校验时该缺陷不暴露。
TEST_F(ModulateGradTiling, test_modulate_grad_tiling_optional_dtype_mismatch_rejected)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ModulateGrad");
    ASSERT_NE(opImpl, nullptr);

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 72, "socVersion": "Ascend950"}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    struct ModulateGradCompileInfo {
        int32_t totalCoreNum = 72;
        uint64_t ubSizePlatForm = 0;
    } compileInfo;

    auto tilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(tilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape gradOutputShape = {{2, 16, 4}, {2, 16, 4}};
    gert::StorageShape inputShape = {{2, 16, 4}, {2, 16, 4}};
    gert::StorageShape scaleShape = {{2, 4}, {2, 4}};
    gert::StorageShape shiftShape = {{2, 4}, {2, 4}};
    gert::StorageShape gradInputShape = {{2, 16, 4}, {2, 16, 4}};
    gert::StorageShape gradScaleShape = {{2, 4}, {2, 4}};
    gert::StorageShape gradShiftShape = {{2, 4}, {2, 4}};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ModulateGrad")
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({&gradOutputShape, &inputShape, &scaleShape, &shiftShape})
                      .OutputShapes({&gradInputShape, &gradScaleShape, &gradShiftShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    auto tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    EXPECT_EQ(opImpl->tiling(tilingContext), ge::GRAPH_FAILED);
}

// 在场可选输入 format 非 ND 时必须拒绝（回归：L2 exc_format_012 FRACTAL_NZ shift 未被
// 拦截）。原实现仅校验必选输入 format。
TEST_F(ModulateGradTiling, test_modulate_grad_tiling_optional_format_rejected)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ModulateGrad");
    ASSERT_NE(opImpl, nullptr);

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 72, "socVersion": "Ascend950"}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    struct ModulateGradCompileInfo {
        int32_t totalCoreNum = 72;
        uint64_t ubSizePlatForm = 0;
    } compileInfo;

    auto tilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(tilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape gradOutputShape = {{2, 16, 4}, {2, 16, 4}};
    gert::StorageShape inputShape = {{2, 16, 4}, {2, 16, 4}};
    gert::StorageShape scaleShape = {{2, 4}, {2, 4}};
    gert::StorageShape shiftShape = {{2, 4}, {2, 4}};
    gert::StorageShape gradInputShape = {{2, 16, 4}, {2, 16, 4}};
    gert::StorageShape gradScaleShape = {{2, 4}, {2, 4}};
    gert::StorageShape gradShiftShape = {{2, 4}, {2, 4}};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ModulateGrad")
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({&gradOutputShape, &inputShape, &scaleShape, &shiftShape})
                      .OutputShapes({&gradInputShape, &gradScaleShape, &gradShiftShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    auto tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    EXPECT_EQ(opImpl->tiling(tilingContext), ge::GRAPH_FAILED);
}

// 零尺寸可选输入按“视为缺席”约定跳过 dtype 校验（回归：L2 exc_dtype_007 bool 空批
// scale）。缺席占位不携带有效 dtype，不得因占位 dtype 误拦正常执行。
TEST_F(ModulateGradTiling, test_modulate_grad_tiling_zero_size_optional_dtype_not_validated)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ModulateGrad");
    ASSERT_NE(opImpl, nullptr);

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 72, "socVersion": "Ascend950"}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    struct ModulateGradCompileInfo {
        int32_t totalCoreNum = 72;
        uint64_t ubSizePlatForm = 0;
    } compileInfo;

    auto tilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(tilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape gradOutputShape = {{2, 0, 4}, {2, 0, 4}};
    gert::StorageShape inputShape = {{2, 0, 4}, {2, 0, 4}};
    gert::StorageShape scaleShape = {{2, 0}, {2, 0}};
    gert::StorageShape shiftShape = {{}, {}};
    gert::StorageShape gradInputShape = {{2, 0, 4}, {2, 0, 4}};
    gert::StorageShape gradScaleShape = {{2, 0}, {2, 0}};
    gert::StorageShape gradShiftShape = {{}, {}};
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ModulateGrad")
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({&gradOutputShape, &inputShape, &scaleShape, &shiftShape})
                      .OutputShapes({&gradInputShape, &gradScaleShape, &gradShiftShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    auto tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    ASSERT_EQ(opImpl->tiling(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_GE(tilingContext->GetRawTilingData()->GetDataSize(), sizeof(ModulateGradRegbaseTilingData));
    const auto* result = reinterpret_cast<const ModulateGradRegbaseTilingData*>(
        tilingContext->GetRawTilingData()->GetData());
    EXPECT_EQ(result->has_scale, 0U);
    EXPECT_EQ(result->has_shift, 0U);
}

// TEST_F(ModulateGradTiling, test_modulate_grad_tiling_001) {
//     std::string opType("ModulateGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
//     auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
//     auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

//     string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                                                        "Intrinsic_fix_pipe_l0c2out": false,
//                                                        "Intrinsic_data_move_l12ub": true,
//                                                        "Intrinsic_data_move_l0c2ub": true,
//                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
//                                                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                                                        "CORE_NUM": 48}
//                                     })";
//     map<string, string> socInfos;
//     map<string, string> aicoreSpec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platformInfo;
//     platformInfo.Init();
//     // compile info
//     struct ModulateGradCompileInfo {
//       int32_t totalCoreNum = 30;
//       uint64_t ubSizePlatForm = 0;
//     } compileInfo;
//     // tilingParseFunc simulate
//     auto kernelHolder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
//             .Outputs({&compileInfo})
//             .Build();
//     ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
//     kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
//     kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//                                                                                             intrinsics);
//     ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
//     auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
//     gert::StorageShape gradoutputShape = {{32, 8, 5120}, {32, 8, 5120}};
//     gert::StorageShape inputShape = {{32, 8, 5120}, {32, 8, 5120}};
//     gert::StorageShape scaleShape = {{32, 5120}, {32, 5120}};
//     gert::StorageShape shiftShape = {{32, 5120}, {32, 5120}};
//     gert::StorageShape gradinputShape = {{32, 8, 5120}, {32, 8, 5120}};
//     gert::StorageShape gradscaleShape = {{32, 5120}, {32, 5120}};
//     gert::StorageShape gradshiftShape = {{32, 5120}, {32, 5120}};

//     // tilingParseFunc simulate
//     auto holder =
//         gert::TilingContextFaker()
//           .SetOpType("ModulateGrad")
//           .NodeIoNum(3, 1)
//           .IrInstanceNum({1, 1, 1})
//           .InputShapes({&gradoutputShape, &inputShape, &scaleShape, &shiftShape})
//           .OutputShapes({&gradinputShape, &gradscaleShape, &gradshiftShape})
//           .CompileInfo(&compileInfo)
//           .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
//           .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .TilingData(param.get())
//           .Workspace(wsSize)
//           .Build();

//     gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);

//     // workspaces nullptr return failed
//     EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
// }

// TEST_F(ModulateGradTiling, test_modulate_grad_tiling_002) {
//     std::string opType("ModulateGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
//     auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
//     auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

//     string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                                                        "Intrinsic_fix_pipe_l0c2out": false,
//                                                        "Intrinsic_data_move_l12ub": true,
//                                                        "Intrinsic_data_move_l0c2ub": true,
//                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
//                                                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                                                        "CORE_NUM": 48}
//                                     })";
//     map<string, string> socInfos;
//     map<string, string> aicoreSpec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

//     // platform info
//     fe::PlatFormInfos platformInfo;
//     platformInfo.Init();
//     // compile info
//     struct ModulateGradCompileInfo {
//       int32_t totalCoreNum = 30;
//       uint64_t ubSizePlatForm = 0;
//     } compileInfo;
//     // tilingParseFunc simulate
//     auto kernelHolder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
//             .Outputs({&compileInfo})
//             .Build();
//     ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
//     kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
//     kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//                                                                                             intrinsics);
//     ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
//     auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
//     gert::StorageShape gradoutputShape = {{8, 4, 20480}, {8, 4, 20480}};
//     gert::StorageShape inputShape = {{8, 4, 20480}, {8, 4, 20480}};
//     gert::StorageShape scaleShape = {{8, 20480}, {8, 20480}};
//     gert::StorageShape shiftShape = {{8, 20480}, {8, 20480}};
//     gert::StorageShape gradinputShape = {{8, 4, 20480}, {8, 4, 20480}};
//     gert::StorageShape gradscaleShape = {{8, 20480}, {8, 20480}};
//     gert::StorageShape gradshiftShape = {{8, 20480}, {8, 20480}};

//     // tilingParseFunc simulate
//     auto holder =
//         gert::TilingContextFaker()
//           .SetOpType("ModulateGrad")
//           .NodeIoNum(3, 1)
//           .IrInstanceNum({1, 1, 1})
//           .InputShapes({&gradoutputShape, &inputShape, &scaleShape, &shiftShape})
//           .OutputShapes({&gradinputShape,&gradscaleShape, &gradshiftShape})
//           .CompileInfo(&compileInfo)
//           .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
//           .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//           .TilingData(param.get())
//           .Workspace(wsSize)
//           .Build();

//     gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);

//     // workspaces nullptr return failed
//     EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
// }
