/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "../../../op_host/add_rms_norm_tiling.h"
namespace optiling::addRmsNormRegbase {
ge::graphStatus TilingAddRmsNormRegbase(gert::TilingContext* context);
}
namespace {
struct Resources {
    gert::StorageShape input;
    gert::StorageShape gamma;
    gert::StorageShape rstd;
    decltype(gert::TilingData::CreateCap(4096)) tiling = gert::TilingData::CreateCap(4096);
    decltype(gert::ContinuousVector::Create<size_t>(4096)) workspace = gert::ContinuousVector::Create<size_t>(4096);
    optiling::AddRmsNormCompileInfo compileInfo{};
    fe::PlatFormInfos platform{};

    Resources(uint32_t coreCount, bool initializePlatform) : input{{1, 64}, {1, 64}}, gamma{{64}, {64}}, rstd{{1}, {1}}
    {
        compileInfo.totalCoreNum = coreCount;
        compileInfo.totalUbSize = 196608ULL;
        if (initializePlatform) {
            platform.Init();
        }
    }

    gert::KernelRunContextHolder Context()
    {
        return gert::TilingContextFaker()
            .SetOpType("AddRmsNorm")
            .NodeIoNum(3, 3)
            .IrInstanceNum({1, 1, 1})
            .InputShapes({&input, &input, &gamma})
            .OutputShapes({&input, &rstd, &input})
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1.0e-6F)}})
            .CompileInfo(&compileInfo)
            .PlatformInfo(reinterpret_cast<char*>(&platform))
            .TilingData(tiling.get())
            .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
            .Build();
    }
};
} // namespace
TEST(AddRmsNormPlatformErrorReporting, MissingPlatformReportsOneFormalError)
{
    Resources resources(32U, true);
    auto holder = resources.Context();
    auto* context = holder.GetContext<gert::TilingContext>();
    auto* kernel = holder.GetContext<gert::KernelContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetComputeNodeInfo(), nullptr);
    const size_t slot = context->GetComputeNodeInfo()->GetInputsNum() + context->GetComputeNodeInfo()->GetOutputsNum() +
                        1U;
    ASSERT_NE(kernel->MutableInput(slot), nullptr);
    kernel->MutableInput(slot)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetPlatformInfo(), nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_EQ(optiling::addRmsNormRegbase::TilingAddRmsNormRegbase(context), ge::GRAPH_FAILED);
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Missing platform metadata for vector resources");
}
