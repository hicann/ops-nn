/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "../../../../op_kernel/arch35/psroi_pooling_grad_v2_d_tiling_data.h"
#include "../../../../op_kernel/arch35/psroi_pooling_grad_v2_d_tiling_key.h"

namespace {
constexpr char OP_NAME[] = "PSROIPoolingGradV2D";
constexpr uint64_t MAX_USER_WORKSPACE_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr int64_t UINT32_LIMIT = std::numeric_limits<uint32_t>::max();

struct TensorFormat {
    ge::Format origin = ge::FORMAT_ND;
    ge::Format storage = ge::FORMAT_ND;
};

struct TilingCase {
    int64_t batch = 1;
    int64_t rois = 1;
    int64_t outputDim = 1;
    int64_t group = 1;
    int64_t height = 2;
    int64_t width = 2;
    int64_t outputWidthDelta = 0;
    std::array<TensorFormat, 3> formats{}; // x, rois, y
    ge::DataType xType = ge::DT_FLOAT;
    ge::DataType roisType = ge::DT_FLOAT;
    ge::DataType outputType = ge::DT_FLOAT;
};

class PSROIPoolingGradV2DTiling : public testing::Test {
protected:
    void Run(const TilingCase& c, ge::graphStatus expectedStatus, uint32_t expectedMode = INDEX_MODE_32)
    {
        // Metadata only: neither tensor data nor the requested user workspace is allocated.
        const int64_t channels = c.outputDim * c.group * c.group;
        gert::StorageShape xShape{{c.batch * c.rois, c.outputDim, c.group, c.group},
                                  {c.batch * c.rois, c.outputDim, c.group, c.group}};
        gert::StorageShape roisShape{{c.batch, 5, c.rois}, {c.batch, 5, c.rois}};
        gert::StorageShape yShape{{c.batch, channels, c.height, c.width + c.outputWidthDelta},
                                  {c.batch, channels, c.height, c.width + c.outputWidthDelta}};
        fe::PlatFormInfos platform;
        ASSERT_TRUE(platform.Init());
        std::map<std::string, std::string> socInfo{
            {"ai_core_cnt", "64"}, {"vector_core_cnt", "64"}, {"core_type_list", "AICore"}};
        std::map<std::string, std::string> coreSpec{{"ub_size", "262144"}};
        std::map<std::string, std::string> version{{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
        platform.SetPlatformRes("SoCInfo", socInfo);
        platform.SetPlatformRes("AICoreSpec", coreSpec);
        platform.SetPlatformRes("version", version);
        platform.SetCoreNumByCoreType("AICore");
        platform_ascendc::PlatformAscendC ascendPlatform(&platform);
        systemWorkspaceBytes = ascendPlatform.GetLibApiWorkSpaceSize();

        struct CompileInfo {
        } compileInfo;
        auto tilingBuffer = gert::TilingData::CreateCap(sizeof(PSROIPoolingGradV2DTilingData));
        auto workspaceBuffer = gert::ContinuousVector::Create<size_t>(1);
        ASSERT_NE(tilingBuffer, nullptr);
        ASSERT_NE(workspaceBuffer, nullptr);
        auto holder = gert::TilingContextFaker()
                          .SetOpType(OP_NAME)
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({1, 1})
                          .InputShapes({&xShape, &roisShape})
                          .OutputShapes({&yShape})
                          .NodeInputTd(0, c.xType, c.formats[0].origin, c.formats[0].storage)
                          .NodeInputTd(1, c.roisType, c.formats[1].origin, c.formats[1].storage)
                          .NodeOutputTd(0, c.outputType, c.formats[2].origin, c.formats[2].storage)
                          .NodeAttrs({{"spatial_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0F)},
                                      {"output_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(c.outputDim)},
                                      {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(c.group)},
                                      {"input_size",
                                       Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({c.height, c.width})}})
                          .CompileInfo(&compileInfo)
                          .PlatformInfo(&platform)
                          .TilingData(tilingBuffer.get())
                          .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspaceBuffer.get()))
                          .Build();
        auto* context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(context, nullptr);
        const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_NAME);
        ASSERT_NE(impl, nullptr);
        ASSERT_NE(impl->tiling, nullptr);
        ASSERT_EQ(impl->tiling(context), expectedStatus);
        if (expectedStatus != ge::GRAPH_SUCCESS) {
            return;
        }
        EXPECT_EQ(context->GetTilingKey(), GET_TPL_TILING_KEY(expectedMode));
        data = *reinterpret_cast<const PSROIPoolingGradV2DTilingData*>(context->GetRawTilingData()->GetData());
        workspaceBytes = context->GetWorkspaceSizes(1)[0];
        EXPECT_EQ(data.batchSize, static_cast<uint64_t>(c.batch));
        EXPECT_EQ(data.roiCount, static_cast<uint64_t>(c.rois));
        EXPECT_EQ(data.totalOutputElements, static_cast<uint64_t>(c.batch) * channels * c.height * c.width);
        EXPECT_EQ(context->GetBlockDim(), data.needCoreNum == 0 ? 1U : data.needCoreNum);
        EXPECT_LE(data.needCoreNum, 64U);
        if (expectedMode == INDEX_MODE_32 || expectedMode == INDEX_MODE_64) {
            EXPECT_EQ(workspaceBytes, systemWorkspaceBytes);
            EXPECT_EQ(data.boundsOffset, 0U);
            EXPECT_EQ(data.contributionOffset, 0U);
        } else {
            EXPECT_GT(workspaceBytes, systemWorkspaceBytes);
            EXPECT_LE(workspaceBytes - systemWorkspaceBytes, MAX_USER_WORKSPACE_BYTES);
            EXPECT_EQ(data.boundsOffset % 128, 0U);
        }
    }

    PSROIPoolingGradV2DTilingData data{};
    uint64_t systemWorkspaceBytes = 0;
    uint64_t workspaceBytes = 0;
};

TEST_F(PSROIPoolingGradV2DTiling, Direct32And64OutputThreshold)
{
    TilingCase c;
    c.height = 1;
    c.width = UINT32_LIMIT;
    Run(c, ge::GRAPH_SUCCESS, INDEX_MODE_32);
    c.width = UINT32_LIMIT + 1;
    Run(c, ge::GRAPH_SUCCESS, INDEX_MODE_64);
}

TEST_F(PSROIPoolingGradV2DTiling, Precompute32And64)
{
    TilingCase c;
    c.rois = 16384;
    c.height = 1;
    c.width = 129;
    Run(c, ge::GRAPH_SUCCESS, PRECOMPUTE_MODE_32);
    EXPECT_GT(data.stableQOffset, 0U);
    EXPECT_EQ(data.contributionOffset, 0U);
    c.height = 65536;
    c.width = 65536;
    Run(c, ge::GRAPH_SUCCESS, PRECOMPUTE_MODE_64);
    EXPECT_GT(data.stableQOffset, 0U);
}

TEST_F(PSROIPoolingGradV2DTiling, PrecomputeRoiThresholds)
{
    TilingCase c;
    c.height = 1;
    c.width = 32;
    c.batch = 1023; // Plenty of cross-batch reuse, but Q is below the precompute minimum.
    Run(c, ge::GRAPH_SUCCESS, INDEX_MODE_32);
    c.batch = 1024; // Modeled work selects precompute at Q=1024.
    Run(c, ge::GRAPH_SUCCESS, PRECOMPUTE_MODE_32);
    c.batch = 1;
    c.width = 129;
    c.rois = 16383; // With B=1 the modeled-work rule does not select precompute.
    Run(c, ge::GRAPH_SUCCESS, INDEX_MODE_32);
    c.rois = 16384; // The high-boundary-reuse rule now selects it.
    Run(c, ge::GRAPH_SUCCESS, PRECOMPUTE_MODE_32);
}

TEST_F(PSROIPoolingGradV2DTiling, Materialize32AndWorkspaceFallbackToPrecompute)
{
    TilingCase c;
    c.rois = 16384;
    c.height = 1;
    c.width = 128;
    Run(c, ge::GRAPH_SUCCESS, MATERIALIZE_MODE_32);
    EXPECT_GT(data.contributionOffset, data.outputMetaOffset);
    EXPECT_GT(data.addMaskOffset, data.contributionOffset);
    c.rois = 1000000; // Y*Q contributions exceed 512 MiB; the smaller precompute workspace fits.
    Run(c, ge::GRAPH_SUCCESS, PRECOMPUTE_MODE_32);
    EXPECT_EQ(data.contributionOffset, 0U);
}

TEST_F(PSROIPoolingGradV2DTiling, PrecomputeWorkspaceFallbackToDirect)
{
    TilingCase c;
    c.rois = 23000000; // 24 bytes/ROI plus aligned batch metadata exceeds 512 MiB.
    c.height = 1;
    c.width = 129;
    Run(c, ge::GRAPH_SUCCESS, INDEX_MODE_32);
}

TEST_F(PSROIPoolingGradV2DTiling, Materialize64CandidateExceedsWorkspaceLimit)
{
    TilingCase c;
    c.height = 1;
    c.width = 1;
    c.rois = UINT32_LIMIT / 5;
    Run(c, ge::GRAPH_SUCCESS, INDEX_MODE_32);
    c.rois += 1; // rois elements cross UINT32_MAX; both workspace modes must fall back.
    Run(c, ge::GRAPH_SUCCESS, INDEX_MODE_64);
    // MATERIALIZE_MODE_64 cannot be selected under the 512 MiB cap: with Y<=128,
    // any 64-bit materialize index requires multi-GiB bounds/contribution/ROI storage.
    // This covers its host candidate/fallback, not execution of MATERIALIZE_MODE_64.
}

TEST_F(PSROIPoolingGradV2DTiling, EmptyBatchAndEmptyRois)
{
    TilingCase c;
    c.batch = 0;
    Run(c, ge::GRAPH_SUCCESS);
    EXPECT_EQ(data.needCoreNum, 0U);
    c.rois = 0;
    Run(c, ge::GRAPH_SUCCESS);
    EXPECT_EQ(data.needCoreNum, 0U);
    c.batch = 2;
    Run(c, ge::GRAPH_SUCCESS);
    EXPECT_EQ(data.totalOutputElements, 8U);
    EXPECT_GT(data.needCoreNum, 0U);
}

TEST_F(PSROIPoolingGradV2DTiling, AcceptsRegisteredStorageFormats)
{
    TilingCase c;
    Run(c, ge::GRAPH_SUCCESS);
    c.formats[0] = c.formats[2] = {ge::FORMAT_NCHW, ge::FORMAT_NCHW};
    Run(c, ge::GRAPH_SUCCESS);
    c.formats[0] = c.formats[2] = {ge::FORMAT_NCHW, ge::FORMAT_ND};
    Run(c, ge::GRAPH_SUCCESS);
    // Origin metadata does not change the storage layout checked by tiling.
    for (const size_t index : {0U, 2U}) {
        c = TilingCase{};
        c.formats[index].origin = ge::FORMAT_NCHW;
        Run(c, ge::GRAPH_SUCCESS);
    }
}

TEST_F(PSROIPoolingGradV2DTiling, RejectsMixedFeatureStorageFormatsInBothDirections)
{
    for (const size_t index : {0U, 2U}) {
        TilingCase c;
        c.formats[index].storage = ge::FORMAT_NCHW;
        Run(c, ge::GRAPH_FAILED);
    }
}

TEST_F(PSROIPoolingGradV2DTiling, RejectsUnsupportedStorageFormats)
{
    for (const ge::Format invalidFormat : {ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0}) {
        TilingCase c;
        c.formats[0].storage = c.formats[2].storage = invalidFormat;
        Run(c, ge::GRAPH_FAILED);
        for (const size_t index : {0U, 2U}) {
            c = TilingCase{};
            c.formats[index].storage = invalidFormat;
            Run(c, ge::GRAPH_FAILED);
        }
    }
    TilingCase c;
    c.formats[1].storage = ge::FORMAT_NCHW;
    Run(c, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingGradV2DTiling, RejectsNonFloatInputsAndOutput)
{
    TilingCase c;
    c.xType = ge::DT_FLOAT16;
    Run(c, ge::GRAPH_FAILED);
    c = TilingCase{};
    c.roisType = ge::DT_FLOAT16;
    Run(c, ge::GRAPH_FAILED);
    c = TilingCase{};
    c.outputType = ge::DT_FLOAT16;
    Run(c, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingGradV2DTiling, RejectsMismatchedOutputAndUnresolvedShape)
{
    TilingCase c;
    c.outputWidthDelta = 1;
    Run(c, ge::GRAPH_FAILED);
    c = TilingCase{};
    c.batch = -1;
    Run(c, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingGradV2DTiling, RejectsOutputElementCountOverflow)
{
    TilingCase c;
    c.rois = 0;
    c.height = std::numeric_limits<int64_t>::max();
    c.width = 3;
    Run(c, ge::GRAPH_FAILED);
}
} // namespace
