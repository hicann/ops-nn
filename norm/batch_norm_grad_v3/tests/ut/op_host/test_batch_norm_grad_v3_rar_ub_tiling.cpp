/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include "norm/batch_norm_grad_v3/op_host/batch_norm_grad_v3_tiling.h"
#include "op_host/tiling_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "tests/ut/common/ut_op_util.h"

namespace {
class BatchNormGradV3RarUb : public testing::TestWithParam<std::string> {
protected:
    void Run(const std::vector<int64_t>& dimensions, ge::Format format, bool training, ge::graphStatus expectedStatus,
             uint64_t expectedKey = 0, ge::DataType dyDtype = ge::DT_FLOAT, int64_t expectedChannelsPerLoop = -1)
    {
        SCOPED_TRACE(GetParam());
        gert::Shape shape;
        for (auto dim : dimensions) {
            shape.AppendDim(dim);
        }
        const int64_t
            channels = dimensions[format == ge::FORMAT_NHWC || format == ge::FORMAT_NDHWC ? dimensions.size() - 1 : 1];
        gert::StorageShape tensorShape;
        tensorShape.MutableOriginShape() = shape;
        tensorShape.MutableStorageShape() = shape;
        gert::StorageShape channelShape = {{channels}, {channels}};
        std::string compileInfoJson = R"({"hardware_info":{"UB_SIZE":253952,"L2_SIZE":33554432,
            "L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":131072,"CORE_NUM":64,
            "vector_core_cnt":64,"socVersion":")" +
                                      GetParam() + R"("}})";
        std::map<std::string, std::string> socInfo, coreSpec, intrinsics, version;
        GetPlatFormInfos(compileInfoJson.c_str(), socInfo, coreSpec, intrinsics, version);
        fe::PlatFormInfos platform;
        ASSERT_TRUE(platform.Init());
        platform.SetPlatformRes("version", version);
        platform.SetPlatformRes("SoCInfo", socInfo);
        platform.SetPlatformRes("AICoreSpec", coreSpec);
        platform.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
        platform.SetCoreNumByCoreType("AICore");
        optiling::BatchNormGradV3CompileInfo compileInfo = {64, 253952, 32, 64};
        auto data = gert::TilingData::CreateCap(4096);
        auto workspace = gert::ContinuousVector::Create<size_t>(1);
        auto holder = gert::TilingContextFaker()
                          .SetOpType("BatchNormGradV3")
                          .NodeIoNum(7, 3)
                          .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                          .InputShapes({&tensorShape, &tensorShape, &channelShape, &channelShape, &channelShape,
                                        &channelShape, &channelShape})
                          .OutputShapes({&tensorShape, &channelShape, &channelShape})
                          .CompileInfo(&compileInfo)
                          .PlatformInfo(reinterpret_cast<char*>(&platform))
                          .NodeInputTd(0, dyDtype, format, format)
                          .NodeInputTd(1, dyDtype, format, format)
                          .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, dyDtype, format, format)
                          .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(training)},
                                      {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5)}})
                          .TilingData(data.get())
                          .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                          .Build();
        auto context = holder.GetContext<gert::TilingContext>();
        ASSERT_TRUE(Ops::NN::OpTiling::IsRegbaseSocVersion(context));
        auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNormGradV3");
        ASSERT_NE(impl, nullptr);
        EXPECT_EQ(impl->tiling(context), expectedStatus);
        if (expectedStatus == ge::GRAPH_SUCCESS) {
            if (expectedKey != 0) {
                EXPECT_EQ(context->GetTilingKey(), expectedKey);
            } else {
                // A valid reduction exceeding full-load capacity must still reach a split template.
                EXPECT_NE(context->GetTilingKey(), uint64_t{10000000});
            }
            if (context->GetTilingKey() == 10000000) {
                // Audit the serialized allocation against the Kernel InitBuffer layout, including alignment
                // and both binary-add regions; do not reuse the Host admission estimate as the oracle.
                std::array<int64_t, 12> fields{};
                ASSERT_GE(context->GetRawTilingData()->GetDataSize(), sizeof(fields));
                std::memcpy(fields.data(), context->GetRawTilingData()->GetData(), sizeof(fields));
                const int64_t rAlign = fields[3];
                const int64_t quotient = fields[8];
                const int64_t channelsPerLoop = fields[11];
                if (expectedChannelsPerLoop >= 0) {
                    EXPECT_EQ(channelsPerLoop, expectedChannelsPerLoop);
                }
                auto align = [](uint64_t bytes, uint64_t block) { return (bytes + block - 1) / block * block; };
                const uint64_t dyBytes = ge::GetSizeByDataType(dyDtype);
                const uint64_t mainQueues = align(rAlign * dyBytes, 32) * channelsPerLoop +
                                            align(rAlign * sizeof(float), 64) * channelsPerLoop;
                const uint64_t channelQueues = 5 * align(channelsPerLoop * sizeof(float), 32);
                const uint64_t binaryRegions = 2 * align(quotient / compileInfo.vlFp32 * sizeof(float), 32);
                uint64_t availableUb = 0;
                platform_ascendc::PlatformAscendC(&platform).GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                                                            availableUb);
                EXPECT_LE(2 * (mainQueues + channelQueues) + binaryRegions, availableUb);
            }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(TargetPlatforms, BatchNormGradV3RarUb,
                         testing::Values("Ascend950", "Ascend960DT", "Ascend960PR"));

TEST_P(BatchNormGradV3RarUb, FullLoadBinaryScratchAdmission)
{
    // Old admission accepted these shapes although the actual double-buffer allocation exceeded UB.
    Run({7893, 1, 2, 1}, ge::FORMAT_NCHW, true, ge::GRAPH_SUCCESS);
    Run({10481, 1, 2, 1}, ge::FORMAT_NCHW, true, ge::GRAPH_SUCCESS, 0, ge::DT_FLOAT16);
}

TEST_P(BatchNormGradV3RarUb, FullLoadBinaryScratchChannelsPerLoop)
{
    // FullLoad still fits, but the old per-loop channel factors (3/2) over-allocated UB.
    Run({2633, 192, 2, 1}, ge::FORMAT_NCHW, true, ge::GRAPH_SUCCESS, 10000000, ge::DT_FLOAT, 2);
    Run({5257, 128, 2, 1}, ge::FORMAT_NCHW, true, ge::GRAPH_SUCCESS, 10000000, ge::DT_FLOAT16, 1);
}

} // namespace
