/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_UT_COMMON_H
#define SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_UT_COMMON_H

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_graph/swiglu_backward_group_quant_with_dual_axis_proto.h"
#include "../../../../op_host/arch35/swiglu_backward_group_quant_with_dual_axis_tiling.h"

using namespace ge;
using namespace ut_util;

struct SwigluTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    SwigluBackwardGroupQuantWithDualAxisMxTilingData data{};
};

struct SwigluTilingCase {
    std::vector<gert::StorageShape*> inputs;
    std::vector<gert::StorageShape*> outputs;
    std::vector<uint32_t> irInstances;
    std::vector<ge::DataType> inputTypes;
    std::vector<ge::DataType> outputTypes;
    float clampLimit = -1.0f;
    float alpha = 1.702f;
    float bias = 0.5f;
    int64_t quantMode = 1;
    int64_t dstType = 36;
};

class SwigluBackwardGroupQuantWithDualAxisTilingTest : public testing::Test {
protected:
    void SetUp() override
    {
        const std::string info = R"({"hardware_info":{
            "BT_SIZE":0,"load3d_constraints":"1",
            "Intrinsic_fix_pipe_l0c2out":false,
            "Intrinsic_data_move_l12ub":true,
            "Intrinsic_data_move_l0c2ub":true,
            "Intrinsic_data_move_out2l1_nd2nz":false,
            "UB_SIZE":262144,"L2_SIZE":33554432,"L1_SIZE":524288,
            "L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":131072,
            "CORE_NUM":40}})";
        GetPlatFormInfos(info.c_str(), socInfos_, aicoreSpec_, intrinsics_);
        platformInfo_.Init();
    }

    SwigluTilingResult Run(SwigluTilingCase& testCase)
    {
        auto tilingData = gert::TilingData::CreateCap(4096);
        auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
        auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
        gert::TilingContextFaker faker;
        faker.SetOpType("SwigluBackwardGroupQuantWithDualAxis")
            .NodeIoNum(testCase.inputs.size(), 5)
            .IrInstanceNum(testCase.irInstances)
            .InputShapes(testCase.inputs)
            .OutputShapes(testCase.outputs)
            .CompileInfo(&compileInfo_)
            .PlatformInfo(reinterpret_cast<char*>(&platformInfo_))
            .NodeAttrs({{"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(testCase.clampLimit)},
                        {"alpha", Ops::NN::AnyValue::CreateFrom<float>(testCase.alpha)},
                        {"bias", Ops::NN::AnyValue::CreateFrom<float>(testCase.bias)},
                        {"quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(testCase.quantMode)},
                        {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(testCase.dstType)}})
            .TilingData(tilingData.get())
            .Workspace(workspace);
        for (size_t i = 0; i < testCase.inputTypes.size(); ++i) {
            faker.NodeInputTd(i, testCase.inputTypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
        }
        for (size_t i = 0; i < testCase.outputTypes.size(); ++i) {
            faker.NodeOutputTd(i, testCase.outputTypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
        }
        auto holder = faker.Build();
        auto context = holder.GetContext<gert::TilingContext>();
        EXPECT_NE(context, nullptr);
        if (context == nullptr) {
            return {};
        }
        EXPECT_NE(context->GetPlatformInfo(), nullptr);
        context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos_);
        context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec_);
        context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
        context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics_);
        SwigluTilingResult result;
        result.status = optiling::TilingForSwigluBackwardGroupQuantWithDualAxisMx(context);
        if (result.status == ge::GRAPH_SUCCESS) {
            auto raw = context->GetRawTilingData();
            EXPECT_NE(raw, nullptr);
            EXPECT_GE(raw->GetDataSize(), sizeof(result.data));
            std::memcpy(&result.data, raw->GetData(), sizeof(result.data));
        }
        return result;
    }

private:
    std::map<std::string, std::string> socInfos_;
    std::map<std::string, std::string> aicoreSpec_;
    std::map<std::string, std::string> intrinsics_;
    fe::PlatFormInfos platformInfo_;
    uint8_t compileInfo_ = 0;
};

#endif
