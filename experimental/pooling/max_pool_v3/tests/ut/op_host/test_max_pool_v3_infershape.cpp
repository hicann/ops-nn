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
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "platform/platform_info.h"

#include "../../../op_graph/max_pool_v3_proto.h"

class MaxPoolV3 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPoolV3 Proto Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MaxPoolV3 Proto Test TearDown" << std::endl; }

    void SetUpPlatform()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_95";
        optiCompilationInfo.soc_version = "Ascend910_95";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_95"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

// case_001: 基础 max pool, kernel=2x2, stride=2x2, 无padding
// [1,1,4,4] → [1,1,2,2]
TEST_F(MaxPoolV3, case_001_basic)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->infer_shape;
    gert::Shape xShape = {1, 1, 4, 4};
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({
                          {"ksize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 2, 2})},
                          {"strides", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 2, 2})},
                          {"pads", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0, 0, 0})},
                          {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                      })
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// case_002: kernel=3x3, stride=1x1
// [1,2,6,6] → [1,2,4,4]
TEST_F(MaxPoolV3, case_002_kernel3_stride1)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->infer_shape;
    gert::Shape xShape = {1, 2, 6, 6};
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({
                          {"ksize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 3, 3})},
                          {"strides", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0, 0, 0})},
                          {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                      })
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// case_003: 带padding
// [1,1,4,4], kernel=2x2, stride=2x2, pad[0,1,0,1] → [1,1,3,3]
TEST_F(MaxPoolV3, case_003_with_padding)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->infer_shape;
    gert::Shape xShape = {1, 1, 4, 4};
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({
                          {"ksize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 2, 2})},
                          {"strides", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 2, 2})},
                          {"pads", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 1, 0, 1})},
                          {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                      })
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// case_004: ceil_mode=true
// [1,1,5,5], kernel=2x2, stride=2x2 → [1,1,3,3]
TEST_F(MaxPoolV3, case_004_ceil_mode)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->infer_shape;
    gert::Shape xShape = {1, 1, 5, 5};
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({
                          {"ksize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 2, 2})},
                          {"strides", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 2, 2})},
                          {"pads", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0, 0, 0})},
                          {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                      })
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// case_005: 大批量 + 多通道
// [4,16,32,32], kernel=2x2, stride=2x2 → [4,16,16,16]
TEST_F(MaxPoolV3, case_005_large_batch)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->infer_shape;
    gert::Shape xShape = {4, 16, 32, 32};
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({
                          {"ksize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 2, 2})},
                          {"strides", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 2, 2})},
                          {"pads", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0, 0, 0})},
                          {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                      })
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// case_006: 1x1 kernel (恒等)
// [1,3,7,7], kernel=1x1, stride=1x1 → [1,3,7,7]
TEST_F(MaxPoolV3, case_006_1x1_kernel)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->infer_shape;
    gert::Shape xShape = {1, 3, 7, 7};
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({
                          {"ksize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 1, 1})},
                          {"strides", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0, 0, 0})},
                          {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                      })
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// case_007: stride 大于 kernel
// [1,1,6,6], kernel=2x2, stride=3x3 → [1,1,2,2]
TEST_F(MaxPoolV3, case_007_stride_larger_than_kernel)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->infer_shape;
    gert::Shape xShape = {1, 1, 6, 6};
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({
                          {"ksize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 2, 2})},
                          {"strides", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 3, 3})},
                          {"pads", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0, 0, 0})},
                          {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                      })
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// case_008: BF16 dtype
// [1,1,4,4], kernel=2x2, stride=2x2 → [1,1,2,2]
TEST_F(MaxPoolV3, case_008_bfloat16)
{
    SetUpPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->infer_shape;
    gert::Shape xShape = {1, 1, 4, 4};
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({
                          {"ksize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 2, 2})},
                          {"strides", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1, 2, 2})},
                          {"pads", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0, 0, 0})},
                          {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                      })
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}
