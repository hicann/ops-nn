/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/trans_quant_param_v2_proto.h"
#include "platform/platform_info.h"

class TransQuantParamV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TransQuantParamV2 test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TransQuantParamV2 test TearDown" << std::endl;
    }
};

TEST_F(TransQuantParamV2Test, trans_quant_paramV2_test_case_1)
{
    ge::op::TransQuantParamV2 op;
    ge::TensorDesc tensor_scale;
    ge::Shape shape({1});
    tensor_scale.SetDataType(ge::DT_FLOAT);
    tensor_scale.SetShape(shape);
    tensor_scale.SetOriginShape(shape);

    ge::TensorDesc tensor_offset;
    ge::Shape shape2({1024});
    tensor_offset.SetDataType(ge::DT_FLOAT);
    tensor_offset.SetShape(shape2);
    tensor_offset.SetOriginShape(shape2);

    op.UpdateInputDesc("scale", tensor_scale);
    op.UpdateInputDesc("offset", tensor_offset);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

TEST_F(TransQuantParamV2Test, trans_quant_paramV2_test_case_2)
{
    ge::op::TransQuantParamV2 op;
    ge::TensorDesc tensor_scale;
    ge::Shape shape({1, 64});
    tensor_scale.SetDataType(ge::DT_FLOAT);
    tensor_scale.SetShape(shape);
    tensor_scale.SetOriginShape(shape);

    op.UpdateInputDesc("scale", tensor_scale);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    std::vector<int64_t> expected_y_shape = {1, 64};

    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
}

TEST_F(TransQuantParamV2Test, trans_quant_paramV2_test_case_3)
{
    ge::op::TransQuantParamV2 op;
    ge::TensorDesc tensor_scale;
    ge::Shape shape({1});
    tensor_scale.SetDataType(ge::DT_FLOAT);
    tensor_scale.SetShape(shape);
    tensor_scale.SetOriginShape(shape);

    ge::TensorDesc tensor_offset;
    ge::Shape shape2({1, 1024});
    tensor_offset.SetDataType(ge::DT_FLOAT);
    tensor_offset.SetShape(shape2);
    tensor_offset.SetOriginShape(shape2);

    op.UpdateInputDesc("scale", tensor_scale);
    op.UpdateInputDesc("offset", tensor_offset);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

TEST_F(TransQuantParamV2Test, ascend910_95_test_offset_greater_than_scale_1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ge::op::TransQuantParamV2 op;
    ge::TensorDesc tensor_scale;
    ge::Shape shape({1});
    tensor_scale.SetDataType(ge::DT_FLOAT);
    tensor_scale.SetShape(shape);
    tensor_scale.SetOriginShape(shape);

    ge::TensorDesc tensor_offset;
    ge::Shape shape2({1, 1024});
    tensor_offset.SetDataType(ge::DT_FLOAT);
    tensor_offset.SetShape(shape2);
    tensor_offset.SetOriginShape(shape2);

    op.UpdateInputDesc("scale", tensor_scale);
    op.UpdateInputDesc("offset", tensor_offset);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

TEST_F(TransQuantParamV2Test, ascend910_95_test_offset_greater_than_scale_2)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ge::op::TransQuantParamV2 op;
    ge::TensorDesc tensor_scale;
    ge::Shape shape({1});
    tensor_scale.SetDataType(ge::DT_FLOAT);
    tensor_scale.SetShape(shape);
    tensor_scale.SetOriginShape(shape);

    ge::TensorDesc tensor_offset;
    ge::Shape shape2({64});
    tensor_offset.SetDataType(ge::DT_FLOAT);
    tensor_offset.SetShape(shape2);
    tensor_offset.SetOriginShape(shape2);

    op.UpdateInputDesc("scale", tensor_scale);
    op.UpdateInputDesc("offset", tensor_offset);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

TEST_F(TransQuantParamV2Test, ascend910_95_test_offset_less_than_scale_3)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ge::op::TransQuantParamV2 op;
    ge::TensorDesc tensor_scale;
    ge::Shape shape({64});
    tensor_scale.SetDataType(ge::DT_FLOAT);
    tensor_scale.SetShape(shape);
    tensor_scale.SetOriginShape(shape);

    ge::TensorDesc tensor_offset;
    ge::Shape shape2({1});
    tensor_offset.SetDataType(ge::DT_FLOAT);
    tensor_offset.SetShape(shape2);
    tensor_offset.SetOriginShape(shape2);

    op.UpdateInputDesc("scale", tensor_scale);
    op.UpdateInputDesc("offset", tensor_offset);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}