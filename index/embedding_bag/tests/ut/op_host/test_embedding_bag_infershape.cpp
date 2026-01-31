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
#include <gtest/gtest.h>
#include "../../../op_graph/embedding_bag_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"


using namespace ge;
using namespace op;

class EmbeddingBagProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EmbeddingBagProtoTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EmbeddingBagProtoTest TearDown" << std::endl;
    }
};

TEST_F(EmbeddingBagProtoTest, infer_shape_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    
    ge::op::EmbeddingBag op;
    op.UpdateInputDesc("weight", create_desc({1024, 4096}, ge::DT_FLOAT));
    op.UpdateInputDesc("indices", create_desc({1024}, ge::DT_INT32));
    op.UpdateInputDesc("offsets", create_desc({1024}, ge::DT_INT32));
    op.SetAttr("mode", 1);
    op.SetAttr("scale_grad_by_freq", false);
    op.SetAttr("sparse", false);
    op.SetAttr("include_last_offset", false);
    op.SetAttr("padding_idx", -1);

    std::vector<int64_t> expectedValueShape = {1024, 4096};
    Runtime2TestParam param{{"mode", "scale_grad_by_freq", "sparse", "include_last_offset", "padding_idx"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputValue = op.GetOutputDescByName("y");
    EXPECT_EQ(outputValue.GetShape().GetDims(), expectedValueShape);
}

TEST_F(EmbeddingBagProtoTest, infer_dtype_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingBag"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingBag")->infer_datatype;

    if (dataTypeFunc != nullptr) {
        ge::DataType inputRef1 = ge::DT_FLOAT;
        ge::DataType inputRef2 = ge::DT_INT32;
        ge::DataType inputRef3 = ge::DT_INT32;
        ge::DataType outputRef1 = ge::DT_FLOAT;
        ge::DataType outputRef2 = ge::DT_INT32;
        ge::DataType outputRef3 = ge::DT_INT32;
        ge::DataType outputRef4 = ge::DT_INT32;

        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .NodeIoNum(3, 4)
                                 .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .InputDataTypes({&inputRef1, &inputRef2, &inputRef3})
                                 .OutputDataTypes({&outputRef1, &outputRef2, &outputRef3, &outputRef4})
                                 .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), outputRef1);
    }
}
