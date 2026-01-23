/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
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

class inplace_index_add:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"inplace_index_add Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"inplace_index_add Proto Test TearDown"<<std::endl;
        }
};

TEST_F(inplace_index_add, inplace_index_add_infershape_diff_test) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_95";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_95"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceIndexAdd")->infer_shape;
    gert::Shape varShape = {-1, 8, 375};
    gert::Shape indicesShape = {-1};
    gert::Shape updatesShape = {-1};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&varShape, &indicesShape, &updatesShape})
                      .OutputShapes({&varShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {-1, 8, 375};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}
