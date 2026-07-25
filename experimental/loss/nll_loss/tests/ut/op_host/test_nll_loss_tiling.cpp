/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_nll_loss_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "nll_loss_tiling.h"
#include "../../../op_kernel/nll_loss_tiling_data.h"
#include "../../../op_kernel/nll_loss_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class NllLossTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "NllLossTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "NllLossTiling TearDown" << endl; }
};

TEST_F(NllLossTiling, ascend910b_test_tiling_fp32_mean)
{
    optiling::NllLossCompileInfo compileInfo;
    gert::StorageShape xShape = {{4, 8}, {4, 8}};
    gert::StorageShape targetShape = {{4}, {4}};
    gert::StorageShape weightShape = {{8}, {8}};
    gert::StorageShape yShape = {{1}, {1}};
    gert::StorageShape twShape = {{1}, {1}};
    std::vector<gert::TilingContextPara::TensorDescription> inputs({
        {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {targetShape, ge::DT_INT32, ge::FORMAT_ND},
        {weightShape, ge::DT_FLOAT, ge::FORMAT_ND},
    });
    std::vector<gert::TilingContextPara::TensorDescription> outputs({
        {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {twShape, ge::DT_FLOAT, ge::FORMAT_ND},
    });
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    attrs.push_back(gert::TilingContextPara::OpAttr("reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")));
    attrs.push_back(gert::TilingContextPara::OpAttr("ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)));
    gert::TilingContextPara tilingContextPara("NllLoss", inputs, outputs, attrs, &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(NLLLOSS_TPL_SCH_MODE_1));
}

TEST_F(NllLossTiling, ascend910b_test_tiling_fp16_sum)
{
    optiling::NllLossCompileInfo compileInfo;
    gert::StorageShape xShape = {{16, 32}, {16, 32}};
    gert::StorageShape targetShape = {{16}, {16}};
    gert::StorageShape weightShape = {{32}, {32}};
    gert::StorageShape yShape = {{1}, {1}};
    gert::StorageShape twShape = {{1}, {1}};
    std::vector<gert::TilingContextPara::TensorDescription> inputs({
        {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        {targetShape, ge::DT_INT32, ge::FORMAT_ND},
        {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
    });
    std::vector<gert::TilingContextPara::TensorDescription> outputs({
        {yShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        {twShape, ge::DT_FLOAT16, ge::FORMAT_ND},
    });
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    attrs.push_back(gert::TilingContextPara::OpAttr("reduction", Ops::NN::AnyValue::CreateFrom<std::string>("sum")));
    attrs.push_back(gert::TilingContextPara::OpAttr("ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)));
    gert::TilingContextPara tilingContextPara("NllLoss", inputs, outputs, attrs, &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(NLLLOSS_TPL_SCH_MODE_0));
}
