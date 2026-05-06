/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <iostream>

#include "../../../op_kernel/softplus_v2_grad_tiling_data.h"
#include "../../../op_kernel/softplus_v2_grad_tiling_key.h"
#include "softplus_v2_grad_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace optiling;

class SoftplusV2GradTiling : public testing::Test {
   protected:
    static void SetUpTestCase() { cout << "SoftplusV2GradTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "SoftplusV2GradTiling TearDown " << endl; }
};

TEST_F(SoftplusV2GradTiling, ascend9101_test_tiling_fp16_001) {
    // float beta;
    // float threshold;
    optiling::SoftplusV2GradCompileInfo compileInfo = {40, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "SoftplusV2Grad",
        {
            {{{2, 8}, {2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // input
            {{{2, 8}, {2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // input
        },
        {
            {{{2, 8}, {2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // output
        },
        {
            gert::TilingContextPara::OpAttr("beta", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
            gert::TilingContextPara::OpAttr("threshold", Ops::Math::AnyValue::CreateFrom<float>(20.0)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 512 1 1 5376 256 512 0 5376 5376 512 256 0 2 4728779609804374016 ";
    std::vector<size_t> expectWorkspaces = {512};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
