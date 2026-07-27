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
 * \file test_sync_batch_norm_backward_reduce_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "sync_batch_norm_backward_reduce_tiling.h"
#include "../../../op_kernel/sync_batch_norm_backward_reduce_tiling_data.h"
#include "../../../op_kernel/sync_batch_norm_backward_reduce_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class SyncBatchNormBackwardReduceTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SyncBatchNormBackwardReduceTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "SyncBatchNormBackwardReduceTiling TearDown" << endl; }
};

static std::vector<gert::TilingContextPara::TensorDescription> MakeInputs(ge::DataType dtype)
{
    gert::StorageShape s = {{256}, {256}};
    return std::vector<gert::TilingContextPara::TensorDescription>({
        {s, dtype, ge::FORMAT_ND},
        {s, dtype, ge::FORMAT_ND},
        {s, dtype, ge::FORMAT_ND},
        {s, dtype, ge::FORMAT_ND},
    });
}

static std::vector<gert::TilingContextPara::TensorDescription> MakeOutputs(ge::DataType dtype)
{
    gert::StorageShape s = {{256}, {256}};
    return std::vector<gert::TilingContextPara::TensorDescription>({
        {s, dtype, ge::FORMAT_ND},
        {s, dtype, ge::FORMAT_ND},
    });
}

TEST_F(SyncBatchNormBackwardReduceTiling, ascend910b_test_tiling_fp16)
{
    optiling::SyncBatchNormBackwardReduceCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT16);
    auto outputs = MakeOutputs(ge::DT_FLOAT16);
    gert::TilingContextPara tilingContextPara("SyncBatchNormBackwardReduce", inputs, outputs, &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(SYNCBNBR_TPL_SCH_MODE_0));
}

TEST_F(SyncBatchNormBackwardReduceTiling, ascend910b_test_tiling_fp32)
{
    optiling::SyncBatchNormBackwardReduceCompileInfo compileInfo;
    auto inputs = MakeInputs(ge::DT_FLOAT);
    auto outputs = MakeOutputs(ge::DT_FLOAT);
    gert::TilingContextPara tilingContextPara("SyncBatchNormBackwardReduce", inputs, outputs, &compileInfo);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(SYNCBNBR_TPL_SCH_MODE_1));
}
