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
#include "infershape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {
gert::InfershapeContextPara MakeInferShapeContext(const gert::StorageShape& xShape, int64_t dim)
{
    return gert::InfershapeContextPara("NanMedian",
                                       {
                                           {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                       },
                                       {
                                           {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       },
                                       {
                                           {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dim)},
                                       });
}
} // namespace

TEST(NanMedianInferShapeTest, KeepsReducedDimension)
{
    auto context = MakeInferShapeContext({{2, 3, 4}, {2, 3, 4}}, -2);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 1, 4}, {2, 1, 4}});
}

TEST(NanMedianInferShapeTest, RejectsOutOfRangeDimension)
{
    auto context = MakeInferShapeContext({{2, 3, 4}, {2, 3, 4}}, 3);
    ExecuteTestCase(context, ge::GRAPH_FAILED, {{}, {}});
}

TEST(NanMedianInferShapeTest, RejectsScalarInput)
{
    auto context = MakeInferShapeContext({{}, {}}, -1);
    ExecuteTestCase(context, ge::GRAPH_FAILED, {{}, {}});
}

TEST(NanMedianInferShapeTest, PreservesUnknownRank)
{
    auto context = MakeInferShapeContext({{-2}, {-2}}, -1);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-2}, {-2}});
}
