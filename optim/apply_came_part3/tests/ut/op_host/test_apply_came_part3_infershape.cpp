/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <utility>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {
constexpr size_t kInputCount = 7;
constexpr size_t kOutputCount = 4;

struct ShapeCase {
    gert::StorageShape u{{65, 67}, {65, 67}};
    gert::StorageShape m{{65, 67}, {65, 67}};
    gert::StorageShape scalar{{1}, {1}};
    gert::StorageShape globalShape{{2}, {2}};
    gert::StorageShape mOut{{}, {}};
    gert::StorageShape sumUR{{}, {}};
    gert::StorageShape sumUC{{}, {}};
    gert::StorageShape sumURC{{}, {}};
};

ge::graphStatus RunInferShape(ShapeCase& shapes)
{
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart3");
    if (impl == nullptr || impl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputCount, kOutputCount)
                      .IrInstanceNum({kInputCount, kOutputCount})
                      .InputShapes({&shapes.u, &shapes.m, &shapes.scalar, &shapes.scalar, &shapes.scalar,
                                    &shapes.scalar, &shapes.globalShape})
                      .OutputShapes({&shapes.mOut, &shapes.sumUR, &shapes.sumUC, &shapes.sumURC})
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    const auto status = impl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        std::array<gert::StorageShape*, kOutputCount> outputs = {&shapes.mOut, &shapes.sumUR, &shapes.sumUC,
                                                                 &shapes.sumURC};
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i]->MutableStorageShape() = *context->GetOutputShape(i);
            outputs[i]->MutableOriginShape() = *context->GetOutputShape(i);
        }
    }
    return status;
}

void RunInferDataType(ge::DataType mType)
{
    std::array<ge::DataType, kInputCount> inputs = {ge::DT_FLOAT, mType,        ge::DT_FLOAT, ge::DT_FLOAT,
                                                    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64};
    std::array<ge::DataType, kOutputCount> outputs = {ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                                                      ge::DT_UNDEFINED};
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(kInputCount)
                      .NodeIoNum(kInputCount, kOutputCount)
                      .IrInstanceNum({kInputCount, kOutputCount})
                      .InputDataTypes(
                          {&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5], &inputs[6]})
                      .OutputDataTypes({&outputs[0], &outputs[1], &outputs[2], &outputs[3]})
                      .Build();

    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart3");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), mType);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT);
}
} // namespace

TEST(ApplyCamePart3InferShape, ValidRankTwoInputs)
{
    ShapeCase shapes;
    ASSERT_EQ(RunInferShape(shapes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(shapes.mOut.GetStorageShape(), gert::Shape({65, 67}));
    EXPECT_EQ(shapes.sumUR.GetStorageShape(), gert::Shape({65}));
    EXPECT_EQ(shapes.sumUC.GetStorageShape(), gert::Shape({67}));
    EXPECT_EQ(shapes.sumURC.GetStorageShape(), gert::Shape({1}));
}

TEST(ApplyCamePart3InferShape, UnknownRankPropagatesToAllOutputs)
{
    ShapeCase shapes;
    shapes.u = gert::StorageShape({{-2}, {-2}});
    shapes.m = gert::StorageShape({{-2}, {-2}});
    ASSERT_EQ(RunInferShape(shapes), ge::GRAPH_SUCCESS);
    for (const auto* output : {&shapes.mOut, &shapes.sumUR, &shapes.sumUC, &shapes.sumURC}) {
        ASSERT_EQ(output->GetStorageShape().GetDimNum(), 1);
        EXPECT_EQ(output->GetStorageShape().GetDim(0), -2);
    }
}

TEST(ApplyCamePart3InferShape, RejectsMismatchedShapes)
{
    ShapeCase shapes;
    shapes.m = gert::StorageShape({{65, 66}, {65, 66}});
    EXPECT_EQ(RunInferShape(shapes), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart3InferShape, RejectsOneDimensionalInput)
{
    ShapeCase shapes;
    shapes.u = gert::StorageShape({{0}, {0}});
    shapes.m = gert::StorageShape({{0}, {0}});
    EXPECT_EQ(RunInferShape(shapes), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart3InferShape, RejectsZeroRowDimension)
{
    ShapeCase shapes;
    shapes.u = gert::StorageShape({{0, 67}, {0, 67}});
    shapes.m = gert::StorageShape({{0, 67}, {0, 67}});
    EXPECT_EQ(RunInferShape(shapes), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart3InferShape, RejectsZeroColumnDimension)
{
    ShapeCase shapes;
    shapes.u = gert::StorageShape({{65, 0}, {65, 0}});
    shapes.m = gert::StorageShape({{65, 0}, {65, 0}});
    EXPECT_EQ(RunInferShape(shapes), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart3InferShape, RejectsBothZeroDimensions)
{
    ShapeCase shapes;
    shapes.u = gert::StorageShape({{0, 0}, {0, 0}});
    shapes.m = gert::StorageShape({{0, 0}, {0, 0}});
    EXPECT_EQ(RunInferShape(shapes), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart3InferShape, ResolvesCompatibleDynamicDimensions)
{
    const std::array<std::pair<gert::StorageShape, gert::StorageShape>, 2> cases = {
        std::make_pair(gert::StorageShape({{-1, 67}, {-1, 67}}), gert::StorageShape({{65, -1}, {65, -1}})),
        std::make_pair(gert::StorageShape({{-1, -1}, {-1, -1}}), gert::StorageShape({{-1, -1}, {-1, -1}}))};
    const std::array<gert::Shape, 2> expected = {gert::Shape({65, 67}), gert::Shape({-1, -1})};
    for (size_t i = 0; i < cases.size(); ++i) {
        ShapeCase shapes;
        shapes.u = cases[i].first;
        shapes.m = cases[i].second;
        ASSERT_EQ(RunInferShape(shapes), ge::GRAPH_SUCCESS);
        EXPECT_EQ(shapes.mOut.GetStorageShape(), expected[i]);
        EXPECT_EQ(shapes.sumUR.GetStorageShape(), gert::Shape({expected[i].GetDim(0)}));
        EXPECT_EQ(shapes.sumUC.GetStorageShape(), gert::Shape({expected[i].GetDim(1)}));
        EXPECT_EQ(shapes.sumURC.GetStorageShape(), gert::Shape({1}));
    }
}

TEST(ApplyCamePart3InferShape, RejectsKnownConflictsInDynamicShapes)
{
    for (const auto& [uShape, mShape] : std::array<std::pair<gert::StorageShape, gert::StorageShape>, 3>{
             std::make_pair(gert::StorageShape({{-1, 67}, {-1, 67}}), gert::StorageShape({{65, 66}, {65, 66}})),
             std::make_pair(gert::StorageShape({{64, -1}, {64, -1}}), gert::StorageShape({{65, 67}, {65, 67}})),
             std::make_pair(gert::StorageShape({{-1, 67}, {-1, 67}}), gert::StorageShape({{0, 67}, {0, 67}}))}) {
        ShapeCase shapes;
        shapes.u = uShape;
        shapes.m = mShape;
        EXPECT_EQ(RunInferShape(shapes), ge::GRAPH_FAILED);
    }
}

TEST(ApplyCamePart3InferShape, ValidatesControlInputShape)
{
    const std::array<std::pair<gert::StorageShape, ge::graphStatus>, 9> cases = {
        std::make_pair(gert::StorageShape({{}, {}}), ge::GRAPH_SUCCESS),
        std::make_pair(gert::StorageShape({{1}, {1}}), ge::GRAPH_SUCCESS),
        std::make_pair(gert::StorageShape({{1, 1}, {1, 1}}), ge::GRAPH_SUCCESS),
        std::make_pair(gert::StorageShape({{2}, {2}}), ge::GRAPH_FAILED),
        std::make_pair(gert::StorageShape({{2147483648LL}, {2147483648LL}}), ge::GRAPH_FAILED),
        std::make_pair(gert::StorageShape({{4294967296LL}, {4294967296LL}}), ge::GRAPH_FAILED),
        std::make_pair(gert::StorageShape({{-1}, {-1}}), ge::GRAPH_SUCCESS),
        std::make_pair(gert::StorageShape({{-1, 2}, {-1, 2}}), ge::GRAPH_FAILED),
        std::make_pair(gert::StorageShape({{-2}, {-2}}), ge::GRAPH_SUCCESS)};
    for (const auto& [scalarShape, expected] : cases) {
        ShapeCase shapes;
        shapes.scalar = scalarShape;
        ASSERT_EQ(RunInferShape(shapes), expected);
        if (expected != ge::GRAPH_SUCCESS) {
            continue;
        }
        EXPECT_EQ(shapes.mOut.GetStorageShape(), gert::Shape({65, 67}));
        EXPECT_EQ(shapes.sumUR.GetStorageShape(), gert::Shape({65}));
        EXPECT_EQ(shapes.sumUC.GetStorageShape(), gert::Shape({67}));
        EXPECT_EQ(shapes.sumURC.GetStorageShape(), gert::Shape({1}));
    }
}

TEST(ApplyCamePart3InferShape, UnknownInputRankDoesNotHideInvalidControls)
{
    ShapeCase shapes;
    shapes.u = gert::StorageShape({{-2}, {-2}});
    shapes.m = gert::StorageShape({{-2}, {-2}});
    shapes.scalar = gert::StorageShape({{2}, {2}});
    EXPECT_EQ(RunInferShape(shapes), ge::GRAPH_FAILED);
}

TEST(ApplyCamePart3InferShape, ValidatesDynamicGlobalShape)
{
    for (const auto& [globalShape, expected] : std::array<std::pair<gert::StorageShape, ge::graphStatus>, 5>{
             std::make_pair(gert::StorageShape({{2}, {2}}), ge::GRAPH_SUCCESS),
             std::make_pair(gert::StorageShape({{0}, {0}}), ge::GRAPH_FAILED),
             std::make_pair(gert::StorageShape({{-1}, {-1}}), ge::GRAPH_SUCCESS),
             std::make_pair(gert::StorageShape({{-2}, {-2}}), ge::GRAPH_SUCCESS),
             std::make_pair(gert::StorageShape({{3}, {3}}), ge::GRAPH_FAILED)}) {
        ShapeCase shapes;
        shapes.globalShape = globalShape;
        EXPECT_EQ(RunInferShape(shapes), expected);
    }
}

TEST(ApplyCamePart3InferShape, InfersAllOutputDataTypes)
{
    RunInferDataType(ge::DT_FLOAT16);
    RunInferDataType(ge::DT_BF16);
    RunInferDataType(ge::DT_FLOAT);
}
