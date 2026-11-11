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
 * \file test_max_pool_v2_infershape.cpp
 * \brief
 */

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "../../../op_graph/max_pool_v2_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

namespace {
class MaxPoolV2 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPoolV2 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaxPoolV2 TearDown" << std::endl; }
};

static std::string Shape2String(const gert::Shape& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static std::unique_ptr<uint8_t[]> MakeConstInt32Tensor(const std::vector<int32_t>& data, gert::Tensor** tensorOut)
{
    size_t totalSize = 0;
    auto holder = gert::Tensor::CreateFollowing(static_cast<int64_t>(data.size()), ge::DT_INT32, totalSize);
    auto* tensor = reinterpret_cast<gert::Tensor*>(holder.get());
    tensor->MutableStorageShape().AppendDim(static_cast<int64_t>(data.size()));
    tensor->MutableOriginShape().AppendDim(static_cast<int64_t>(data.size()));
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), data.data(),
                   data.size() * sizeof(int32_t));
    *tensorOut = tensor;
    return holder;
}
} // namespace

TEST_F(MaxPoolV2, MaxPoolV2_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 3, 8, 8}, {1, 3, 8, 8}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 1, 3, 3}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 1, 2, 2}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 3, 4, 4]");
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_test2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 3, 8, 8}, {1, 3, 8, 8}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 1, 3, 3}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 1, 2, 2}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 3, 3, 3]");
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_nhwc)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 8, 8, 3}, {1, 8, 8, 3}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 3, 3, 1}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 2, 2, 1}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 4, 4, 3]");
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_dynamic_dim)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 3, -1, 8}, {1, 3, -1, 8}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 1, 3, 3}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 1, 2, 2}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 3, -1, 3]");
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 1, 3, 3}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 1, 2, 2}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[-1, -1, -1, -1]");
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_fail_invalid_data_format)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 3, 8, 8}, {1, 3, 8, 8}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 1, 3, 3}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 1, 2, 2}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_fail_nhwc_ksize0_not_one)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 8, 8, 3}, {1, 8, 8, 3}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({2, 3, 3, 1}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 2, 2, 1}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_fail_nhwc_strides3_not_one)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 8, 8, 3}, {1, 8, 8, 3}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 3, 3, 1}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 2, 2, 2}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_fail_nchw_ksize1_not_one)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 3, 8, 8}, {1, 3, 8, 8}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 2, 3, 3}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 1, 2, 2}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_fail_nchw_strides1_not_one)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 3, 8, 8}, {1, 3, 8, 8}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 1, 3, 3}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 2, 2, 2}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_fail_invalid_padding)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 3, 8, 8}, {1, 3, 8, 8}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 1, 3, 3}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 1, 2, 2}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("EXPLICIT")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_fail_ksize_length)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 3, 8, 8}, {1, 3, 8, 8}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 1}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 1, 2, 2}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_fail_strides_length)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 3, 8, 8}, {1, 3, 8, 8}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 1, 3, 3}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 2, 2}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_nd_nhwc)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 8, 8, 3}, {1, 8, 8, 3}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 3, 3, 1}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 2, 2, 1}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 4, 4, 3]");
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_nd_nhwc_valid)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 8, 8, 3}, {1, 8, 8, 3}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 3, 3, 1}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 2, 2, 1}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 3, 3, 3]");
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_nchw_fmt_nhwc_attr)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 8, 8, 3}, {1, 8, 8, 3}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 3, 3, 1}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 2, 2, 1}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 4, 4, 3]");
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_default_data_format)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 3, 3, 1}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 2, 2, 1}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[2, 4, 4, 3]");
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_fail_ksize_hw_product_over_255)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 3, 8, 8}, {1, 3, 8, 8}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 1, 16, 16}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 1, 1, 1}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_fail_strides_over_63)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 8, 8, 3}, {1, 8, 8, 3}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 3, 3, 1}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 64, 2, 1}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_boundary_ksize_255_strides_63)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 64, 64, 3}, {1, 64, 64, 3}};
    gert::StorageShape yShape = {{}, {}};

    gert::Tensor* ksizeTensor = nullptr;
    auto ksizeHolder = MakeConstInt32Tensor({1, 15, 17, 1}, &ksizeTensor);
    gert::Tensor* stridesTensor = nullptr;
    auto stridesHolder = MakeConstInt32Tensor({1, 63, 1, 1}, &stridesTensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
                      .NodeAttrs({{"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 2, 64, 3]");
}
