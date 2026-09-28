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
 * \file test_wts_arq_infershape.cpp
 * \brief WtsARQ InferShape / InferDataType UT.
 *
 * Coverage (spec.yaml shape_constraints / dtype_policy / error_codes):
 *   - normal: no-broadcast / restricted broadcast / scalar(rank0) / empty tensor
 *   - dynamic: unknown rank(-2) / unknown dim(-1) defer only the relations they
 *     participate in; rank>8, rank equality and w_min/w_max shape equality stay
 *     enforceable for the inputs whose ranks are known
 *   - error: rank > 8 / rank mismatch / w_min != w_max shape / illegal broadcast dim
 *   - error: w elements > 2^31 (SHAPE_SIZE_LIMIT), and the == 2^31 boundary passes
 *   - error: attr num_bits != 8 rejected by both InferShape and InferDataType
 *   - dtype: fp16/fp32 pass, unsupported dtype / w_min / w_max dtype mismatch fail
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "infer_datatype_context_faker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "../../../../../tests/ut/common/any_value.h"

namespace {

constexpr const char* kOpType = "WtsARQ";

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (const int64_t d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

ge::graphStatus RunInferShape(const std::vector<int64_t>& w, const std::vector<int64_t>& wMin,
                              const std::vector<int64_t>& wMax, ge::DataType dt, gert::Shape& yShape,
                              int64_t numBits = 8)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType)->infer_shape;
    gert::StorageShape wShape = MakeStorageShape(w);
    gert::StorageShape wMinShape = MakeStorageShape(wMin);
    gert::StorageShape wMaxShape = MakeStorageShape(wMax);
    gert::StorageShape yStorageShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&wShape, &wMinShape, &wMaxShape})
                      .OutputShapes({&yStorageShape})
                      .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Attr("num_bits", numBits)
                      .Build();
    auto ret = inferShapeFunc(holder.GetContext<gert::InferShapeContext>());
    if (ret == ge::GRAPH_SUCCESS) {
        yShape = *holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    }
    return ret;
}

ge::graphStatus RunInferDataType(ge::DataType wDt, ge::DataType wMinDt, ge::DataType wMaxDt, ge::DataType& yDt,
                                 int64_t numBits = 8)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType)->infer_datatype;
    ge::DataType yDtRef = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(3)
                      .NodeIoNum(3, 1)
                      .NodeInputTd(0, wDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, wMinDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, wMaxDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&wDt, &wMinDt, &wMaxDt})
                      .OutputDataTypes({&yDtRef})
                      .Attr("num_bits", numBits)
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    auto ret = inferDataTypeFunc(context);
    if (ret == ge::GRAPH_SUCCESS) {
        yDt = context->GetOutputDataType(0);
    }
    return ret;
}

} // namespace

class WtsArqInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "WtsArqInferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "WtsArqInferShapeTest TearDown" << std::endl; }
};

TEST_F(WtsArqInferShapeTest, infershape_no_broadcast_success)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT16, y), ge::GRAPH_SUCCESS);
    gert::Shape expected = {2, 3};
    ASSERT_EQ(Ops::Base::ToString(y), Ops::Base::ToString(expected));
}

TEST_F(WtsArqInferShapeTest, infershape_restricted_broadcast_success)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({2, 3}, {2, 1}, {2, 1}, ge::DT_FLOAT, y), ge::GRAPH_SUCCESS);
    gert::Shape expected = {2, 3};
    ASSERT_EQ(Ops::Base::ToString(y), Ops::Base::ToString(expected));
}

TEST_F(WtsArqInferShapeTest, infershape_scalar_rank0_success)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({}, {}, {}, ge::DT_FLOAT, y), ge::GRAPH_SUCCESS);
    gert::Shape expected = {};
    ASSERT_EQ(Ops::Base::ToString(y), Ops::Base::ToString(expected));
}

TEST_F(WtsArqInferShapeTest, infershape_empty_tensor_success)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({0, 3}, {1, 3}, {1, 3}, ge::DT_FLOAT, y), ge::GRAPH_SUCCESS);
    gert::Shape expected = {0, 3};
    ASSERT_EQ(Ops::Base::ToString(y), Ops::Base::ToString(expected));
}

TEST_F(WtsArqInferShapeTest, infershape_unknown_rank_success)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({-2}, {-2}, {-2}, ge::DT_FLOAT, y), ge::GRAPH_SUCCESS);
    gert::Shape expected = {-2};
    ASSERT_EQ(Ops::Base::ToString(y), Ops::Base::ToString(expected));
}

TEST_F(WtsArqInferShapeTest, infershape_unknown_dim_success)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({-1, 3}, {1, 3}, {1, 3}, ge::DT_FLOAT, y), ge::GRAPH_SUCCESS);
    gert::Shape expected = {-1, 3};
    ASSERT_EQ(Ops::Base::ToString(y), Ops::Base::ToString(expected));
}

TEST_F(WtsArqInferShapeTest, infershape_rank_over_limit_failed)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2},
                            ge::DT_FLOAT, y),
              ge::GRAPH_FAILED);
}

TEST_F(WtsArqInferShapeTest, infershape_rank_mismatch_failed)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({2, 3}, {3}, {3}, ge::DT_FLOAT, y), ge::GRAPH_FAILED);
}

TEST_F(WtsArqInferShapeTest, infershape_wmin_wmax_shape_mismatch_failed)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({2, 3}, {2, 1}, {1, 3}, ge::DT_FLOAT, y), ge::GRAPH_FAILED);
}

TEST_F(WtsArqInferShapeTest, infershape_illegal_broadcast_dim_failed)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({2, 3}, {2, 2}, {2, 2}, ge::DT_FLOAT, y), ge::GRAPH_FAILED);
}

// A dynamic dim only defers the axes it participates in: dim1 is fully known and
// illegal (w=3 vs w_min=w_max=2) even though dim0 is unknown.
TEST_F(WtsArqInferShapeTest, infershape_dynamic_illegal_broadcast_failed)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({-1, 3}, {2, 2}, {2, 2}, ge::DT_FLOAT, y), ge::GRAPH_FAILED);
}

// Same for the w_min == w_max relation when the deciding axis is known.
TEST_F(WtsArqInferShapeTest, infershape_dynamic_wmin_wmax_mismatch_failed)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({-1, 3}, {2, 1}, {1, 3}, ge::DT_FLOAT, y), ge::GRAPH_FAILED);
}

// Known dims that are legal must still pass while another axis stays dynamic.
TEST_F(WtsArqInferShapeTest, infershape_dynamic_legal_known_dims_success)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({-1, 3}, {-1, 1}, {2, 1}, ge::DT_FLOAT, y), ge::GRAPH_SUCCESS);
    gert::Shape expected = {-1, 3};
    ASSERT_EQ(Ops::Base::ToString(y), Ops::Base::ToString(expected));
}

// Unknown rank on a sibling must not excuse a decidable rank>8 violation of w.
TEST_F(WtsArqInferShapeTest, infershape_rank_over_limit_with_unknownrank_sibling_failed)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape(std::vector<int64_t>(9, 2), {-2}, {-2}, ge::DT_FLOAT, y), ge::GRAPH_FAILED);
}

// Unknown rank on w must not excuse a decidable w_min != w_max shape equality violation.
TEST_F(WtsArqInferShapeTest, infershape_unknownrank_w_wmin_wmax_mismatch_failed)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({-2}, {2, 2}, {3, 3}, ge::DT_FLOAT, y), ge::GRAPH_FAILED);
}

// Known-rank siblings of an unknown-rank w must still have equal ranks.
TEST_F(WtsArqInferShapeTest, infershape_unknownrank_w_known_rank_mismatch_failed)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({-2}, {2, 2}, {2, 2, 2}, ge::DT_FLOAT, y), ge::GRAPH_FAILED);
}

TEST_F(WtsArqInferShapeTest, infershape_shape_size_over_limit_failed)
{
    gert::Shape y;
    // 65536 * 32769 = 2^31 + 65536 > 2^31
    ASSERT_EQ(RunInferShape({65536, 32769}, {1, 32769}, {1, 32769}, ge::DT_FLOAT, y), ge::GRAPH_FAILED);
}

TEST_F(WtsArqInferShapeTest, infershape_shape_size_at_limit_success)
{
    gert::Shape y;
    // 65536 * 32768 = 2^31, boundary is allowed ("smaller than or equal to 2^31")
    ASSERT_EQ(RunInferShape({65536, 32768}, {1, 32768}, {1, 32768}, ge::DT_FLOAT, y), ge::GRAPH_SUCCESS);
    gert::Shape expected = {65536, 32768};
    ASSERT_EQ(Ops::Base::ToString(y), Ops::Base::ToString(expected));
}

TEST_F(WtsArqInferShapeTest, infershape_num_bits_not_8_failed)
{
    gert::Shape y;
    ASSERT_EQ(RunInferShape({2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT, y, 7), ge::GRAPH_FAILED);
}

TEST_F(WtsArqInferShapeTest, inferdatatype_fp16_success)
{
    ge::DataType yDt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, yDt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(yDt, ge::DT_FLOAT16);
}

TEST_F(WtsArqInferShapeTest, inferdatatype_fp32_success)
{
    ge::DataType yDt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, yDt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(yDt, ge::DT_FLOAT);
}

TEST_F(WtsArqInferShapeTest, inferdatatype_unsupported_dtype_failed)
{
    ge::DataType yDt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, yDt), ge::GRAPH_FAILED);
}

TEST_F(WtsArqInferShapeTest, inferdatatype_wmin_dtype_mismatch_failed)
{
    ge::DataType yDt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, yDt), ge::GRAPH_FAILED);
}

TEST_F(WtsArqInferShapeTest, inferdatatype_wmax_dtype_mismatch_failed)
{
    ge::DataType yDt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, yDt), ge::GRAPH_FAILED);
}

TEST_F(WtsArqInferShapeTest, inferdatatype_num_bits_not_8_failed)
{
    ge::DataType yDt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, yDt, 7), ge::GRAPH_FAILED);
}
