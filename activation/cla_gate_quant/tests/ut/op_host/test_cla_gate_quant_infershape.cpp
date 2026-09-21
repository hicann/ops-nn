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
 * \file test_cla_gate_quant_infershape.cpp
 * \brief ClaGateQuant infershape / inferdatatype UT
 *
 * Covers:
 *   - InferShapeForClaGateQuant: dual-axis / single-axis, D=128 / D=256,
 *     row/col scale ceil-div rounding, dynamic dimensions/rank, T=1 boundary,
 *     error paths (rank != 3, non-positive dim, missing attrs).
 *   - InferDataTypeForClaGateQuant: all four dst_type branches
 *     (35/36/40/41) plus the invalid dst_type error path.
 */

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;

namespace {
constexpr int64_t ROW_BLOCK = 64;
constexpr int64_t SCALE_LAST_DIM = 2;
const char* OP_TYPE = "ClaGateQuant";

// Attr order must match op_host/cla_gate_quant_def.cpp:
// dst_type(0), round_mode(1), scale_alg(2), input_attn_layout(3), dual_axis_flag(4)
std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeAttrs(int64_t dstType, const std::string& roundMode,
                                                                 int64_t scaleAlg, const std::string& layout,
                                                                 bool dualAxisFlag)
{
    return {{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dstType)},
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>(roundMode)},
            {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(scaleAlg)},
            {"input_attn_layout", Ops::NN::AnyValue::CreateFrom<std::string>(layout)},
            {"dual_axis_flag", Ops::NN::AnyValue::CreateFrom<bool>(dualAxisFlag)}};
}

std::vector<int64_t> DimsOf(const gert::Shape* shape)
{
    std::vector<int64_t> dims;
    if (shape == nullptr) {
        return dims;
    }
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        dims.push_back(shape->GetDim(i));
    }
    return dims;
}

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
} // namespace

class ClaGateQuantInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ClaGateQuantInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ClaGateQuantInferShapeTest TearDown" << std::endl; }
};

// --------------------------------------------------------------------------- //
// InferShape
// --------------------------------------------------------------------------- //
TEST_F(ClaGateQuantInferShapeTest, infershape_dual_axis_d128)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape globalAttn = {4, 16, 128};
    gert::Shape localAttn = {4, 16, 128};
    gert::Shape globalGate = {4, 16};
    gert::Shape localGate = {4, 16};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&globalAttn, &localAttn, &globalGate, &localGate})
                      .NodeAttrs(MakeAttrs(36, "rint", 1, "TND", true))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    const int64_t k = 16 * 128;
    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{4, k}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(1)), (std::vector<int64_t>{4, CeilDiv(k, ROW_BLOCK), SCALE_LAST_DIM}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(2)), (std::vector<int64_t>{4, k}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(3)), (std::vector<int64_t>{CeilDiv(4, ROW_BLOCK), k, SCALE_LAST_DIM}));
}

TEST_F(ClaGateQuantInferShapeTest, infershape_single_axis_d128_col_empty)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape globalAttn = {13, 11, 128};
    gert::Shape localAttn = {13, 11, 128};
    gert::Shape globalGate = {13, 11};
    gert::Shape localGate = {13, 11};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&globalAttn, &localAttn, &globalGate, &localGate})
                      .NodeAttrs(MakeAttrs(36, "rint", 1, "TND", false))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    const int64_t k = 11 * 128;
    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{13, k}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(1)), (std::vector<int64_t>{13, CeilDiv(k, ROW_BLOCK), SCALE_LAST_DIM}));
    // single-axis: col_data / col_scale are empty [0]
    EXPECT_EQ(DimsOf(context->GetOutputShape(2)), (std::vector<int64_t>{0}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(3)), (std::vector<int64_t>{0}));
}

TEST_F(ClaGateQuantInferShapeTest, infershape_dual_axis_d256)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape globalAttn = {2, 3, 256};
    gert::Shape localAttn = {2, 3, 256};
    gert::Shape globalGate = {2, 3};
    gert::Shape localGate = {2, 3};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&globalAttn, &localAttn, &globalGate, &localGate})
                      .NodeAttrs(MakeAttrs(40, "rint", 0, "TND", true))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    const int64_t k = 3 * 256;
    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{2, k}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(1)), (std::vector<int64_t>{2, CeilDiv(k, ROW_BLOCK), SCALE_LAST_DIM}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(2)), (std::vector<int64_t>{2, k}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(3)), (std::vector<int64_t>{CeilDiv(2, ROW_BLOCK), k, SCALE_LAST_DIM}));
}

// T=1 forces colScale dim0 == 1 (ceil-div rounding path), N=1 forces k == D.
TEST_F(ClaGateQuantInferShapeTest, infershape_t1_n1_boundary)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape globalAttn = {1, 1, 256};
    gert::Shape localAttn = {1, 1, 256};
    gert::Shape globalGate = {1, 1};
    gert::Shape localGate = {1, 1};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&globalAttn, &localAttn, &globalGate, &localGate})
                      .NodeAttrs(MakeAttrs(41, "floor", 0, "TND", true))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{1, 256}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(1)), (std::vector<int64_t>{1, 4, SCALE_LAST_DIM}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(2)), (std::vector<int64_t>{1, 256}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(3)), (std::vector<int64_t>{1, 256, SCALE_LAST_DIM}));
}

// Large T makes colScale dim0 > 1 and rowScale dim1 > 1.
TEST_F(ClaGateQuantInferShapeTest, infershape_large_shape_rounding)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape globalAttn = {129, 7, 128};
    gert::Shape localAttn = {129, 7, 128};
    gert::Shape globalGate = {129, 7};
    gert::Shape localGate = {129, 7};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&globalAttn, &localAttn, &globalGate, &localGate})
                      .NodeAttrs(MakeAttrs(35, "rint", 0, "TND", true))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    const int64_t k = 7 * 128;
    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{129, k}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(1)), (std::vector<int64_t>{129, CeilDiv(k, ROW_BLOCK), SCALE_LAST_DIM}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(2)), (std::vector<int64_t>{129, k}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(3)), (std::vector<int64_t>{CeilDiv(129, ROW_BLOCK), k, SCALE_LAST_DIM}));
}

TEST_F(ClaGateQuantInferShapeTest, infershape_dual_axis_unknown_dims)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape globalAttn = {-1, -1, -1};
    gert::Shape localAttn = {-1, -1, -1};
    gert::Shape globalGate = {-1, -1};
    gert::Shape localGate = {-1, -1};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&globalAttn, &localAttn, &globalGate, &localGate})
                      .NodeAttrs(MakeAttrs(35, "rint", 1, "TND", true))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{-1, -1}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(1)), (std::vector<int64_t>{-1, -1, SCALE_LAST_DIM}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(2)), (std::vector<int64_t>{-1, -1}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(3)), (std::vector<int64_t>{-1, -1, SCALE_LAST_DIM}));
}

TEST_F(ClaGateQuantInferShapeTest, infershape_single_axis_partially_unknown_dims)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape globalAttn = {-1, 16, 128};
    gert::Shape localAttn = {-1, 16, 128};
    gert::Shape globalGate = {-1, 16};
    gert::Shape localGate = {-1, 16};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&globalAttn, &localAttn, &globalGate, &localGate})
                      .NodeAttrs(MakeAttrs(36, "rint", 0, "TND", false))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    const int64_t k = 16 * 128;
    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{-1, k}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(1)), (std::vector<int64_t>{-1, CeilDiv(k, ROW_BLOCK), SCALE_LAST_DIM}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(2)), (std::vector<int64_t>{0}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(3)), (std::vector<int64_t>{0}));
}

TEST_F(ClaGateQuantInferShapeTest, infershape_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape globalAttn = {-2};
    gert::Shape localAttn = {-2};
    gert::Shape globalGate = {-2};
    gert::Shape localGate = {-2};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&globalAttn, &localAttn, &globalGate, &localGate})
                      .NodeAttrs(MakeAttrs(36, "rint", 0, "TND", true))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{-2}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(1)), (std::vector<int64_t>{-2}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(2)), (std::vector<int64_t>{-2}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(3)), (std::vector<int64_t>{-2}));
}

TEST_F(ClaGateQuantInferShapeTest, infershape_invalid_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape globalAttn = {4, 16}; // rank 2, invalid
    gert::Shape localAttn = {4, 16};
    gert::Shape globalGate = {4, 16};
    gert::Shape localGate = {4, 16};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&globalAttn, &localAttn, &globalGate, &localGate})
                      .NodeAttrs(MakeAttrs(36, "rint", 1, "TND", true))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(ClaGateQuantInferShapeTest, infershape_non_positive_dim)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape globalAttn = {4, 16, 0}; // D == 0, invalid
    gert::Shape localAttn = {4, 16, 0};
    gert::Shape globalGate = {4, 16};
    gert::Shape localGate = {4, 16};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&globalAttn, &localAttn, &globalGate, &localGate})
                      .NodeAttrs(MakeAttrs(36, "rint", 1, "TND", true))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

// No attrs at all: dualAxisFlagPtr is nullptr, so the infershape must fall back
// to the default single-axis branch and emit empty col outputs.
TEST_F(ClaGateQuantInferShapeTest, infershape_missing_attrs_defaults_single_axis)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape globalAttn = {4, 16, 128};
    gert::Shape localAttn = {4, 16, 128};
    gert::Shape globalGate = {4, 16};
    gert::Shape localGate = {4, 16};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&globalAttn, &localAttn, &globalGate, &localGate})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    const int64_t k = 16 * 128;
    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{4, k}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(1)), (std::vector<int64_t>{4, CeilDiv(k, ROW_BLOCK), SCALE_LAST_DIM}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(2)), (std::vector<int64_t>{0}));
    EXPECT_EQ(DimsOf(context->GetOutputShape(3)), (std::vector<int64_t>{0}));
}

// --------------------------------------------------------------------------- //
// InferDataType
// --------------------------------------------------------------------------- //
namespace {
ge::graphStatus RunInferDataType(int64_t dstType, std::vector<ge::DataType>& outDtypes)
{
    auto func = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_datatype;
    if (func == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType inRef0 = ge::DT_FLOAT16;
    ge::DataType inRef1 = ge::DT_FLOAT16;
    ge::DataType inRef2 = ge::DT_FLOAT16;
    ge::DataType inRef3 = ge::DT_FLOAT16;
    ge::DataType outRef0 = ge::DT_UNDEFINED;
    ge::DataType outRef1 = ge::DT_UNDEFINED;
    ge::DataType outRef2 = ge::DT_UNDEFINED;
    ge::DataType outRef3 = ge::DT_UNDEFINED;

    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType(OP_TYPE)
                      .IrInputNum(4)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(MakeAttrs(dstType, "rint", 1, "TND", true))
                      .InputDataTypes({&inRef0, &inRef1, &inRef2, &inRef3})
                      .OutputDataTypes({&outRef0, &outRef1, &outRef2, &outRef3})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto status = func(context);
    outDtypes.clear();
    if (status == ge::GRAPH_SUCCESS) {
        for (size_t i = 0; i < 4; ++i) {
            outDtypes.push_back(context->GetOutputDataType(i));
        }
    }
    return status;
}
} // namespace

TEST_F(ClaGateQuantInferShapeTest, inferdatatype_fp8_e5m2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE), nullptr);
    std::vector<ge::DataType> outDtypes;
    EXPECT_EQ(RunInferDataType(35, outDtypes), ge::GRAPH_SUCCESS);
    ASSERT_EQ(outDtypes.size(), 4U);
    EXPECT_EQ(outDtypes[0], ge::DT_FLOAT8_E5M2);
    EXPECT_EQ(outDtypes[1], ge::DT_FLOAT8_E8M0);
    EXPECT_EQ(outDtypes[2], ge::DT_FLOAT8_E5M2);
    EXPECT_EQ(outDtypes[3], ge::DT_FLOAT8_E8M0);
}

TEST_F(ClaGateQuantInferShapeTest, inferdatatype_fp8_e4m3fn)
{
    std::vector<ge::DataType> outDtypes;
    EXPECT_EQ(RunInferDataType(36, outDtypes), ge::GRAPH_SUCCESS);
    ASSERT_EQ(outDtypes.size(), 4U);
    EXPECT_EQ(outDtypes[0], ge::DT_FLOAT8_E4M3FN);
    EXPECT_EQ(outDtypes[1], ge::DT_FLOAT8_E8M0);
    EXPECT_EQ(outDtypes[2], ge::DT_FLOAT8_E4M3FN);
    EXPECT_EQ(outDtypes[3], ge::DT_FLOAT8_E8M0);
}

TEST_F(ClaGateQuantInferShapeTest, inferdatatype_fp4_e2m1)
{
    std::vector<ge::DataType> outDtypes;
    EXPECT_EQ(RunInferDataType(40, outDtypes), ge::GRAPH_SUCCESS);
    ASSERT_EQ(outDtypes.size(), 4U);
    EXPECT_EQ(outDtypes[0], ge::DT_FLOAT4_E2M1);
    EXPECT_EQ(outDtypes[1], ge::DT_FLOAT8_E8M0);
    EXPECT_EQ(outDtypes[2], ge::DT_FLOAT4_E2M1);
    EXPECT_EQ(outDtypes[3], ge::DT_FLOAT8_E8M0);
}

TEST_F(ClaGateQuantInferShapeTest, inferdatatype_fp4_e1m2)
{
    std::vector<ge::DataType> outDtypes;
    EXPECT_EQ(RunInferDataType(41, outDtypes), ge::GRAPH_SUCCESS);
    ASSERT_EQ(outDtypes.size(), 4U);
    EXPECT_EQ(outDtypes[0], ge::DT_FLOAT4_E1M2);
    EXPECT_EQ(outDtypes[1], ge::DT_FLOAT8_E8M0);
    EXPECT_EQ(outDtypes[2], ge::DT_FLOAT4_E1M2);
    EXPECT_EQ(outDtypes[3], ge::DT_FLOAT8_E8M0);
}

TEST_F(ClaGateQuantInferShapeTest, inferdatatype_invalid_dst_type)
{
    std::vector<ge::DataType> outDtypes;
    EXPECT_EQ(RunInferDataType(99, outDtypes), ge::GRAPH_FAILED);
}
