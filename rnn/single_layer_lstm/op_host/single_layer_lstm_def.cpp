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
 * \file single_layer_lstm_def.cpp
 * \brief Single-layer forward with homogeneous floating-point inputs and outputs.
 */

#include "register/op_def_registry.h"

namespace ops {

class SingleLayerLstm : public OpDef {
public:
    explicit SingleLayerLstm(const char* name) : OpDef(name)
    {
        /* Each dtype-list position defines one complete input/output combination. */
        auto input = [this](const char* n) -> OpParamDef& {
            return this->Input(n)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        };
        auto output = [this](const char* n) -> OpParamDef& {
            return this->Output(n)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        };

        input("x"); // [T, B, I]
        input("w"); // [I+H, 4H]  fused, input rows first, hidden rows already transposed

        input("b"); // [4H], fused bias when bias_hh is absent; otherwise input bias

        input("init_h"); // [B, H]
        input("init_c"); // [B, H]

        /* Optional scalar maximum sequence length; outputs at t >= seq_length are zero. */
        /* Repeat fixed dtypes for all three positional combinations. */
        this->Input("seq_length")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL);

        this->Input("bias_hh")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        /* Saved states have the same dtype as the inputs and y. */
        output("y");        // [T, B, H]   the layer's result, at x's width
        output("output_h"); // [T, B, H], same quantity and dtype as y
        output("output_c"); // [T, B, H]
        output("i");        // [T, B, H], sigmoid(input gate)
        output("j");        // [T, B, H], tanh(cell input gate)
        output("f");        // [T, B, H], sigmoid(forget gate)
        output("o");        // [T, B, H], sigmoid(output gate)
        output("tanhc");    // [T, B, H], tanh(c_t)

        /* Tiling validates direction and gate_order; unsupported values are rejected. */
        this->Attr("direction").AttrType(OPTIONAL).String("UNIDIRECTIONAL");
        this->Attr("gate_order").AttrType(OPTIONAL).String("ifjo");
        // -1 preserves callers whose logical and physical extents coincide. Explicit values
        // describe zero-padded storage without changing tensor shapes or allocation sizes.
        this->Attr("logical_input_size").AttrType(OPTIONAL).Int(-1);
        this->Attr("logical_hidden_size").AttrType(OPTIONAL).Int(-1);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opInterface.value", "single_layer_lstm")
            .ExtendCfgInfo("opFile.value", "single_layer_lstm");

        /* Pass the configuration explicitly so ValueDepend and compile flags are retained. */
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(SingleLayerLstm);

} // namespace ops
