/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Operator definition for LSTMBlockCellGrad (verbatim TF OpDef name).
 * Registers the operator's metadata (inputs, outputs, attr, supported
 * dtypes/formats, AICore configuration) with the CANN operator definition
 * registry.
 *
 *   16 Inputs  : x, cs_prev, h_prev, w, wci, wcf, wco, b,
 *                i, cs, f, o, ci, co, cs_grad, h_grad        (all REQUIRED)
 *   5 Outputs  : cs_prev_grad, dicfo, wci_grad, wcf_grad, wco_grad (all REQUIRED)
 *   1 Attr     : use_peephole (bool, OPTIONAL, default false)
 *   dtypes     : {ge::DT_FLOAT, ge::DT_FLOAT16} on every tensor (no BF16)
 *   format     : FORMAT_ND everywhere (known + unknown shapes)
 */

#include "register/op_def_registry.h"

namespace ops {

/**
 * class LSTMBlockCellGrad : public OpDef
 *
 * Input/output names and their order mirror the TF OpDef REGISTER_OP(
 * "LSTMBlockCellGrad") input_arg / output_arg lists verbatim (no aliases),
 * so the tf_plugin AutoMappingByOpFn positional/attribute mapping is a
 * straight pass-through.
 */
class LSTMBlockCellGrad : public OpDef {
public:
    explicit LSTMBlockCellGrad(const char* name) : OpDef(name)
    {
        // --- Input declarations: 16 inputs, all REQUIRED ---
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("cs_prev")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("h_prev")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("wci")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("wcf")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("wco")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("i")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("cs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("f")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("o")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("ci")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("co")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("cs_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("h_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // --- Output declarations: 5 outputs, all REQUIRED ---
        this->Output("cs_prev_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("dicfo")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("wci_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("wcf_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("wco_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // --- Attr declaration: the single attribute use_peephole (bool) ---
        // TF OpDef requires it (no default); CANN GEIR attrs need a default,
        // taken as false (aligned with the forward LSTMBlockCell).
        this->Attr("use_peephole").AttrType(OPTIONAL).Bool(false);

        // --- AICore configuration ---
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig
            .DynamicCompileStaticFlag(true) // single static-shape kernel, TilingData-driven
            .DynamicFormatFlag(false)       // fixed ND, no dynamic format conversion
            .DynamicRankSupportFlag(false)  // rank fixed (12 rank-2 + 4 rank-1 inputs)
            .DynamicShapeSupportFlag(true)  // batch dim variable (incl. B==0 empty tensors)
            .NeedCheckSupportFlag(false)    // GEIR execution framework self-checks dtype/format/arch
            .PrecisionReduceFlag(false)     // output dtype contract bound to input dtype, no auto reduce
            // kernel entry file (matches CMakeLists KERNEL_SRC arch35/lstm_block_cell_grad.cpp)
            .ExtendCfgInfo("opFile.value", "lstm_block_cell_grad")
            // opInterface binds the kernel ENTRY FUNCTION name: the default
            // derivation would snake-case the OpType into "lstm_block_cell_grad"
            // and the generated wrapper would then fail to find the PascalCase
            // template entry.  Does not alter the IO/Attr prototype above.
            .ExtendCfgInfo("opInterface.value", "LSTMBlockCellGrad");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
        // Ascend950PR / Ascend950DT are both DAV_3510 (arch35): the single
        // "ascend950" config covers every supported chip.
    }
};

/**
 * Static registration (line below) — instantiates the
 * OpDef subclass and registers it with the global operator definition
 * registry so op_build can discover it and generate aic-*-ops-info.ini.
 */
OP_ADD(LSTMBlockCellGrad);

} // namespace ops
