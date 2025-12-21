/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_rnn_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "runtime/infer_shape_context.h"

 namespace {
    template <typename T>
    std::string TbeGetName(const T& op) {
    ge::AscendString op_ascend_name;
    ge::graphStatus ret = op.GetName(op_ascend_name);
    if (ret != ge::GRAPH_SUCCESS) {
        std::string op_name = "None";
        return op_name;
    }
    return op_ascend_name.GetString();
    }

 }
namespace optiling {
ge::graphStatus OpSelectFormat([[maybe_unused]]const ge::Operator& op , ge::AscendString& result) {
  std::string resJsonStr =
      "{\"input0\":{\"name\":\"x\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"input1\":{\"name\":\"w\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"input2\":{\"name\":\"b\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"input3\":{\"name\":\"seq_length\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"input4\":{\"name\":\"init_h\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"input5\":{\"name\":\"init_c\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"input6\":{\"name\":\"wci\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"input7\":{\"name\":\"wcf\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"input8\":{\"name\":\"wco\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"input9\":{\"name\":\"mask\", \"dtype\":\"uint8,uint8\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"output0\":{\"name\":\"y\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"output1\":{\"name\":\"output_h\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"output2\":{\"name\":\"output_c\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"output3\":{\"name\":\"i\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"output4\":{\"name\":\"j\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"output5\":{\"name\":\"f\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"output6\":{\"name\":\"o\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"},"
      "\"output7\":{\"name\":\"tanhc\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", \"unknownshape_format\":\"ND,ND\"}}";
  result = ge::AscendString(resJsonStr.c_str());

  return ge::GRAPH_SUCCESS;
}
}

namespace ops
{
  class DynamicRNN : public OpDef
  {
  public:
    explicit DynamicRNN(const char *name) : OpDef(name)
    {
      this->Input("x")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("w")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("b")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("seq_length")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("init_h")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("init_c")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("wci")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("wcf")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("wco")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("mask")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_UINT8, ge::DT_UINT8})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("y")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("output_h")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("output_c")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("i")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("j")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("f")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("o")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("tanhc")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
      this->Attr("cell_type")
          .AttrType(REQUIRED)
          .String("LSTM");
      this->Attr("direction")
          .AttrType(REQUIRED)
          .String("UNIDIRECTIONAL");
      this->Attr("cell_depth")
          .AttrType(REQUIRED)
          .Int(1);
      this->Attr("use_peephole")
          .AttrType(REQUIRED)
          .Bool(false);
      this->Attr("keep_prob")
          .AttrType(REQUIRED)
          .Float(1.0);
      this->Attr("cell_clip")
          .AttrType(REQUIRED)
          .Float(-1.0);
      this->Attr("num_proj")
          .AttrType(REQUIRED)
          .Int(0);
      this->Attr("time_major")
          .AttrType(REQUIRED)
          .Bool(true);
      this->Attr("activation")
          .AttrType(REQUIRED)
          .String("tanh");
      this->Attr("forget_bias")
          .AttrType(REQUIRED)
          .Float(0.0);
      this->Attr("gate_order")
          .AttrType(REQUIRED)
          .String("ifjo");
      this->Attr("is_training")
          .AttrType(REQUIRED)
          .Bool(true);
      this->AICore().SetOpSelectFormat(optiling::OpSelectFormat);

      OpAICoreConfig aicore_config;
      aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .ExtendCfgInfo("opInterface.value", "dynamic_rnn")
        .ExtendCfgInfo("opFile.value", "dynamic_rnn_910b");

      this->AICore().AddConfig("ascend910b", aicore_config);
      this->AICore().AddConfig("ascend910_93", aicore_config);
    }
  };

  OP_ADD(DynamicRNN);
} // namespace ops