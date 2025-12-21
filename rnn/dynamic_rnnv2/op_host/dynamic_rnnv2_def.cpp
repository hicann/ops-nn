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

namespace {
constexpr int32_t DYNAMIC_DIM = -1;
constexpr int32_t UNKNOW_DIM = -2;
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
ge::graphStatus OpSelectFormatRnnV2(const ge::Operator& op, ge::AscendString& result) {
  ge::Shape inputShape = op.GetInputDesc(0).GetShape();
  size_t dimNum = inputShape.GetDimNum();
  bool isDynamic = false;
  for (size_t i = 0; i < dimNum; i++) {
    if (inputShape.GetDim(i) == DYNAMIC_DIM || inputShape.GetDim(i) == UNKNOW_DIM) {
      isDynamic = true;
      break;
    }
  }
  std::string resJsonStr;
  if (isDynamic) {
      resJsonStr =
      "{\"input0\":{\"name\":\"x\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"input1\":{\"name\":\"weight_input\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"input2\":{\"name\":\"weight_hidden\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"input3\":{\"name\":\"b\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"input4\":{\"name\":\"seq_length\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"input5\":{\"name\":\"init_h\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"input6\":{\"name\":\"init_c\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"input7\":{\"name\":\"wci\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"input8\":{\"name\":\"wcf\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"input9\":{\"name\":\"wco\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"input10\":{\"name\":\"mask\", \"dtype\":\"uint8,uint8\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"output0\":{\"name\":\"y\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"output1\":{\"name\":\"output_h\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"output2\":{\"name\":\"output_c\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"output3\":{\"name\":\"i\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"output4\":{\"name\":\"j\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"output5\":{\"name\":\"f\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"output6\":{\"name\":\"o\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"},"
      "\"output7\":{\"name\":\"tanhc\", \"dtype\":\"float32,float16\", \"format\":\"ND,ND\", "
      "\"unknownshape_format\":\"ND,ND\"}}";
  } else {
      std::string xFormat = "\"FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ\"";
      std::string wFormat = "\"FRACTAL_ZN_RNN,FRACTAL_ZN_RNN,FRACTAL_ZN_RNN,FRACTAL_ZN_RNN\"";
      std::string bFormat = "\"ND_RNN_BIAS,ND_RNN_BIAS,ND_RNN_BIAS,ND_RNN_BIAS\"";
      std::string seqFormat = "\"ND,ND,FRACTAL_NZ,FRACTAL_NZ\"";
      std::string maskFormat = "\"ND,ND,ND,ND\"";

      std::string xDtype = "\"float16,float16,float16,float16\"";
      std::string bDtype = "\"float32,float16,float32,float16\"";
      std::string seqDtype = "\"int32,int32,float16,float16\"";
      std::string maskDtype = "\"uint8,uint8,uint8,uint8\"";

      resJsonStr =
      "{\"input0\":{\"name\":\"x\", \"dtype\":" + xDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"input1\":{\"name\":\"weight_input\", \"dtype\":" + xDtype + ", \"format\":" + wFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"input2\":{\"name\":\"weight_hidden\", \"dtype\":" + xDtype + ", \"format\":" + wFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"input3\":{\"name\":\"b\", \"dtype\":" + bDtype + ", \"format\":" + bFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"input4\":{\"name\":\"seq_length\", \"dtype\":" + seqDtype + ", \"format\":" + seqFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"input5\":{\"name\":\"init_h\", \"dtype\":" + xDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"input6\":{\"name\":\"init_c\", \"dtype\":" + bDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"input7\":{\"name\":\"wci\", \"dtype\":" + xDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"input8\":{\"name\":\"wcf\", \"dtype\":" + xDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"input9\":{\"name\":\"wco\", \"dtype\":" + xDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"input10\":{\"name\":\"mask\", \"dtype\":" + maskDtype + ", \"format\":" + maskFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"

      "\"output0\":{\"name\":\"y\", \"dtype\":" + bDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"output1\":{\"name\":\"output_h\", \"dtype\":" + xDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"output2\":{\"name\":\"output_c\", \"dtype\":" + bDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"output3\":{\"name\":\"i\", \"dtype\":" + bDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"output4\":{\"name\":\"j\", \"dtype\":" + bDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"output5\":{\"name\":\"f\", \"dtype\":" + bDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"output6\":{\"name\":\"o\", \"dtype\":" + bDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"},"
      "\"output7\":{\"name\":\"tanhc\", \"dtype\":" + bDtype + ", \"format\":" + xFormat +
      ", \"unknownshape_format\":\"ND,ND,ND,ND\"}}";
  }
  result = ge::AscendString(resJsonStr.c_str());

  return ge::GRAPH_SUCCESS;
}
}

namespace ops {

class DynamicRNNV2 : public OpDef {
public:
    explicit DynamicRNNV2(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("weight_input")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("weight_hidden")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("b")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("seq_length")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("init_h")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("init_c")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("wci")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("wcf")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("wco")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("mask")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_UINT8, ge::DT_UINT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("output_h")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("output_c")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("i")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("j")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("f")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("o")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("tanhc")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("cell_type").AttrType(REQUIRED).String("LSTM");
    this->Attr("direction").AttrType(REQUIRED).String("UNIDIRECTIONAL");
    this->Attr("cell_depth").AttrType(REQUIRED).Int(1);
    this->Attr("use_peephole").AttrType(REQUIRED).Bool(false);
    this->Attr("keep_prob").AttrType(REQUIRED).Float(1.0);
    this->Attr("cell_clip").AttrType(REQUIRED).Float(-1.0);
    this->Attr("num_proj").AttrType(REQUIRED).Int(0);
    this->Attr("time_major").AttrType(REQUIRED).Bool(true);
    this->Attr("activation").AttrType(REQUIRED).String("tanh");
    this->Attr("recurrent_activation").AttrType(REQUIRED).String("sigmoid");
    this->Attr("forget_bias").AttrType(REQUIRED).Float(0.0);
    this->Attr("gate_order").AttrType(REQUIRED).String("ijfo");
    this->Attr("stateful").AttrType(REQUIRED).Bool(false);
    this->Attr("merge_mode").AttrType(REQUIRED).String("concat");
    this->Attr("is_training").AttrType(REQUIRED).Bool(true);
    this->AICore().SetOpSelectFormat(optiling::OpSelectFormatRnnV2);

      OpAICoreConfig aicore_config;
      aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .ExtendCfgInfo("opInterface.value", "dynamic_rnn_v2")
        .ExtendCfgInfo("opFile.value", "dynamic_rnn_v2_910b");

    this->AICore().AddConfig("ascend910b", aicore_config);
    this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(DynamicRNNV2);
}  // namespace ops