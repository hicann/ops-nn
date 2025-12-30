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
 * \file quant_batch_matmulV3_onnx_plugin.cpp
 * \brief
 */

#include "onnx_common.h"

namespace domi
{

using NodeProto = ge::onnx::NodeProto;

static Status  ParseParamsWeightBatchQuantMatMulV2(const Message *opSrc, ge::Operator &opDst) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(opSrc);
  if (node == nullptr) {
    OP_LOGE(GetOpName(opDst).c_str(), "Dynamic cast opSrc to NodeProto failed.");
    return FAILED;
  }
  int antiquant_group_size = 0;
  int dtype = -1;
  for (auto attr : node->attribute()) {
    if (attr.name() == "antiquant_group_size" && attr.type() == ge::onnx::AttributeProto::INT) {
      antiquant_group_size = attr.i();
      continue;
    }
    if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
      dtype = attr.i();
      continue;
    }
  }
  // onnx doesn't have transpose attr
  opDst.SetAttr("transpose_x", false);
  opDst.SetAttr("transpose_weight", false);
  opDst.SetAttr("antiquant_group_size", antiquant_group_size);
  opDst.SetAttr("dtype", dtype);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("WeightQuantBatchMatmulV2")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::WeightQuantBatchMatmulV2"),
                 ge::AscendString("ai.onnx::11::NPUWeightQuantBatchMatmulV2"),
                 ge::AscendString("ai.onnx::12::NPUWeightQuantBatchMatmulV2"),
                 ge::AscendString("ai.onnx::13::NPUWeightQuantBatchMatmulV2"),
                 ge::AscendString("ai.onnx::14::NPUWeightQuantBatchMatmulV2"),
                 ge::AscendString("ai.onnx::15::NPUWeightQuantBatchMatmulV2"),
                 ge::AscendString("ai.onnx::16::NPUWeightQuantBatchMatmulV2"),
                 ge::AscendString("ai.onnx::17::NPUWeightQuantBatchMatmulV2"),
                 ge::AscendString("ai.onnx::18::NPUWeightQuantBatchMatmulV2"),
                 ge::AscendString("ai.onnx::19::NPUWeightQuantBatchMatmulV2")})
  .ParseParamsFn(ParseParamsWeightBatchQuantMatMulV2)
  .ImplyType(ImplyType::TVM);
}