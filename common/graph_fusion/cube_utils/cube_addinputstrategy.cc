/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cube_utils/cube_addinputstrategy.h"
#include <memory>
#include <sstream>
#include <set>
#include <mutex>

#include "cube_utils/cube_math_util.h"
#include "runtime/expand_dims_type.h"
#include "transformer/transfer_shape_according_to_format_ext.h"
#include "graph/attr_value.h"

namespace ops {
namespace {
const std::string ATTR_OUTDTYPE = "dequantouttype";
const ge::AscendString kAscendOutDtypeAsc(ATTR_OUTDTYPE.c_str());
const std::string kVectorMulScalar = "VectorMulScalar";
const std::string kSetQuantScale = "SetQuantScale";
const std::string kSetQuantOffsetScale = "SetQuantOffsetScale";
constexpr char const *kConstOutput = "y";
constexpr char const *kAttrSingleOp = "_is_single_op";
const ge::AscendString kAscendSingleOpAsc(kAttrSingleOp);
constexpr char const *kAttrReluOPType = "_relu_op_type";
const ge::AscendString kAscendReluOPTypeAsc(kAttrReluOPType);
constexpr char const *kAttrQuantOutputDtype = "_quant_output_data_type";
const ge::AscendString kAscendQuantOutputDtypeAsc(kAttrQuantOutputDtype);
constexpr char const *kAttrInputStrategy = "_input_strategy";
const ge::AscendString kAscendInputStrategyAsc(kAttrInputStrategy);
constexpr char const *kConst = "Const";
constexpr char const *kAttrHiddenSize = "hidden_size";
constexpr char const *kAttrInputSize = "input_size";
constexpr char const *kAttrStateSize = "state_size";
constexpr int64_t kInputStrategy21 = 21;
constexpr int64_t kInputStrategy51 = 51;
constexpr int64_t kInputStrategy52 = 52;
const uint32_t kCantMuls = 0U;
const uint32_t kHasScalarShape = 1U;
const uint32_t kIsSameShape = 2U;
const uint32_t kCanUseBrocastMuls = 3U;
const std::set<std::string> kLutOpTypeSet = {"Tanh", "Elu", "Sigmoid"};

static uint32_t kConstNodeId = 0;
static std::mutex kConstIdMtx;

template <typename T>
ge::GNode CreateScalarConstNode(const T value, const ge::DataType data_type, ge::Graph &graph) {
  ge::Shape scalar_shape({1});
  ge::TensorDesc tensor_desc(scalar_shape, ge::FORMAT_ND, data_type);
  tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc.SetOriginShape(scalar_shape);

  std::string const_op_name;
  {
    std::lock_guard<std::mutex> lock(kConstIdMtx);
    const_op_name = "dynamic_const_" + std::to_string(kConstNodeId);
    kConstNodeId++;
  }
  ge::GNode newopnode = ge::es::CompliantNodeBuilder(&graph)
                        .OpType(kConst)
                        .Name(const_op_name.c_str())
                        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();

  newopnode.UpdateOutputDesc(0, tensor_desc);
  ge::Tensor const_tensor(tensor_desc, reinterpret_cast<const uint8_t *>(&value), sizeof(T));
  newopnode.SetAttr(kAscendNameWeightAsc, const_tensor);
  return newopnode;
}
}

template <typename T>
ge::graphStatus FixPipeAddInputBase::UpdateSalarInput(ge::TensorDesc tensor_desc, T value,
                                             ge::Tensor tensornode, const ge::DataType &data_type) const {
  int data_size = ge::GetSizeByDataType(data_type);
  if (data_size == 0) {
    return ge::GRAPH_FAILED;
  }
  ge::Shape shape{};
  OPS_LOG_D("Fixpipe", "value = %f data size = %d", static_cast<float>(value), data_size);
  tensor_desc.SetDataType(data_type);
  tensor_desc.SetShape(shape);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(ge::FORMAT_ND);
  tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  if (tensornode.SetData(reinterpret_cast<uint8_t *>(&value), data_size) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::GNode FixPipeAddInputBase::CreateNewDataNodeOnly(ge::Graph &graph, ge::TensorDesc tensor_desc,
                                                         ge::Tensor tensornode, const std::string &op_name) const {
  OPS_LOG_D("Fixpipe", "fixpipenode name = %s", op_name.c_str());
  ge::GNode newopnode = ge::es::CompliantNodeBuilder(&graph)
                        .OpType(kConst)
                        .Name(op_name.c_str())
                        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();

  bool tmp_bool1 = true;
  newopnode.SetAttr(kAscendSingleOpAsc, tmp_bool1);
  bool tmp_bool2 = true;
  newopnode.SetAttr(kAscendOriginalInputAsc, tmp_bool2);
  if (newopnode.UpdateOutputDesc(0, tensor_desc) != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "UpdateOutputDesc Failed");
    return ge::GNode();
  }
  newopnode.SetAttr(kAscendNameWeightAsc, tensornode);
  return newopnode;
}

ge::graphStatus FixPipeAddInputBase::CreateNewDataNodeDirect(ge::Graph &graph, ge::TensorDesc tensor_desc,
                                                    ge::Tensor tensornode,
                                                    const FixPipeFunctionParamPtr &functpara) const {
  const ge::GNodePtr &fixpipenode = functpara->GetFixpipeNode();
  int input_index = functpara->GetParaIndex();
  std::string op_name = std::string(GNodeGetName(fixpipenode).GetString()) + "_" +  functpara->GetInputName();
  auto newdatanode = CreateNewDataNodeOnly(graph, tensor_desc, tensornode, op_name);
  if (newdatanode.GetOutputsSize() == 0 || fixpipenode->GetInputsSize() < static_cast<size_t>(input_index)) {
    OPS_LOG_D("Fixpipe", "GetOutDataAnchor GetInDataAnchor = null ");
    return ge::GRAPH_FAILED;
  }
  if (graph.AddDataEdge(newdatanode, 0, *fixpipenode, input_index) != ge::GRAPH_SUCCESS) {
    REPORT_OPS_ERROR("[GraphOpt][FixpipePss][RelkDataEdge] Fail to add edge between src node[%s] and dst node[%s].",
                    GNodeGetName(newdatanode).GetString(), GNodeGetName(fixpipenode).GetString());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

bool FixPipeAddInputBase::IsScalar(const ge::Shape &origin_shape) {
  if (origin_shape.GetShapeSize() == 0 || origin_shape.GetShapeSize() == 1) {
    return true;
  }
  return false;
}
ge::graphStatus FixPipeAddInputBase::UpdateVectorMulsOutputTensorDesc(const ge::TensorDesc &prenode_inputdesc,
                                                             const ge::TensorDesc &postnode_inputdesc,
                                                             ge::TensorDesc &out_tensor_desc) const {
  const ge::Shape prenode_input_shape = prenode_inputdesc.GetOriginShape();
  const ge::Shape postnode_input_shape = postnode_inputdesc.GetOriginShape();
  out_tensor_desc.SetDataType(ge::DT_FLOAT);
  OPS_LOG_D("Fixpipe", "desc1 dims = %s %s  desc2 dims = %s %s",
          FixpipeComm::ShapeToString(prenode_inputdesc.GetOriginShape()).c_str(),
          FixpipeComm::ShapeToString(prenode_inputdesc.GetShape()).c_str(),
          FixpipeComm::ShapeToString(postnode_inputdesc.GetOriginShape()).c_str(),
          FixpipeComm::ShapeToString(postnode_inputdesc.GetShape()).c_str());
  auto canContinueMuls = IsSameShape(prenode_inputdesc, postnode_inputdesc);
  if (canContinueMuls == kCantMuls) {
  OPS_LOG_W("Fixpipe", "shape1 != shape2");
    return ge::GRAPH_FAILED;
  }
  if (canContinueMuls == kCanUseBrocastMuls) {
    OPS_LOG_D("Fixpipe", "UpdateVectorMulsOutputTensorDesc can use brocast");
    if (IsScalar(prenode_input_shape)) {
      OPS_LOG_D("Fixpipe", "UpdateVectorMulsOutputTensorDesc prenode inputshape is scalar");
      out_tensor_desc.SetOriginShape(postnode_input_shape);
    } else {
      OPS_LOG_D("Fixpipe", "UpdateVectorMulsOutputTensorDesc postnode inputshape is scalar");
      out_tensor_desc.SetOriginShape(prenode_input_shape);
    }
    auto broadcastshape = GetBroadcastShape(prenode_inputdesc.GetShape(), postnode_inputdesc.GetShape());
    OPS_LOG_D("Fixpipe", "UpdateVectorMulsOutputTensorDesc can use brocast:%s", FixpipeComm::ShapeToString(broadcastshape).c_str());
    out_tensor_desc.SetShape(broadcastshape);
  } else if (canContinueMuls == kIsSameShape) {
       OPS_LOG_D("Fixpipe", "UpdateVectorMulsOutputTensorDesc issameshape");
    out_tensor_desc.SetShape(prenode_inputdesc.GetShape());
    out_tensor_desc.SetOriginShape(prenode_input_shape);
  } else {
    OPS_LOG_D("Fixpipe", "UpdateVectorMulsOutputTensorDesc one inputshape isscalar");
    if (IsScalar(prenode_input_shape)) {
      OPS_LOG_D("Fixpipe", "UpdateVectorMulsOutputTensorDesc prenode inputshape isscalar");
      out_tensor_desc.SetShape(postnode_inputdesc.GetShape());
      out_tensor_desc.SetOriginShape(postnode_input_shape);
    } else {
      OPS_LOG_D("Fixpipe", "UpdateVectorMulsOutputTensorDesc postnode inputshape isscalar");
      out_tensor_desc.SetShape(prenode_inputdesc.GetShape());
      out_tensor_desc.SetOriginShape(prenode_input_shape);
    }
  }
  OPS_LOG_D("Fixpipe", "UpdateVectorMulsOutputTensorDesc successfully");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixPipeAddInputBase::CreateAndUpdateVectorMulsInput(ge::Graph &graph,
                                                             const FixPipeFunctionParamPtr &functpara,
                                                             const FixPipeNodeInfo &postfuzednode,
                                                             const FixPipeNodeInfo &tofuzednode,
                                                             std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  OPS_LOG_D("Fixpipe", "CreateAndUpdateVectorMulsInput start");
  auto tofuze_node = tofuzednode.GetNode();
  auto post_fuze_node = postfuzednode.GetNode();
  if (FixpipeComm::CheckPeerOutNode(tofuze_node, 1) != ge::GRAPH_SUCCESS ||
      FixpipeComm::CheckPeerOutNode(post_fuze_node, 1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &fixpipenode = functpara->GetFixpipeNode();
  const char* vector_muls_type = "VectorMuls";
  std::string vector_muls_name = std::string(GNodeGetName(fixpipenode).GetString()) + "VectorMuls" + functpara->GetInputName();
  ge::GNode newopnode = ge::es::CompliantNodeBuilder(&graph)
                        .OpType(vector_muls_type)
                        .Name(vector_muls_name.c_str())
                        .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"x2", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
  bool tmp_bool = true;
  newopnode.SetAttr(kAscendSingleOpAsc, tmp_bool);

  ge::TensorDesc tufuzenode_inputdesc;
  ge::TensorDesc postnode_inputdesc;
  if (tofuze_node->GetInputDesc(1, tufuzenode_inputdesc) != ge::GRAPH_SUCCESS ||
      post_fuze_node->GetInputDesc(1, postnode_inputdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc out_tensor_desc = tufuzenode_inputdesc;
  if (UpdateVectorMulsOutputTensorDesc(tufuzenode_inputdesc, postnode_inputdesc, out_tensor_desc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  newopnode.UpdateOutputDesc(0, out_tensor_desc);
  newopnode.UpdateInputDesc(0, tufuzenode_inputdesc);
  newopnode.UpdateInputDesc(1, postnode_inputdesc);

  auto inputnode = tofuze_node->GetInDataNodesAndPortIndexs(1);
  auto anotherinput = post_fuze_node->GetInDataNodesAndPortIndexs(1);
  int input_index = functpara->GetParaIndex();
  fixpipenode->UpdateInputDesc(input_index, out_tensor_desc);
  if (graph.AddDataEdge(*inputnode.first, inputnode.second, newopnode, 0) != ge::GRAPH_SUCCESS ||
      graph.AddDataEdge(*anotherinput.first, anotherinput.second, newopnode, 1) != ge::GRAPH_SUCCESS ||
      graph.AddDataEdge(newopnode, 0, *fixpipenode, input_index) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus FixPipeAddInputBase::CreateAndUpdateVectorMulScalarInput(ge::Graph &graph,
                                                                  const FixPipeFunctionParamPtr &functpara,
                                                                  const FixPipeNodeInfo &prefuzednode, const T &value,
                                                                  std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  auto pre_fuzed_node = prefuzednode.GetNode();
  if (FixpipeComm::CheckPeerOutNode(pre_fuzed_node, 1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  int input_index = functpara->GetParaIndex();
  const ge::GNodePtr &fixpipenode = functpara->GetFixpipeNode();
  std::string vector_mul_scalar_name = std::string(GNodeGetName(fixpipenode).GetString()) + kVectorMulScalar + functpara->GetInputName();
  ge::GNode newopnode = ge::es::CompliantNodeBuilder(&graph)
                        .OpType(kVectorMulScalar.c_str())
                        .Name(vector_mul_scalar_name.c_str())
                        .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
  bool tmp_bool = true;
  newopnode.SetAttr(kAscendSingleOpAsc, tmp_bool);
  ge::TensorDesc prenode_inputdesc;
  if (pre_fuzed_node->GetInputDesc(1, prenode_inputdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc out_tensor_desc = prenode_inputdesc;
  if (functpara->GetDataType() != ge::DT_UNDEFINED) {
    out_tensor_desc.SetDataType(functpara->GetDataType());
  }
  newopnode.UpdateOutputDesc(0, out_tensor_desc);
  newopnode.UpdateInputDesc(0, prenode_inputdesc);

  auto inputnode = pre_fuzed_node->GetInDataNodesAndPortIndexs(1);
  if (graph.AddDataEdge(*inputnode.first, inputnode.second, newopnode, 0) != ge::GRAPH_SUCCESS ||
      graph.AddDataEdge(newopnode, 0, *fixpipenode, input_index) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::GNode FixPipeAddInputBase::CreateVectorMulsOpDesc(ge::Graph &graph, const std::string &op_name,
                                const ge::GNodePtr &pre_op_desc, const ge::GNodePtr &post_op_desc) const {
  OPS_LOG_D("Fixpipe", "Begin to create VectorMuls host op[%s].", op_name.c_str());
  const char* vector_muls_type = "VectorMuls";
  ge::GNode newopnode = ge::es::CompliantNodeBuilder(&graph)
                        .OpType(vector_muls_type)
                        .Name(op_name.c_str())
                        .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"x2", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
  bool tmp_bool = true;
  newopnode.SetAttr(kAscendSingleOpAsc, tmp_bool);
  ge::TensorDesc tufuzenode_inputdesc;
  ge::TensorDesc postnode_inputdesc;
  if (pre_op_desc->GetInputDesc(1, tufuzenode_inputdesc) != ge::GRAPH_SUCCESS ||
      post_op_desc->GetInputDesc(1, postnode_inputdesc) != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "Input1 of pre node or post node is null");
    return ge::GNode();
  }
  ge::TensorDesc out_tensor_desc = tufuzenode_inputdesc;
  if (UpdateVectorMulsOutputTensorDesc(tufuzenode_inputdesc, postnode_inputdesc, out_tensor_desc) != ge::GRAPH_SUCCESS) {
    return ge::GNode();
  }
  newopnode.UpdateInputDesc(0, tufuzenode_inputdesc);
  newopnode.UpdateInputDesc(1, postnode_inputdesc);
  newopnode.UpdateOutputDesc(0, out_tensor_desc);
  OPS_LOG_D("Fixpipe", "Host op[%s, %s] has been created.", GNodeGetName(newopnode).GetString(), GNodeGetType(newopnode).GetString());
  return newopnode;
}

ge::GNode FixPipeAddInputBase::CreateVectorMulScalarOpDesc(ge::Graph &graph, const std::string &op_name,
                                                                 const ge::GNodePtr &pre_op_desc,
                                                                 const ge::GNodePtr &post_op_desc,
                                                                 const ge::DataType &data_type) {
  OPS_LOG_D("Fixpipe", "Begin to create VectorMulScalar host op[%s].", op_name.c_str());
  ge::GNode newopnode = ge::es::CompliantNodeBuilder(&graph)
                        .OpType(kVectorMulScalar.c_str())
                        .Name(op_name.c_str())
                        .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();

  bool tmp_bool1 = true;
  newopnode.SetAttr(kAscendSingleOpAsc, tmp_bool1);
  float negative_slope = 0.0;
  post_op_desc->GetAttr(kAscendNegativeSlopeAsc, negative_slope);
  OPS_LOG_D("Fixpipe", "Post fuzed node is [%s, %s], negative_slope[%f].",
          GNodeGetName(post_op_desc).GetString(), GNodeGetType(post_op_desc).GetString(), negative_slope);
  float tmp_float = negative_slope;
  newopnode.SetAttr(kAscendScaleAsc, tmp_float);

  ge::TensorDesc prenode_inputdesc;
  if (pre_op_desc->GetInputDesc(1, prenode_inputdesc) != ge::GRAPH_SUCCESS) {
    return ge::GNode();
  }
  ge::TensorDesc out_tensor_desc = prenode_inputdesc;
  if (data_type != ge::DT_UNDEFINED) {
    out_tensor_desc.SetDataType(data_type);
  }
  newopnode.UpdateOutputDesc(0, out_tensor_desc);
  newopnode.UpdateInputDesc(0, prenode_inputdesc);
  OPS_LOG_D("Fixpipe", "Host op[%s, %s] has been created.", GNodeGetName(newopnode).GetString(), GNodeGetType(newopnode).GetString());
  return newopnode;
}

ge::graphStatus FixPipeAddInputBase::CreateAndRelinkCastNode(ge::Graph &graph,
                                                     const ge::GNodePtr &inputnode,
                                                     const ge::GNodePtr &outputnode,
                                                     const int &input_index,
                                                     std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  if (inputnode == nullptr || outputnode == nullptr) {
    return ge::GRAPH_FAILED;
  }
  std::string cast_op_name = std::string(GNodeGetName(outputnode).GetString()) + "INSERTCAST" + std::to_string(input_index);
  ge::GNode newopnode = ge::es::CompliantNodeBuilder(&graph)
                        .OpType(CAST.c_str())
                        .Name(cast_op_name.c_str())
                        .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
  bool tmp_bool = true;
  newopnode.SetAttr(kAscendSingleOpAsc, tmp_bool);
  ge::TensorDesc prenode_outputdesc;
  ge::TensorDesc outnode_inputdesc;
  if (inputnode->GetOutputDesc(0, prenode_outputdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (outputnode->GetInputDesc(input_index, outnode_inputdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  newopnode.UpdateOutputDesc(0, outnode_inputdesc);
  newopnode.UpdateInputDesc(0, prenode_outputdesc);
  if (graph.AddDataEdge(*inputnode, 0, newopnode, 0) != ge::GRAPH_SUCCESS ||
      graph.AddDataEdge(newopnode, 0, *outputnode, input_index) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixPipeAddInputBase::CloneVectorInput(ge::Graph &graph,
                                              const FixPipeNodeInfo &tofuzednode,
                                              const FixPipeFunctionParamPtr &functpara,
                                              std::vector<ge::GNodePtr> &new_nodes) const {
  const ge::GNodePtr &fixpipenode = functpara->GetFixpipeNode();
  auto tofuze_node = tofuzednode.GetNode();
  ge::TensorDesc tufuzenode_inputdesc;
  if (tofuze_node->GetInputDesc(1, tufuzenode_inputdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  int input_index = functpara->GetParaIndex();
  ge::TensorDesc cur_tensor_desc = tufuzenode_inputdesc;
  if (FixpipeComm::CheckPeerOutNode(tofuze_node, 1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  auto inputnode = tofuze_node->GetInDataNodesAndPortIndexs(1);
  bool need_insert_cast = false;
  if (functpara->GetDataType() != ge::DT_UNDEFINED &&
      cur_tensor_desc.GetDataType() != functpara->GetDataType()) {
    cur_tensor_desc.SetDataType(functpara->GetDataType());
    need_insert_cast = true;
  }
  fixpipenode->UpdateInputDesc(input_index, cur_tensor_desc);
  if (!need_insert_cast) {
    if (graph.AddDataEdge(*inputnode.first, inputnode.second, *fixpipenode, input_index) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  } else {
    return CreateAndRelinkCastNode(graph, inputnode.first, fixpipenode, input_index, new_nodes);
  }
}

template <typename T>
ge::graphStatus FixPipeAddInputBase::CreateAndUpdateSalarInput(ge::Graph &graph, const FixPipeFunctionParamPtr &functpara,
                                                      T value, const ge::DataType &data_type,
                                                      std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  const ge::GNodePtr &fixpipenode = functpara->GetFixpipeNode();
  int input_index = functpara->GetParaIndex();
  ge::TensorDesc tensor_desc;
  bool has_desc = true;
  if (fixpipenode->GetInputDesc(input_index, tensor_desc) != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "input = %d", input_index);
    ge::Shape shape{};
    tensor_desc.Update(shape, ge::FORMAT_ND, data_type);
    has_desc = false;
  }
  ge::Tensor tensor_node(tensor_desc);
  if (UpdateSalarInput<T>(tensor_desc, value, tensor_node, data_type) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (CreateNewDataNodeDirect(graph, tensor_desc, tensor_node, functpara) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (!has_desc) {
    fixpipenode->UpdateInputDesc(input_index, tensor_desc);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixPipeAddInputBase::CreateScalarInputNode(ge::Graph &graph, const FixPipeFunctionParamPtr &functpara,
                                                   const ge::GNodePtr &first_node, const int64_t input_strategy) const {
  OPS_CHECK_NOTNULL(first_node);
  float scale = 0.0;
  float offset = 0.0;
  GNodeGetAttr(first_node, ATTR_SCALE, scale);
  GNodeGetAttr(first_node, ATTR_OFFSET, offset);
  OPS_LOG_D("Fixpipe", "Offset and scale of node[%s, %s] is [%f] and [%f].",
            GNodeGetName(first_node).GetString(), GNodeGetType(first_node).GetString(), offset, scale);
  ge::GNode offset_node = CreateScalarConstNode(offset, ge::DT_FLOAT, graph);
  ge::GNode scale_node = CreateScalarConstNode(scale, ge::DT_FLOAT, graph);

  const ge::GNodePtr &fixpipe_node = functpara->GetFixpipeNode();
  OPS_CHECK_NOTNULL(fixpipe_node);

  // create set quant offset scale host op
  std::string set_data_op_name = std::string(GNodeGetName(fixpipe_node).GetString()) + kSetQuantOffsetScale + functpara->GetInputName();
  // create set quant offset scale node
  ge::GNode set_quant_data_node = ge::es::CompliantNodeBuilder(&graph)
                                  .OpType(kSetQuantOffsetScale.c_str())
                                  .Name(set_data_op_name.c_str())
                                  .IrDefInputs({{ATTR_OFFSET, ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                                {ATTR_SCALE, ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                                  .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                  .Build();
  ge::TensorDesc offset_output_desc;
  ge::TensorDesc scale_output_desc;
  ge::TensorDesc first_output_desc;
  if (offset_node.GetOutputDesc(0, offset_output_desc) != ge::GRAPH_SUCCESS ||
      scale_node.GetOutputDesc(0, scale_output_desc) != ge::GRAPH_SUCCESS ||
      first_node->GetOutputDesc(0, first_output_desc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  (void)set_quant_data_node.UpdateInputDesc(0, offset_output_desc);
  (void)set_quant_data_node.UpdateInputDesc(1, scale_output_desc);
  ge::TensorDesc output_desc = offset_output_desc;
  output_desc.SetDataType(ge::DT_UINT64);
  (void)set_quant_data_node.UpdateOutputDesc(0, output_desc);
  int64_t tmp_int1 = static_cast<int64_t>(first_output_desc.GetDataType());
  set_quant_data_node.SetAttr(kAscendQuantOutputDtypeAsc, tmp_int1);
  OPS_LOG_D("Fixpipe", "Output data type of first node[%s, %s] is [%s].", GNodeGetName(first_node).GetString(), GNodeGetType(first_node).GetString(),
          ge::TypeUtils::DataTypeToSerialString(first_output_desc.GetDataType()).c_str());
  int64_t tmp_int2 = input_strategy;
  set_quant_data_node.SetAttr(kAscendInputStrategyAsc, tmp_int2);
  OPS_LOG_D("Fixpipe", "Set input strategy[%ld] for op[%s, %s].", input_strategy,
            set_data_op_name.c_str(), kSetQuantOffsetScale.c_str());

  // update fixpipe input tensor desc
  (void)fixpipe_node->UpdateInputDesc(static_cast<uint32_t>(functpara->GetParaIndex()), output_desc);

  // add edge
  (void)graph.AddDataEdge(offset_node, 0, set_quant_data_node, 0);
  (void)graph.AddDataEdge(scale_node, 0, set_quant_data_node, 1);
  (void)graph.AddDataEdge(set_quant_data_node, 0, *fixpipe_node, static_cast<uint32_t>(functpara->GetParaIndex()));
  return ge::GRAPH_SUCCESS;
}

void FixPipeAddInputBase::SetClipValue6(ge::Graph &graph, const FixPipeFunctionParamPtr &functpara,
                                        ge::DataType dst_datatype, std::vector<ge::GNodePtr> &new_nodes) const {
  if (dst_datatype == ge::DT_FLOAT) {
    float clipvalue = kRelu6Value;
    CreateAndUpdateSalarInput<float>(graph, functpara, clipvalue, dst_datatype, new_nodes);
  } else if (dst_datatype == ge::DT_FLOAT16) {
    float relu6value_tmp = kRelu6Value;
    ops::fp16_t clipvalue(relu6value_tmp);
    OPS_LOG_D("Fixpipe", "clipvalue = %u", clipvalue.ToUInt16());
    CreateAndUpdateSalarInput<ops::fp16_t>(graph, functpara, clipvalue, dst_datatype, new_nodes);
  } else if (dst_datatype == ge::DT_INT8 || dst_datatype == ge::DT_INT4) {
    dst_datatype = ge::DT_FLOAT16;
    float relu6value_tmp = kRelu6Value;
    ops::fp16_t clipvalue(relu6value_tmp);
    OPS_LOG_D("Fixpipe", "clipvalue = %u", clipvalue.ToUInt16());
    CreateAndUpdateSalarInput<ops::fp16_t>(graph, functpara, clipvalue, dst_datatype, new_nodes);
  }
}

ge::GNodePtr FixPipeAddInputBase::GetQuantScaleOffset(const FixPipePassInfo &match_pass,
                                                      const uint32_t &index,
                                                      float &scale, float &offset) const {
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, index)) {
    return nullptr;
  }
  auto node = match_pass.m_opnodes[index].GetNode();
  if (node == nullptr) {
    return nullptr;
  }
  GNodeGetAttr(node, ATTR_SCALE, scale);
  GNodeGetAttr(node, ATTR_OFFSET, offset);
  return node;
}

ge::GNode FixPipeAddInputBase::CreateQuantScaleOpDesc(ge::Graph &graph, const std::string &op_name,
                                                           const ge::GNodePtr &pre_op_desc,
                                                           const ge::GNodePtr &post_op_desc,
                                                           const ge::GNodePtr &input2_op_desc) {
  OPS_LOG_D("Fixpipe", "Begin to create SetQuantScale Host op[%s].", op_name.c_str());
  // create set quant scale host op
  ge::GNode newopnode = ge::es::CompliantNodeBuilder(&graph)
                        .OpType(kSetQuantScale.c_str())
                        .Name(op_name.c_str())
                        .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"x2", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
  ge::TensorDesc prenode_inputdesc;
  ge::TensorDesc prenode_outputdesc;
  if (pre_op_desc->GetInputDesc(1, prenode_inputdesc) != ge::GRAPH_SUCCESS ||
           pre_op_desc->GetOutputDesc(0, prenode_outputdesc) != ge::GRAPH_SUCCESS) {
    return ge::GNode();
  }

  ge::TensorDesc out_tensor_desc = prenode_inputdesc;
  out_tensor_desc.SetDataType(ge::DT_UINT64);
  newopnode.UpdateInputDesc(0, prenode_inputdesc);
  if (input2_op_desc != nullptr) {
    ge::TensorDesc input2_tensor_desc;
    (void)input2_op_desc->GetOutputDesc(0, input2_tensor_desc);
    newopnode.UpdateInputDesc(1, input2_tensor_desc);
  }
  newopnode.UpdateOutputDesc(0, out_tensor_desc);

  // set attr
  float offset = 0.0;
  if (GNodeGetAttr(pre_op_desc, ATTR_OFFSET, offset) == ge::GRAPH_SUCCESS) {
    float tmp_float = offset;
    newopnode.SetAttr(kAscendOffsetAsc, tmp_float);
    OPS_LOG_D("Fixpipe", "Set offset value [%f] for op[%s]", offset, op_name.c_str());
  }
  int64_t tmp_int1 = static_cast<int64_t>(prenode_outputdesc.GetDataType());
  newopnode.SetAttr(kAscendOutDtypeAsc, tmp_int1);
  bool tmp_bool = true;
  newopnode.SetAttr(kAscendSingleOpAsc, tmp_bool);
  // set relu_op_type
  if (post_op_desc != nullptr) {
    ge::AscendString tmp_str = GNodeGetType(post_op_desc);
    newopnode.SetAttr(kAscendReluOPTypeAsc, tmp_str);
    OPS_LOG_D("Fixpipe", "Set attr[%s] value[%s] for op[%s]", kAttrReluOPType, tmp_str.GetString(), op_name.c_str());
  }
  OPS_LOG_D("Fixpipe", "Host op[%s, %s] has been created.", GNodeGetName(newopnode).GetString(), GNodeGetType(newopnode).GetString());
  return newopnode;
}
ge::graphStatus FixPipeAddInputBase::DoWithClipReluInputWithSingleRelu6(ge::Graph &graph,
                                                                  const FixPipePassInfo &match_pass,
                                                                  const FixPipeFunctionParamPtr &functpara,
                                                                  std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategyBase DoWithClipReluInputWithSingleRelu6");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  if (first_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc firstnode_outputdesc;
  if (first_node->GetOutputDesc(0, firstnode_outputdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  ge::DataType dst_datatype = firstnode_outputdesc.GetDataType();
  SetClipValue6(graph, functpara, dst_datatype, new_nodes);
  return ge::GRAPH_SUCCESS;
}

bool FixPipeAddInputBase::CanShapeBroadcast(const ge::Shape &shape1, const ge::Shape &shape2) {
  OPS_LOG_D("Fixpipe", "Begin to check can two shape broadcast, shape1 dims = %zu %s, shape2 dims = %zu %s",
          shape1.GetDimNum(), FixpipeComm::ShapeToString(shape1).c_str(), shape2.GetDimNum(), FixpipeComm::ShapeToString(shape2).c_str());
  // The shapes are broadcastable if for each dimension either the sizes match
  // or one of the sizes is 1.
  if (shape1.GetDimNum() != shape2.GetDimNum()) {
    return false;
  }

  for (size_t i = 0U; i < shape1.GetDimNum(); ++i) {
    int64_t dim1 = shape1.GetDim(i);
    int64_t dim2 = shape2.GetDim(i);
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      OPS_LOG_D("Fixpipe", "two shape can not broadcast");
      return false;
    }
  }
  OPS_LOG_D("Fixpipe", "two shape can broadcast");
  return true;
}

bool FixPipeAddInputBase::CanBroadcast(const ge::TensorDesc &tensor_desc1,
                                       const ge::TensorDesc &tensor_desc2) {
  return CanShapeBroadcast(tensor_desc1.GetShape(), tensor_desc2.GetShape());
}

void GeShapeToRtShape(const ge::Shape &ge_shape, gert::Shape &rt_shape) {
  rt_shape.SetDimNum(0);
  for (size_t i = 0; i < ge_shape.GetDimNum(); ++i) {
    rt_shape.AppendDim(ge_shape.GetDim(i));
  }
}

void RtShapeToGeShape(const gert::Shape &rt_shape, ge::Shape &ge_shape) {
  std::vector<int64_t> dims;
  for (size_t i = 0; i < rt_shape.GetDimNum(); ++i) {
    dims.push_back(rt_shape.GetDim(i));
  }
  ge_shape = ge::Shape(dims);
}

void FixPipeAddInputBase::GetShapeAfterExpandDims(const ge::GNode &node, const ge::TensorDesc &tensor_desc, ge::Shape &shape) {
  int64_t reshape_type_mask = 0;
  ge::AttrValue attr_value;
  gert::Shape inner_shape;
  GeShapeToRtShape(shape, inner_shape);
  if (node.GetInputAttr(kAscendReshapeTypeMaskAsc, 1, attr_value) == ge::GRAPH_SUCCESS && 
                       attr_value.GetAttrValue(reshape_type_mask) == ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "Begin to expand dims, reshape type[%ld], shape[%s].",
              reshape_type_mask, FixpipeComm::ShapeToString(shape).c_str());
    if (reshape_type_mask == 0) {
      return;
    }
    gert::ExpandDimsType expand_dims_type(reshape_type_mask);
    expand_dims_type.Expand(inner_shape);
  } else {
    int64_t input_size = 1;
    int64_t hidden_size = 1;
    int64_t state_size = -1;
    ge::AscendString attr_input(kAttrInputSize);
    (void)node.GetAttr(attr_input, input_size);
    ge::AscendString attr_hidden(kAttrHiddenSize);
    (void)node.GetAttr(attr_hidden, hidden_size);
    ge::AscendString attr_state(kAttrStateSize);
    (void)node.GetAttr(attr_state, state_size);
    transformer::ExtAxisOpValue op_value{input_size, hidden_size, state_size};
    transformer::ShapeTransferAccordingToFormatExt::TransferShape(tensor_desc.GetOriginFormat(),
                              tensor_desc.GetFormat(), tensor_desc.GetDataType(), inner_shape, op_value);
  }
  RtShapeToGeShape(inner_shape, shape);
  OPS_LOG_D("Fixpipe", "After expanding dims, shape[%s].", FixpipeComm::ShapeToString(shape).c_str());
}

bool FixPipeAddInputBase::GetShapeByFormat(const ge::Format old_format, const ge::Format new_format,
                          const ge::DataType data_type, const ge::Shape &old_shape, ge::Shape &new_shape) {
  new_shape = old_shape;
  gert::Shape inner_new_shape;
  gert::Shape inner_old_shape;
  GeShapeToRtShape(old_shape, inner_old_shape);
  GeShapeToRtShape(new_shape, inner_new_shape);

  if (static_cast<ge::Format>(GetPrimaryFormat(static_cast<int32_t>(old_format))) ==
      static_cast<ge::Format>(GetPrimaryFormat(static_cast<int32_t>(new_format)))) {
    OPS_LOG_D("Fixpipe", "Origin formt and formt is same, no need to transfer shape.");
    return true;
  }
  if (old_shape.GetDimNum() == 0U) {
    OPS_LOG_D("Fixpipe", "Do not need to do shape transformation for unknown rank case.");
    return true;
  }
  // default value
  transformer::ExtAxisOpValue ext_axis{1, 1, -1};
  bool ret = transformer::ShapeTransferAccordingToFormatExt::TransferShape(old_format, new_format,
                           data_type, inner_old_shape, inner_new_shape, ext_axis);
  RtShapeToGeShape(inner_new_shape, new_shape);
  return ret;
}

ge::Shape FixPipeAddInputBase::GetBroadcastShape(const ge::Shape &shape1, const ge::Shape &shape2) {
  if (!CanShapeBroadcast(shape1, shape2)) {
    return ge::Shape();
  }
  auto rank = shape1.GetDimNum();
  auto dim_num1 = shape1.GetDims();
  auto dim_num2 = shape2.GetDims();
  // init broadcasted shape:0
  std::vector<int64_t> broadcastShape(rank, 0);
  // begin from last vetcor
  for (size_t i = 0U; i < rank; ++i) {
    auto dim1 = dim_num1[rank - 1 - i];
    auto dim2 = dim_num2[rank - 1 - i];
    if (dim1 == dim2) {
      broadcastShape[rank - 1 - i] = dim1;
    }
    else if (dim1 == 1) {
      broadcastShape[rank - 1 - i] = dim2;
    }
    else if (dim2 == 1) {
      broadcastShape[rank - 1 - i] = dim1;
    }
    else {
      return ge::Shape(); // return empty means cant broadcast
    }
  }
  OPS_LOG_D("Fixpipe", "Get broadcastShape successfully.");
  return ge::Shape(broadcastShape);
}

uint32_t FixPipeAddInputBase::IsSameShape(const ge::TensorDesc &tensor_desc1,
                                          const ge::TensorDesc &tensor_desc2) {
  if (FixpipeComm::IsShapeEqual(tensor_desc1.GetShape(), tensor_desc2.GetShape())) {
    OPS_LOG_D("Fixpipe", "two tensor is same shape %s", FixpipeComm::ShapeToString(tensor_desc1.GetShape()).c_str());
    return kIsSameShape;
  } else if (tensor_desc1.GetOriginShape().GetShapeSize() == tensor_desc2.GetOriginShape().GetShapeSize()) {
    if (CanBroadcast(tensor_desc1, tensor_desc2)) {
      OPS_LOG_D("Fixpipe", "canbrocastshape with originshape size is same");
      return kCanUseBrocastMuls;
    }
    OPS_LOG_W("Fixpipe", "cant continue vectormuls with originshape is same");
    return kCantMuls;
  } else if (IsScalar(tensor_desc1.GetOriginShape()) || IsScalar(tensor_desc2.GetOriginShape())) {
    if (IsScalar(tensor_desc1.GetShape()) || IsScalar(tensor_desc2.GetShape())) {
      OPS_LOG_D("Fixpipe", "one shape is scalar");
      return kHasScalarShape;
    } else if (CanBroadcast(tensor_desc1, tensor_desc2)) {
      OPS_LOG_D("Fixpipe", "canbrocastshape with one shape is scalar");
      return kCanUseBrocastMuls;
    }
    OPS_LOG_W("Fixpipe", "cant continue vectormuls with one shape is scalar");
    return kCantMuls;
  } else if (CanBroadcast(tensor_desc1, tensor_desc2)) {
    OPS_LOG_D("Fixpipe", "just canbrocastshape");
    return kCanUseBrocastMuls;
  }
  OPS_LOG_W("Fixpipe", "cant continue vectormuls");
  return kCantMuls;
}

// quant
ge::graphStatus FixPipeAddInputStrategy21::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy21 precove quant");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    OPS_LOG_W("Fixpipe", "First node index[%u] is invalid for size of matched nodes is [%zu].",
            functpara->GetFirstIndex(), match_pass.m_opnodes.size());
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  ge::graphStatus ret = CreateScalarInputNode(graph, functpara, first_node, kInputStrategy21);
  OPS_LOG_D("Fixpipe", "Finish FixPipeAddInputStrategy21 precove quant, ret is [%u].", ret);
  return ret;
}

// dequant/requant
ge::graphStatus FixPipeAddInputStrategy22::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                              const FixPipeFunctionParamPtr &functpara,
                                              std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy22 dequant requant");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto prefuzednode = match_pass.m_opnodes[functpara->GetFirstIndex()];
  auto pre_fuzed_node = prefuzednode.GetNode();
  if (FixpipeComm::CheckPeerOutNode(pre_fuzed_node, 1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  int input_index = functpara->GetParaIndex();
  ge::TensorDesc prenode_inputdesc;
  ge::TensorDesc prenode_outdesc;
  if (pre_fuzed_node->GetInputDesc(1, prenode_inputdesc) != ge::GRAPH_SUCCESS ||
      pre_fuzed_node->GetOutputDesc(0, prenode_outdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &fixpipenode = functpara->GetFixpipeNode();
  const char* set_m1_dequant_type = "SetM1Dequant";
  std::string set_m1_dequant_name = std::string(GNodeGetName(fixpipenode).GetString()) + "SetM1Dequant" + functpara->GetInputName();
  ge::GNode newopnode = ge::es::CompliantNodeBuilder(&graph)
                        .OpType(set_m1_dequant_type)
                        .Name(set_m1_dequant_name.c_str())
                        .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
  bool tmp_bool = true;
  newopnode.SetAttr(kAscendSingleOpAsc, tmp_bool);
  float offset = 0.0;
  if (GNodeGetAttr(pre_fuzed_node, ATTR_OFFSET, offset)) {
    float tmp_offset = offset;
    newopnode.SetAttr(kAscendOffsetAsc, tmp_offset);
    OPS_LOG_D("Fixpipe", "offset value = %f", offset);
  }
  int64_t tmp_dtype = static_cast<int>(prenode_outdesc.GetDataType());
  newopnode.SetAttr(kAscendOutDtypeAsc, tmp_dtype);

  ge::TensorDesc pre_tensor_desc = prenode_inputdesc;
  pre_tensor_desc.SetDataType(ge::DT_UINT64);
  newopnode.UpdateOutputDesc(0, pre_tensor_desc);
  newopnode.UpdateInputDesc(0, prenode_inputdesc);
  auto inputnode = pre_fuzed_node->GetInDataNodesAndPortIndexs(1);
  fixpipenode->UpdateInputDesc(input_index, pre_tensor_desc);
  if (graph.AddDataEdge(*inputnode.first, inputnode.second, newopnode, 0) != ge::GRAPH_SUCCESS ||
      graph.AddDataEdge(newopnode, 0, *fixpipenode, input_index) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

/**
 * Dequant + (relu) + (Quant)
 */
ge::graphStatus FixPipeAddInputStrategy23::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy23 add inputs begin.");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &pre_fuze_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  OPS_CHECK_NOTNULL(pre_fuze_node);
  OPS_LOG_D("Fixpipe", "Pre fuzed node is [%s, %s]", GNodeGetName(pre_fuze_node).GetString(), GNodeGetType(pre_fuze_node).GetString());
  if (FixpipeComm::CheckPeerOutNode(pre_fuze_node, 1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &fixpipe_node = functpara->GetFixpipeNode();
  OPS_CHECK_NOTNULL(fixpipe_node);
  OPS_LOG_D("Fixpipe", "Fixpipe node is [%s, %s]", GNodeGetName(fixpipe_node).GetString(), GNodeGetType(fixpipe_node).GetString());

  ge::GNodePtr post_fuze_node;
  if (FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex())) {
    post_fuze_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  }

  // create set quant scale host op
  std::string quant_scale_op_name = std::string(GNodeGetName(fixpipe_node).GetString()) + kSetQuantScale + functpara->GetInputName();
  ge::GNode quant_scale_node = CreateQuantScaleOpDesc(graph, quant_scale_op_name, pre_fuze_node,
                                                             post_fuze_node, nullptr);
  
  OPS_LOG_D("Fixpipe", "Node[%s, %s] has been add to graph.",
          GNodeGetName(quant_scale_node).GetString(), GNodeGetType(quant_scale_node).GetString());

  ge::TensorDesc quant_scale_outdesc;
  quant_scale_node.GetOutputDesc(0, quant_scale_outdesc);
  fixpipe_node->UpdateInputDesc(static_cast<uint32_t>(functpara->GetParaIndex()),
                                quant_scale_outdesc);
  // add edge
  int input_index = functpara->GetParaIndex();
  auto input_node = pre_fuze_node->GetInDataNodesAndPortIndexs(1);
  (void)graph.AddDataEdge(*input_node.first, input_node.second, quant_scale_node, 0);
  (void)graph.AddDataEdge(quant_scale_node, 0, *fixpipe_node, input_index);

  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy23 finish adding inputs.");
  return ge::GRAPH_SUCCESS;
}

/**
 * Dequant + LeakyRelu + (Quant)
 */
ge::graphStatus FixPipeAddInputStrategy24::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy24 add inputs begin.");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex()) ||
      !FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex())) {
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &pre_fuze_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  OPS_CHECK_NOTNULL(pre_fuze_node);
  OPS_LOG_D("Fixpipe", "Pre fuzed node is [%s, %s]", GNodeGetName(pre_fuze_node).GetString(), GNodeGetType(pre_fuze_node).GetString());
  if (FixpipeComm::CheckPeerOutNode(pre_fuze_node, 1) != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "Node[%s, %s] does not have input1.", GNodeGetName(pre_fuze_node).GetString(), GNodeGetType(pre_fuze_node).GetString());
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &post_fuze_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  OPS_CHECK_NOTNULL(post_fuze_node);

  const ge::GNodePtr &fixpipe_node = functpara->GetFixpipeNode();
  OPS_CHECK_NOTNULL(fixpipe_node);
  OPS_LOG_D("Fixpipe", "Fixpipe node is [%s, %s]", GNodeGetName(fixpipe_node).GetString(), GNodeGetType(fixpipe_node).GetString());

  // create vector muls host op
  std::string vec_mul_scalar_op_name = std::string(GNodeGetName(fixpipe_node).GetString()) + kVectorMulScalar + functpara->GetInputName();
  // Currently, only DT_FLOAT data formats are supported, and the UINT64 data format is additionally supported on 035.
  // If there is a scenario where UINT64 to DT_FLOAT data conversion exists, the accuracy impact is limited.
  ge::GNode vec_mul_scalar_node = CreateVectorMulScalarOpDesc(graph, vec_mul_scalar_op_name, pre_fuze_node,
                                                                     post_fuze_node, ge::DT_FLOAT);
  
  OPS_LOG_D("Fixpipe", "Node[%s, %s] has been add to graph.",
          GNodeGetName(vec_mul_scalar_node).GetString(), GNodeGetType(vec_mul_scalar_node).GetString());

  // add edge for vector_muls_node's inputs
  auto input_node = pre_fuze_node->GetInDataNodesAndPortIndexs(1);
  (void)graph.AddDataEdge(*input_node.first, input_node.second, vec_mul_scalar_node, 0);

  // create set quant scale host op
  std::string quant_scale_op_name = std::string(GNodeGetName(fixpipe_node).GetString()) + kSetQuantScale + functpara->GetInputName();
  ge::GNodePtr vec_mul_scalar_node_ptr;
  OPS_MAKE_SHARED(vec_mul_scalar_node_ptr = std::make_shared<ge::GNode>(vec_mul_scalar_node), return ge::GRAPH_FAILED);
  ge::GNode quant_scale_node = CreateQuantScaleOpDesc(graph, quant_scale_op_name, pre_fuze_node,
                                                             post_fuze_node,
                                                             vec_mul_scalar_node_ptr);
  
  OPS_LOG_D("Fixpipe", "Node[%s, %s] has been add to graph.",
          GNodeGetName(quant_scale_node).GetString(), GNodeGetType(quant_scale_node).GetString());
  ge::TensorDesc quant_scale_outdesc;
  quant_scale_node.GetOutputDesc(0, quant_scale_outdesc);
  fixpipe_node->UpdateInputDesc(static_cast<uint32_t>(functpara->GetParaIndex()),
                                quant_scale_outdesc);

  int input_index = functpara->GetParaIndex();
  (void)graph.AddDataEdge(*input_node.first, input_node.second, quant_scale_node, 0);
  (void)graph.AddDataEdge(vec_mul_scalar_node, 0, quant_scale_node, 1);
  (void)graph.AddDataEdge(quant_scale_node, 0, *fixpipe_node, input_index);
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy24 finish adding inputs.");
  return ge::GRAPH_SUCCESS;
}

/**
 * Dequant + PRelu + (Quant)
 */
ge::graphStatus FixPipeAddInputStrategy25::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy25 add inputs begin.");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex()) ||
      !FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex())) {
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &pre_fuze_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  OPS_CHECK_NOTNULL(pre_fuze_node);
  OPS_LOG_D("Fixpipe", "Pre fuzed node is [%s, %s]", GNodeGetName(pre_fuze_node).GetString(), GNodeGetType(pre_fuze_node).GetString());
  if (FixpipeComm::CheckPeerOutNode(pre_fuze_node, 1) != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "Node[%s, %s] does not have input1.", GNodeGetName(pre_fuze_node).GetString(), GNodeGetType(pre_fuze_node).GetString());
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &post_fuze_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  OPS_CHECK_NOTNULL(post_fuze_node);
  OPS_LOG_D("Fixpipe", "Post fuzed node is [%s, %s]", GNodeGetName(post_fuze_node).GetString(), GNodeGetType(post_fuze_node).GetString());
  if (FixpipeComm::CheckPeerOutNode(post_fuze_node, 1) != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "Node[%s, %s] does not have input1.", GNodeGetName(post_fuze_node).GetString(), GNodeGetType(post_fuze_node).GetString());
    return ge::GRAPH_FAILED;
  }

  const ge::GNodePtr &fixpipe_node = functpara->GetFixpipeNode();
  OPS_CHECK_NOTNULL(fixpipe_node);
  OPS_LOG_D("Fixpipe", "Fixpipe node is [%s, %s]", GNodeGetName(fixpipe_node).GetString(), GNodeGetType(fixpipe_node).GetString());

  UpdatePostNodeShape(*pre_fuze_node, *post_fuze_node);
  // create vector muls host op
  std::string vector_muls_op_name = std::string(GNodeGetName(fixpipe_node).GetString()) + "VectorMuls" + functpara->GetInputName();
  ge::GNode vector_muls_node = CreateVectorMulsOpDesc(graph, vector_muls_op_name, pre_fuze_node,
                                                             post_fuze_node);
  
  OPS_LOG_D("Fixpipe", "Node[%s, %s] has been add to graph.",
          GNodeGetName(vector_muls_node).GetString(), GNodeGetType(vector_muls_node).GetString());

  // add edge for vector_muls_node's inputs
  auto pre_input_node = pre_fuze_node->GetInDataNodesAndPortIndexs(1);
  auto post_input_node = post_fuze_node->GetInDataNodesAndPortIndexs(1);
  (void)graph.AddDataEdge(*pre_input_node.first, pre_input_node.second, vector_muls_node, 0);
  (void)graph.AddDataEdge(*post_input_node.first, post_input_node.second, vector_muls_node, 1);

  std::string quant_scale_op_name = std::string(GNodeGetName(fixpipe_node).GetString()) + kSetQuantScale + functpara->GetInputName();
    ge::GNodePtr vector_muls_node_ptr;
  OPS_MAKE_SHARED(vector_muls_node_ptr = std::make_shared<ge::GNode>(vector_muls_node), return ge::GRAPH_FAILED);
  ge::GNode quant_scale_node = CreateQuantScaleOpDesc(graph, quant_scale_op_name, pre_fuze_node,
                                                             post_fuze_node,
                                                             vector_muls_node_ptr);
  
  OPS_LOG_D("Fixpipe", "Node[%s, %s] has been add to graph.",
          GNodeGetName(quant_scale_node).GetString(), GNodeGetType(quant_scale_node).GetString());

  ge::TensorDesc quant_scale_outdesc;
  quant_scale_node.GetOutputDesc(0, quant_scale_outdesc);
  fixpipe_node->UpdateInputDesc(static_cast<uint32_t>(functpara->GetParaIndex()),
                                quant_scale_outdesc);

  int input_index = functpara->GetParaIndex();

  auto input_node = pre_fuze_node->GetInDataNodesAndPortIndexs(1);
  (void)graph.AddDataEdge(*input_node.first, input_node.second, quant_scale_node, 0);
  (void)graph.AddDataEdge(vector_muls_node, 0, quant_scale_node, 1);
  (void)graph.AddDataEdge(quant_scale_node, 0, *fixpipe_node, input_index);
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy25 finish adding inputs.");
  return ge::GRAPH_SUCCESS;
}

void FixPipeAddInputStrategy25::UpdateQuantScaleNodeShape(const ge::GNode &quant_scale_node) {
  ge::TensorDesc quant_scale_node_input0_opdesc;
  quant_scale_node.GetInputDesc(0, quant_scale_node_input0_opdesc);
  ge::TensorDesc quant_scale_node_input1_opdesc;
  quant_scale_node.GetInputDesc(1, quant_scale_node_input1_opdesc);
  ge::TensorDesc quant_scale_node_output_opdesc;
  quant_scale_node.GetOutputDesc(0, quant_scale_node_output_opdesc);
  const ge::Shape quant_scale_input0_shape = quant_scale_node_input0_opdesc.GetShape();
  const ge::Shape quant_scale_input1_shape = quant_scale_node_input1_opdesc.GetShape();
  if (FixpipeComm::IsShapeEqual(quant_scale_input0_shape, quant_scale_input1_shape)) {
    OPS_LOG_D("Fixpipe", "Node[%s, %s] x1 shape is same with x2, no need to broadcast.", GNodeGetName(quant_scale_node).GetString(),
            GNodeGetType(quant_scale_node).GetString());
    return;
  }
  auto broadcast_shape = GetBroadcastShape(quant_scale_input0_shape, quant_scale_input1_shape);
  if (!FixpipeComm::IsShapeEqual(quant_scale_input0_shape, broadcast_shape)) {
    OPS_LOG_D("Fixpipe", "Broadcast node[%s, %s] output shape from %s to new shape %s.", GNodeGetName(quant_scale_node).GetString(),
            GNodeGetType(quant_scale_node).GetString(), FixpipeComm::ShapeToString(quant_scale_input0_shape).c_str(),
            FixpipeComm::ShapeToString(broadcast_shape).c_str());
    quant_scale_node_output_opdesc.SetShape(broadcast_shape);
  }
}

void FixPipeAddInputStrategy25::UpdatePostNodeShape(const ge::GNode &pre_node, const ge::GNode &post_node) {
  ge::TensorDesc pre_tensor_desc;
  ge::TensorDesc post_tensor_desc;
  pre_node.GetInputDesc(1, pre_tensor_desc);
  post_node.GetInputDesc(1, post_tensor_desc);
  if (pre_tensor_desc.GetFormat() == post_tensor_desc.GetFormat()) {
    OPS_LOG_D("Fixpipe", "The format of pre_node's second input is same with post_node's second input.");
    return;
  }
  ge::Shape pre_shape = pre_tensor_desc.GetOriginShape();
  GetShapeAfterExpandDims(pre_node, pre_tensor_desc, pre_shape);
  ge::Shape post_shape = post_tensor_desc.GetOriginShape();
  GetShapeAfterExpandDims(post_node, post_tensor_desc, post_shape);
  if (!CanShapeBroadcast(pre_shape, post_shape)) {
    return;
  }
  ge::Shape new_shape;
  if (!GetShapeByFormat(post_tensor_desc.GetOriginFormat(),
                        pre_tensor_desc.GetFormat(), pre_tensor_desc.GetDataType(), post_shape, new_shape)) {
    OPS_LOG_D("Fixpipe", "Can not get new shape for node[%s, %s].", GNodeGetName(post_node).GetString(), GNodeGetType(post_node).GetString());
    return;
  }
  OPS_LOG_D("Fixpipe", "New shape for node[%s, %s]'s second input is [%s].",
          GNodeGetName(post_node).GetString(), GNodeGetType(post_node).GetString(), FixpipeComm::ShapeToString(new_shape).c_str());
  post_tensor_desc.SetShape(new_shape);
  post_tensor_desc.SetFormat(pre_tensor_desc.GetFormat());

  auto peer_node_pair = post_node.GetInDataNodesAndPortIndexs(1);
  auto peer_node = peer_node_pair.first;
  auto peer_port = peer_node_pair.second;
  if (peer_node == nullptr) {
    return;
  }
  OPS_LOG_D("Fixpipe", "Update shape of peer node[%s, %s]'s output[%d] to [%s].",
          GNodeGetName(peer_node).GetString(), GNodeGetType(peer_node).GetString(), peer_port, FixpipeComm::ShapeToString(new_shape).c_str());
  ge::TensorDesc peer_out_tensor_desc;
  peer_node->GetOutputDesc(peer_port, peer_out_tensor_desc);
  peer_out_tensor_desc.SetShape(new_shape);
  peer_out_tensor_desc.SetFormat(pre_tensor_desc.GetFormat());
}

/**
 * Dequant + Tanh/Elu/Sigmoid + (Quant)
 * Constant folding operators SetQuantScale are used to calculate offset and M1
 */
ge::graphStatus AddInputStrategyDequntLut::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  OPS_LOG_D("Fixpipe", "AddInputStrategyDequntLut add inputs begin.");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex()) ||
      !FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex())) {
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &pre_fuze_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  OPS_CHECK_NOTNULL(pre_fuze_node);
  OPS_LOG_D("Fixpipe", "Pre fuzed node is [%s, %s]", GNodeGetName(pre_fuze_node).GetString(), GNodeGetType(pre_fuze_node).GetString());
  if (FixpipeComm::CheckPeerOutNode(pre_fuze_node, 1) != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "Node[%s, %s] does not have input1.", GNodeGetName(pre_fuze_node).GetString(), GNodeGetType(pre_fuze_node).GetString());
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &post_fuze_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  OPS_CHECK_NOTNULL(post_fuze_node);
  const ge::GNodePtr &fixpipe_node = functpara->GetFixpipeNode();
  OPS_CHECK_NOTNULL(fixpipe_node);
  OPS_LOG_D("Fixpipe", "Fixpipe node is [%s, %s]", GNodeGetName(fixpipe_node).GetString(), GNodeGetType(fixpipe_node).GetString());

  // create set quant scale host op
  std::string quant_scale_op_name = std::string(GNodeGetName(fixpipe_node).GetString()) + kSetQuantScale + functpara->GetInputName();
  ge::GNode quant_scale_node = CreateQuantScaleOpDesc(graph, quant_scale_op_name, pre_fuze_node, post_fuze_node, nullptr);
  
  OPS_LOG_D("Fixpipe", "Node[%s, %s] has been add to graph.", GNodeGetName(quant_scale_node).GetString(),
          GNodeGetType(quant_scale_node).GetString());
  ge::TensorDesc quant_scale_outdesc;
  quant_scale_node.GetOutputDesc(0, quant_scale_outdesc);
  fixpipe_node->UpdateInputDesc(static_cast<uint32_t>(functpara->GetParaIndex()),
                                             quant_scale_outdesc);

  int input_index = functpara->GetParaIndex();
  auto input_node = pre_fuze_node->GetInDataNodesAndPortIndexs(1);
  (void)graph.AddDataEdge(*input_node.first, input_node.second, quant_scale_node, 0);
  (void)graph.AddDataEdge(quant_scale_node, 0, *fixpipe_node, input_index);
  OPS_LOG_D("Fixpipe", "AddInputStrategyDequntLut finish adding inputs.");
  return ge::GRAPH_SUCCESS;
}

// prelu+quant
ge::graphStatus FixPipeAddInputStrategy31::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                              const FixPipeFunctionParamPtr &functpara,
                                              std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy31 preact prlu+quant");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex()) ||
      !FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto second_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  if (second_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc input_desc;
  if (second_node->GetInputDesc(1, input_desc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  float scale = 0.0;
  float offset = 0.0;
  auto first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  if (first_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  GNodeGetAttr(first_node, ATTR_SCALE, scale);
  GNodeGetAttr(first_node, ATTR_OFFSET, offset);
  CreateAndUpdateVectorMulScalarInput<float>(graph, functpara, match_pass.m_opnodes[functpara->GetSecondIndex()],
                                            scale, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// prelu+dequant/requant
ge::graphStatus FixPipeAddInputStrategy32::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy32 preact prelu+derequant");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex()) ||
      !FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  CreateAndUpdateVectorMulsInput(graph, functpara, match_pass.m_opnodes[functpara->GetSecondIndex()],
                                 match_pass.m_opnodes[functpara->GetFirstIndex()], new_nodes);
  return ge::GRAPH_SUCCESS;
}

// prelu
ge::graphStatus FixPipeAddInputStrategy33::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy33 preact, prelu");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  CloneVectorInput(graph, match_pass.m_opnodes[functpara->GetFirstIndex()], functpara, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// lrelu+quant
ge::graphStatus FixPipeAddInputStrategy34::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy34 preact lrelu+quant");
  float scale = 0.0;
  float offset = 0.0;
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex()) ||
      !FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto first_node = GetQuantScaleOffset(match_pass, functpara->GetFirstIndex(), scale, offset);
  if (first_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto second_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  if (second_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  float attr_negative_slope_a = 0.0;
  GNodeGetAttr(second_node, ATTR_NEGATIVE_SLOPE, attr_negative_slope_a);
  attr_negative_slope_a *= scale;
  CreateAndUpdateSalarInput<float>(graph, functpara, attr_negative_slope_a, ge::DT_FLOAT, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// lrelu+dequant/requant
ge::graphStatus FixPipeAddInputStrategy35::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy35 preact lrelu+derequant");
  float attr_negative_slope_a = 0.0;
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex()) ||
      !FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto second_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  if (second_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  GNodeGetAttr(second_node, ATTR_NEGATIVE_SLOPE, attr_negative_slope_a);
  CreateAndUpdateVectorMulScalarInput<float>(graph, functpara, match_pass.m_opnodes[functpara->GetFirstIndex()],
                                              attr_negative_slope_a, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// lrlu
ge::graphStatus FixPipeAddInputStrategy36::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                              const FixPipeFunctionParamPtr &functpara,
                                              std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy36 preactalone, lrelu");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  if (first_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc output_desc;
  if (first_node->GetOutputDesc(0, output_desc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  float attr_negative_slope_a = 0.0;
  GNodeGetAttr(first_node, ATTR_NEGATIVE_SLOPE, attr_negative_slope_a);
  OPS_LOG_D("Fixpipe", "slop_a = %f index = %u op type = %s", attr_negative_slope_a, functpara->GetFirstIndex(),
          GNodeGetType(first_node).GetString());
  CreateAndUpdateSalarInput<float>(graph, functpara, attr_negative_slope_a, ge::DT_FLOAT, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// cast + prelu
ge::graphStatus FixPipeAddInputStrategy37::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy37 preact, cast+prelu");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex())) {
    return ge::GRAPH_FAILED;
  }
  CloneVectorInput(graph, match_pass.m_opnodes[functpara->GetSecondIndex()], functpara, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// cast + lrlu
ge::graphStatus FixPipeAddInputStrategy38::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                               const FixPipeFunctionParamPtr &functpara,
                                               std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy38 preactalone, cast+lrelu");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto second_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  if (second_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc output_desc;
  if (second_node->GetOutputDesc(0, output_desc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  float attr_negative_slope_a = 0.0;
  GNodeGetAttr(second_node, ATTR_NEGATIVE_SLOPE, attr_negative_slope_a);
  CreateAndUpdateSalarInput<float>(graph, functpara, attr_negative_slope_a, ge::DT_FLOAT, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// relu6
ge::graphStatus FixPipeAddInputStrategy41::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                              const FixPipeFunctionParamPtr &functpara,
                                              std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy41 clipvalue, relu6");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  if (first_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc firstnode_outputdesc;
  if (first_node->GetOutputDesc(0, firstnode_outputdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  ge::DataType dst_datatype = ge::DT_FLOAT16;
  SetClipValue6(graph, functpara, dst_datatype, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// quant + relu6
ge::graphStatus FixPipeAddInputStrategy42::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy42 clipvalue relu6+ quant");
  float scale = 0.0;
  float offset = 0.0;
  auto first_node = GetQuantScaleOffset(match_pass, functpara->GetFirstIndex(), scale, offset);
  if (first_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto second_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  if (second_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc secondnode_outputdesc;
  if (second_node->GetOutputDesc(0, secondnode_outputdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  ge::DataType dst_datatype = ge::DT_FLOAT16;
  float value = kRelu6Value * scale + offset;
  uint16_t value_tmp = Fp32ToFp16(value);
  CreateAndUpdateSalarInput<uint16_t>(graph, functpara, value_tmp, dst_datatype, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// relu6+dequant
ge::graphStatus FixPipeAddInputStrategy43::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                              const FixPipeFunctionParamPtr &functpara,
                                              std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy43 clipvalue relu6+dequant");
  float offset = 0.0;
  float scale = 0.0;
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  if (first_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc dequantoutdesc;
  if (first_node->GetOutputDesc(0, dequantoutdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  ge::DataType dst_datatype = ge::DT_FLOAT16;
  if (GNodeGetAttr(first_node, ATTR_SCALE, scale) == ge::GRAPH_SUCCESS &&
      GNodeGetAttr(first_node, ATTR_OFFSET, offset) == ge::GRAPH_SUCCESS) {
    float value = kRelu6Value * scale + offset;
    uint16_t value_tmp = Fp32ToFp16(value);
    CreateAndUpdateSalarInput<uint16_t>(graph, functpara, value_tmp, dst_datatype, new_nodes);
  } else {
    SetClipValue6(graph, functpara, dst_datatype, new_nodes);
  }
  return ge::GRAPH_SUCCESS;
}

// cast+relu6
ge::graphStatus FixPipeAddInputStrategy44::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                              const FixPipeFunctionParamPtr &functpara,
                                              std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy44 clipvalue, cast+relu6");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto second_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  if (second_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc second_outdesc;
  if (second_node->GetOutputDesc(0, second_outdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  ge::DataType dst_datatype = ge::DT_FLOAT16;
  SetClipValue6(graph, functpara, dst_datatype, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// quant
ge::graphStatus FixPipeAddInputStrategy51::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy51 post quant");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    OPS_LOG_W("Fixpipe", "First node index[%u] is invalid for size of matched nodes is [%zu].",
            functpara->GetFirstIndex(), match_pass.m_opnodes.size());
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  ge::graphStatus ret = CreateScalarInputNode(graph, functpara, first_node, kInputStrategy51);
  OPS_LOG_D("Fixpipe", "Finish FixPipeAddInputStrategy51 post quant, ret is [%u].", ret);
  return ret;
}

// tanh/elh/sigmoid  + quant
ge::graphStatus FixPipeAddInputStrategy52::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  (void)new_nodes;
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy52 post quant");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    OPS_LOG_W("Fixpipe", "First node index[%u] is invalid for size of matched nodes is [%zu].",
            functpara->GetFirstIndex(), match_pass.m_opnodes.size());
    return ge::GRAPH_FAILED;
  }
  const ge::GNodePtr &first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  OPS_CHECK_NOTNULL(first_node);
  const auto in_nodes = first_node->GetInDataNodesAndPortIndexs(0);
  if (first_node->GetInputsSize() == 0 || kLutOpTypeSet.count(std::string(GNodeGetType(in_nodes.first).GetString())) == 0) {
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus ret = CreateScalarInputNode(graph, functpara, first_node, kInputStrategy52);
  OPS_LOG_D("Fixpipe", "Finish FixPipeAddInputStrategy52 post quant, ret is [%u].", ret);
  return ret;
}

// prelu+quant
ge::graphStatus FixPipeAddInputStrategy61::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy61 postact prlu+quant");
  float scale = 0.0;
  float offset = 0.0;
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex()) ||
      !FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto second_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  if (second_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  GNodeGetAttr(second_node, ATTR_SCALE, scale);
  GNodeGetAttr(second_node, ATTR_OFFSET, offset);
  CreateAndUpdateVectorMulScalarInput<float>(graph, functpara, match_pass.m_opnodes[functpara->GetFirstIndex()], scale,
                                              new_nodes);
  return ge::GRAPH_SUCCESS;
}

// prelu
ge::graphStatus FixPipeAddInputStrategy62::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy62 postactalone, prelu");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  CloneVectorInput(graph, match_pass.m_opnodes[functpara->GetFirstIndex()], functpara, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// lrelu+quant
ge::graphStatus FixPipeAddInputStrategy63::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy63 postact lrelu+quant");
  float scale = 0.0;
  float offset = 0.0;
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex()) ||
      !FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  if (first_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto second_node = GetQuantScaleOffset(match_pass, functpara->GetSecondIndex(), scale, offset);
  if (second_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  float attr_negative_slope_a = 0.0;
  GNodeGetAttr(first_node, ATTR_NEGATIVE_SLOPE, attr_negative_slope_a);
  attr_negative_slope_a *= scale;
  CreateAndUpdateSalarInput<float>(graph, functpara, attr_negative_slope_a, ge::DT_FLOAT, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// lrelu
ge::graphStatus FixPipeAddInputStrategy64::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy64 postactalone, lrelu");
  float attr_negative_slope_a = 0.0;
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  if (first_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  GNodeGetAttr(first_node, ATTR_NEGATIVE_SLOPE, attr_negative_slope_a);
  CreateAndUpdateSalarInput<float>(graph, functpara, attr_negative_slope_a, ge::DT_FLOAT, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// relu6
ge::graphStatus FixPipeAddInputStrategy71::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy71 clip1value, relu6");
  return DoWithClipReluInputWithSingleRelu6(graph, match_pass, functpara, new_nodes);
}

// relu6+quant
ge::graphStatus FixPipeAddInputStrategy72::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                              const FixPipeFunctionParamPtr &functpara,
                                              std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy72 clip1value relu6 + quant");
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetSecondIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto second_node = match_pass.m_opnodes[functpara->GetSecondIndex()].GetNode();
  if (second_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc secondnode_outputdesc;
  if (second_node->GetOutputDesc(0, secondnode_outputdesc) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  ge::DataType dst_datatype = secondnode_outputdesc.GetDataType();
  SetClipValue6(graph, functpara, dst_datatype, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// eltwise
ge::graphStatus FixPipeAddInputStrategy81::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy81 eltwise scale");
  float scale_tmp = 0.0;
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  if (first_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  (void)GNodeGetAttr(first_node, ATTR_SCALE, scale_tmp);
  ops::fp16_t insert_value(scale_tmp);
  CreateAndUpdateSalarInput<ops::fp16_t>(graph, functpara, insert_value, ge::DT_FLOAT16, new_nodes);
  return ge::GRAPH_SUCCESS;
}

// eltwise
ge::graphStatus FixPipeAddInputStrategy91::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                             const FixPipeFunctionParamPtr &functpara,
                                             std::vector<ge::GNodePtr> &new_nodes) const {
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategy91 eltwise offset");
  float offset_a = 0.0;
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes, functpara->GetFirstIndex())) {
    return ge::GRAPH_FAILED;
  }
  auto first_node = match_pass.m_opnodes[functpara->GetFirstIndex()].GetNode();
  if (first_node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  (void)GNodeGetAttr(first_node, ATTR_OFFSET, offset_a);
  offset_a = offset_a > 128 ? 128 : offset_a;
  offset_a = offset_a < -127 ? -127 : offset_a;
  ops::fp16_t revert_offset(-offset_a);
  if (functpara->GetDataType() == ge::DT_INT8 ||
      functpara->GetDataType() == ge::DT_INT4) {
    CreateAndUpdateSalarInput<ops::fp16_t>(graph, functpara, revert_offset, ge::DT_FLOAT16, new_nodes);
  } else {
    CreateAndUpdateSalarInput<float>(graph, functpara, offset_a, ge::DT_FLOAT, new_nodes);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixPipeAddInputStrategyDefault::DoAddInput(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                                  const FixPipeFunctionParamPtr &functpara,
                                                  std::vector<ge::GNodePtr> &new_nodes) const {
  (void)graph;
  (void)match_pass;
  (void)functpara;
  (void)new_nodes;
  OPS_LOG_D("Fixpipe", "FixPipeAddInputStrategyDefault");
  return ge::GRAPH_SUCCESS;
}
}  // namespace ops
