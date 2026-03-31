/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cube_utils/cube_common.h"
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <securec.h>

namespace ops {
using std::queue;
using std::set;
using std::unordered_set;
namespace {
const std::string kFixpipeConfigKey = "Intrinsic_fix_pipe_";
const std::string kUnitListName = "unit_list";
const std::string kFuncList = "_func_list";
const std::string kDependUnits = "_depend_unit";
const std::string kClipRelu = "clip_relu";
constexpr char const *kCast = "Cast";
constexpr char const *kRelu = "Relu";
constexpr char const *kRelu6 = "Relu6";
constexpr char const *kTransData = "TransData";
constexpr char const *kConst = "Const";
constexpr uint32_t kMaxConfigMatchSize = 5;
constexpr uint32_t kTranConfigIndex = 3;
constexpr uint32_t kTranConfigSecondIndex = 4;
const ge::AscendString kAscendSatuateModeAsc("ge.satuateMode");
}
const std::string kPatternConv = "conv_pattern";
const std::string kPatternRelu = "relu_pattern";
const std::string kPatternQuant = "quant_pattern";
const std::string kConv2DBackpropInputD = "Conv2DBackpropInputD";
const std::string kConv2DBackpropFilterD = "Conv2DBackpropFilterD";
const std::string kConv3D = "Conv3D";
const std::string kConv3DBackpropInputD = "Conv3DBackpropInputD";
const std::string kConv3DTransposeD = "Conv3DTransposeD";
const std::string kDepthwiseConv2D = "DepthwiseConv2D";
const std::string kMatMul = "MatMul";
const std::string kMatMulV2 = "MatMulV2";
const std::string kMatMulV3 = "MatMulV3";
const std::string kBatchMatMulV2 = "BatchMatMulV2";
const std::string kBatchMatMulV3 = "BatchMatMulV3";
const std::string kDepthwiseConv2DBackpropFilterD = "DepthwiseConv2DBackpropFilterD";
const std::string kDepthwiseConv2DBackpropInputD = "DepthwiseConv2DBackpropInputD";
const std::string kMerge = "Merge";
const std::string kAscendDequant = "AscendDequant";
const std::string kAscendQuant = "AscendQuant";
const std::string kAscendAntiQuant = "AscendAntiQuant";
const std::string kAdd = "Add";
const std::string kSub = "Sub";
const std::string kLeakyRelu = "LeakyRelu";
const std::string kPRelu = "PRelu";
const std::string kSigmoid = "Sigmoid";
const std::string kElu = "Elu";
const std::string kTanh = "Tanh";
const std::string kFixPipe = "FixPipe";
const std::string kFusionOpList = "fusion_op_list";
const std::string kUnitList = "unit_list";
const std::string kEltwiseMode = "eltwise_mode";
const std::string kCubeUnit = "CUBE_UNIT";
const std::string kPreConv = "pre_conv";
const std::string kPreAct = "pre_act";
const std::string kPostEltwise = "post_eltwise";
const std::string kPostAct = "post_act";
const std::string kPostQuant = "post_quant";
const std::string kPostTransform = "post_transform";
const std::string kAttrScale = "scalarattr";
const std::string kAttrNeedJudgeDtype = "_need_judge_dtype";
const std::string kAttrEltwiseMode = "mode";
const std::string kAttrFakeCubeNode = "_fake_cube_node";
const std::string kAttrNameWeight = "value";
const std::string kAttrNameOriginalInput = "_is_original_input";
const std::string kAttrNameReshapeTypeMask = "_reshape_type_mask";
const std::string kAttrDumpAble = "_dump_able";

const std::string CONV2D = "Conv2D";
const std::string DECONVOLUTION = "Deconvolution";
const std::string MATMULV2OP = "MatMulV2";
const std::string DEPTHWISECONV2D = "DepthwiseConv2D";
const std::string kFullyConnection = "FullyConnection";
const std::string kConv2DTranspose = "Conv2DTranspose";
const std::string kConv2DTransposeD = "Conv2DTransposeD";
const std::string kBatchMatMul = "BatchMatMul";
const std::string kConv2DCompress = "Conv2DCompress";
const std::string kFullyConnectionCompress = "FullyConnectionCompress";
const std::string kMatMulV2Compress = "MatMulV2Compress";
const std::string kConv2DTransposeDCompress = "Conv2DTransposeDCompress";
const std::string kBatchMatMulCompress = "BatchMatMulCompress";
const std::string CAST = "Cast";
const std::string TRANSDATA = "TransData";
const std::string CONSTANT = "Const";
const std::string RELU = "Relu";
const std::string RELU6 = "Relu6";
const std::string LEAKY_RELU = "LeakyRelu";
const std::string ASCEND_QUANT = "AscendQuant";
const std::string ASCEND_DEQUANT = "AscendDequant";
const std::string kAscendRequant = "AscendRequant";
const std::string kInfNan = "INF_NAN";
const std::string SWITCH = "Switch";
const std::string ATTR_NEGATIVE_SLOPE = "negative_slope";
const std::string ELTWISE = "Eltwise";
const std::string kSupportFixPipeAbility = "support_fixpipe_ability";
const std::string kNotSupportFixpipeFusion = "_not_support_fixpipe_fusion";
const std::string TRANSDATA_INPUT_NAME = "src";
const std::string TRANSDATA_OUTPUT_NAME = "dst";
const std::string UNSQUEEZE_V2_INPUT_NAME = "x";
const std::string UNSQUEEZE_V2_OUTPUT_NAME = "y";
const std::string ATTR_NAME_SRC_FORMAT = "src_format";
const std::string ATTR_NAME_DST_FORMAT = "dst_format";
const std::string UNSQUEEZE_V2 = "UnsqueezeV2";
const std::string AXIS_ATTR_NAME = "axis";
const std::string ATTR_SCALE = "scale"; 
const std::string ATTR_OFFSET = "offset";

const size_t kFixpipeNodeLimited = 2;
const size_t kPassMinSize = 2;
const size_t kMaxOpNmaLen = 512U;
const uint32_t kPassFlag = 2;
const uint32_t kMaxDepth = 7;
const uint32_t kEltMaxDepth = 3;
const uint32_t kNumber0 = 0;
const uint32_t kNumber1 = 1;
const uint32_t kNumber2 = 2;
const int64_t SHAPE_NUMBER_16 = 16;
const float kRelu6Value = 6.0;

// AscendString 常量定义（用于比较操作，避免重复构造）
// 命名规则：添加 Asc 后缀以区分 std::string 版本
const ge::AscendString kAscendEltwiseAsc(ELTWISE.c_str());
const ge::AscendString kAscendAddAsc(kAdd.c_str());
const ge::AscendString kAscendConv2DAsc(CONV2D.c_str());
const ge::AscendString kAscendMergeAsc(kMerge.c_str());
const ge::AscendString kAscendFixPipeAsc(kFixPipe.c_str());
const ge::AscendString kAscendLeakyReluAsc(LEAKY_RELU.c_str());
const ge::AscendString kAscendPReluAsc(kPRelu.c_str());
const ge::AscendString kAscendRelu6Asc(RELU6.c_str());
const ge::AscendString kAscendQuantAsc(kAscendQuant.c_str());
const ge::AscendString kAscendDequantAsc(kAscendDequant.c_str());
const ge::AscendString kAscendAntiQuantAsc(kAscendAntiQuant.c_str());
const ge::AscendString kAscendRequantAsc(kAscendRequant.c_str());
const ge::AscendString kAscendConstantAsc(CONSTANT.c_str());
const ge::AscendString kAscendTransdataAsc(TRANSDATA.c_str());
const ge::AscendString kAscendSwitchAsc(SWITCH.c_str());
const ge::AscendString kAscendOffsetAsc(ATTR_OFFSET.c_str());
const ge::AscendString kAscendNegativeSlopeAsc(ATTR_NEGATIVE_SLOPE.c_str());
const ge::AscendString kAscendReshapeTypeMaskAsc(kAttrNameReshapeTypeMask.c_str());
const ge::AscendString kAscendInfNanAsc(kInfNan.c_str());
const ge::AscendString kAscendNameWeightAsc(kAttrNameWeight.c_str());
const ge::AscendString kAscendOriginalInputAsc(kAttrNameOriginalInput.c_str());
const ge::AscendString kAscendScaleAsc(kAttrScale.c_str());
const ge::AscendString kAscendNeedJudgeDtypeAsc(kAttrNeedJudgeDtype.c_str());
const ge::AscendString kAscendFusionOpListAsc(kFusionOpList.c_str());
const ge::AscendString kAscendUnitListAsc(kUnitList.c_str());
const ge::AscendString kAscendEltwiseModeAsc(kEltwiseMode.c_str());

const std::vector<std::string> kSupportFixpipeCubeTypeVec = {kDepthwiseConv2D,
                                                             CONV2D,
                                                             kConv3D,
                                                             kMatMul,
                                                             kMatMulV2,
                                                             kMatMulV3,
                                                             kConv2DBackpropInputD,
                                                             DECONVOLUTION,
                                                             kConv2DTransposeD,
                                                             kConv2DTranspose,
                                                             kConv2DBackpropFilterD,
                                                             kConv3DTransposeD,
                                                             kConv3DBackpropInputD,
                                                             kBatchMatMul,
                                                             kBatchMatMulV2,
                                                             kBatchMatMulV3,
                                                             kFullyConnection,
                                                             kDepthwiseConv2DBackpropFilterD,
                                                             kDepthwiseConv2DBackpropInputD,
                                                             kMerge};
const std::unordered_set<std::string> kSupportMultipleOutputCubeSet = {CONV2D,
                                                                       kDepthwiseConv2D};
const std::unordered_set<std::string> kMatMulSet = {kMatMul,
                                                    kMatMulV2,
                                                    kMatMulV3,
                                                    kBatchMatMul,
                                                    kBatchMatMulV2};
const std::unordered_set<std::string> kAtomicWriteCubeSet = {kConv2DBackpropFilterD,
                                                             kDepthwiseConv2DBackpropFilterD};
const std::unordered_set<std::string> kCandidateReluTypes = {kPRelu,
                                                             RELU6,
                                                             LEAKY_RELU,
                                                             RELU};
const std::unordered_set<ge::Format> kMatmulNotSupportFixpipeFormatSet = {ge::FORMAT_ND,
                                                                          ge::FORMAT_NCHW,
                                                                          ge::FORMAT_NHWC,
                                                                          ge::FORMAT_HWCN,
                                                                          ge::FORMAT_NCDHW};
const std::unordered_set<std::string> CUBE_ABLE_COMPRESS_SET = {CONV2D,
                                                                kMatMulV2,
                                                                kFullyConnection,
                                                                kConv2DTransposeD};
const unordered_set<std::string> kCubeCompressOpList = {kConv2DCompress,
                                                        kFullyConnectionCompress,
                                                        kMatMulV2Compress,
                                                        kConv2DTransposeDCompress,
                                                        kBatchMatMulCompress};
const std::map<std::string, std::string> kFixpipeInstruction2OpTypeMap{{"requant", kAscendRequant},
                                                                       {"quant", kAscendQuant},
                                                                       {"dequant", kAscendDequant},
                                                                       {"cast", kCast},
                                                                       {"add", kAdd},
                                                                       {"nz2nd", kTransData},
                                                                       {"anti_sub", kAscendAntiQuant},
                                                                       {"anti_add", kAscendAntiQuant},
                                                                       {"sub", kSub},
                                                                       {"normal_relu", kRelu},
                                                                       {"scalar_relu", kLeakyRelu},
                                                                       {"vector_relu", kPRelu},
                                                                       {"clip_relu", kRelu6},
                                                                       {"sigmoid", kSigmoid},
                                                                       {"elu", kElu},
                                                                       {"tanh", kTanh}};
const std::array<FixpipeAbilityAttr,
    static_cast<size_t>(FixpipeAbilityType::FixpipeAbilityTypeBottom)> kFixpipeAbilityAttrs {
        kSupportPostEltwiseBroadcast, kSupportMultipleOutput, kUseGmAtomicAdd, kNodeCantAccess
};

const std::vector<std::string> kSupportFixpipeUtilFuncCubeTypeVec = {CONV2D};

FixPipeUnit::~FixPipeUnit() {
  dependunits_.clear();
  dependunitsindex_.clear();
  opnodes_.clear();
}

FixPipeNodeInfo::~FixPipeNodeInfo() {
  node_tofixpipelist_.clear();
}

ge::DataType FixpipeComm::TranferString(const std::string &configstr) {
  if (configstr == "s4") {
    return ge::DT_INT4;
  } else if (configstr == "s8") {
    return ge::DT_INT8;
  } else if (configstr == "s16") {
    return ge::DT_INT16;
  } else if (configstr == "s32") {
    return ge::DT_INT32;
  } else if (configstr == "s64") {
    return ge::DT_INT64;
  } else if (configstr == "u8") {
    return ge::DT_UINT8;
  } else if (configstr == "u16") {
    return ge::DT_UINT16;
  } else if (configstr == "u32") {
    return ge::DT_UINT32;
  } else if (configstr == "u64") {
    return ge::DT_UINT64;
  } else if (configstr == "f16") {
    return ge::DT_FLOAT16;
  } else if (configstr == "bf16") {
    return ge::DT_BF16;
  } else if (configstr == "f32") {
    return ge::DT_FLOAT;
  }
  return ge::DT_UNDEFINED;
}

CONFIGDTYPE FixpipeComm::TransFerConfig2Dtype(const std::string &configstr) {
  CONFIGDTYPE ret;
  ret.has_output_dtype = false;
  ret.input_dtype = ge::DT_UNDEFINED;
  ret.output_dtype = ge::DT_UNDEFINED;
  if (configstr.size() >= kMaxConfigMatchSize) {
    std::string tmpsubstr = configstr.substr(0, kTranConfigIndex);
    if (tmpsubstr == "f32" || tmpsubstr == "f16" || tmpsubstr == "s32") {
      if (configstr.at(kTranConfigIndex) == '2') {
        std::string secondstr = configstr.substr(kTranConfigSecondIndex);
        ret.has_output_dtype = true;
        ret.input_dtype = TranferString(tmpsubstr);
        ret.output_dtype = TranferString(secondstr);
      } else {
        ret.input_dtype = TranferString(tmpsubstr);
      }
    }
  } else {
    ret.input_dtype = TranferString(configstr);
  }
  return ret;
}

bool FixpipeComm::ReadPlatFormConfig(const ge::CustomPassContext &context, const bool &skip_trans,
    std::vector<std::string> &unit_list, std::map<std::string, std::vector<std::string>> &depends_list,
    std::map<std::string, std::map<std::string, std::vector<CONFIGDTYPE>>> &fixpipe_map) {
  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos optional_infos;
  if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
                                          platform_infos, optional_infos) != ge::GRAPH_SUCCESS) {
    OPS_LOG_W("Fixpipe", "Fail to get platform info without soc version.");
    return false;
  }
  // inputmap first key is startwith "Intrinsic_fix_pipe_" second fixpipe config value
  std::map<std::string, std::vector<std::string>> input_map = optional_infos.GetFixPipeDtypeMap();
  for (auto &iter : input_map[kFixpipeConfigKey + kUnitListName]) {
    if (skip_trans && iter == kPostTransform) {
      continue;
    }
    std::vector<std::string> &oplist = input_map[kFixpipeConfigKey + iter + kFuncList];
    std::map<std::string, std::vector<CONFIGDTYPE>> outputinnermap;
    for (auto &opname : oplist) {
      if (opname == kClipRelu) {
        ge::AscendString satuate_mode;
        ge::graphStatus status = context.GetOptionValue(kAscendSatuateModeAsc, satuate_mode);
        OPS_LOG_D("Fixpipe", "The option value[ge.satuateMode] in ge context is %s.", satuate_mode.GetString());
        if (status == ge::GRAPH_SUCCESS && satuate_mode == kAscendInfNanAsc) {
          continue;
        }
      }
      std::string outstring;
      auto item = kFixpipeInstruction2OpTypeMap.find(opname);
      if (item != kFixpipeInstruction2OpTypeMap.end()) {
        outstring = item->second;
      }
      vector<std::string> &dtypelist = input_map[kFixpipeConfigKey + iter + "_" + opname];
      std::vector<CONFIGDTYPE> outputsinner;
      for (auto &dtype : dtypelist) {
        CONFIGDTYPE dtypestr = TransFerConfig2Dtype(dtype);
        outputsinner.push_back(dtypestr);
      }
      outputinnermap.emplace(make_pair(outstring, outputsinner));
    }
    std::vector<std::string> &depends_units = input_map[kFixpipeConfigKey + iter + kDependUnits];
    depends_list.emplace(make_pair(iter, depends_units));
    fixpipe_map.emplace(make_pair(iter, outputinnermap));
    unit_list.push_back(iter);
  }
  return true;
}

ge::graphStatus FixpipeComm::CheckPeerOutNode(const ge::GNodePtr &vectornode, const uint32_t &input_index) {
  OPS_LOG_D("Fixpipe", "name = %s type = %s index = %d", GNodeGetName(vectornode).GetString(), GNodeGetType(vectornode).GetString(), input_index);
  ge::TensorDesc input_desc;
  if (vectornode->GetInputDesc(input_index, input_desc) != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "GetInputDesc() = failed");
    return ge::GRAPH_FAILED;
  }

  auto cube_node = vectornode->GetInDataNodesAndPortIndexs(input_index);
  if (cube_node.first == nullptr) {
    OPS_LOG_D("Fixpipe", "GetInDataNodes() = null");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::GNodePtr FixpipeComm::GetConstNode(const ge::GNodePtr &vectornode, uint32_t &depth) {
  if (vectornode == nullptr) {
    return nullptr;
  }
  OPS_LOG_D("Fixpipe", "name = %s type = %s depth = %d", GNodeGetName(vectornode).GetString(), GNodeGetType(vectornode).GetString(), depth);
  if (depth >= kMaxDepth) {
    return nullptr;
  }
  if (GNodeGetType(vectornode) == kAscendConstantAsc) {
    OPS_LOG_D("Fixpipe", "name = %s type = %s is constant", GNodeGetName(vectornode).GetString(), GNodeGetType(vectornode).GetString());
    return vectornode;
  } else {
    if (CheckPeerOutNode(vectornode, 0) == ge::GRAPH_SUCCESS) {
      depth++;
      auto node = vectornode->GetInDataNodesAndPortIndexs(0);
      OPS_LOG_D("Fixpipe", "name = %s type = %s depth = %d", GNodeGetName(node.first).GetString(), GNodeGetType(node.first).GetString(), depth);
      return GetConstNode(node.first, depth);
    }
    return nullptr;
  }
}

ge::GNodePtr FixpipeComm::GetConstNode(const ge::GNodePtr &vectornode) {
  if (CheckPeerOutNode(vectornode, 1) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  auto input_node = vectornode->GetInDataNodesAndPortIndexs(1);
  if (GNodeGetType(input_node.first) == kAscendConstantAsc) {
    return input_node.first;
  }
  uint32_t depth = 0;
  return GetConstNode(input_node.first, depth);
}

template <typename T>
bool FixpipeComm::JudgedataImpl(uint8_t *data, const size_t &data_size) {
  T *shape_data = const_cast<T *>(reinterpret_cast<const T *>(data));
  for (size_t i = 0; i < static_cast<size_t>(data_size / sizeof(T)); i++) {
    if (shape_data[i] < 0) {
      OPS_LOG_D("Fixpipe", "shape_data = %f", static_cast<float>(shape_data[i]));
      return false;
    }
  }
  return true;
}

bool FixpipeComm::JudegedataUInt64(uint8_t *data, const size_t &data_size) {
  uint64_t *shape_data = const_cast<uint64_t *>(reinterpret_cast<const uint64_t *>(data));
  for (size_t i = 0; i < static_cast<size_t>(data_size / sizeof(uint64_t)); i++) {
    uint32_t scale_deq = GET_DEQUANT_SCALE_DEQ(shape_data[i]);
    float scale = 0;
    if (memcpy_s(&scale, sizeof(scale), &scale_deq, sizeof(uint32_t)) != 0) {
      return false;
    }
    if (scale < 0) {
      return false;
    }
  }
  return true;
}

bool FixpipeComm::JudgedataFp16(uint8_t *data, const size_t &data_size) {
  ops::fp16_t *shape_data = const_cast<ops::fp16_t *>(reinterpret_cast<const ops::fp16_t *>(data));
  for (size_t i = 0; i < (data_size / sizeof(int16_t)); i++) {
    if (shape_data[i].ToInt16() < 0) {
      return false;
    }
  }
  return true;
}

bool FixpipeComm::Judgedata(uint8_t *data, const size_t &data_size, const ge::DataType &data_type) {
  if (data == nullptr || data_size == 0) {
    return false;
  }
  if (data_type == ge::DT_UINT64) {
    return JudegedataUInt64(data, data_size);
  } else if (data_type == ge::DT_FLOAT16) {
    return JudgedataFp16(data, data_size);
  } else {
    switch (data_type) {
      case ge::DT_INT32:
        return JudgedataImpl<int32_t>(data, data_size);
      case ge::DT_FLOAT:
        return JudgedataImpl<float>(data, data_size);
      case ge::DT_DOUBLE:
        return JudgedataImpl<double>(data, data_size);
      case ge::DT_INT8:
        return JudgedataImpl<int8_t>(data, data_size);
      case ge::DT_INT16:
        return JudgedataImpl<int16_t>(data, data_size);
      case ge::DT_BOOL:
        return JudgedataImpl<int8_t>(data, data_size);
      case ge::DT_UINT8:
      case ge::DT_UINT16:
      case ge::DT_UINT32:
        return true;
      default:
        return false;
    }
  }
}

bool FixpipeComm::CheckConstValueData(const ge::GNodePtr &vectornode) {
  auto const_node = GetConstNode(vectornode);
  if (const_node == nullptr) {
    OPS_LOG_D("Fixpipe", "Node[%s, %s] does not have const input.", GNodeGetName(vectornode).GetString(), GNodeGetType(vectornode).GetString());
    return false;
  }
  ge::TensorDesc cur_tensor_desc;
  if (const_node->GetOutputDesc(0, cur_tensor_desc) != ge::GRAPH_SUCCESS) {
    return false;
  }
  ge::Tensor weight_value;
  GNodeGetAttr(const_node, kAttrNameWeight, weight_value);
  auto datatype = cur_tensor_desc.GetDataType();
  OPS_LOG_D("Fixpipe", "datatype = %d", static_cast<uint32_t>(datatype));
  return Judgedata(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(weight_value.GetData())),
                   weight_value.GetSize(), datatype);
}

bool FixpipeComm::CheckIsInVector(const std::vector<FixPipeNodeInfo> &m_opnodes, const uint32_t &index) {
  if (m_opnodes.size() == 0) {
    return false;
  }
  if (index <= (m_opnodes.size() - 1)) {
    return true;
  }
  return false;
}

ge::GNodePtr FixpipeComm::GetMergeNodeByCube(const ge::GNodePtr &node_ptr) {
  ge::GNodePtr merge_node = GetMergeNode(node_ptr);
  if (merge_node == nullptr) {
    return nullptr;
  }
  if (!CheckMergeInput(merge_node)) {
    return nullptr;
  }
  return merge_node;
}

ge::GNodePtr FixpipeComm::GetMergeNode(const ge::GNodePtr &node_ptr) {
  for (size_t idx = 0; idx < node_ptr->GetOutputsSize(); ++idx) {
    auto outputNodesPairs = node_ptr->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &outputPair : outputNodesPairs) {
      if (GNodeGetType(outputPair.first) == kAscendMergeAsc) {
        return outputPair.first;
      }
    }
  }

  return nullptr;
}

bool FixpipeComm::CheckMergeInput(const ge::GNodePtr &merge_node) {
  if (CheckPeerOutNode(merge_node, 0) != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "name = %s type = %s hast input 0node ", GNodeGetName(merge_node).GetString(), GNodeGetType(merge_node).GetString());
    return false;
  }
  if (CheckPeerOutNode(merge_node, 1) != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "name = %s type = %s hast input 1 node", GNodeGetName(merge_node).GetString(), GNodeGetType(merge_node).GetString());
    return false;
  }
  auto inputnode0 = merge_node->GetInDataNodesAndPortIndexs(0);
  auto inputnode1 = merge_node->GetInDataNodesAndPortIndexs(1);
  if (CUBE_ABLE_COMPRESS_SET.count(std::string(GNodeGetType(inputnode0.first).GetString())) == 0 &&
      kCubeCompressOpList.count(std::string(GNodeGetType(inputnode0.first).GetString())) == 0) {
    return false;
  }
  if (CUBE_ABLE_COMPRESS_SET.count(std::string(GNodeGetType(inputnode1.first).GetString())) == 0 &&
      kCubeCompressOpList.count(std::string(GNodeGetType(inputnode1.first).GetString())) == 0) {
    return false;
  }
  OPS_LOG_D("Fixpipe", "GetFixpipeCubeType cube canbe replace name = %s type = %s",
          GNodeGetName(merge_node).GetString(), GNodeGetType(merge_node).GetString());
  return true;
}

FixpipeCubeType FixpipeComm::GetFixpipeCubeType(const ge::GNodePtr &node_ptr) {
  bool fake_cube = false;
  GNodeGetAttr(node_ptr, kAttrFakeCubeNode, fake_cube);
  if (fake_cube) {
    return FixpipeCubeType::Cube;
  }
  std::string node_type(GNodeGetType(node_ptr).GetString());
  auto iter = std::find(kSupportFixpipeCubeTypeVec.begin(), kSupportFixpipeCubeTypeVec.end(), node_type);
  if (iter == kSupportFixpipeCubeTypeVec.end()) {
    OPS_LOG_D("Fixpipe", "GetFixpipeCubeType isn't node name = %s type = %s", GNodeGetName(node_ptr).GetString(), node_type.c_str());
    return FixpipeCubeType::NotCube;
  }
  auto merge_node = GetMergeNodeByCube(node_ptr);
  if (merge_node == nullptr) {
    return FixpipeCubeType::Cube;
  }
  return FixpipeCubeType::CubeMerge;
}

bool FixpipeComm::IsEltwiseNode(const ge::GNodePtr &node) {
  const std::string op_type = GNodeGetType(node).GetString();
  if (op_type == ELTWISE || op_type == kAdd || op_type == kSub) {
    return true;
  }
  return false;
}

void FixpipeComm::SetFixpipeAbilityAttr(const ge::GNodePtr &node_ptr, const FixpipeAbilityType &fixpipe_ability_type) {
  FixpipeAbilityAttr fixpipe_ability_attr = kNoFixpipeAbility;
  GNodeGetAttr(node_ptr, kSupportFixPipeAbility, fixpipe_ability_attr);
  fixpipe_ability_attr = fixpipe_ability_attr | kFixpipeAbilityAttrs[static_cast<size_t>(fixpipe_ability_type)];
  int64_t tmp_int = fixpipe_ability_attr;
  GNodeSetAttr(node_ptr, kSupportFixPipeAbility, tmp_int);
}

void FixpipeComm::UnSetFixpipeAbilityAttr(const ge::GNodePtr &node_ptr,
                                           const FixpipeAbilityType &fixpipe_ability_type) {
  FixpipeAbilityAttr fixpipe_ability_attr = kNoFixpipeAbility;
  GNodeGetAttr(node_ptr, kSupportFixPipeAbility, fixpipe_ability_attr);
  fixpipe_ability_attr = fixpipe_ability_attr & (~(1 << static_cast<size_t>(fixpipe_ability_type)));
  int64_t tmp_int = fixpipe_ability_attr;
  GNodeSetAttr(node_ptr, kSupportFixPipeAbility, tmp_int);
}

bool FixpipeComm::CheckFixpipeAbilityAttr(const ge::GNodePtr &node_ptr,
                                           const FixpipeAbilityType &fixpipe_ability_type) {
  FixpipeAbilityAttr fixpipe_ability_attr = kNoFixpipeAbility;
  GNodeGetAttr(node_ptr, kSupportFixPipeAbility, fixpipe_ability_attr);
  return ((fixpipe_ability_attr >> static_cast<size_t>(fixpipe_ability_type)) & 1) == 1;
}

bool FixpipeComm::HasControlEdge(const ge::GNodePtr &src_node_ptr, const ge::GNodePtr &dst_node_ptr) {
  if (src_node_ptr == nullptr || dst_node_ptr == nullptr) {
    return false;
  }
  bool src_res = false;
  for (const auto &it : src_node_ptr->GetOutControlNodes()) {
    if (it == dst_node_ptr) {
      src_res = true;
      break;
    }
  }
  bool dst_res = false;
  for (const auto &it : dst_node_ptr->GetInControlNodes()) {
    if (it == src_node_ptr) {
      dst_res = true;
      break;
    }
  }
  return src_res && dst_res;
}

std::string FixpipeComm::GetStrByDataTypeVec(const std::vector<ge::DataType>& data_type_vec) {
  std::string result;
  size_t size = data_type_vec.size();
  for (size_t i = 0; i < size; ++i) {
    std::string data_type = ge::TypeUtils::DataTypeToSerialString(data_type_vec[i]);
    result += data_type;
    if (i != size - 1) {
      result += ",";
    }
  }
  return result;
}

ge::graphStatus FixpipeComm::GetConv2DReluSwapForbiddenFlag(const ge::GNodePtr &head_node) {
  bool supportOut2L1Dn2Nz = false;
  if (!FixpipeComm::GetSupportDN2NZSoc(supportOut2L1Dn2Nz)) {
    return ge::GRAPH_NOT_CHANGED;
  }

  bool isConv2DFlag = FixpipeComm::GetConv2DOpType(head_node);
  if (supportOut2L1Dn2Nz && isConv2DFlag) {
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

bool FixpipeComm::GetSupportDN2NZSoc(bool &supportOut2L1Dn2Nz)
{
  // do soc check
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optionalInfo;
  if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "ConvFusionPassUtils", "Can't get platformInfo.");
    return false;
  }
  supportOut2L1Dn2Nz = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_data_move_out2l1_dn2nz") !=
    platformInfo.ai_core_intrinsic_dtype_map.end();
  return true;
}

bool FixpipeComm::GetConv2DOpType(const ge::GNodePtr &head_node) {
  // head_node null already check in Fusion
  if (GNodeGetType(head_node) != kAscendConv2DAsc) {
    return false;
  }
  return true;
}

ge::graphStatus FixpipeComm::GetFixpipeUtilFuncSupportFlag(const ge::GNodePtr &head_node) {
  bool supportOut2L1Dn2Nz = false;
  if (!FixpipeComm::GetSupportDN2NZSoc(supportOut2L1Dn2Nz)) { // regardless of which version, always be true
    return ge::GRAPH_NOT_CHANGED; // get platform failed
  }
  if (!supportOut2L1Dn2Nz) {
    return ge::GRAPH_SUCCESS;
  }

  bool isInFixpipeUtilFuncOpWhiteListFlag = false;
  const std::string op_type = GNodeGetType(head_node).GetString();
  auto iter = std::find(kSupportFixpipeUtilFuncCubeTypeVec.begin(), kSupportFixpipeUtilFuncCubeTypeVec.end(),
                        op_type);
  if (iter != kSupportFixpipeUtilFuncCubeTypeVec.end()) {
    OPS_LOG_D("Fixpipe", "Op in SupportFixpipeUtilFuncCubeTypeVec is node name = %s type = %s",
            GNodeGetName(head_node).GetString(), op_type.c_str());
    isInFixpipeUtilFuncOpWhiteListFlag = true;
  }
  if (!isInFixpipeUtilFuncOpWhiteListFlag) {
    return ge::GRAPH_NOT_CHANGED;
  }
  return ge::GRAPH_SUCCESS;
}

uint32_t FixpipeComm::GetOutDataNodesSize(const ge::GNodePtr &node) {
  uint32_t res = 0;
  for (size_t idx = 0; idx < node->GetOutputsSize(); ++idx) {
    const auto outputNodesPairs = node->GetOutDataNodesAndPortIndexs(idx);
    res += outputNodesPairs.size();
  }
  return res;
}

bool FixpipeComm::IsShapeEqual(const ge::Shape &shape1, const ge::Shape &shape2) {
  if (shape1.GetDimNum() != shape2.GetDimNum()) {
    return false;
  }
  for (size_t i = 0; i < shape1.GetDimNum(); ++i) {
    if (shape1.GetDim(i) != shape2.GetDim(i)) {
      return false;
    }
  }
  return true;
}

std::string FixpipeComm::ShapeToString(const ge::Shape &shape) {
  if (shape.GetDimNum() == 0) {
    return "";
  }
  std::stringstream ss;
  ss << shape.GetDim(0);
  for (size_t i = 1UL; i < shape.GetDimNum(); ++i) {
    ss << ", " << shape.GetDim(i);
  }
  return ss.str();
}

std::vector<string> FixpipeComm::Split(const string &str, const string &pattern) {
  std::vector<string> res_vec;
  if (str.empty()) {
    return res_vec;
  }
  string str_and_pattern = str + pattern;
  size_t pos = str_and_pattern.find(pattern);
  size_t size = str_and_pattern.size();
  while (pos != string::npos) {
    string sub_str = str_and_pattern.substr(0, pos);
    res_vec.push_back(sub_str);
    str_and_pattern = str_and_pattern.substr(pos + pattern.size(), size);
    pos = str_and_pattern.find(pattern);
  }
  return res_vec;
}
}  // namespace ops
