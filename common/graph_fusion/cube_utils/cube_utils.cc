/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cube_utils/cube_utils.h"
#include <memory>
#include <sstream>
#include <vector>
#include <string>
#include <initializer_list>
namespace ops {
namespace {
// maps aic version to ISA arch VERSION
const std::map<int32_t, ISAArchVersion> kAicIsaArchVersionMap {
        {100, ISAArchVersion::EN_ISA_ARCH_V100},
        {200, ISAArchVersion::EN_ISA_ARCH_V200},
        {202, ISAArchVersion::EN_ISA_ARCH_V200},
        {210, ISAArchVersion::EN_ISA_ARCH_V200},
        {220, ISAArchVersion::EN_ISA_ARCH_V220},
        {300, ISAArchVersion::EN_ISA_ARCH_V300},
        {310, ISAArchVersion::EN_ISA_ARCH_V300},
        {350, ISAArchVersion::EN_ISA_ARCH_V350}
};
const std::map<ISAArchVersion, std::string> kIsaArchVersionMapStr {
        {ISAArchVersion::EN_ISA_ARCH_V100, "v100"},
        {ISAArchVersion::EN_ISA_ARCH_V200, "v200"},
        {ISAArchVersion::EN_ISA_ARCH_V220, "v220"},
        {ISAArchVersion::EN_ISA_ARCH_V300, "v300"},
        {ISAArchVersion::EN_ISA_ARCH_V350, "v350"}
};

ge::GNodePtr GetCubeNode(const std::stack<FixPipeNodeInfo> &cur_pass) {
  std::stack<FixPipeNodeInfo> tmp_pass(cur_pass);
  ge::GNodePtr node;
  while (!tmp_pass.empty()) {
    node = tmp_pass.top().GetNode();
    tmp_pass.pop();
  }
  return node;
}

void GetPath(const ge::GNodePtr &input_node, const ge::GNodePtr &cube_node, int32_t &path_size) {
  int32_t max_depth = kEltMaxDepth;
  ge::GNodePtr cur_node = input_node;
  while (max_depth > 0) {
    if (cur_node == nullptr) {
      break;
    }
    if (cur_node == cube_node) {
      path_size++;
      break;
    }
    auto cur_in_node = cur_node->GetInDataNodesAndPortIndexs(0);
    if (cur_in_node.first != nullptr) {
      cur_node = cur_in_node.first;
    } else {
      break;
    }
    max_depth--;
  }
}

std::vector<std::string> FindNonSubstrings(const std::vector<std::string>& strings) {
  std::vector<std::string> result;
  for (const auto &str : strings) {
    bool is_substring = false;
    for (const auto &candidate : strings) {
      if (str == candidate) continue;
      if (candidate.find(str) != std::string::npos) {
        is_substring = true;
        break;
      }
    }
    if (!is_substring) {
      result.push_back(str);
      OPS_LOG_D("Fixpipe", "Add selected pass_hash = %s", str.c_str());
    }
  }
  return result;
}
} // namespace

constexpr uint32_t FIXPIPE_INPUT_2_INDEX = 2;
constexpr uint32_t FIXPIPE_INPUT_3_INDEX = 3;
constexpr uint32_t FIXPIPE_INPUT_4_INDEX = 4;
constexpr uint32_t FIXPIPE_INPUT_5_INDEX = 5;
constexpr uint32_t FIXPIPE_INPUT_6_INDEX = 6;
constexpr uint32_t FIXPIPE_INPUT_7_INDEX = 7;
constexpr uint32_t FIXPIPE_INPUT_8_INDEX = 8;
constexpr uint32_t FIXPIPE_INPUT_9_INDEX = 9;
constexpr uint32_t ELTWISE_SUB_TYPE = 2;
constexpr uint32_t ELTWISE_ADD_TYPE = 1;
constexpr uint32_t INSERTWOINDEX = 2;
constexpr uint32_t NODECANTACCES = 2;
constexpr uint32_t ELTWISEOPSUBSTRINDEX = 3;
constexpr uint32_t TWOINPUTSIZE =  2U;
const std::string FIXPIPE_TMP_PRE_PASS_NODE = "_fixpipe_tmp_pre_pass_node";
const std::string kSameFixpipeNodeScope = "_fixpipe_scope";
constexpr uint32_t PROBE_DEPTH_MAX = 5;
constexpr size_t kAicVersionSize = 3;

std::string FixpipeUtils::GetIsaArchVersionStr(const ISAArchVersion isa_arch_version) {
  std::string isa_version_str;
  auto iter = kIsaArchVersionMapStr.find(isa_arch_version);
  if (iter != kIsaArchVersionMapStr.end()) {
    isa_version_str = iter->second;
  }
  return isa_version_str;
}

void FixpipeUtils::ParseIsaArchVersion(fe::PlatFormInfos &platform_infos) {
  isa_arch_version_ = ISAArchVersion::EN_ISA_ARCH_V100;
  // aic version, ISAArchVersion
  std::string aic_version_str;
  if (!platform_infos.GetPlatformRes("version", "AIC_version", aic_version_str) || aic_version_str.empty()) {
    OPS_LOG_W("Fixpipe", "Aic version is empty.");
    return;
  }
  OPS_LOG_D("Fixpipe", "Aic version is [%s].", aic_version_str.c_str());
  std::vector<string> aic_version_vec = FixpipeComm::Split(aic_version_str, "-");
  if (aic_version_vec.size() < kAicVersionSize) {
    OPS_LOG_W("Fixpipe", "The AIC version [%s] is invalid.", aic_version_str.c_str());
    return;
  }
  int32_t aic_version = atoi(aic_version_vec[2].c_str());
  auto iter_aic = kAicIsaArchVersionMap.find(aic_version);
  if (iter_aic != kAicIsaArchVersionMap.end()) {
    isa_arch_version_ = iter_aic->second;
  }
  OPS_LOG_I("Fixpipe", "ISA arch version is [%s].", GetIsaArchVersionStr(isa_arch_version_).c_str());
  return;
}

bool FixpipeUtils::ReadConfig(const ge::CustomPassContext &context) {
  OPS_LOG_D("Fixpipe", "Begin to read config.");
  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos optional_infos;
  if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) != ge::GRAPH_SUCCESS) {
    OPS_LOG_W("Fixpipe", "Fail to get platform info without soc version.");
    return false;
  }
  ParseIsaArchVersion(platform_infos);
  std::vector<CONFIGDTYPE> cubeconfigtype;
  std::map<std::string, std::vector<CONFIGDTYPE>> cubmap;
  for (auto &iter : kSupportFixpipeCubeTypeVec) {
    cubmap.emplace(make_pair(iter, cubeconfigtype));
  }
  bool skip_trans = isa_arch_version_ == ISAArchVersion::EN_ISA_ARCH_V350;
  if (skip_trans) {
    cubmap.emplace(TRANSDATA, cubeconfigtype);
  }
  FixPipeUnit cube_ops(kCubeUnit, cubmap);
  m_idxtonodetypes_.push_back(cube_ops);
  unitmapindex_.emplace(make_pair(kCubeUnit, 0));
  uint32_t index = 1;
  std::map<std::string, std::map<std::string, std::vector<CONFIGDTYPE>>> fixpipe_map;
  std::vector<std::string> unit_list;
  std::map<std::string, std::vector<std::string>> depends_units;
  FixpipeComm::ReadPlatFormConfig(context, skip_trans, unit_list, depends_units, fixpipe_map);
  if (unit_list.empty()) {
    return false;
  }
  for (auto &iter : unit_list) {
    FixPipeUnit unitops(iter, fixpipe_map[iter]);
    unitmapindex_.emplace(make_pair(iter, index));
    index++;
    OPS_LOG_D("Fixpipe", "unit name = %s", iter.c_str());
    for (auto &depends : depends_units[iter]) {
      OPS_LOG_D("Fixpipe", "depends = %s", depends.c_str());
    }
    unitops.SetDependUnits(depends_units[iter]);
    m_idxtonodetypes_.push_back(unitops);
  }
  for (auto &unit : m_idxtonodetypes_) {
    const std::vector<std::string> depend_unitsname = unit.GetDependsUnits();
    if (depend_unitsname.empty()) {
      continue;
    }
    std::vector<uint32_t> depend_unitindex;
    for (auto &unitname : depend_unitsname) {
      OPS_LOG_D("Fixpipe", "match unit name = %s index = [%u] GetName() = %s ", unit.GetName().c_str(), unitmapindex_[unitname],
              unitname.c_str());
      depend_unitindex.push_back(unitmapindex_[unitname]);
    }
    unit.SetDependUnitsIndex(depend_unitindex);
  }
  return true;
}

bool FixpipeUtils::IsConfictWithSkipConfig(const FixPipePassInfo &cur_pass, const uint32_t &ret_index) const {
  std::vector<uint32_t> cur_index;
  for (auto &index : cur_pass.m_opnodes) {
    OPS_LOG_D("Fixpipe", "cur_index = %u", index.GetBelongUnitIndex());
    cur_index.push_back(index.GetBelongUnitIndex());
  }
  return IsConfictWithSkipConfig(cur_index, ret_index);
}

bool FixpipeUtils::IsConfictWithSkipConfig(const std::vector<uint32_t> &index, const uint32_t &ret_index) const {
  std::vector<uint32_t> depend_indexs = m_idxtonodetypes_[ret_index].GetDependsUnitsIndex();
  OPS_LOG_D("Fixpipe", "ret_index = %u", ret_index);
  for (auto &depend_index : depend_indexs) {
    if (std::find(index.begin(), index.end(), depend_index) == index.end()) {
      OPS_LOG_D("Fixpipe", "depend_index isn't = %u", depend_index);
      return false;
    }
  }
  return true;
}

bool FixpipeUtils::IsConfictWithSkipConfig(std::stack<uint32_t> index, const uint32_t &ret_index) const {
  std::vector<uint32_t> cur_index;
  while (!index.empty()) {
    auto node = index.top();
    OPS_LOG_D("Fixpipe", "cur_index = %d", node);
    cur_index.push_back(static_cast<uint32_t>(node));
    index.pop();
  }
  return IsConfictWithSkipConfig(cur_index, ret_index);
}

bool FixpipeUtils::JudgeCachePass(const FixPipeNodeInfo &node, std::stack<uint32_t> &index, uint32_t &ret_index) const {
  uint32_t cur_index;
  OPS_LOG_D("Fixpipe", "JudgeCachePass start node name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
          GNodeGetType(node.GetNode()).GetString());
  if (index.empty()) {
    cur_index = 0;
  } else {
    cur_index = index.top();
  }
  OPS_LOG_D("Fixpipe", "JudgeCachePass  cur_index= %d", cur_index);
  bool find_flag = false;
  if (index.empty() && node.GetIsHeadNode() &&
      FixpipeComm::GetFixpipeCubeType(node.GetNode()) != FixpipeCubeType::NotCube) {
    ret_index = 0;
    OPS_LOG_D("Fixpipe", "JudgeCachePass is headcubenode node = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return true;
  }
  for (size_t i = cur_index + 1; i < m_idxtonodetypes_.size(); i++) {
    find_flag = GetNodeIndex(node, static_cast<uint32_t>(i));
    if (find_flag) {
      ret_index = i;
      break;
    }
  }
  if (!find_flag) {
    return false;
  }
  OPS_LOG_D("Fixpipe", "JudgeCachePass node can fixpipe name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
          GNodeGetType(node.GetNode()).GetString());
  return IsConfictWithSkipConfig(index, ret_index);
}

bool FixpipeUtils::GetNodeIndex(const FixPipeNodeInfo &node, const uint32_t &index) const {
  for (auto &nodetype : m_idxtonodetypes_[index].GetNode()) {
    std::string node_type;
    if (GNodeGetType(node.GetNode()) == kAscendEltwiseAsc) {
      node_type = GetEltWiseType(node);
    } else {
      node_type = GNodeGetType(node.GetNode()).GetString();
    }
    if (nodetype.first == node_type) {
      OPS_LOG_D("Fixpipe", "GetNodeIndex node is name = %s type = %s index= %u", GNodeGetName(node.GetNode()).GetString(),
              GNodeGetType(node.GetNode()).GetString(), index);
      return true;
    }
  }
  return false;
}

bool FixpipeUtils::PreCachePass(const FixPipePassInfo &cur_pass, const FixPipeNodeInfo &node) const {
  if (FiltrNodeStrategy(node) != ge::GRAPH_SUCCESS) {
    return false;
  }
  if (!FixpipeComm::CheckIsInVector(cur_pass.m_opnodes)) {
    return false;
  }
  ge::graphStatus ret = FiltrNodeStrategyForQuant(node, cur_pass.m_opnodes[cur_pass.m_opnodes.size() - 1]);
  if (ret != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "PreCachePass post relu+quant node can't be fixpipe name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return false;
  }
  bool find_flag = false;
  uint32_t ret_index = 0;
  if (!cur_pass.m_opnodes.empty()) {
    for (uint32_t i = cur_pass.unit_index + 1; i < static_cast<uint32_t>(m_idxtonodetypes_.size()); i++) {
      find_flag = GetNodeIndex(node, i);
      if (find_flag) {
        ret_index = i;
        break;
      }
    }
  }
  if (!find_flag) {
    return false;
  }
  return IsConfictWithSkipConfig(cur_pass, ret_index);
}

bool FixpipeUtils::PreMatchAcorrdingToPass(const FixPipePassInfo &cur_pass, const FixPipeNodeInfo &node) const {
  bool not_support_fixpipe_node = false;
  GNodeGetAttr(node.GetNode(), kNotSupportFixpipeFusion, not_support_fixpipe_node);
  if (not_support_fixpipe_node) {
    return false;
  }
  if (!IsInWhitelist(node)) {
    OPS_LOG_D("Fixpipe", "node isn't IsInWhitelist name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return false;
  }
  if (!node.GetIsHeadNode() && FixpipeComm::GetFixpipeCubeType(node.GetNode()) != FixpipeCubeType::NotCube) {
    OPS_LOG_D("Fixpipe", "node isnt headcube name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return false;
  }
  if (!PreCachePass(cur_pass, node)) {
    OPS_LOG_D("Fixpipe", "node isn't PreCachePass name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return false;
  }
  OPS_LOG_D("Fixpipe", "node isn' name = %s type = %s", GNodeGetName(node.GetNode()).GetString(), GNodeGetType(node.GetNode()).GetString());
  return true;
}

bool FixpipeUtils::NeedToCutPass(FixPipePassInfo &m_pass) const {
  if (m_pass.m_opnodes.size() == 1) {
    OPS_LOG_D("Fixpipe", "only has headnode");
    return true;
  }
  FixPipeNodeInfo node = m_pass.m_opnodes[m_pass.m_opnodes.size() - 1];
  if (node.GetNode()->GetOutputsSize() == 0) {
    OPS_LOG_D("Fixpipe", "GetOutDataNodes.empty");
    m_pass.m_flag = 2;  // m_flag = 2 DONT NEED, 1NEED, 0 UNKOWN
    return false;
  }
  std::string cube_type = GetMergeInputNodeType(m_pass.m_opnodes[0].GetNode());
  if (cube_type == kConv2DTransposeD && m_pass.m_opnodes[0].GetNode()->GetOutputsSize() > 1) {
    if (GNodeGetType(m_pass.m_opnodes[1].GetNode()) == kAscendQuantAsc ||
        GNodeGetType(m_pass.m_opnodes[1].GetNode()) == kAscendDequantAsc) {
      OPS_LOG_D("Fixpipe", "Headnode is kConv2DTransposeD and second node is kAscendQuant or kAscendDequant, dont need to cut");
      m_pass.m_flag = 2;
      return false;
    }
    if (GNodeGetType(m_pass.m_opnodes[1].GetNode()) != kAscendQuantAsc  &&
        GNodeGetType(m_pass.m_opnodes[1].GetNode()) != kAscendDequantAsc) {
      OPS_LOG_D("Fixpipe", "Headnode is kConv2DTransposeD but second node is not kAscendQuant or kAscendDequant, need to cut");
      m_pass.m_flag = 1;
      return true;
    }
  }
  if (cube_type != CONV2D && node.GetNode()->GetOutputsSize() > 1) {
    m_pass.m_flag = 2;  // m_flag = 2 DONT NEED, 1NEED, 0 UNKOWN
    return false;
  }

  for (size_t idx = 0; idx < node.GetNode()->GetOutputsSize(); ++idx) {
    const auto outputNodesPairs = node.GetNode()->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &outputPair : outputNodesPairs) {
      FixPipeNodeInfo grandnode(outputPair.first);
      if (!PreMatchAcorrdingToPass(m_pass, grandnode)) {
        OPS_LOG_D("Fixpipe", "Has a can't fixpipe outputnode name = %s type = %s", GNodeGetName(grandnode.GetNode()).GetString(),
                GNodeGetType(grandnode.GetNode()).GetString());
        m_pass.m_flag = 2;  // m_flag = 2 DONT NEED, 1NEED, 0 UNKOWN
        return false;
      }
    }
  }
  OPS_LOG_D("Fixpipe", " needto cut, passid = %d", m_pass.pass_index);
  m_pass.m_flag = 1;  // m_flag = 2 DONT NEED, 1NEED, 0 UNKOWN
  return true;
}

std::string FixpipeUtils::GetEltWiseType(const FixPipeNodeInfo &node) const {
  uint32_t real_eltwisetype = 0;
  if (GNodeGetAttr(node.GetNode(), kAttrEltwiseMode, real_eltwisetype)) {
    if (real_eltwisetype == ELTWISE_ADD_TYPE) {
      return kAdd;
    }
    if (real_eltwisetype == ELTWISE_SUB_TYPE) {
      return kSub;
    }
  }
  return ELTWISE;
}

bool FixpipeUtils::IsInWhitelist(const FixPipeNodeInfo &node) const {
  for (auto &unit : m_idxtonodetypes_) {
    for (auto &nodes : unit.GetNode()) {
      std::string node_type;
      if (GNodeGetType(node.GetNode()) == kAscendEltwiseAsc) {
        node_type = GetEltWiseType(node);
      } else {
        node_type = GNodeGetType(node.GetNode()).GetString();
      }
      if (nodes.first == node_type) {
        OPS_LOG_D("Fixpipe", "Node isinwhitelist name = %s type = %s", GNodeGetName(node.GetNode()).GetString(), node_type.c_str());
        return true;
      }
    }
  }
  return false;
}
template <typename... Args>
void FixpipeUtils::PrintNodeFilterReason(const FixPipeNodeInfo &node, const Args &...args) const {
  std::stringstream ss;
  (void)std::initializer_list<int>{(ss << args, 0)...};
  ss.flush();
  OPS_LOG_D("Fixpipe", "node[%s:%s] in fixpipe pass filter by %s", GNodeGetName(node.GetNode()).GetString(),
          GNodeGetType(node.GetNode()).GetString(), ss.str().c_str());
}
// extrafilterforhardware
ge::graphStatus FixpipeUtils::FiltrNodeStrategy(const FixPipeNodeInfo &node) const {
  bool fake_cube = false;
  GNodeGetAttr(node.GetNode(), kAttrFakeCubeNode, fake_cube);
  std::string node_type = std::string(GNodeGetType(node.GetNode()).GetString());
  if (node_type == TRANSDATA && !fake_cube) {
    return FiltrNodeStrategyForTransData(node);
  }
  if (node_type == kAdd || node_type == kSub || node_type == ELTWISE) {
    return FiltrNodeStrategyForEltWise(node);
  }
  if (node_type == kPRelu || node_type == kLeakyRelu
     || node_type == RELU6) {
    return FiltrNodeStrategyForRelu(node);
  }
  if (node_type == CAST) {
    return FiltrNodeStrategyForCast(node);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::FiltrNodeStrategyForQuant(const FixPipeNodeInfo &cur_node, const FixPipeNodeInfo &prenode) const {
  if (cur_node.GetNode() != nullptr) {
    if (GNodeGetType(cur_node.GetNode()) != kAscendQuantAsc) {
      return ge::GRAPH_SUCCESS;
    }
    bool is_dump_able = false;
    if (GNodeGetAttr(cur_node.GetNode(), kAttrDumpAble, is_dump_able) == ge::GRAPH_SUCCESS && is_dump_able) {
      OPS_LOG_D("Fixpipe", "Node[%s, %s] is dump able, can not be fused.",
              GNodeGetName(cur_node.GetNode()).GetString(), GNodeGetType(cur_node.GetNode()).GetString());
      return ge::GRAPH_SUCCESS;
    }
    if (prenode.GetNode() == nullptr) {
      return ge::GRAPH_SUCCESS;
    }
  }
  float scale = 0.0;
  GNodeGetAttr(cur_node.GetNode(), ATTR_SCALE, scale);
  float offset = 0.0;
  GNodeGetAttr(cur_node.GetNode(), ATTR_OFFSET, offset);
  if (scale < 0) {
    return ge::GRAPH_FAILED;
  }
  if (GNodeGetType(prenode.GetNode()) == kAscendLeakyReluAsc) {
    float attr_negative_slope_a = 0.0;
    GNodeGetAttr(prenode.GetNode(), ATTR_NEGATIVE_SLOPE, attr_negative_slope_a);
    if (attr_negative_slope_a < 0) {
      return ge::GRAPH_FAILED;
    }
  }
  if (GNodeGetType(prenode.GetNode()) == kAscendPReluAsc) {
    if (!FixpipeComm::CheckConstValueData(prenode.GetNode())) {
      return ge::GRAPH_FAILED;
    }
  }
  if (GNodeGetType(prenode.GetNode()) == kAscendRelu6Asc) {
    if (scale * kRelu6Value + offset < 0) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::FiltrNodeStrategyForRelu(const FixPipeNodeInfo &node) const {
  ge::TensorDesc input_desc;
  if (node.GetNode()->GetInputDesc(0, input_desc) != ge::GRAPH_SUCCESS) {
    PrintNodeFilterReason(node, "input 0 is empty");
    return ge::GRAPH_FAILED;
  }
  if (input_desc.GetDataType() == ge::DT_FLOAT16 ||
      input_desc.GetDataType() == ge::DT_INT8) {
    return ge::GRAPH_SUCCESS;
  }
  PrintNodeFilterReason(node, "input data type is not fp16 or int8, cur input datatype is ",
      FixpipeComm::GetStrByDataTypeVec({input_desc.GetDataType()}));
  return ge::GRAPH_FAILED;
}

ge::graphStatus FixpipeUtils::FiltrNodeStrategyForCast(const FixPipeNodeInfo &node) const {
  ge::TensorDesc input_desc;
  ge::TensorDesc output_desc;
  if (node.GetNode()->GetInputDesc(0, input_desc) != ge::GRAPH_SUCCESS ||
      node.GetNode()->GetOutputDesc(0, output_desc) != ge::GRAPH_SUCCESS) {
    PrintNodeFilterReason(node, "input or output is null");
    return ge::GRAPH_FAILED;
  }
  if (input_desc.GetDataType() != ge::DT_FLOAT) {
    PrintNodeFilterReason(node, "input dtype is not fp32, cur dtype is ",
                          FixpipeComm::GetStrByDataTypeVec({input_desc.GetDataType()}));
    return ge::GRAPH_FAILED;
  }
  auto out_data_type = output_desc.GetDataType();
  if (out_data_type != ge::DT_BF16 && out_data_type != ge::DT_FLOAT16) {
    PrintNodeFilterReason(node, "output dtype is not bf16 or fp16, output dtype is ",
                          FixpipeComm::GetStrByDataTypeVec({out_data_type}));
    return ge::GRAPH_FAILED;
  }
  if (node.GetCubeNode() != nullptr &&
      FixpipeComm::CheckFixpipeAbilityAttr(node.GetCubeNode(), FixpipeAbilityType::UseGmAtomicAdd)) {
    PrintNodeFilterReason(node, "Cube node has using atomic write.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::FiltrNodeStrategyForTransData(const FixPipeNodeInfo &node) const {
  ge::TensorDesc input_desc;
  ge::TensorDesc output_desc;
  if (node.GetNode()->GetInputDesc(0, input_desc) != ge::GRAPH_SUCCESS) {
    PrintNodeFilterReason(node, "input is empty");
    return ge::GRAPH_FAILED;
  }
  if (node.GetNode()->GetOutputDesc(0, output_desc) != ge::GRAPH_SUCCESS) {
    PrintNodeFilterReason(node, "output is empty");
    return ge::GRAPH_FAILED;
  }
  auto input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(input_desc.GetFormat()));
  auto output_format = static_cast<ge::Format>(ge::GetPrimaryFormat(output_desc.GetFormat()));
  if (input_format == ge::FORMAT_FRACTAL_NZ && output_format == ge::FORMAT_ND) {
    return ge::GRAPH_SUCCESS;
  }
  if (input_format == ge::FORMAT_NC1HWC0 && output_format == ge::FORMAT_NHWC) {
    return ge::GRAPH_SUCCESS;
  }
  if (input_format == ge::FORMAT_NDC1HWC0 && output_format == ge::FORMAT_NDHWC) {
    return ge::GRAPH_SUCCESS;
  }
  PrintNodeFilterReason(node,
                        "only support input:output format is [FRACTAL_NZ:FORMAT_ND, NC1HWC0:NHWC, NDC1HWC0, NDHWC]");
  return ge::GRAPH_FAILED;
}

ge::graphStatus FixpipeUtils::CheckEltWiseShapeIsSame(
    const FixPipeNodeInfo &node, const ge::TensorDesc &input_desc0, const ge::TensorDesc &input_desc1) const
{
  // attr check
  int64_t fixpipe_support_attr = 0UL;
  GNodeGetAttr(node.GetNode(), kSupportFixPipeAbility, fixpipe_support_attr);
  OPS_LOG_D("Fixpipe", "node %s support_fixpipe_ability is 0x%lx", GNodeGetName(node.GetNode()).GetString(), fixpipe_support_attr);
  if (FixpipeComm::CheckFixpipeAbilityAttr(node.GetNode(),
                                            FixpipeAbilityType::SupportPostEltwiseBroadcast)) {
    return ge::GRAPH_SUCCESS;
  }

  if (!FixpipeComm::IsShapeEqual(input_desc0.GetShape(), input_desc1.GetShape())) {
    PrintNodeFilterReason(node, "input1 or input2 shape is not same, ", "input0 is ",
                          FixpipeComm::ShapeToString(input_desc0.GetShape()).c_str(), "input1 is ",
                          FixpipeComm::ShapeToString(input_desc1.GetShape()).c_str());
    return ge::GRAPH_FAILED;
  }
  if (!FixpipeComm::IsShapeEqual(input_desc0.GetOriginShape(), input_desc1.GetOriginShape())) {
    PrintNodeFilterReason(node, "input1 or input2 origin shape is not same, ", "input0 is ",
                          FixpipeComm::ShapeToString(input_desc0.GetShape()).c_str(), "input1 is ",
                          FixpipeComm::ShapeToString(input_desc1.GetShape()).c_str());
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::FiltrNodeStrategyForEltWise(const FixPipeNodeInfo &node) const {
  if (node.GetNode()->GetInputsSize() != TWOINPUTSIZE) {
    PrintNodeFilterReason(node, "input size is not 2");
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc input_desc0;
  ge::TensorDesc input_desc1;
  if (node.GetNode()->GetInputDesc(0, input_desc0) != ge::GRAPH_SUCCESS ||
      node.GetNode()->GetInputDesc(1, input_desc1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (input_desc0.GetFormat() != input_desc1.GetFormat()) {
    PrintNodeFilterReason(node, "input1 or input2 format is not same, ", "input0 is ",
                          static_cast<uint32_t>(input_desc0.GetFormat()), "input1 is ",
                          static_cast<uint32_t>(input_desc1.GetFormat()));
    return ge::GRAPH_FAILED;
  }
  if (CheckEltWiseShapeIsSame(node, input_desc0, input_desc1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if ((input_desc0.GetDataType() != ge::DT_FLOAT16) || (input_desc1.GetDataType() != ge::DT_FLOAT16)) {
    PrintNodeFilterReason(node, "input1 or input2 data type is not fp16, ", "input0 is ",
                          FixpipeComm::GetStrByDataTypeVec({input_desc0.GetDataType()}), "input1 is ",
                          FixpipeComm::GetStrByDataTypeVec({input_desc1.GetDataType()}));
    return ge::GRAPH_FAILED;
  }
  if (GNodeGetType(node.GetNode()) == kAscendAddAsc) {
    return ge::GRAPH_SUCCESS;
  } else if (GNodeGetType(node.GetNode()) == kAscendEltwiseAsc) {
    auto node_type = GetEltWiseType(node);
    if (node_type == kAdd || node_type == ELTWISE) {
      return ge::GRAPH_SUCCESS;
    }
  }
  ge::AscendString cur_pass_pre_node;
  (void)GNodeGetAttr(node.GetNode(), FIXPIPE_TMP_PRE_PASS_NODE, cur_pass_pre_node);
  if (FixpipeComm::CheckPeerOutNode(node.GetNode(), 1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  auto input1_node = node.GetNode()->GetInDataNodesAndPortIndexs(1);
  if (GNodeGetName(input1_node.first) == cur_pass_pre_node) {
    PrintNodeFilterReason(node, "input1 node is same wiht cur pass pre node, ", "input0 is ",
                GNodeGetName(input1_node.first).GetString(), "cur pass pre node is ", cur_pass_pre_node.GetString());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::JudgeIsMatch(const FixPipeNodeInfo &node, std::stack<uint32_t> &cur_index, uint32_t &ret_index) const {
  if (!IsInWhitelist(node)) {
    OPS_LOG_D("Fixpipe", "geIsMatch node isn't inwhitelist name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  if (!node.GetIsHeadNode() && FixpipeComm::GetFixpipeCubeType(node.GetNode()) != FixpipeCubeType::NotCube) {
    OPS_LOG_D("Fixpipe", "JudgeIsMatch node iscube but node headnode name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  if (cur_index.size() > m_idxtonodetypes_.size()) {
    OPS_LOG_D("Fixpipe", "JudgeIsMatch cur_index size = %zu  m_idxtonodetypes_ size = %zu", cur_index.size(),
            m_idxtonodetypes_.size());
    return ge::GRAPH_FAILED;
  }
  if (!cur_index.empty() && cur_index.top() >= static_cast<uint32_t>(m_idxtonodetypes_.size() - 1)) {
    OPS_LOG_D("Fixpipe", "JudgeIsMatch cur_index is last top = %u  m_idxtonodetypes_ size = %zu", cur_index.top(),
            m_idxtonodetypes_.size());
    return ge::GRAPH_FAILED;
  }
  if (!JudgeCachePass(node, cur_index, ret_index)) {
    OPS_LOG_D("Fixpipe", "JudgeIsMatch node isnt JudgeCachePass name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  if (FiltrNodeStrategy(node) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  OPS_LOG_D("Fixpipe", "JudgeIsMatch node is true name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
          GNodeGetType(node.GetNode()).GetString());
  return ge::GRAPH_SUCCESS;
}

void FixpipeUtils::GetFusionNodes(const std::stack<FixPipeNodeInfo> &cur_pass,
                                 std::vector<ge::GNodePtr> &fusion_nodes) const {
  std::stack<FixPipeNodeInfo> tmp_pass(cur_pass);
  while (!tmp_pass.empty()) {
    fusion_nodes.emplace_back(tmp_pass.top().GetNode());
    tmp_pass.pop();
  }
}

std::string FixpipeUtils::GetCubeType(const std::stack<FixPipeNodeInfo> &cur_pass) const {
  std::stack<FixPipeNodeInfo> tmp_pass(cur_pass);
  ge::GNodePtr node;
  while (!tmp_pass.empty()) {
    node = tmp_pass.top().GetNode();
    tmp_pass.pop();
  }
  return GetMergeInputNodeType(node);
}

std::string FixpipeUtils::GetMergeInputNodeType(const ge::GNodePtr &merge_node) const {
  std::string merge_nodetype = GNodeGetType(merge_node).GetString();
  if (merge_nodetype != kMerge) {
    if (merge_nodetype == TRANSDATA) {
      auto input_node = merge_node->GetInDataNodesAndPortIndexs(0);
      return GNodeGetType(input_node.first).GetString();
    } else {
      return merge_nodetype;
    }
  } else {
    for (size_t idx = 0; idx < merge_node->GetInputsSize(); ++idx) {
      const auto input_node = merge_node->GetInDataNodesAndPortIndexs(idx);
      if (input_node.first != nullptr && GNodeGetType(input_node.first) == kAscendConv2DAsc) {
        return CONV2D;
      }
    }
    return merge_nodetype;
  }
}

void FixpipeUtils::GenerateMatchedPassesImpl(FixPipeNodeInfo &node, FixpipeMatchParams &fp_mch_params) {
  uint32_t tmp_cur_index = 0;
  const auto &fixpipe_node = node.GetNode();
  OPS_LOG_D("Fixpipe", "GenerateMatchedPassesImpl start node name = %s type = %s", GNodeGetName(fixpipe_node).GetString(),
          GNodeGetType(fixpipe_node).GetString());
  ge::graphStatus ret = JudgeIsMatch(node, fp_mch_params.cur_index, tmp_cur_index);
  if (ret != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("Fixpipe", "GenerateMatchedPassesImpl node can't be fixpipe name =%s type = %s", GNodeGetName(fixpipe_node).GetString(),
            GNodeGetType(fixpipe_node).GetString());
    return;
  }
  if (fp_mch_params.cur_index.size() > 1) {
    ret = FiltrNodeStrategyForQuant(node, fp_mch_params.cur_pass.top());
    if (ret != ge::GRAPH_SUCCESS) {
      OPS_LOG_D("Fixpipe", "GenerateMatchedPassesImpl post relu+quant node can't be fixpipe name = %s type = %s",
              GNodeGetName(fixpipe_node).GetString(), GNodeGetType(fixpipe_node).GetString());
      return;
    }
  }
  /* check eltwise node has cycle or not
   * case:    Conv2D--->relu--->eltwise
   *             |_____________/
   * if eltwise node fusion as fixpipe, there is a cycle
   * in case, only relu can fusion
   */
  if (m_idxtonodetypes_[tmp_cur_index].GetName() == kPostEltwise) {
    ge::GNodePtr cube_node = GetCubeNode(fp_mch_params.cur_pass);
    int32_t path_size = 0;
    for (size_t idx = 0; idx < fixpipe_node->GetInputsSize(); ++idx) {
      const auto input_node = fixpipe_node->GetInDataNodesAndPortIndexs(idx);
      GetPath(input_node.first, cube_node, path_size);
    }
    if (path_size > 1) {
      bool tmp_bool = true;
      GNodeSetAttr(fixpipe_node, kNotSupportFixpipeFusion, tmp_bool);
      OPS_LOG_D("Fixpipe", "GenerateMatchedPassesImpl node(%s) attr(not_support_fixpipe_fusion) is true.",
              GNodeGetName(fixpipe_node).GetString());
      not_support_fixpipe_fusion_nodes_.emplace_back(fixpipe_node);
      return;
    }
  }

  node.SetNodeFixpipeability(1);
  node.SetBelongUnitType(m_idxtonodetypes_[tmp_cur_index].GetName());
  node.SetBelongUnitIndex(tmp_cur_index);
  fp_mch_params.cur_pass.push(node);
  fp_mch_params.cur_index.push(tmp_cur_index);
  OPS_LOG_D("Fixpipe", "GenerateMatchedPassesImpl node can fixpipe name = %s type = %s belong_unitname = %s index = %d",
          GNodeGetName(fixpipe_node).GetString(), GNodeGetType(fixpipe_node).GetString(), node.GetBelongUnitType().c_str(),
          tmp_cur_index);

  GenerateMatchedPassesFromStack(fp_mch_params.cur_pass, fp_mch_params.fixpipe_index++, tmp_cur_index);
  if (fixpipe_node->GetOutputsSize() > 1) {
    if (!FixpipeComm::CheckFixpipeAbilityAttr(node.GetCubeNode(),
                                              FixpipeAbilityType::SupportMultipleOutput)) {
      OPS_LOG_D("Fixpipe", "GenerateMatchedPassesImpl node does not support multiple output fixpipe name = %s type = %s",
              GNodeGetName(fixpipe_node).GetString(), GNodeGetType(fixpipe_node).GetString());
      fp_mch_params.cur_index.pop();
      fp_mch_params.cur_pass.pop();
      return;
    }
  }
  for (size_t idx = 0; idx < fixpipe_node->GetOutputsSize(); ++idx) {
    const auto outputNodesPairs = fixpipe_node->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &out_node : outputNodesPairs) {
      OPS_LOG_D("Fixpipe", " GenerateMatchedPassesImpl node outnode fixpipe name = %s type = %s", GNodeGetName(out_node.first).GetString(),
              GNodeGetType(out_node.first).GetString());
      ge::AscendString fixpipe_node_name = GNodeGetName(fixpipe_node);
      GNodeSetAttr(out_node.first, FIXPIPE_TMP_PRE_PASS_NODE, fixpipe_node_name);
      FixPipeNodeInfo grandnode(out_node.first, node.GetCubeNode());
      GenerateMatchedPassesImpl(grandnode, fp_mch_params);
    }
  }
  fp_mch_params.cur_index.pop();
  fp_mch_params.cur_pass.pop();
  return;
}

void FixpipeUtils::GenerateMatchedPassesFromStack(const std::stack<FixPipeNodeInfo> &cur_pass, const uint32_t &pass_index,
                                                 const uint32_t &cur_index) {
  std::stack<FixPipeNodeInfo> print_pass(cur_pass);
  FixPipePassInfo tmp_pass;
  tmp_pass.m_flag = 0;
  tmp_pass.pass_index = pass_index;
  tmp_pass.unit_index = cur_index;
  std::stack<FixPipeNodeInfo> res_pass;
  while (!print_pass.empty()) {
    auto node = print_pass.top();
    res_pass.push(node);
    print_pass.pop();
  }
  while (!res_pass.empty()) {
    auto node = res_pass.top();
    tmp_pass.m_opnodes.push_back(node);
    res_pass.pop();
  }
  ChangeOrInsertPass(tmp_pass);
  return;
}

ge::graphStatus FixpipeUtils::ModfiyMatchedPasses(bool firstround_cut) {
  OPS_LOG_D("Fixpipe", "Start");
  if (m_matchpasses_.empty()) {
    return ge::GRAPH_FAILED;
  }
  std::vector<FixPipePassInfo> tmp_passes(m_matchpasses_);
  m_matchpasses_.clear();
  // 第一轮剪枝只保留同一路径可能的最长pass，当pass结束node的下一个还可以加入时也会被删除
  if (firstround_cut) {
    for (auto &tmp_pass : tmp_passes) {
      if (!NeedToCutPass(tmp_pass)) {
        OPS_LOG_D("Fixpipe", "ModfiyMatchedPasses pass dont needtocut id = %d", tmp_pass.pass_index);
        m_matchpasses_.push_back(tmp_pass);
      }
    }
  } else {
    // 不跑第一轮剪枝，则挑出所有路径的最长pass
    std::map<std::string, FixPipePassInfo> tmp_pass_map;
    std::vector<std::string> tmp_pass_hash_vec;
    for (auto &tmp_pass : tmp_passes) {
      if (tmp_pass.m_opnodes.size() == 1) {
        OPS_LOG_D("Fixpipe", "Only has headnode, remove it");
        continue;
      }
      std::string pass_hash = "";
      for (auto &node : tmp_pass.m_opnodes) {
        std::string node_id = "[" +  std::string(GNodeGetName(node.GetNode()).GetString()) + "]";
        pass_hash += node_id;
      }
      tmp_pass_map[pass_hash] = tmp_pass;
      tmp_pass_hash_vec.push_back(pass_hash);
      OPS_LOG_D("Fixpipe", "Add pass_hash = %s", pass_hash.c_str());
    }
    auto selected_passes = FindNonSubstrings(tmp_pass_hash_vec);
    for (auto &selected_pass : selected_passes) {
      m_matchpasses_.push_back(tmp_pass_map[selected_pass]);
    }
  }
  ModfiyMatchedPasseSecRound();
  return ge::GRAPH_SUCCESS;
}

uint32_t FixpipeUtils::GetUnitIndex(const FixPipePassInfo &fixpipe_pass, const FixPipeNodeInfo &fixpipe_node) {
  uint32_t unit_index = 0;
  if (!fixpipe_pass.m_opnodes.empty()) {
    for (uint32_t i = fixpipe_pass.unit_index + 1; i < static_cast<uint32_t>(m_idxtonodetypes_.size()); i++) {
      bool find_flag = GetNodeIndex(fixpipe_node, i);
      if (find_flag) {
        unit_index = i;
        break;
      }
    }
  }
  return unit_index;
}

ge::graphStatus FixpipeUtils::ModfiyMatchedPasseSecRound() {
  OPS_LOG_D("Fixpipe", "Start sec round");
  if (m_matchpasses_.empty()) {
    return ge::GRAPH_SUCCESS;
  }
  ge::GNodePtr cube_node = m_matchpasses_[0].m_opnodes[0].GetNode();
  std::string cube_type = GetMergeInputNodeType(cube_node);
  if (!FixpipeComm::CheckFixpipeAbilityAttr(cube_node, FixpipeAbilityType::SupportMultipleOutput) &&
      m_matchpasses_.size() <= kFixpipeNodeLimited) {
    return ge::GRAPH_SUCCESS;
  } else if (m_matchpasses_.size() == 1) {
    return ge::GRAPH_SUCCESS;
  }
  // each pass, max length is 7
  int32_t min_length = kMaxDepth;
  for (auto &pass : m_matchpasses_) {
    min_length = std::min(min_length, static_cast<int32_t>(pass.m_opnodes.size()));
  }
  std::unordered_set<ge::GNodePtr> fixpipe_nodes;
  std::map<ge::GNodePtr, FixPipeNodeInfo> fixpipe_info;
  for (auto &pass : m_matchpasses_) {
    auto pass_nodes = pass.m_opnodes;
    for (auto &pass_node : pass_nodes) {
      fixpipe_nodes.emplace(pass_node.GetNode());
      fixpipe_info[pass_node.GetNode()] = pass_node;
    }
  }

  std::vector<ge::GNodePtr> out_nodes;
  bool cube_not_fixpipe_out_flag = false;
  for (size_t idx = 0; idx < cube_node->GetOutputsSize(); ++idx) {
    const auto outputNodesPairs = cube_node->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &out_node : outputNodesPairs) {
      if (fixpipe_nodes.count(out_node.first) != 0) {
        out_nodes.emplace_back(out_node.first);
      } else {
        cube_not_fixpipe_out_flag = true;
      }
    }
  }
  std::vector<std::vector<ge::GNodePtr>> matched_pass;
  ge::GNodePtr branch_node;
  if (out_nodes.size() > 1) {
    branch_node = cube_node;
  } else {
    for (int32_t i = 1; i < min_length; i++) {
      if (m_matchpasses_[0].m_opnodes[i].GetNode()->GetOutputsSize() > 1) {
        branch_node = m_matchpasses_[0].m_opnodes[i].GetNode();
        break;
      }
    }
  }
  if (branch_node == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  if (cube_not_fixpipe_out_flag) {
    GenerateSinglePass(cube_node, matched_pass, fixpipe_nodes);
  } else {
    GeneratePass(cube_node, branch_node, matched_pass, fixpipe_nodes);
  }
  GenerateFixpipePasses(matched_pass, fixpipe_info);
  return ge::GRAPH_SUCCESS;
}

void FixpipeUtils::GenerateSinglePass(const ge::GNodePtr &cube_node,
                                     std::vector<std::vector<ge::GNodePtr>> &matched_pass,
                                     std::unordered_set<ge::GNodePtr> &fixpipe_nodes) const {
  ge::GNodePtr cur_node;
  std::vector<ge::GNodePtr> pass;
  pass.emplace_back(cube_node);
  for (size_t idx = 0; idx < cube_node->GetOutputsSize(); ++idx) {
    const auto outputNodesPairs = cube_node->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &out_node : outputNodesPairs) {
      if (!matched_pass.empty()) {
        return;
      }
      if (fixpipe_nodes.count(out_node.first) > 0) {
        matched_pass.push_back(pass);
        cur_node = out_node.first;
        while (fixpipe_nodes.count(cur_node) > 0) {
          matched_pass.back().emplace_back(cur_node);
          if (FixpipeComm::GetOutDataNodesSize(cur_node) != 1) {
            break;
          }
          const auto curOutputNodesPairs = cur_node->GetOutDataNodesAndPortIndexs(0);
          cur_node = curOutputNodesPairs[0].first;
        }
      }
    }
  }
}

void FixpipeUtils::GeneratePass(const ge::GNodePtr &cube_node, const ge::GNodePtr &branch_node,
                               std::vector<std::vector<ge::GNodePtr>> &matched_pass,
                               std::unordered_set<ge::GNodePtr> &fixpipe_nodes) const {
  ge::GNodePtr cur_node = cube_node;
  std::vector<ge::GNodePtr> pass;
  while (cur_node != branch_node) {
    pass.emplace_back(cur_node);
    const auto curOutputNodesPairs = cur_node->GetOutDataNodesAndPortIndexs(0);
    cur_node = curOutputNodesPairs[0].first;
  }
  pass.emplace_back(branch_node);
  size_t count = 0;
  if (FixpipeComm::GetOutDataNodesSize(branch_node) > static_cast<uint32_t>(kFixpipeNodeLimited)) {
    matched_pass.push_back(pass);
    count++;
  }
  for (size_t idx = 0; idx < branch_node->GetOutputsSize(); ++idx) {
    const auto outputNodesPairs = branch_node->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &out_node : outputNodesPairs) {
      if (count >= kFixpipeNodeLimited) {
        return;
      }
      if (fixpipe_nodes.count(out_node.first) > 0) {
        matched_pass.push_back(pass);
        count++;
        cur_node = out_node.first;
        while (fixpipe_nodes.count(cur_node) > 0) {
          matched_pass.back().emplace_back(cur_node);
          if (FixpipeComm::GetOutDataNodesSize(cur_node) != 1) {
            break;
          }
          const auto curOutputNodesPairs = cur_node->GetOutDataNodesAndPortIndexs(0);
          cur_node = curOutputNodesPairs[0].first;
        }
      }
    }
  }
  if (matched_pass.size() < kFixpipeNodeLimited) {
    matched_pass.push_back(pass);
  }
}

void FixpipeUtils::GenerateFixpipePasses(std::vector<std::vector<ge::GNodePtr>> &matched_pass,
                                        std::map<ge::GNodePtr, FixPipeNodeInfo> &fixpipe_info) {
  m_matchpasses_.clear();
  uint32_t index_count = 0;
  for (auto &pass : matched_pass) {
    if (pass.size() < kPassMinSize) {
      continue;
    }
    FixPipePassInfo fixpipe_pass;
    fixpipe_pass.m_flag = kPassFlag;
    fixpipe_pass.unit_index = 0;
    fixpipe_pass.pass_index = index_count;
    for (auto &node : pass) {
      auto fixpipe_node = fixpipe_info[node];
      uint32_t unit_index = GetUnitIndex(fixpipe_pass, fixpipe_node);
      fixpipe_pass.m_opnodes.emplace_back(fixpipe_node);
      fixpipe_pass.unit_index = unit_index;
    }
    index_count++;
    m_matchpasses_.emplace_back(fixpipe_pass);
  }
}

void FixpipeUtils::ChangeOrInsertPass(FixPipePassInfo &tmp_pass) {
  bool find_flag = false;
  for (auto &m_already_pass : m_matchpasses_) {
    if (m_already_pass.m_opnodes.size() != tmp_pass.m_opnodes.size()) {
      continue;
    }
    bool m_find_flag = true;
    for (uint32_t index = 0; index < static_cast<uint32_t>(m_already_pass.m_opnodes.size()); index++) {
      if (GNodeGetName(m_already_pass.m_opnodes[index].GetNode()) != GNodeGetName(tmp_pass.m_opnodes[index].GetNode())) {
        m_find_flag = false;
        OPS_LOG_D("Fixpipe", "ChangeOrInsertPass false index= %d", index);
        break;
      }
    }
    if (m_find_flag) {
      OPS_LOG_D("Fixpipe", "ChangeOrInsertPass find_flag = true");
      find_flag = true;
      break;
    }
  }
  if (!find_flag) {
    m_matchpasses_.push_back(tmp_pass);
  }
}

ge::graphStatus FixpipeUtils::GenerateMatchedPasses(FixPipeNodeInfo &conv_node) {
  if (!m_matchpasses_.empty()) {
    OPS_LOG_I("Fixpipe", "GenerateMatchedPasses WARNNIGN passes already exist.");
  }
  FixpipeMatchParams fp_mch_params;
  GenerateMatchedPassesImpl(conv_node, fp_mch_params);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::RelinkHeadEdges(ge::Graph &graph, FixPipeNodeInfo &head_node, FixPipeNodeInfo &fixpipeenhancenode) const {
  if (head_node.GetNode()->GetOutputsSize() == 0) {
    REPORT_OPS_ERROR("[GraphOpt][FixpipePss][RelinkHeadEdges] head_node[%s] outdataanchor empty.",
                    GNodeGetName(head_node.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  // todo
  if (graph.AddDataEdge(*head_node.GetNode(), 0, *fixpipeenhancenode.GetNode(), 0) != ge::GRAPH_SUCCESS) {
    REPORT_OPS_ERROR("[GraphOpt][FixpipePss][RelinkHeadEdges] Fail to add edge between src node [%s] and dst node[%s].",
                    GNodeGetName(head_node.GetNode()).GetString(), GNodeGetName(fixpipeenhancenode.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc output_desc;
  if (head_node.GetNode()->GetOutputDesc(0, output_desc) == ge::GRAPH_SUCCESS) {
    fixpipeenhancenode.GetNode()->UpdateInputDesc(0, output_desc);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::RelinkOutputEdges(ge::Graph &graph, const FixPipePassInfo &match_pass, FixPipeNodeInfo &fixpipeenhancenode) const {
  std::vector<FixPipeNodeInfo> fixednodeids(match_pass.m_opnodes.begin() + 1, match_pass.m_opnodes.end());
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes)) {
    return ge::GRAPH_FAILED;
  }
  FixPipeNodeInfo last_tofuzednode = match_pass.m_opnodes[match_pass.m_opnodes.size() - 1];
  OPS_LOG_D("Fixpipe", "RelinkOpEdges 4 size = %zu", FixpipeComm::GetOutDataNodesSize(last_tofuzednode.GetNode()));

  const auto outputNodesPairs = last_tofuzednode.GetNode()->GetOutDataNodesAndPortIndexs(0);
  for (const auto &outchild_node : outputNodesPairs) {
    if (outchild_node.first == nullptr) {
      continue;
    }

    if (graph.RemoveEdge(*last_tofuzednode.GetNode(), 0, *outchild_node.first, outchild_node.second) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    if (graph.AddDataEdge(*fixpipeenhancenode.GetNode(), 0, *outchild_node.first, outchild_node.second) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    OPS_LOG_D("Fixpipe", "src_node = %s dst_node = %s", GNodeGetName(fixpipeenhancenode.GetNode()).GetString(),
              GNodeGetName(outchild_node.first).GetString());
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::RelinkAntiEltWiseEdges(ge::Graph &graph, const ge::GNodePtr &inputfather_node, const FixPipeNodeInfo &tofuzednode,
                                           FixPipeNodeInfo &fixpipeenhancenode) {
  FixPipeNodePair nodepair(inputfather_node, tofuzednode.GetNode());
  m_toantiquantnodes_.emplace(inputfather_node);
  // todo 
  if (inputfather_node->GetInputsSize() != 0) {
    auto input_node = inputfather_node->GetInDataNodesAndPortIndexs(0);
    if (graph.AddDataEdge(*input_node.first, input_node.second, *fixpipeenhancenode.GetNode(), 1) != ge::GRAPH_SUCCESS) {
      REPORT_OPS_ERROR(
          "[GraphOpt][FixpipePss][RelinkAntiEltWiseEdges]  Fail to add  edge between src node [%s] and dst node[%s].",
          GNodeGetName(inputfather_node).GetString(), GNodeGetName(fixpipeenhancenode.GetNode()).GetString());
      return ge::GRAPH_FAILED;
    }
    ge::TensorDesc output_desc;
    if (input_node.first->GetOutputDesc(0, output_desc) == ge::GRAPH_SUCCESS) {
      fixpipeenhancenode.GetNode()->UpdateInputDesc(1, output_desc);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::RelinkOtherEltWiseEdges(ge::Graph &graph, const int32_t &src_port, const FixPipeNodeInfo &tofuzednode,
                                            FixPipeNodeInfo &fixpipeenhancenode, const ge::GNodePtr &inputfather_node) const {
  // todo
  if (inputfather_node != nullptr) {
    if (graph.AddDataEdge(*inputfather_node, src_port, *fixpipeenhancenode.GetNode(), 1) != ge::GRAPH_SUCCESS) {
      REPORT_OPS_ERROR(
          "[GraphOpt][FixpipePss][RelinkOtherEltWiseEdges] Fail to add  edge between src node [%s] and dst node[%s].",
          GNodeGetName(inputfather_node).GetString(), GNodeGetName(fixpipeenhancenode.GetNode()).GetString());
      return ge::GRAPH_FAILED;
    }
    if (!fixpipeenhancenode.GetNode()->GetInControlNodes().empty() &&
        FixpipeComm::HasControlEdge(inputfather_node, tofuzednode.GetNode())) {
      if (graph.AddControlEdge(*inputfather_node, *fixpipeenhancenode.GetNode()) != ge::GRAPH_SUCCESS) {
        REPORT_OPS_ERROR(
            "[GraphOpt][FixpipePss][RelinkOtherEltWiseEdges] Fail to add  edge between src node [%s] and dst node[%s].",
            GNodeGetName(inputfather_node).GetString(), GNodeGetName(fixpipeenhancenode.GetNode()).GetString());
        return ge::GRAPH_FAILED;
      }
    }
  }
  if (!fixpipeenhancenode.GetNode()->GetInControlNodes().empty() &&
      FixpipeComm::HasControlEdge(inputfather_node, tofuzednode.GetNode())) {
    if (graph.AddControlEdge(*inputfather_node, *fixpipeenhancenode.GetNode()) != ge::GRAPH_SUCCESS) {
      REPORT_OPS_ERROR(
          "[GraphOpt][FixpipePss][RelinkOtherEltWiseEdges] Fail to add  edge between src node [%s] and dst node[%s].",
          GNodeGetName(inputfather_node).GetString(), GNodeGetName(fixpipeenhancenode.GetNode()).GetString());
      return ge::GRAPH_FAILED;
    }
  }

  ge::TensorDesc output_desc;
  if (inputfather_node->GetOutputDesc(0, output_desc) == ge::GRAPH_SUCCESS) {
    fixpipeenhancenode.GetNode()->UpdateInputDesc(1, output_desc);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::RelinkEltWiseEdges(ge::Graph &graph, const FixPipePassInfo &match_pass, FixPipeNodeInfo &fixpipeenhancenode) {
  std::vector<FixPipeNodeInfo> fixednodeids(match_pass.m_opnodes.begin() + 1, match_pass.m_opnodes.end());
  OPS_LOG_D("Fixpipe", "RelinkOpEdges 3");
  for (auto &tofuzednode : fixednodeids) {
    if (tofuzednode.GetBelongUnitType() == kPostEltwise) {
      return RelinkEltWiseEdgesImpl(graph, match_pass, tofuzednode, fixpipeenhancenode);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::RelinkEltWiseEdgesImpl(ge::Graph &graph, const FixPipePassInfo &match_pass, FixPipeNodeInfo &tofuzednode,
                                           FixPipeNodeInfo &fixpipeenhancenode) {
  OPS_LOG_D("Fixpipe", "RelinkOpEdges 3.1");
  for (size_t idx = 0; idx < tofuzednode.GetNode()->GetInputsSize(); ++idx) {
    auto in_node = tofuzednode.GetNode()->GetInDataNodesAndPortIndexs(idx);
    if (in_node.first == nullptr) {
      continue;
    }
    if (IsNodeInPass(match_pass.m_opnodes, in_node.first)) {
      OPS_LOG_D("Fixpipe", "inputfathernode is infixpipenode name = %s", GNodeGetName(in_node.first).GetString());
      continue;
    }
    if (GNodeGetType(in_node.first) == kAscendAntiQuantAsc) {
      return RelinkAntiEltWiseEdges(graph, in_node.first, tofuzednode, fixpipeenhancenode);
    } else {
      return RelinkOtherEltWiseEdges(graph, in_node.second, tofuzednode, fixpipeenhancenode, in_node.first);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::RelinkOpEdges(ge::Graph &graph, FixPipeNodeInfo &head_node,
                                   const FixPipePassInfo &match_pass, FixPipeNodeInfo &fixpipeenhancenode) {
  if (RelinkHeadEdges(graph, head_node, fixpipeenhancenode) != ge::GRAPH_SUCCESS) {
    REPORT_OPS_ERROR("[GraphOpt][FixpipePss][RelinkOpEdges]  Fail to add edge between head node [%s] and dst node[%s].",
                    GNodeGetName(head_node.GetNode()).GetString(), GNodeGetName(fixpipeenhancenode.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  if (RelinkEltWiseEdges(graph, match_pass, fixpipeenhancenode) != ge::GRAPH_SUCCESS) {
    REPORT_OPS_ERROR("[GraphOpt][FixpipePss][RelinkOpEdges]  Fail to add edge between eltwise and dst node[%s].",
                    GNodeGetName(fixpipeenhancenode.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  if (RelinkOutputEdges(graph, match_pass, fixpipeenhancenode) != ge::GRAPH_SUCCESS) {
    REPORT_OPS_ERROR("[GraphOpt][FixpipePss][RelinkOpEdges]  Fail to add edge between output and dst node[%s].",
                    GNodeGetName(fixpipeenhancenode.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

bool FixpipeUtils::IsNodeInPass(const std::vector<FixPipeNodeInfo> &fixednodeids, const ge::GNodePtr &input_node) const {
  bool found = (fixednodeids.end() != std::find(fixednodeids.begin(), fixednodeids.end(), input_node));
  OPS_LOG_D("Fixpipe", "IsNodeInPass is %u name = %s type = %s", found, GNodeGetName(input_node).GetString(),
          GNodeGetType(input_node).GetString());
  return found;
}

ge::graphStatus FixpipeUtils::AddInputs(ge::Graph &graph, const FixPipePassInfo &match_pass,
                              const ge::GNodePtr &fixpipenode, std::vector<ge::GNodePtr> &new_nodes) {
  for (uint32_t i = FIXPIPE_INPUT_2_INDEX; i < 10; i++) {
    FixPipeFunctionParamPtr funtcparam;
    OPS_MAKE_SHARED(funtcparam = std::make_shared<FixPipeFunctionParam>("Dummy", fixpipenode, i), return ge::GRAPH_FAILED);
    FixPipeAddInputPtr addinputptr = AddInputStrategy(match_pass, funtcparam);
    if (addinputptr != nullptr) {
      if (addinputptr->DoAddInput(graph, match_pass, funtcparam, new_nodes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::UpdateInputDesc(ge::GNode &cur_new_fixpipenode) const {
  ge::Shape shape{};
  ge::TensorDesc fakedesc(shape, ge::FORMAT_RESERVED, ge::DT_UNDEFINED);
  cur_new_fixpipenode.UpdateInputDesc(0, fakedesc);
  cur_new_fixpipenode.UpdateInputDesc(1, fakedesc);
  cur_new_fixpipenode.UpdateInputDesc(FIXPIPE_INPUT_2_INDEX, fakedesc);
  cur_new_fixpipenode.UpdateInputDesc(FIXPIPE_INPUT_3_INDEX, fakedesc);
  cur_new_fixpipenode.UpdateInputDesc(FIXPIPE_INPUT_4_INDEX, fakedesc);
  cur_new_fixpipenode.UpdateInputDesc(FIXPIPE_INPUT_5_INDEX, fakedesc);
  cur_new_fixpipenode.UpdateInputDesc(FIXPIPE_INPUT_6_INDEX, fakedesc);
  cur_new_fixpipenode.UpdateInputDesc(FIXPIPE_INPUT_7_INDEX, fakedesc);
  cur_new_fixpipenode.UpdateInputDesc(FIXPIPE_INPUT_8_INDEX, fakedesc);
  cur_new_fixpipenode.UpdateInputDesc(FIXPIPE_INPUT_9_INDEX, fakedesc);
  return ge::GRAPH_SUCCESS;
}

ge::GNodePtr FixpipeUtils::CreateFixpipeNode(const FixPipePassInfo &match_pass, const FixPipeNodeInfo &head_node,
                                           ge::Graph &graph) const {
  OPS_LOG_D("Fixpipe", "CreateFixpipeNode 1 ");
  bool need_judge = true;
  std::string op_name = std::string(GNodeGetName(head_node.GetNode()).GetString()) + "_pass." + std::to_string(match_pass.pass_index);
  ge::GNode fixpipenode = ge::es::CompliantNodeBuilder(&graph)
                        .OpType(kFixPipe.c_str())
                        .Name(op_name.c_str())
                        .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"x2", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                      {"quant_scale_0", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                      {"relu_weight_0", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                      {"clip_value_0", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                      {"quant_scale_1", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                      {"relu_weight_1", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                      {"clip_value_1", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                      {"anti_quant_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                      {"anti_quant_offset", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                        .IrDefOutputs({{"output", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();

  fixpipenode.SetAttr(kAscendNeedJudgeDtypeAsc, need_judge);
  std::vector<FixPipeNodeInfo> fixednodeids(match_pass.m_opnodes.begin() + 1, match_pass.m_opnodes.end());
  std::vector<std::string> fuzedoptypes;
  std::vector<std::string> activatefixpipeunits;
  std::string eltwiseops;
  for (auto &tofuzednode : fixednodeids) {
    fuzedoptypes.push_back(GNodeGetType(tofuzednode.GetNode()).GetString());
    activatefixpipeunits.push_back(tofuzednode.GetBelongUnitType());
    if (tofuzednode.GetBelongUnitType() == kPostEltwise) {
      if (GNodeGetType(tofuzednode.GetNode()) == kAscendEltwiseAsc) {
        eltwiseops = GetEltWiseType(tofuzednode);
      } else {
        eltwiseops = GNodeGetType(tofuzednode.GetNode()).GetString();
      }
    }
  }
  if (GNodeGetType(fixednodeids[0].GetNode()) == kAscendDequantAsc &&
                   fixednodeids[0].GetNode()->HasAttr(kAscendOffsetAsc)) {
    if (fixednodeids.size() > 1 && fixednodeids[1].GetBelongUnitType() == kPreAct) {
      fuzedoptypes.insert(fuzedoptypes.cbegin() + INSERTWOINDEX, ASCEND_QUANT);
    } else {
      fuzedoptypes.insert(fuzedoptypes.cbegin() + 1, ASCEND_QUANT);
    }
  }
  std::transform(eltwiseops.begin(), eltwiseops.end(), eltwiseops.begin(), ::toupper);
  std::string tmp_str1(eltwiseops.substr(0, ELTWISEOPSUBSTRINDEX));
  ge::AscendString tmp_str1_asc(tmp_str1.c_str());
  std::vector<ge::AscendString> fuzedoptypes_asc;
  std::vector<ge::AscendString> activatefixpipeunits_asc;
  for (const auto &it : fuzedoptypes) {
    fuzedoptypes_asc.push_back(ge::AscendString(it.c_str()));
  }
  for (const auto &it : activatefixpipeunits) {
    activatefixpipeunits_asc.push_back(ge::AscendString(it.c_str()));
  }
  fixpipenode.SetAttr(kAscendEltwiseModeAsc, tmp_str1_asc);
  fixpipenode.SetAttr(kAscendFusionOpListAsc, fuzedoptypes_asc);
  fixpipenode.SetAttr(kAscendUnitListAsc, activatefixpipeunits_asc);
  UpdateInputDesc(fixpipenode);
  if (!FixpipeComm::CheckIsInVector(match_pass.m_opnodes)) {
    return nullptr;
  }
  FixPipeNodeInfo last_tofuzednode = match_pass.m_opnodes[match_pass.m_opnodes.size() - 1];
  ge::TensorDesc output_desc;
  if (last_tofuzednode.GetNode()->GetOutputDesc(0, output_desc) == ge::GRAPH_SUCCESS) {
    fixpipenode.UpdateOutputDesc(0, output_desc);
    OPS_LOG_D("Fixpipe", "CreateFixpipeNode node_name = %s.", op_name.c_str());
  }

  ge::GNodePtr fixpipenode_ptr = graph.FindNodeByName(op_name.c_str());
  return fixpipenode_ptr;
}

ge::graphStatus FixpipeUtils::DeleteToFusedNodeEdge(ge::Graph &graph, const FixPipePassInfo &match_pass,
                                          std::set<ge::GNodePtr> &todeletenode) const {
  std::vector<FixPipeNodeInfo> fixednodeids(match_pass.m_opnodes.begin() + 1, match_pass.m_opnodes.end());
  for (auto &tofuzednode : fixednodeids) {
    if (graph.FindNodeByName(GNodeGetName(tofuzednode.GetNode())) == nullptr) {
      continue;
    }
    OPS_LOG_D("Fixpipe", "DeleteToFusedNodeEdge 2 name = %s type = %s", GNodeGetName(tofuzednode.GetNode()).GetString(),
            GNodeGetType(tofuzednode.GetNode()).GetString());
    todeletenode.emplace(tofuzednode.GetNode());
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::DeleteNode(ge::Graph &graph, const std::set<ge::GNodePtr> &todeletenode) {
  for (auto &node : todeletenode) {
    graph.RemoveNode(*node, 1);
  }
  for (auto &node : m_toantiquantnodes_) {
    if (node == nullptr) {
      continue;
    } else {
      graph.RemoveNode(*node, 1);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::FusionImpl(const string &pass_name, FixPipeNodeInfo &head_node, ge::Graph &graph,
                               std::vector<ge::GNodePtr> &new_nodes) {
  std::vector<ge::GNodePtr> fixpipe_nodes;
  for (auto &pass : m_matchpasses_) {
    std::vector<ge::GNodePtr> original_nodes;
    std::unordered_set<ge::GNodePtr> original_nodes_set;
    for (auto &node : pass.m_opnodes) {
      /* 1. trans node is not original node
       * 2. cube node is not in the fusion node list, so we skip */
      const auto &node_ptr = node.GetNode();
      const std::string &op_type = GNodeGetType(node_ptr).GetString();
      bool is_cube_op = std::find(kSupportFixpipeCubeTypeVec.begin(), kSupportFixpipeCubeTypeVec.end(), op_type) !=
          kSupportFixpipeCubeTypeVec.end();
      if (op_type == CAST || op_type == TRANSDATA || is_cube_op) {
        continue;
      }
      original_nodes_set.emplace(node_ptr);
    }
    OPS_LOG_D("Fixpipe", "FusionImpl 1.1");
    ge::GNodePtr fixpipenode = CreateFixpipeNode(pass, head_node, graph);
    OPS_CHECK_NOTNULL(fixpipenode);
    OPS_LOG_D("Fixpipe", "FusionImpl 1.2");
    FixPipeNodeInfo fixpipeenhancenode(fixpipenode);
    OPS_LOG_D("Fixpipe", "FusionImpl 1.3");
    AddInputs(graph, pass, fixpipenode, new_nodes);
    OPS_LOG_D("Fixpipe", "FusionImpl 1.4");
    RelinkOpEdges(graph, head_node, pass, fixpipeenhancenode);
    OPS_LOG_D("Fixpipe", "FusionImpl 1.5");
    std::vector<ge::GNodePtr> fusion_nodes;
    fusion_nodes.push_back(fixpipenode);
    original_nodes.assign(original_nodes_set.begin(), original_nodes_set.end());
    OPS_LOG_D("Fixpipe", "FusionImpl 1.6");
    fixpipe_nodes.push_back(fixpipenode);
  }
  OPS_LOG_D("Fixpipe", "Fixpipe_nodes size is %zu.", fixpipe_nodes.size());
  std::set<ge::GNodePtr> todeletenode;
  for (auto &pass : m_matchpasses_) {
    DeleteToFusedNodeEdge(graph, pass, todeletenode);
  }
  DeleteNode(graph, todeletenode);
  if (fixpipe_nodes.empty()) {
    return ge::GRAPH_FAILED;
  }
  for (const auto &node : fixpipe_nodes) {
    SetFixpipeRealtiveNodeScopeId(node);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::InitInput2() {
  if (isa_arch_version_ != ISAArchVersion::EN_ISA_ARCH_V350) {
    FixPipeAddInputPtr strategy21ptr = nullptr;
    OPS_MAKE_SHARED(strategy21ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy21>()),
                   return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendQuant, strategy21ptr));
    OPS_LOG_D("Fixpipe", "strategy21 name = %s", ("quant_scale_0:" + kAscendQuant).c_str());
    FixPipeAddInputPtr strategy22ptr = nullptr;
    OPS_MAKE_SHARED(strategy22ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy22>()),
                   return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendRequant, strategy22ptr));
    OPS_LOG_D("Fixpipe", "strategy22 name = %s", ("quant_scale_0:" + kAscendRequant).c_str());
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant, strategy22ptr));
    OPS_LOG_D("Fixpipe", "strategy22 name = %s", ("quant_scale_0:" + kAscendDequant).c_str());
  } else {
    FixPipeAddInputPtr strategy23ptr = nullptr;
    OPS_MAKE_SHARED(strategy23ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy23>()),
                   return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant, strategy23ptr));
    OPS_LOG_D("Fixpipe", "Add strategy23 name = %s", ("quant_scale_0:" + kAscendDequant).c_str());
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + RELU, strategy23ptr));
    OPS_LOG_D("Fixpipe", "Add strategy23 name = %s", ("quant_scale_0:" + kAscendDequant + "_" + RELU).c_str());
    FixPipeAddInputPtr strategy24ptr = nullptr;
    OPS_MAKE_SHARED(strategy24ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy24>()),
                   return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + kLeakyRelu, strategy24ptr));
    OPS_LOG_D("Fixpipe", "Add strategy24 name = %s", ("quant_scale_0:" + kAscendDequant + "_" + kLeakyRelu).c_str());

    FixPipeAddInputPtr strategy25ptr = nullptr;
    OPS_MAKE_SHARED(strategy25ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy25>()),
                 return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + kPRelu, strategy25ptr));
    OPS_LOG_D("Fixpipe", "Add strategy25 name = %s", ("quant_scale_0:" + kAscendDequant + "_" + kPRelu).c_str());

    FixPipeAddInputPtr pattern_dequant_lut_ptr = nullptr;
    OPS_MAKE_SHARED(pattern_dequant_lut_ptr =
                       std::shared_ptr<FixPipeAddInputBase>(std::make_shared<AddInputStrategyDequntLut>()),
                   return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + kTanh, pattern_dequant_lut_ptr));
    OPS_LOG_D("Fixpipe", "Add strategy for pattern [%s]", ("quant_scale_0:" + kAscendDequant + "_" + kTanh).c_str());

    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + kElu, pattern_dequant_lut_ptr));
    OPS_LOG_D("Fixpipe", "Add strategy for pattern [%s]", ("quant_scale_0:" + kAscendDequant + "_" + kElu).c_str());

    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + kSigmoid, pattern_dequant_lut_ptr));
    OPS_LOG_D("Fixpipe", "Add strategy for pattern [%s]", ("quant_scale_0:" + kAscendDequant + "_" + kSigmoid).c_str());
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::InitInput3() {
  if (isa_arch_version_ == ISAArchVersion::EN_ISA_ARCH_V350) {
    OPS_LOG_D("Fixpipe", "There is no need to init strategy for input3.");
    return ge::GRAPH_SUCCESS;
  }
  FixPipeAddInputPtr strategy31ptr = nullptr;
  OPS_MAKE_SHARED(strategy31ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy31>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendQuant + "_" + kPRelu, strategy31ptr));
  OPS_LOG_D("Fixpipe", "strategy31 name = %s", ("relu_weight_0:" + kAscendQuant + "_" + kPRelu).c_str());
  FixPipeAddInputPtr strategy32ptr = nullptr;
  OPS_MAKE_SHARED(strategy32ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy32>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendDequant + "_" + kPRelu, strategy32ptr));
  OPS_LOG_D("Fixpipe", "strategy32 name = %s", ("relu_weight_0:" + kAscendDequant + "_" + kPRelu).c_str());
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendRequant + "_" + kPRelu, strategy32ptr));
  OPS_LOG_D("Fixpipe", "strategy32 name = %s", ("relu_weight_0:" + kAscendRequant + "_" + kPRelu).c_str());
  FixPipeAddInputPtr strategy33ptr = nullptr;
  OPS_MAKE_SHARED(strategy33ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy33>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kPRelu, strategy33ptr));
  OPS_LOG_D("Fixpipe", "strategy33 name = %s", ("relu_weight_0:" + kPRelu).c_str());
  FixPipeAddInputPtr strategy34ptr = nullptr;
  OPS_MAKE_SHARED(strategy34ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy34>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendQuant + "_" + kLeakyRelu, strategy34ptr));
  OPS_LOG_D("Fixpipe", "strategy34 name = %s", ("relu_weight_0:" + kAscendQuant + "_" + kLeakyRelu).c_str());
  FixPipeAddInputPtr strategy35ptr = nullptr;
  OPS_MAKE_SHARED(strategy35ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy35>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendDequant + "_" + kLeakyRelu, strategy35ptr));
  OPS_LOG_D("Fixpipe", "strategy35 name = %s", ("relu_weight_0:" + kAscendDequant + "_" + kLeakyRelu).c_str());
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendRequant + "_" + kLeakyRelu, strategy35ptr));
  OPS_LOG_D("Fixpipe", "strategy35 name = %s", ("relu_weight_0:" + kAscendRequant + "_" + kLeakyRelu).c_str());
  FixPipeAddInputPtr strategy36ptr = nullptr;
  OPS_MAKE_SHARED(strategy36ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy36>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kLeakyRelu, strategy36ptr));
  OPS_LOG_D("Fixpipe", "strategy36 name = %s", ("relu_weight_0:" + kLeakyRelu).c_str());
  FixPipeAddInputPtr strategy37ptr = nullptr;
  OPS_MAKE_SHARED(strategy37ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy37>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + CAST + "_" + kPRelu, strategy37ptr));
  OPS_LOG_D("Fixpipe", "strategy37 name = %s", ("relu_weight_0:" + CAST + "_" + kPRelu).c_str());
  FixPipeAddInputPtr strategy38ptr = nullptr;
  OPS_MAKE_SHARED(strategy38ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy38>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + CAST + "_" + kLeakyRelu, strategy38ptr));
  OPS_LOG_D("Fixpipe", "strategy38 name = %s", ("relu_weight_0:" + CAST + "_" + kLeakyRelu).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::InitInput4() {
  FixPipeAddInputPtr strategy41ptr = nullptr;
  OPS_MAKE_SHARED(strategy41ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy41>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("clip_value_0:" + RELU6, strategy41ptr));
  OPS_LOG_D("Fixpipe", "strategy41 name = %s", ("clip_value_0:" + RELU6).c_str());
  FixPipeAddInputPtr strategy42ptr = nullptr;
  OPS_MAKE_SHARED(strategy42ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy42>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("clip_value_0:" + kAscendQuant + "_" + RELU6, strategy42ptr));
  OPS_LOG_D("Fixpipe", "strategy42 name = %s", ("clip_value_0:" + kAscendQuant + "_" + RELU6).c_str());
  FixPipeAddInputPtr strategy43ptr = nullptr;
  OPS_MAKE_SHARED(strategy43ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy43>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("clip_value_0:" + kAscendDequant + "_" + RELU6, strategy43ptr));
  OPS_LOG_D("Fixpipe", "strategy43 name = %s", ("clip_value_0:" + kAscendDequant + "_" + RELU6).c_str());
  m_opmaps_.emplace(make_pair("clip_value_0:" + kAscendRequant + "_" + RELU6, strategy43ptr));
  OPS_LOG_D("Fixpipe", "strategy43 name = %s", ("clip_value_0:" + kAscendRequant + "_" + RELU6).c_str());
  FixPipeAddInputPtr strategy44ptr = nullptr;
  OPS_MAKE_SHARED(strategy44ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy44>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("clip_value_0:" + CAST + "_" + RELU6, strategy44ptr));
  OPS_LOG_D("Fixpipe", "strategy44 name = %s", ("clip_value_0:" + CAST + "_" + RELU6).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::InitInput5() {
  FixPipeAddInputPtr strategy5xptr = nullptr;
  if (isa_arch_version_ != ISAArchVersion::EN_ISA_ARCH_V350) {
    OPS_MAKE_SHARED(strategy5xptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy51>()),
                   return ge::GRAPH_FAILED);
  } else {
    OPS_MAKE_SHARED(strategy5xptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy52>()),
                   return ge::GRAPH_FAILED);
  }
  m_opmaps_.emplace(make_pair("quant_scale_1:" + kAscendQuant, strategy5xptr));
  OPS_LOG_D("Fixpipe", "strategy5x name = %s", ("quant_scale_1:" + kAscendQuant).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::InitInput6() {
  FixPipeAddInputPtr strategy61ptr = nullptr;
  OPS_MAKE_SHARED(strategy61ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy61>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_1:" + kPRelu + "_" + kAscendQuant, strategy61ptr));
  OPS_LOG_D("Fixpipe", "strategy61 name = %s", ("relu_weight_1:" + kPRelu + "_" + kAscendQuant).c_str());

  FixPipeAddInputPtr strategy62ptr = nullptr;
  OPS_MAKE_SHARED(strategy62ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy62>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_1:" + kPRelu, strategy62ptr));
  OPS_LOG_D("Fixpipe", "strategy62 name = %s", ("relu_weight_1:" + kPRelu).c_str());

  FixPipeAddInputPtr strategy63ptr = nullptr;
  OPS_MAKE_SHARED(strategy63ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy63>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_1:" + kLeakyRelu + "_" + kAscendQuant, strategy63ptr));
  OPS_LOG_D("Fixpipe", "strategy63 name = %s", ("relu_weight_1:" + kLeakyRelu + "_" + kAscendQuant).c_str());
  FixPipeAddInputPtr strategy64ptr = nullptr;
  OPS_MAKE_SHARED(strategy64ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy64>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_1:" + kLeakyRelu, strategy64ptr));
  OPS_LOG_D("Fixpipe", "strategy64 name = %s", ("relu_weight_1:" + kLeakyRelu).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::InitInput7() {
  FixPipeAddInputPtr strategy71ptr = nullptr;
  OPS_MAKE_SHARED(strategy71ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy71>()),
                 return ge::GRAPH_FAILED);
  FixPipeAddInputPtr strategy72ptr = nullptr;
  OPS_MAKE_SHARED(strategy72ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy72>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("clip_value_1:" + RELU6, strategy71ptr));
  OPS_LOG_D("Fixpipe", "strategy71 name = %s", ("clip_value_1:" + RELU6).c_str());
  m_opmaps_.emplace(make_pair("clip_value_1:" + RELU6 + "_" + kAscendQuant, strategy72ptr));
  OPS_LOG_D("Fixpipe", "strategy72 name = %s", ("clip_value_1:" + RELU6 + "_" + kAscendQuant).c_str());
  return ge::GRAPH_SUCCESS;
}


ge::graphStatus FixpipeUtils::InitInput8() {
  FixPipeAddInputPtr strategy81ptr = nullptr;
  OPS_MAKE_SHARED(strategy81ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy81>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("anti_quant_scale:" + kAscendAntiQuant, strategy81ptr));
  OPS_LOG_D("Fixpipe", "strategy81 name = %s", ("anti_quant_scale:" + kAscendAntiQuant).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::InitInput9() {
  FixPipeAddInputPtr strategy91ptr = nullptr;
  OPS_MAKE_SHARED(strategy91ptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategy91>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("anti_quant_offset:" + kAscendAntiQuant, strategy91ptr));
  OPS_LOG_D("Fixpipe", "strategy91 name = %s", ("anti_quant_offset:" + kAscendAntiQuant).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::InitInputDefault() {
  FixPipeAddInputPtr strategydefaultptr = nullptr;
  OPS_MAKE_SHARED(
      strategydefaultptr = std::shared_ptr<FixPipeAddInputBase>(std::make_shared<FixPipeAddInputStrategyDefault>()),
      return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("DEFAULT", strategydefaultptr));
  OPS_LOG_D("Fixpipe", "strategydefault name = DEFAULT");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::InitInput() {
  if (InitInput2() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (InitInput3() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (InitInput4() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (InitInput5() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (InitInput6() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (InitInput7() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (InitInput8() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (InitInput9() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (InitInputDefault() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

void FixpipeUtils::ClearPasses() {
  m_matchpasses_.clear();
  m_toantiquantnodes_.clear();
  m_idxtonodetypes_.clear();
  unitmapindex_.clear();
  not_support_fixpipe_fusion_nodes_.clear();
}

void FixpipeUtils::CollectSwitchMergeNodes(const ge::GNodePtr &cube_node,
                                          std::vector<ge::GNodePtr> &fixpipe_nodes) const {
  fixpipe_nodes.emplace_back(cube_node);
  if (GNodeGetType(cube_node) != kAscendMergeAsc) {
    return;
  }
  auto cube_in_node = cube_node->GetInDataNodesAndPortIndexs(0);
  if (cube_in_node.first == nullptr) {
    return;
  }
  std::vector<ge::GNodePtr> switch_nodes;
  for (size_t idx = 0; idx < cube_in_node.first->GetInputsSize(); ++idx) {
    auto in_node = cube_in_node.first->GetInDataNodesAndPortIndexs(idx);
    if (in_node.first == nullptr) {
      continue;
    }
    if (GNodeGetType(in_node.first) == kAscendSwitchAsc) {
      switch_nodes.emplace_back(in_node.first);
    }
  }

  if (!switch_nodes.empty()) {
    for (size_t idx = 0; idx < cube_node->GetInputsSize(); ++idx) {
      auto all_cube_in_node = cube_node->GetInDataNodesAndPortIndexs(idx);
      fixpipe_nodes.emplace_back(all_cube_in_node.first);
    }
    for (const auto &node : switch_nodes) {
      fixpipe_nodes.emplace_back(node);
    }
  }
  return;
}

void FixpipeUtils::CollectFixpipe(const ge::GNodePtr &cube_node, std::vector<ge::GNodePtr> &fixpipe_nodes) const {
  if (cube_node == nullptr) {
    return;
  }
  for (size_t idx = 0; idx < cube_node->GetOutputsSize(); ++idx) {
    auto outputNodesPairs = cube_node->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &outputPair : outputNodesPairs) {
      if (GNodeGetType(outputPair.first) == kAscendFixPipeAsc) {
        if (outputPair.second == 0) {
          fixpipe_nodes.emplace_back(outputPair.first);
        }
      }
    }
  }
}

void FixpipeUtils::CommonCollectFixpipeRelativeNodes(const ge::GNodePtr &node,
                                                    std::vector<ge::GNodePtr> &fixpipe_nodes) const {
  auto cube_node = node->GetInDataNodesAndPortIndexs(0);
  if (cube_node.first == nullptr) {
    return;
  }
  CollectSwitchMergeNodes(cube_node.first, fixpipe_nodes);
  CollectFixpipe(cube_node.first, fixpipe_nodes);
}

void FixpipeUtils::SetFixpipeRealtiveNodeScopeId(const ge::GNodePtr &node) const {
  if (GNodeGetType(node) != kAscendFixPipeAsc) {
    return;
  }
  int64_t fixpipe_scope_id = 0;
  GNodeGetAttr(node, kSameFixpipeNodeScope, fixpipe_scope_id);
  if (fixpipe_scope_id != 0) {
    return;
  }
  std::vector<ge::GNodePtr> fixpipe_nodes;
  CommonCollectFixpipeRelativeNodes(node, fixpipe_nodes);
  if (fixpipe_nodes.empty()) {
    return;
  }
  fixpipe_scope_id = GetFixpipeAtomicId();
  for (auto setnode : fixpipe_nodes) {
    if (setnode == nullptr) {
      continue;
    }
    int64_t tmp_int = fixpipe_scope_id;
    GNodeSetAttr(setnode, kSameFixpipeNodeScope, tmp_int);
  }
}

uint32_t FixpipeUtils::GetFixpipeAtomicId() const {
  static std::atomic<uint32_t> global_cmo_id(1);
  return global_cmo_id.fetch_add(1, std::memory_order_relaxed);
}

// api start
ge::graphStatus FixpipeUtils::GetFixpipeNodeList(ge::GNodePtr conv_node, const ge::CustomPassContext &context) {
  ClearPasses();
  if (!ReadConfig(context)) {
    return ge::GRAPH_FAILED;
  }
  OPS_CHECK_NOTNULL(conv_node);
  if (GNodeGetType(conv_node) == kAscendMergeAsc) {
    OPS_LOG_D("Fixpipe", "head node name = %s type = %s is merge not do with",
            GNodeGetName(conv_node).GetString(), GNodeGetType(conv_node).GetString());
    return ge::GRAPH_FAILED;
  }
  if (FixpipeComm::CheckFixpipeAbilityAttr(conv_node, FixpipeAbilityType::NodeCantAccess)) {
    OPS_LOG_D("Fixpipe", "Node[%s, %s] can not do fixpipe fusion.", GNodeGetName(conv_node).GetString(),
            GNodeGetType(conv_node).GetString());
    return ge::GRAPH_FAILED;
  }
  ge::GNodePtr origin_conv_node = conv_node;
  OPS_LOG_D("Fixpipe", "convnode_name = %s type = %s", GNodeGetName(conv_node).GetString(), GNodeGetType(conv_node).GetString());
  ge::GNodePtr merge_node = FixpipeComm::GetMergeNodeByCube(conv_node);
  if (merge_node != nullptr) {
    conv_node = merge_node;
    OPS_LOG_D("Fixpipe", "head node has replaced with merge node name = %s type = %s", GNodeGetName(conv_node).GetString(),
          GNodeGetType(conv_node).GetString());
  }
  OPS_LOG_D("Fixpipe", "Fusion 2");
  auto cur_head_info = std::make_shared<FixPipeNodeInfo> (conv_node, origin_conv_node);
  cur_head_info->SetIsHeadNode(true);

  GenerateMatchedPasses(*cur_head_info);
  cur_head_info_ = cur_head_info;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::SelectFixpipeNodeList(bool firstround_cut) {
  ModfiyMatchedPasses(firstround_cut);
  for (auto &pass : m_matchpasses_) {
    OPS_LOG_D("Fixpipe", "Fusion 3 matchedpass passid = %d", pass.pass_index);
    for (auto &node : pass.m_opnodes) {
      OPS_LOG_D("Fixpipe", "Fusion 3 node_name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
              GNodeGetType(node.GetNode()).GetString());
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FixpipeUtils::CreateFixpipeNode(const string &pass_name, ge::Graph &graph,
                                       std::vector<ge::GNodePtr> &new_nodes) {
  OPS_LOG_D("Fixpipe", "Fusion 4");
  if (InitInput() != ge::GRAPH_SUCCESS) {
    OPS_LOG_W("Fixpipe", "The initialization of the fixpipe node construction strategy was not successful");
    return ge::GRAPH_FAILED;
  }
  ge::graphStatus res = FusionImpl(pass_name, *cur_head_info_, graph, new_nodes);
  for (auto &node : not_support_fixpipe_fusion_nodes_) {
    bool not_support = false;
    GNodeSetAttr(node, kNotSupportFixpipeFusion, not_support);
  }
  return res;
}
// api end


FixPipeAddInputPtr FixpipeUtils::AddInputStrategy(const FixPipePassInfo &match_pass,
                                                 FixPipeFunctionParamPtr funtcparam) {
  switch (funtcparam->GetParaIndex()) {
    case FIXPIPE_INPUT_2_INDEX:  // quant_scale0
      return AddInput2Strategy(match_pass, funtcparam);
    case FIXPIPE_INPUT_3_INDEX:  // relu_weight_0
      return AddInput3Strategy(match_pass, funtcparam);
    case FIXPIPE_INPUT_4_INDEX:  // clip_value_0
      return AddInput4Strategy(match_pass, funtcparam);
    case FIXPIPE_INPUT_5_INDEX:  // quant_scale1
      return AddInput5Strategy(match_pass, funtcparam);
    case FIXPIPE_INPUT_6_INDEX:  // relu_weight_1
      return AddInput6Strategy(match_pass, funtcparam);
    case FIXPIPE_INPUT_7_INDEX:  // clip_value_1
      return AddInput7Strategy(match_pass, funtcparam);
    case FIXPIPE_INPUT_8_INDEX:  // eltwise+antiquant
      return AddInput8Strategy(match_pass, funtcparam);
    case FIXPIPE_INPUT_9_INDEX:  // eltwise+antiquant
      return AddInput9Strategy(match_pass, funtcparam);
    default:
      return nullptr;
  }
}

FixPipeAddInputPtr FixpipeUtils::AddInputSingleUnitStrategy(const FixPipePassInfo &match_pass,
                                                           FixPipeFunctionParamPtr funtcparam,
                                                           const std::string &first_unitname) {
  OPS_LOG_D("Fixpipe", "AddInputSingleUnitStrategy funtcparam->GetInputName = %s inputdex = %d first_unitname = %s",
          funtcparam->GetInputName().c_str(), funtcparam->GetParaIndex(), first_unitname.c_str());
  std::string strategy_key = "DEFAULT";
  for (size_t idx = 1; idx < match_pass.m_opnodes.size(); idx++) {
    auto tofuzednode = match_pass.m_opnodes[idx];
    if (tofuzednode.GetBelongUnitType() == first_unitname) {
      funtcparam->SetFirstIndex(idx);
      funtcparam->SetSecondIndex(idx);
      OPS_LOG_D("Fixpipe", "first_index = %d firstname = %s second_index = %d secondname = %s", funtcparam->GetFirstIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString(),
              funtcparam->GetSecondIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
      strategy_key = funtcparam->GetInputName() + ":" + std::string(GNodeGetType(tofuzednode.GetNode()).GetString());
      break;
    }
  }
  OPS_LOG_D("Fixpipe", "AddInputSingleUnitStrategy strategy key is [%s].", strategy_key.c_str());
  auto iter = m_opmaps_.find(strategy_key);
  if (iter != m_opmaps_.end()) {
    return iter->second;
  }
  OPS_LOG_D("Fixpipe", "AddInputSingleUnitStrategy default");
  return m_opmaps_["DEFAULT"];
}

FixPipeAddInputPtr FixpipeUtils::AddInputAntiStrategy(const FixPipePassInfo &match_pass,
                                                     FixPipeFunctionParamPtr funtcparam,
                                                     const std::string &first_unitname) {
  OPS_LOG_D("Fixpipe", "AddInputAntiStrategy funtcparam->GetInputName = %s inputdex = %d first_unitname = %s",
          funtcparam->GetInputName().c_str(), funtcparam->GetParaIndex(), first_unitname.c_str());
  for (size_t idx = 1; idx < match_pass.m_opnodes.size(); idx++) {
    auto tofuzednode = match_pass.m_opnodes[idx];
    if (tofuzednode.GetBelongUnitType() != first_unitname) {
      continue;
    }
    for (size_t indata_idx = 0; indata_idx < tofuzednode.GetNode()->GetInputsSize(); ++indata_idx) {
      auto inputfather_node_pair = tofuzednode.GetNode()->GetInDataNodesAndPortIndexs(indata_idx);
      auto inputfather_node = inputfather_node_pair.first;
      if (inputfather_node == nullptr) {
        REPORT_OPS_ERROR("[GraphOpt][FixpipePss][AddInputAntiStrategy] node [%s] ",
                        GNodeGetName(tofuzednode.GetNode()).GetString());
        continue;
      }
      if (IsNodeInPass(match_pass.m_opnodes, inputfather_node)) {
        continue;
      }
      if (GNodeGetType(inputfather_node) != kAscendAntiQuantAsc) {
        continue;
      }
      float scale_tmp = 0.0;
      if (!GNodeGetAttr(inputfather_node, ATTR_SCALE, scale_tmp)) {
        OPS_LOG_W("Fixpipe", "Get scale attr of quant node[%s] failed!", GNodeGetName(inputfather_node).GetString());
        return m_opmaps_["DEFAULT"];
      }
      float offset_a = 0.0f;
      GNodeGetAttr(inputfather_node, ATTR_OFFSET, offset_a);
      funtcparam->SetFirstIndex(idx);
      funtcparam->SetSecondIndex(idx);
      ge::TensorDesc tensor_desc;
      inputfather_node->GetInputDesc(0, tensor_desc);
      funtcparam->SetDataType(tensor_desc.GetDataType());
      float tmp_float1 = offset_a;
      GNodeSetAttr(tofuzednode.GetNode(), ATTR_OFFSET, tmp_float1);
      float tmp_float2 = scale_tmp;
      GNodeSetAttr(tofuzednode.GetNode(), ATTR_SCALE, tmp_float2);
      OPS_LOG_D("Fixpipe", "first_index = %d firstname = %s second_index = %d secondname = %s", funtcparam->GetFirstIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString(),
              funtcparam->GetSecondIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
      return m_opmaps_[funtcparam->GetInputName() + ":" + std::string(GNodeGetType(inputfather_node).GetString())];
    }
  }
  OPS_LOG_D("Fixpipe", "AddInputAntiStrategy default");
  return m_opmaps_["DEFAULT"];
}

FixPipeAddInputPtr FixpipeUtils::AddInputTwoUnitStrategy(const FixPipePassInfo &match_pass,
                                                        FixPipeFunctionParamPtr funtcparam,
                                                        const std::string &first_unitname,
                                                        const std::string &second_unitname) {
  OPS_LOG_D("Fixpipe", "AddInputTwoUnitStrategy input name = %s input indexdex = %d first_unitname = %s second_unitname = %s",
          funtcparam->GetInputName().c_str(), funtcparam->GetParaIndex(),
          first_unitname.c_str(), second_unitname.c_str());
  bool has_first_unit = false;
  bool has_second_unit = false;
  for (size_t idx = 1; idx < match_pass.m_opnodes.size(); idx++) {
    auto tofuzednode = match_pass.m_opnodes[idx];
    if (tofuzednode.GetBelongUnitType() == first_unitname) {
      funtcparam->SetFirstIndex(idx);
      has_first_unit = true;
    }
    if (tofuzednode.GetBelongUnitType() == second_unitname) {
      funtcparam->SetSecondIndex(idx);
      has_second_unit = true;
    }
  }
  std::string strategy_key = "DEFAULT";
  if (has_second_unit) {
    if (has_first_unit) {
      OPS_LOG_D("Fixpipe", "1.first_index = %d second_index = %d firstname = %ssecondname = %s", funtcparam->GetFirstIndex(),
              funtcparam->GetSecondIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
      strategy_key = funtcparam->GetInputName() + ":" +
                     std::string(GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString()) + "_" +
                     std::string(GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
    } else {
      funtcparam->SetFirstIndex(funtcparam->GetSecondIndex());
      OPS_LOG_D("Fixpipe", "2.second_index = %d secondname = %s", funtcparam->GetSecondIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
      strategy_key = funtcparam->GetInputName() + ":" +
                     std::string(GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
    }
  } else {
    if (has_first_unit) {
      funtcparam->SetSecondIndex(funtcparam->GetFirstIndex());
      OPS_LOG_D("Fixpipe", "3.first_index = %d firstname = %s", funtcparam->GetFirstIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString());
      strategy_key = funtcparam->GetInputName() + ":" +
                     std::string(GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString());
    }
  }
  OPS_LOG_D("Fixpipe", "AddInputTwoUnitStrategy strategy key is [%s]", strategy_key.c_str());
  auto iter = m_opmaps_.find(strategy_key);
  if (iter != m_opmaps_.end()) {
    return iter->second;
  }
  OPS_LOG_D("Fixpipe", "AddInputTwoUnitStrategy default");
  return m_opmaps_["DEFAULT"];
}

FixPipeAddInputPtr FixpipeUtils::AddInput2Strategy(const FixPipePassInfo &match_pass,
                                                  FixPipeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("quant_scale_0");
  funtcparam->SetDataType(ge::DT_UINT64);
  if (isa_arch_version_ == ISAArchVersion::EN_ISA_ARCH_V350) {
    return AddInputTwoUnitStrategy(match_pass, funtcparam, kPreConv, kPreAct);
  } else {
    return AddInputSingleUnitStrategy(match_pass, funtcparam, kPreConv);
  }
}

FixPipeAddInputPtr FixpipeUtils::AddInput3Strategy(const FixPipePassInfo &match_pass,
                                                  FixPipeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("relu_weight_0");
  funtcparam->SetDataType(ge::DT_FLOAT);
  return AddInputTwoUnitStrategy(match_pass, funtcparam, kPreConv, kPreAct);
}

FixPipeAddInputPtr FixpipeUtils::AddInput4Strategy(const FixPipePassInfo &match_pass,
                                                  FixPipeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("clip_value_0");
  return AddInputTwoUnitStrategy(match_pass, funtcparam, kPreConv, kPreAct);
}

FixPipeAddInputPtr FixpipeUtils::AddInput5Strategy(const FixPipePassInfo &match_pass,
                                                  FixPipeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("quant_scale_1");
  funtcparam->SetDataType(ge::DT_UINT64);
  return AddInputSingleUnitStrategy(match_pass, funtcparam, kPostQuant);
}

FixPipeAddInputPtr FixpipeUtils::AddInput6Strategy(const FixPipePassInfo &match_pass,
                                                  FixPipeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("relu_weight_1");
  funtcparam->SetDataType(ge::DT_FLOAT);
  return AddInputTwoUnitStrategy(match_pass, funtcparam, kPostAct, kPostQuant);
}

FixPipeAddInputPtr FixpipeUtils::AddInput7Strategy(const FixPipePassInfo &match_pass,
                                                  FixPipeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("clip_value_1");
  funtcparam->SetDataType(ge::DT_FLOAT16);
  return AddInputTwoUnitStrategy(match_pass, funtcparam, kPostAct, kPostQuant);
}

FixPipeAddInputPtr FixpipeUtils::AddInput8Strategy(const FixPipePassInfo &match_pass,
                                                  FixPipeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("anti_quant_scale");
  funtcparam->SetDataType(ge::DT_FLOAT16);
  return AddInputAntiStrategy(match_pass, funtcparam, kPostEltwise);
}

FixPipeAddInputPtr FixpipeUtils::AddInput9Strategy(const FixPipePassInfo &match_pass,
                                                  FixPipeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("anti_quant_offset");
  return AddInputAntiStrategy(match_pass, funtcparam, kPostEltwise);
}
}  // namespace ops
