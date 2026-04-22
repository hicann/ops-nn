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

ge::GNodePtr GetCubeNode(const std::stack<PostCubeNodeInfo> &cur_pass) {
  std::stack<PostCubeNodeInfo> tmp_pass(cur_pass);
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
      OPS_LOG_D("PostCube", "Add selected pass_hash = %s", str.c_str());
    }
  }
  return result;
}
} // namespace

constexpr uint32_t POST_CUBE_INPUT_2_INDEX = 2;
constexpr uint32_t POST_CUBE_INPUT_3_INDEX = 3;
constexpr uint32_t POST_CUBE_INPUT_4_INDEX = 4;
constexpr uint32_t POST_CUBE_INPUT_5_INDEX = 5;
constexpr uint32_t POST_CUBE_INPUT_6_INDEX = 6;
constexpr uint32_t POST_CUBE_INPUT_7_INDEX = 7;
constexpr uint32_t POST_CUBE_INPUT_8_INDEX = 8;
constexpr uint32_t POST_CUBE_INPUT_9_INDEX = 9;
constexpr uint32_t ELTWISE_SUB_TYPE = 2;
constexpr uint32_t ELTWISE_ADD_TYPE = 1;
constexpr uint32_t INSERTWOINDEX = 2;
constexpr uint32_t NODECANTACCES = 2;
constexpr uint32_t ELTWISEOPSUBSTRINDEX = 3;
constexpr uint32_t TWOINPUTSIZE =  2U;
const std::string POST_CUBE_TMP_PRE_PASS_NODE = "_post_cube_tmp_pre_pass_node";
const std::string kSamePostCubeNodeScope = "_post_cube_scope";
constexpr uint32_t PROBE_DEPTH_MAX = 5;
constexpr size_t kAicVersionSize = 3;

std::string PostCubeUtils::GetIsaArchVersionStr(const ISAArchVersion isa_arch_version) {
  std::string isa_version_str;
  auto iter = kIsaArchVersionMapStr.find(isa_arch_version);
  if (iter != kIsaArchVersionMapStr.end()) {
    isa_version_str = iter->second;
  }
  return isa_version_str;
}

void PostCubeUtils::ParseIsaArchVersion(fe::PlatFormInfos &platform_infos) {
  isa_arch_version_ = ISAArchVersion::EN_ISA_ARCH_V100;
  // aic version, ISAArchVersion
  std::string aic_version_str;
  if (!platform_infos.GetPlatformRes("version", "AIC_version", aic_version_str) || aic_version_str.empty()) {
    OPS_LOG_W("PostCube", "Aic version is empty.");
    return;
  }
  OPS_LOG_D("PostCube", "Aic version is [%s].", aic_version_str.c_str());
  std::vector<string> aic_version_vec = PostCubeComm::Split(aic_version_str, "-");
  if (aic_version_vec.size() < kAicVersionSize) {
    OPS_LOG_W("PostCube", "The AIC version [%s] is invalid.", aic_version_str.c_str());
    return;
  }
  int32_t aic_version = atoi(aic_version_vec[2].c_str());
  auto iter_aic = kAicIsaArchVersionMap.find(aic_version);
  if (iter_aic != kAicIsaArchVersionMap.end()) {
    isa_arch_version_ = iter_aic->second;
  }
  OPS_LOG_I("PostCube", "ISA arch version is [%s].", GetIsaArchVersionStr(isa_arch_version_).c_str());
  return;
}

bool PostCubeUtils::InitAndBuildUnits(const ge::CustomPassContext &context) {
  std::vector<CONFIGDTYPE> cubeconfigtype;
  std::map<std::string, std::vector<CONFIGDTYPE>> cubmap;
  for (auto &iter : kSupportPostCubeCubeTypeVec) {
    cubmap.emplace(make_pair(iter, cubeconfigtype));
  }
  bool skip_trans = isa_arch_version_ == ISAArchVersion::EN_ISA_ARCH_V350;
  if (skip_trans) {
    cubmap.emplace(TRANSDATA, cubeconfigtype);
  }
  PostCubeUnit cube_ops(kCubeUnit, cubmap);
  m_idxtonodetypes_.push_back(cube_ops);
  unitmapindex_.emplace(make_pair(kCubeUnit, 0));
  uint32_t index = 1;
  std::map<std::string, std::map<std::string, std::vector<CONFIGDTYPE>>> post_cube_map;
  std::vector<std::string> unit_list;
  std::map<std::string, std::vector<std::string>> depends_units;
  PostCubeComm::ReadPlatFormConfig(context, skip_trans, unit_list, depends_units, post_cube_map);
  if (unit_list.empty()) {
    return false;
  }
  for (auto &iter : unit_list) {
    PostCubeUnit unitops(iter, post_cube_map[iter]);
    unitmapindex_.emplace(make_pair(iter, index));
    index++;
    OPS_LOG_D("PostCube", "unit name = %s", iter.c_str());
    for (auto &depends : depends_units[iter]) {
      OPS_LOG_D("PostCube", "depends = %s", depends.c_str());
    }
    unitops.SetDependUnits(depends_units[iter]);
    m_idxtonodetypes_.push_back(unitops);
  }
  return true;
}

void PostCubeUtils::ResolveUnitDependencies() {
  for (auto &unit : m_idxtonodetypes_) {
    const std::vector<std::string> depend_unitsname = unit.GetDependsUnits();
    if (depend_unitsname.empty()) {
      continue;
    }
    std::vector<uint32_t> depend_unitindex;
    for (auto &unitname : depend_unitsname) {
      OPS_LOG_D("PostCube", "match unit name = %s index = [%u] GetName() = %s ", unit.GetName().c_str(), unitmapindex_[unitname],
              unitname.c_str());
      depend_unitindex.push_back(unitmapindex_[unitname]);
    }
    unit.SetDependUnitsIndex(depend_unitindex);
  }
}

bool PostCubeUtils::ReadConfig(const ge::CustomPassContext &context) {
  OPS_LOG_D("PostCube", "Begin to read config.");
  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos optional_infos;
  if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) != ge::GRAPH_SUCCESS) {
    OPS_LOG_W("PostCube", "Fail to get platform info without soc version.");
    return false;
  }
  ParseIsaArchVersion(platform_infos);
  if (!InitAndBuildUnits(context)) {
    return false;
  }
  ResolveUnitDependencies();
  return true;
}

bool PostCubeUtils::IsConfictWithSkipConfig(const PostCubePassInfo &cur_pass, const uint32_t &ret_index) const {
  std::vector<uint32_t> cur_index;
  for (auto &index : cur_pass.m_opnodes) {
    OPS_LOG_D("PostCube", "cur_index = %u", index.GetBelongUnitIndex());
    cur_index.push_back(index.GetBelongUnitIndex());
  }
  return IsConfictWithSkipConfig(cur_index, ret_index);
}

bool PostCubeUtils::IsConfictWithSkipConfig(const std::vector<uint32_t> &index, const uint32_t &ret_index) const {
  std::vector<uint32_t> depend_indexs = m_idxtonodetypes_[ret_index].GetDependsUnitsIndex();
  OPS_LOG_D("PostCube", "ret_index = %u", ret_index);
  for (auto &depend_index : depend_indexs) {
    if (std::find(index.begin(), index.end(), depend_index) == index.end()) {
      OPS_LOG_D("PostCube", "depend_index isn't = %u", depend_index);
      return false;
    }
  }
  return true;
}

bool PostCubeUtils::IsConfictWithSkipConfig(std::stack<uint32_t> index, const uint32_t &ret_index) const {
  std::vector<uint32_t> cur_index;
  while (!index.empty()) {
    auto node = index.top();
    OPS_LOG_D("PostCube", "cur_index = %d", node);
    cur_index.push_back(static_cast<uint32_t>(node));
    index.pop();
  }
  return IsConfictWithSkipConfig(cur_index, ret_index);
}

bool PostCubeUtils::JudgeCachePass(const PostCubeNodeInfo &node, std::stack<uint32_t> &index, uint32_t &ret_index) const {
  uint32_t cur_index;
  OPS_LOG_D("PostCube", "JudgeCachePass start node name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
          GNodeGetType(node.GetNode()).GetString());
  if (index.empty()) {
    cur_index = 0;
  } else {
    cur_index = index.top();
  }
  OPS_LOG_D("PostCube", "JudgeCachePass  cur_index= %d", cur_index);
  bool find_flag = false;
  if (index.empty() && node.GetIsHeadNode() &&
      PostCubeComm::GetPostCubeCubeType(node.GetNode()) != PostCubeCubeType::NotCube) {
    ret_index = 0;
    OPS_LOG_D("PostCube", "JudgeCachePass is headcubenode node = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
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
  OPS_LOG_D("PostCube", "JudgeCachePass node can post_cube name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
          GNodeGetType(node.GetNode()).GetString());
  return IsConfictWithSkipConfig(index, ret_index);
}

bool PostCubeUtils::GetNodeIndex(const PostCubeNodeInfo &node, const uint32_t &index) const {
  for (auto &nodetype : m_idxtonodetypes_[index].GetNode()) {
    std::string node_type;
    if (GNodeGetType(node.GetNode()) == kAscendEltwiseAsc) {
      node_type = GetEltWiseType(node);
    } else {
      node_type = GNodeGetType(node.GetNode()).GetString();
    }
    if (nodetype.first == node_type) {
      OPS_LOG_D("PostCube", "GetNodeIndex node is name = %s type = %s index= %u", GNodeGetName(node.GetNode()).GetString(),
              GNodeGetType(node.GetNode()).GetString(), index);
      return true;
    }
  }
  return false;
}

bool PostCubeUtils::PreCachePass(const PostCubePassInfo &cur_pass, const PostCubeNodeInfo &node) const {
  if (FiltrNodeStrategy(node) != ge::GRAPH_SUCCESS) {
    return false;
  }
  if (!PostCubeComm::CheckIsInVector(cur_pass.m_opnodes)) {
    return false;
  }
  ge::graphStatus ret = FiltrNodeStrategyForQuant(node, cur_pass.m_opnodes[cur_pass.m_opnodes.size() - 1]);
  if (ret != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("PostCube", "PreCachePass post relu+quant node can't be post_cube name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
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

bool PostCubeUtils::PreMatchAcorrdingToPass(const PostCubePassInfo &cur_pass, const PostCubeNodeInfo &node) const {
  bool not_support_post_cube_node = false;
  GNodeGetAttr(node.GetNode(), kNotSupportPostCubeFusion, not_support_post_cube_node);
  if (not_support_post_cube_node) {
    return false;
  }
  if (!IsInWhitelist(node)) {
    OPS_LOG_D("PostCube", "node isn't IsInWhitelist name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return false;
  }
  if (!node.GetIsHeadNode() && PostCubeComm::GetPostCubeCubeType(node.GetNode()) != PostCubeCubeType::NotCube) {
    OPS_LOG_D("PostCube", "node isnt headcube name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return false;
  }
  if (!PreCachePass(cur_pass, node)) {
    OPS_LOG_D("PostCube", "node isn't PreCachePass name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return false;
  }
  OPS_LOG_D("PostCube", "node isn' name = %s type = %s", GNodeGetName(node.GetNode()).GetString(), GNodeGetType(node.GetNode()).GetString());
  return true;
}

bool PostCubeUtils::NeedToCutPass(PostCubePassInfo &m_pass) const {
  if (m_pass.m_opnodes.size() == 1) {
    OPS_LOG_D("PostCube", "only has headnode");
    return true;
  }
  PostCubeNodeInfo node = m_pass.m_opnodes[m_pass.m_opnodes.size() - 1];
  if (node.GetNode()->GetOutputsSize() == 0) {
    OPS_LOG_D("PostCube", "GetOutDataNodes.empty");
    m_pass.m_flag = 2;  // m_flag = 2 DONT NEED, 1NEED, 0 UNKOWN
    return false;
  }
  std::string cube_type = GetMergeInputNodeType(m_pass.m_opnodes[0].GetNode());
  if (cube_type == kConv2DTransposeD && m_pass.m_opnodes[0].GetNode()->GetOutputsSize() > 1) {
    if (GNodeGetType(m_pass.m_opnodes[1].GetNode()) == kAscendQuantAsc ||
        GNodeGetType(m_pass.m_opnodes[1].GetNode()) == kAscendDequantAsc) {
      OPS_LOG_D("PostCube", "Headnode is kConv2DTransposeD and second node is kAscendQuant or kAscendDequant, dont need to cut");
      m_pass.m_flag = 2;
      return false;
    }
    if (GNodeGetType(m_pass.m_opnodes[1].GetNode()) != kAscendQuantAsc  &&
        GNodeGetType(m_pass.m_opnodes[1].GetNode()) != kAscendDequantAsc) {
      OPS_LOG_D("PostCube", "Headnode is kConv2DTransposeD but second node is not kAscendQuant or kAscendDequant, need to cut");
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
      PostCubeNodeInfo grandnode(outputPair.first);
      if (!PreMatchAcorrdingToPass(m_pass, grandnode)) {
        OPS_LOG_D("PostCube", "Has a can't post_cube outputnode name = %s type = %s", GNodeGetName(grandnode.GetNode()).GetString(),
                GNodeGetType(grandnode.GetNode()).GetString());
        m_pass.m_flag = 2;  // m_flag = 2 DONT NEED, 1NEED, 0 UNKOWN
        return false;
      }
    }
  }
  OPS_LOG_D("PostCube", " needto cut, passid = %d", m_pass.pass_index);
  m_pass.m_flag = 1;  // m_flag = 2 DONT NEED, 1NEED, 0 UNKOWN
  return true;
}

std::string PostCubeUtils::GetEltWiseType(const PostCubeNodeInfo &node) const {
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

bool PostCubeUtils::IsInWhitelist(const PostCubeNodeInfo &node) const {
  for (auto &unit : m_idxtonodetypes_) {
    for (auto &nodes : unit.GetNode()) {
      std::string node_type;
      if (GNodeGetType(node.GetNode()) == kAscendEltwiseAsc) {
        node_type = GetEltWiseType(node);
      } else {
        node_type = GNodeGetType(node.GetNode()).GetString();
      }
      if (nodes.first == node_type) {
        OPS_LOG_D("PostCube", "Node isinwhitelist name = %s type = %s", GNodeGetName(node.GetNode()).GetString(), node_type.c_str());
        return true;
      }
    }
  }
  return false;
}
template <typename... Args>
void PostCubeUtils::PrintNodeFilterReason(const PostCubeNodeInfo &node, const Args &...args) const {
  std::stringstream ss;
  (void)std::initializer_list<int>{(ss << args, 0)...};
  ss.flush();
  OPS_LOG_D("PostCube", "node[%s:%s] in post_cube pass filter by %s", GNodeGetName(node.GetNode()).GetString(),
          GNodeGetType(node.GetNode()).GetString(), ss.str().c_str());
}
// extrafilterforhardware
ge::graphStatus PostCubeUtils::FiltrNodeStrategy(const PostCubeNodeInfo &node) const {
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

ge::graphStatus PostCubeUtils::FiltrNodeStrategyForQuant(const PostCubeNodeInfo &cur_node, const PostCubeNodeInfo &prenode) const {
  if (cur_node.GetNode() != nullptr) {
    if (GNodeGetType(cur_node.GetNode()) != kAscendQuantAsc) {
      return ge::GRAPH_SUCCESS;
    }
    bool is_dump_able = false;
    if (GNodeGetAttr(cur_node.GetNode(), kAttrDumpAble, is_dump_able) == ge::GRAPH_SUCCESS && is_dump_able) {
      OPS_LOG_D("PostCube", "Node[%s, %s] is dump able, can not be fused.",
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
    if (!PostCubeComm::CheckConstValueData(prenode.GetNode())) {
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

ge::graphStatus PostCubeUtils::FiltrNodeStrategyForRelu(const PostCubeNodeInfo &node) const {
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
      PostCubeComm::GetStrByDataTypeVec({input_desc.GetDataType()}));
  return ge::GRAPH_FAILED;
}

ge::graphStatus PostCubeUtils::FiltrNodeStrategyForCast(const PostCubeNodeInfo &node) const {
  ge::TensorDesc input_desc;
  ge::TensorDesc output_desc;
  if (node.GetNode()->GetInputDesc(0, input_desc) != ge::GRAPH_SUCCESS ||
      node.GetNode()->GetOutputDesc(0, output_desc) != ge::GRAPH_SUCCESS) {
    PrintNodeFilterReason(node, "input or output is null");
    return ge::GRAPH_FAILED;
  }
  if (input_desc.GetDataType() != ge::DT_FLOAT) {
    PrintNodeFilterReason(node, "input dtype is not fp32, cur dtype is ",
                          PostCubeComm::GetStrByDataTypeVec({input_desc.GetDataType()}));
    return ge::GRAPH_FAILED;
  }
  auto out_data_type = output_desc.GetDataType();
  if (out_data_type != ge::DT_BF16 && out_data_type != ge::DT_FLOAT16) {
    PrintNodeFilterReason(node, "output dtype is not bf16 or fp16, output dtype is ",
                          PostCubeComm::GetStrByDataTypeVec({out_data_type}));
    return ge::GRAPH_FAILED;
  }
  if (node.GetCubeNode() != nullptr &&
      PostCubeComm::CheckPostCubeAbilityAttr(node.GetCubeNode(), PostCubeAbilityType::UseGmAtomicAdd)) {
    PrintNodeFilterReason(node, "Cube node has using atomic write.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::FiltrNodeStrategyForTransData(const PostCubeNodeInfo &node) const {
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

ge::graphStatus PostCubeUtils::CheckEltWiseShapeIsSame(
    const PostCubeNodeInfo &node, const ge::TensorDesc &input_desc0, const ge::TensorDesc &input_desc1) const
{
  // attr check
  int64_t post_cube_support_attr = 0UL;
  GNodeGetAttr(node.GetNode(), kSupportPostCubeAbility, post_cube_support_attr);
  OPS_LOG_D("PostCube", "node %s support_post_cube_ability is 0x%lx", GNodeGetName(node.GetNode()).GetString(), post_cube_support_attr);
  if (PostCubeComm::CheckPostCubeAbilityAttr(node.GetNode(),
                                            PostCubeAbilityType::SupportPostEltwiseBroadcast)) {
    return ge::GRAPH_SUCCESS;
  }

  if (!PostCubeComm::IsShapeEqual(input_desc0.GetShape(), input_desc1.GetShape())) {
    PrintNodeFilterReason(node, "input1 or input2 shape is not same, ", "input0 is ",
                          PostCubeComm::ShapeToString(input_desc0.GetShape()).c_str(), "input1 is ",
                          PostCubeComm::ShapeToString(input_desc1.GetShape()).c_str());
    return ge::GRAPH_FAILED;
  }
  if (!PostCubeComm::IsShapeEqual(input_desc0.GetOriginShape(), input_desc1.GetOriginShape())) {
    PrintNodeFilterReason(node, "input1 or input2 origin shape is not same, ", "input0 is ",
                          PostCubeComm::ShapeToString(input_desc0.GetShape()).c_str(), "input1 is ",
                          PostCubeComm::ShapeToString(input_desc1.GetShape()).c_str());
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::FiltrNodeStrategyForEltWise(const PostCubeNodeInfo &node) const {
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
                          PostCubeComm::GetStrByDataTypeVec({input_desc0.GetDataType()}), "input1 is ",
                          PostCubeComm::GetStrByDataTypeVec({input_desc1.GetDataType()}));
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
  (void)GNodeGetAttr(node.GetNode(), POST_CUBE_TMP_PRE_PASS_NODE, cur_pass_pre_node);
  if (PostCubeComm::CheckPeerOutNode(node.GetNode(), 1) != ge::GRAPH_SUCCESS) {
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

ge::graphStatus PostCubeUtils::JudgeIsMatch(const PostCubeNodeInfo &node, std::stack<uint32_t> &cur_index, uint32_t &ret_index) const {
  if (!IsInWhitelist(node)) {
    OPS_LOG_D("PostCube", "geIsMatch node isn't inwhitelist name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  if (!node.GetIsHeadNode() && PostCubeComm::GetPostCubeCubeType(node.GetNode()) != PostCubeCubeType::NotCube) {
    OPS_LOG_D("PostCube", "JudgeIsMatch node iscube but node headnode name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  if (cur_index.size() > m_idxtonodetypes_.size()) {
    OPS_LOG_D("PostCube", "JudgeIsMatch cur_index size = %zu  m_idxtonodetypes_ size = %zu", cur_index.size(),
            m_idxtonodetypes_.size());
    return ge::GRAPH_FAILED;
  }
  if (!cur_index.empty() && cur_index.top() >= static_cast<uint32_t>(m_idxtonodetypes_.size() - 1)) {
    OPS_LOG_D("PostCube", "JudgeIsMatch cur_index is last top = %u  m_idxtonodetypes_ size = %zu", cur_index.top(),
            m_idxtonodetypes_.size());
    return ge::GRAPH_FAILED;
  }
  if (!JudgeCachePass(node, cur_index, ret_index)) {
    OPS_LOG_D("PostCube", "JudgeIsMatch node isnt JudgeCachePass name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
            GNodeGetType(node.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  if (FiltrNodeStrategy(node) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  OPS_LOG_D("PostCube", "JudgeIsMatch node is true name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
          GNodeGetType(node.GetNode()).GetString());
  return ge::GRAPH_SUCCESS;
}

void PostCubeUtils::GetFusionNodes(const std::stack<PostCubeNodeInfo> &cur_pass,
                                 std::vector<ge::GNodePtr> &fusion_nodes) const {
  std::stack<PostCubeNodeInfo> tmp_pass(cur_pass);
  while (!tmp_pass.empty()) {
    fusion_nodes.emplace_back(tmp_pass.top().GetNode());
    tmp_pass.pop();
  }
}

std::string PostCubeUtils::GetCubeType(const std::stack<PostCubeNodeInfo> &cur_pass) const {
  std::stack<PostCubeNodeInfo> tmp_pass(cur_pass);
  ge::GNodePtr node;
  while (!tmp_pass.empty()) {
    node = tmp_pass.top().GetNode();
    tmp_pass.pop();
  }
  return GetMergeInputNodeType(node);
}

std::string PostCubeUtils::GetMergeInputNodeType(const ge::GNodePtr &merge_node) const {
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

bool PostCubeUtils::CheckEltWiseCycle(const PostCubeNodeInfo &node, const PostCubeMatchParams &fp_mch_params,
                                      uint32_t cur_index) {
  const auto &post_cube_node = node.GetNode();
  if (m_idxtonodetypes_[cur_index].GetName() != kPostEltwise) {
    return false;
  }
  ge::GNodePtr cube_node = GetCubeNode(fp_mch_params.cur_pass);
  int32_t path_size = 0;
  for (size_t idx = 0; idx < post_cube_node->GetInputsSize(); ++idx) {
    const auto input_node = post_cube_node->GetInDataNodesAndPortIndexs(idx);
    GetPath(input_node.first, cube_node, path_size);
  }
  if (path_size > 1) {
    bool tmp_bool = true;
    GNodeSetAttr(post_cube_node, kNotSupportPostCubeFusion, tmp_bool);
    OPS_LOG_D("PostCube", "GenerateMatchedPassesImpl node(%s) attr(not_support_post_cube_fusion) is true.",
            GNodeGetName(post_cube_node).GetString());
    not_support_post_cube_fusion_nodes_.emplace_back(post_cube_node);
    return true;
  }
  return false;
}

void PostCubeUtils::ProcessOutputNodesForPass(const PostCubeNodeInfo &node, PostCubeMatchParams &fp_mch_params) {
  const auto &post_cube_node = node.GetNode();
  for (size_t idx = 0; idx < post_cube_node->GetOutputsSize(); ++idx) {
    const auto outputNodesPairs = post_cube_node->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &out_node : outputNodesPairs) {
      OPS_LOG_D("PostCube", " GenerateMatchedPassesImpl node outnode post_cube name = %s type = %s", GNodeGetName(out_node.first).GetString(),
              GNodeGetType(out_node.first).GetString());
      ge::AscendString post_cube_node_name = GNodeGetName(post_cube_node);
      GNodeSetAttr(out_node.first, POST_CUBE_TMP_PRE_PASS_NODE, post_cube_node_name);
      PostCubeNodeInfo grandnode(out_node.first, node.GetCubeNode());
      GenerateMatchedPassesImpl(grandnode, fp_mch_params);
    }
  }
}

void PostCubeUtils::GenerateMatchedPassesImpl(PostCubeNodeInfo &node, PostCubeMatchParams &fp_mch_params) {
  uint32_t tmp_cur_index = 0;
  const auto &post_cube_node = node.GetNode();
  OPS_LOG_D("PostCube", "GenerateMatchedPassesImpl start node name = %s type = %s", GNodeGetName(post_cube_node).GetString(),
          GNodeGetType(post_cube_node).GetString());
  ge::graphStatus ret = JudgeIsMatch(node, fp_mch_params.cur_index, tmp_cur_index);
  if (ret != ge::GRAPH_SUCCESS) {
    OPS_LOG_D("PostCube", "GenerateMatchedPassesImpl node can't be post_cube name =%s type = %s", GNodeGetName(post_cube_node).GetString(),
            GNodeGetType(post_cube_node).GetString());
    return;
  }
  if (fp_mch_params.cur_index.size() > 1) {
    ret = FiltrNodeStrategyForQuant(node, fp_mch_params.cur_pass.top());
    if (ret != ge::GRAPH_SUCCESS) {
      OPS_LOG_D("PostCube", "GenerateMatchedPassesImpl post relu+quant node can't be post_cube name = %s type = %s",
              GNodeGetName(post_cube_node).GetString(), GNodeGetType(post_cube_node).GetString());
      return;
    }
  }
  if (CheckEltWiseCycle(node, fp_mch_params, tmp_cur_index)) {
    return;
  }

  node.SetNodePostCubeability(1);
  node.SetBelongUnitType(m_idxtonodetypes_[tmp_cur_index].GetName());
  node.SetBelongUnitIndex(tmp_cur_index);
  fp_mch_params.cur_pass.push(node);
  fp_mch_params.cur_index.push(tmp_cur_index);
  OPS_LOG_D("PostCube", "GenerateMatchedPassesImpl node can post_cube name = %s type = %s belong_unitname = %s index = %d",
          GNodeGetName(post_cube_node).GetString(), GNodeGetType(post_cube_node).GetString(), node.GetBelongUnitType().c_str(),
          tmp_cur_index);

  GenerateMatchedPassesFromStack(fp_mch_params.cur_pass, fp_mch_params.post_cube_index++, tmp_cur_index);
  if (post_cube_node->GetOutputsSize() > 1) {
    if (!PostCubeComm::CheckPostCubeAbilityAttr(node.GetCubeNode(),
                                              PostCubeAbilityType::SupportMultipleOutput)) {
      OPS_LOG_D("PostCube", "GenerateMatchedPassesImpl node does not support multiple output post_cube name = %s type = %s",
              GNodeGetName(post_cube_node).GetString(), GNodeGetType(post_cube_node).GetString());
      fp_mch_params.cur_index.pop();
      fp_mch_params.cur_pass.pop();
      return;
    }
  }
  ProcessOutputNodesForPass(node, fp_mch_params);
  fp_mch_params.cur_index.pop();
  fp_mch_params.cur_pass.pop();
  return;
}

void PostCubeUtils::GenerateMatchedPassesFromStack(const std::stack<PostCubeNodeInfo> &cur_pass, const uint32_t &pass_index,
                                                 const uint32_t &cur_index) {
  std::stack<PostCubeNodeInfo> print_pass(cur_pass);
  PostCubePassInfo tmp_pass;
  tmp_pass.m_flag = 0;
  tmp_pass.pass_index = pass_index;
  tmp_pass.unit_index = cur_index;
  std::stack<PostCubeNodeInfo> res_pass;
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

ge::graphStatus PostCubeUtils::ModfiyMatchedPasses(bool firstround_cut) {
  OPS_LOG_D("PostCube", "Start");
  if (m_matchpasses_.empty()) {
    return ge::GRAPH_FAILED;
  }
  std::vector<PostCubePassInfo> tmp_passes(m_matchpasses_);
  m_matchpasses_.clear();
  // 第一轮剪枝只保留同一路径可能的最长pass，当pass结束node的下一个还可以加入时也会被删除
  if (firstround_cut) {
    for (auto &tmp_pass : tmp_passes) {
      if (!NeedToCutPass(tmp_pass)) {
        OPS_LOG_D("PostCube", "ModfiyMatchedPasses pass dont needtocut id = %d", tmp_pass.pass_index);
        m_matchpasses_.push_back(tmp_pass);
      }
    }
  } else {
    // 不跑第一轮剪枝，则挑出所有路径的最长pass
    std::map<std::string, PostCubePassInfo> tmp_pass_map;
    std::vector<std::string> tmp_pass_hash_vec;
    for (auto &tmp_pass : tmp_passes) {
      if (tmp_pass.m_opnodes.size() == 1) {
        OPS_LOG_D("PostCube", "Only has headnode, remove it");
        continue;
      }
      std::string pass_hash = "";
      for (auto &node : tmp_pass.m_opnodes) {
        std::string node_id = "[" +  std::string(GNodeGetName(node.GetNode()).GetString()) + "]";
        pass_hash += node_id;
      }
      tmp_pass_map[pass_hash] = tmp_pass;
      tmp_pass_hash_vec.push_back(pass_hash);
      OPS_LOG_D("PostCube", "Add pass_hash = %s", pass_hash.c_str());
    }
    auto selected_passes = FindNonSubstrings(tmp_pass_hash_vec);
    for (auto &selected_pass : selected_passes) {
      m_matchpasses_.push_back(tmp_pass_map[selected_pass]);
    }
  }
  ModfiyMatchedPasseSecRound();
  return ge::GRAPH_SUCCESS;
}

uint32_t PostCubeUtils::GetUnitIndex(const PostCubePassInfo &post_cube_pass, const PostCubeNodeInfo &post_cube_node) {
  uint32_t unit_index = 0;
  if (!post_cube_pass.m_opnodes.empty()) {
    for (uint32_t i = post_cube_pass.unit_index + 1; i < static_cast<uint32_t>(m_idxtonodetypes_.size()); i++) {
      bool find_flag = GetNodeIndex(post_cube_node, i);
      if (find_flag) {
        unit_index = i;
        break;
      }
    }
  }
  return unit_index;
}

void PostCubeUtils::CollectPostCubeNodesAndInfo(std::unordered_set<ge::GNodePtr> &post_cube_nodes,
                                                std::map<ge::GNodePtr, PostCubeNodeInfo> &post_cube_info) const {
  for (auto &pass : m_matchpasses_) {
    auto pass_nodes = pass.m_opnodes;
    for (auto &pass_node : pass_nodes) {
      post_cube_nodes.emplace(pass_node.GetNode());
      post_cube_info[pass_node.GetNode()] = pass_node;
    }
  }
}

ge::GNodePtr PostCubeUtils::FindBranchNode(const ge::GNodePtr &cube_node, int32_t min_length,
                                           const std::unordered_set<ge::GNodePtr> &post_cube_nodes) const {
  std::vector<ge::GNodePtr> out_nodes;
  for (size_t idx = 0; idx < cube_node->GetOutputsSize(); ++idx) {
    const auto outputNodesPairs = cube_node->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &out_node : outputNodesPairs) {
      if (post_cube_nodes.count(out_node.first) != 0) {
        out_nodes.emplace_back(out_node.first);
      }
    }
  }
  if (out_nodes.size() > 1) {
    return cube_node;
  }
  for (int32_t i = 1; i < min_length; i++) {
    if (m_matchpasses_[0].m_opnodes[i].GetNode()->GetOutputsSize() > 1) {
      return m_matchpasses_[0].m_opnodes[i].GetNode();
    }
  }
  return nullptr;
}

ge::graphStatus PostCubeUtils::ModfiyMatchedPasseSecRound() {
  OPS_LOG_D("PostCube", "Start sec round");
  if (m_matchpasses_.empty()) {
    return ge::GRAPH_SUCCESS;
  }
  ge::GNodePtr cube_node = m_matchpasses_[0].m_opnodes[0].GetNode();
  std::string cube_type = GetMergeInputNodeType(cube_node);
  if (!PostCubeComm::CheckPostCubeAbilityAttr(cube_node, PostCubeAbilityType::SupportMultipleOutput) &&
      m_matchpasses_.size() <= kPostCubeNodeLimited) {
    return ge::GRAPH_SUCCESS;
  } else if (m_matchpasses_.size() == 1) {
    return ge::GRAPH_SUCCESS;
  }
  // each pass, max length is 7
  int32_t min_length = kMaxDepth;
  for (auto &pass : m_matchpasses_) {
    min_length = std::min(min_length, static_cast<int32_t>(pass.m_opnodes.size()));
  }
  std::unordered_set<ge::GNodePtr> post_cube_nodes;
  std::map<ge::GNodePtr, PostCubeNodeInfo> post_cube_info;
  CollectPostCubeNodesAndInfo(post_cube_nodes, post_cube_info);

  std::vector<std::vector<ge::GNodePtr>> matched_pass;
  bool cube_not_post_cube_out_flag = false;
  for (size_t idx = 0; idx < cube_node->GetOutputsSize(); ++idx) {
    const auto outputNodesPairs = cube_node->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &out_node : outputNodesPairs) {
      if (post_cube_nodes.count(out_node.first) == 0) {
        cube_not_post_cube_out_flag = true;
        break;
      }
    }
    if (cube_not_post_cube_out_flag) {
      break;
    }
  }
  ge::GNodePtr branch_node = FindBranchNode(cube_node, min_length, post_cube_nodes);
  if (branch_node == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  if (cube_not_post_cube_out_flag) {
    GenerateSinglePass(cube_node, matched_pass, post_cube_nodes);
  } else {
    GeneratePass(cube_node, branch_node, matched_pass, post_cube_nodes);
  }
  GeneratePostCubePasses(matched_pass, post_cube_info);
  return ge::GRAPH_SUCCESS;
}

void PostCubeUtils::ExtendPassWithPostCubeNodes(std::vector<ge::GNodePtr> &pass,
                                                const ge::GNodePtr &start_node,
                                                const std::unordered_set<ge::GNodePtr> &post_cube_nodes) const {
  auto cur_node = start_node;
  while (post_cube_nodes.count(cur_node) > 0) {
    pass.emplace_back(cur_node);
    if (PostCubeComm::GetOutDataNodesSize(cur_node) != 1) {
      break;
    }
    const auto curOutputNodesPairs = cur_node->GetOutDataNodesAndPortIndexs(0);
    cur_node = curOutputNodesPairs[0].first;
  }
}

void PostCubeUtils::GenerateSinglePass(const ge::GNodePtr &cube_node,
                                     std::vector<std::vector<ge::GNodePtr>> &matched_pass,
                                     std::unordered_set<ge::GNodePtr> &post_cube_nodes) const {
  ge::GNodePtr cur_node;
  std::vector<ge::GNodePtr> pass;
  pass.emplace_back(cube_node);
  for (size_t idx = 0; idx < cube_node->GetOutputsSize(); ++idx) {
    const auto outputNodesPairs = cube_node->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &out_node : outputNodesPairs) {
      if (!matched_pass.empty()) {
        return;
      }
      if (post_cube_nodes.count(out_node.first) > 0) {
        matched_pass.push_back(pass);
        ExtendPassWithPostCubeNodes(matched_pass.back(), out_node.first, post_cube_nodes);
      }
    }
  }
}

void PostCubeUtils::GeneratePass(const ge::GNodePtr &cube_node, const ge::GNodePtr &branch_node,
                               std::vector<std::vector<ge::GNodePtr>> &matched_pass,
                               std::unordered_set<ge::GNodePtr> &post_cube_nodes) const {
  ge::GNodePtr cur_node = cube_node;
  std::vector<ge::GNodePtr> pass;
  while (cur_node != branch_node) {
    pass.emplace_back(cur_node);
    const auto curOutputNodesPairs = cur_node->GetOutDataNodesAndPortIndexs(0);
    cur_node = curOutputNodesPairs[0].first;
  }
  pass.emplace_back(branch_node);
  size_t count = 0;
  if (PostCubeComm::GetOutDataNodesSize(branch_node) > static_cast<uint32_t>(kPostCubeNodeLimited)) {
    matched_pass.push_back(pass);
    count++;
  }
  for (size_t idx = 0; idx < branch_node->GetOutputsSize(); ++idx) {
    const auto outputNodesPairs = branch_node->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &out_node : outputNodesPairs) {
      if (count >= kPostCubeNodeLimited) {
        return;
      }
      if (post_cube_nodes.count(out_node.first) > 0) {
        matched_pass.push_back(pass);
        count++;
        ExtendPassWithPostCubeNodes(matched_pass.back(), out_node.first, post_cube_nodes);
      }
    }
  }
  if (matched_pass.size() < kPostCubeNodeLimited) {
    matched_pass.push_back(pass);
  }
}

void PostCubeUtils::GeneratePostCubePasses(std::vector<std::vector<ge::GNodePtr>> &matched_pass,
                                        std::map<ge::GNodePtr, PostCubeNodeInfo> &post_cube_info) {
  m_matchpasses_.clear();
  uint32_t index_count = 0;
  for (auto &pass : matched_pass) {
    if (pass.size() < kPassMinSize) {
      continue;
    }
    PostCubePassInfo post_cube_pass;
    post_cube_pass.m_flag = kPassFlag;
    post_cube_pass.unit_index = 0;
    post_cube_pass.pass_index = index_count;
    for (auto &node : pass) {
      auto post_cube_node = post_cube_info[node];
      uint32_t unit_index = GetUnitIndex(post_cube_pass, post_cube_node);
      post_cube_pass.m_opnodes.emplace_back(post_cube_node);
      post_cube_pass.unit_index = unit_index;
    }
    index_count++;
    m_matchpasses_.emplace_back(post_cube_pass);
  }
}

void PostCubeUtils::ChangeOrInsertPass(PostCubePassInfo &tmp_pass) {
  bool find_flag = false;
  for (auto &m_already_pass : m_matchpasses_) {
    if (m_already_pass.m_opnodes.size() != tmp_pass.m_opnodes.size()) {
      continue;
    }
    bool m_find_flag = true;
    for (uint32_t index = 0; index < static_cast<uint32_t>(m_already_pass.m_opnodes.size()); index++) {
      if (GNodeGetName(m_already_pass.m_opnodes[index].GetNode()) != GNodeGetName(tmp_pass.m_opnodes[index].GetNode())) {
        m_find_flag = false;
        OPS_LOG_D("PostCube", "ChangeOrInsertPass false index= %d", index);
        break;
      }
    }
    if (m_find_flag) {
      OPS_LOG_D("PostCube", "ChangeOrInsertPass find_flag = true");
      find_flag = true;
      break;
    }
  }
  if (!find_flag) {
    m_matchpasses_.push_back(tmp_pass);
  }
}

ge::graphStatus PostCubeUtils::GenerateMatchedPasses(PostCubeNodeInfo &conv_node) {
  if (!m_matchpasses_.empty()) {
    OPS_LOG_I("PostCube", "GenerateMatchedPasses WARNNIGN passes already exist.");
  }
  PostCubeMatchParams fp_mch_params;
  GenerateMatchedPassesImpl(conv_node, fp_mch_params);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::RelinkHeadEdges(ge::Graph &graph, PostCubeNodeInfo &head_node, PostCubeNodeInfo &post_cubeenhancenode) const {
  if (head_node.GetNode()->GetOutputsSize() == 0) {
    REPORT_OPS_ERROR("[GraphOpt][PostCubePss][RelinkHeadEdges] head_node[%s] outdataanchor empty.",
                    GNodeGetName(head_node.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }

  if (graph.AddDataEdge(*head_node.GetNode(), 0, *post_cubeenhancenode.GetNode(), 0) != ge::GRAPH_SUCCESS) {
    REPORT_OPS_ERROR("[GraphOpt][PostCubePss][RelinkHeadEdges] Fail to add edge between src node [%s] and dst node[%s].",
                    GNodeGetName(head_node.GetNode()).GetString(), GNodeGetName(post_cubeenhancenode.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  ge::TensorDesc output_desc;
  if (head_node.GetNode()->GetOutputDesc(0, output_desc) == ge::GRAPH_SUCCESS) {
    post_cubeenhancenode.GetNode()->UpdateInputDesc(0, output_desc);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::RelinkOutputEdges(ge::Graph &graph, const PostCubePassInfo &match_pass, PostCubeNodeInfo &post_cubeenhancenode) const {
  std::vector<PostCubeNodeInfo> fixednodeids(match_pass.m_opnodes.begin() + 1, match_pass.m_opnodes.end());
  if (!PostCubeComm::CheckIsInVector(match_pass.m_opnodes)) {
    return ge::GRAPH_FAILED;
  }
  PostCubeNodeInfo last_tofuzednode = match_pass.m_opnodes[match_pass.m_opnodes.size() - 1];
  OPS_LOG_D("PostCube", "RelinkOpEdges 4 size = %zu", PostCubeComm::GetOutDataNodesSize(last_tofuzednode.GetNode()));

  const auto outputNodesPairs = last_tofuzednode.GetNode()->GetOutDataNodesAndPortIndexs(0);
  for (const auto &outchild_node : outputNodesPairs) {
    if (outchild_node.first == nullptr) {
      continue;
    }

    if (graph.RemoveEdge(*last_tofuzednode.GetNode(), 0, *outchild_node.first, outchild_node.second) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    if (graph.AddDataEdge(*post_cubeenhancenode.GetNode(), 0, *outchild_node.first, outchild_node.second) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    OPS_LOG_D("PostCube", "src_node = %s dst_node = %s", GNodeGetName(post_cubeenhancenode.GetNode()).GetString(),
              GNodeGetName(outchild_node.first).GetString());
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::RelinkAntiEltWiseEdges(ge::Graph &graph, const ge::GNodePtr &inputfather_node, const PostCubeNodeInfo &tofuzednode,
                                           PostCubeNodeInfo &post_cubeenhancenode) {
  PostCubeNodePair nodepair(inputfather_node, tofuzednode.GetNode());
  m_toantiquantnodes_.emplace(inputfather_node);
  // todo 
  if (inputfather_node->GetInputsSize() != 0) {
    auto input_node = inputfather_node->GetInDataNodesAndPortIndexs(0);
    if (graph.AddDataEdge(*input_node.first, input_node.second, *post_cubeenhancenode.GetNode(), 1) != ge::GRAPH_SUCCESS) {
      REPORT_OPS_ERROR(
          "[GraphOpt][PostCubePss][RelinkAntiEltWiseEdges]  Fail to add  edge between src node [%s] and dst node[%s].",
          GNodeGetName(inputfather_node).GetString(), GNodeGetName(post_cubeenhancenode.GetNode()).GetString());
      return ge::GRAPH_FAILED;
    }
    ge::TensorDesc output_desc;
    if (input_node.first->GetOutputDesc(0, output_desc) == ge::GRAPH_SUCCESS) {
      post_cubeenhancenode.GetNode()->UpdateInputDesc(1, output_desc);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::RelinkOtherEltWiseEdges(ge::Graph &graph, const int32_t &src_port, const PostCubeNodeInfo &tofuzednode,
                                            PostCubeNodeInfo &post_cubeenhancenode, const ge::GNodePtr &inputfather_node) const {
  // todo
  if (inputfather_node != nullptr) {
    if (graph.AddDataEdge(*inputfather_node, src_port, *post_cubeenhancenode.GetNode(), 1) != ge::GRAPH_SUCCESS) {
      REPORT_OPS_ERROR(
          "[GraphOpt][PostCubePss][RelinkOtherEltWiseEdges] Fail to add  edge between src node [%s] and dst node[%s].",
          GNodeGetName(inputfather_node).GetString(), GNodeGetName(post_cubeenhancenode.GetNode()).GetString());
      return ge::GRAPH_FAILED;
    }
    if (!post_cubeenhancenode.GetNode()->GetInControlNodes().empty() &&
        PostCubeComm::HasControlEdge(inputfather_node, tofuzednode.GetNode())) {
      if (graph.AddControlEdge(*inputfather_node, *post_cubeenhancenode.GetNode()) != ge::GRAPH_SUCCESS) {
        REPORT_OPS_ERROR(
            "[GraphOpt][PostCubePss][RelinkOtherEltWiseEdges] Fail to add  edge between src node [%s] and dst node[%s].",
            GNodeGetName(inputfather_node).GetString(), GNodeGetName(post_cubeenhancenode.GetNode()).GetString());
        return ge::GRAPH_FAILED;
      }
    }
  }

  ge::TensorDesc output_desc;
  if (inputfather_node->GetOutputDesc(0, output_desc) == ge::GRAPH_SUCCESS) {
    post_cubeenhancenode.GetNode()->UpdateInputDesc(1, output_desc);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::RelinkEltWiseEdges(ge::Graph &graph, const PostCubePassInfo &match_pass, PostCubeNodeInfo &post_cubeenhancenode) {
  std::vector<PostCubeNodeInfo> fixednodeids(match_pass.m_opnodes.begin() + 1, match_pass.m_opnodes.end());
  OPS_LOG_D("PostCube", "RelinkOpEdges 3");
  for (auto &tofuzednode : fixednodeids) {
    if (tofuzednode.GetBelongUnitType() == kPostEltwise) {
      return RelinkEltWiseEdgesImpl(graph, match_pass, tofuzednode, post_cubeenhancenode);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::RelinkEltWiseEdgesImpl(ge::Graph &graph, const PostCubePassInfo &match_pass, PostCubeNodeInfo &tofuzednode,
                                           PostCubeNodeInfo &post_cubeenhancenode) {
  OPS_LOG_D("PostCube", "RelinkOpEdges 3.1");
  for (size_t idx = 0; idx < tofuzednode.GetNode()->GetInputsSize(); ++idx) {
    auto in_node = tofuzednode.GetNode()->GetInDataNodesAndPortIndexs(idx);
    if (in_node.first == nullptr) {
      continue;
    }
    if (IsNodeInPass(match_pass.m_opnodes, in_node.first)) {
      OPS_LOG_D("PostCube", "inputfathernode is inpost_cubenode name = %s", GNodeGetName(in_node.first).GetString());
      continue;
    }
    if (GNodeGetType(in_node.first) == kAscendAntiQuantAsc) {
      return RelinkAntiEltWiseEdges(graph, in_node.first, tofuzednode, post_cubeenhancenode);
    } else {
      return RelinkOtherEltWiseEdges(graph, in_node.second, tofuzednode, post_cubeenhancenode, in_node.first);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::RelinkOpEdges(ge::Graph &graph, PostCubeNodeInfo &head_node,
                                   const PostCubePassInfo &match_pass, PostCubeNodeInfo &post_cubeenhancenode) {
  if (RelinkHeadEdges(graph, head_node, post_cubeenhancenode) != ge::GRAPH_SUCCESS) {
    REPORT_OPS_ERROR("[GraphOpt][PostCubePss][RelinkOpEdges]  Fail to add edge between head node [%s] and dst node[%s].",
                    GNodeGetName(head_node.GetNode()).GetString(), GNodeGetName(post_cubeenhancenode.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  if (RelinkEltWiseEdges(graph, match_pass, post_cubeenhancenode) != ge::GRAPH_SUCCESS) {
    REPORT_OPS_ERROR("[GraphOpt][PostCubePss][RelinkOpEdges]  Fail to add edge between eltwise and dst node[%s].",
                    GNodeGetName(post_cubeenhancenode.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  if (RelinkOutputEdges(graph, match_pass, post_cubeenhancenode) != ge::GRAPH_SUCCESS) {
    REPORT_OPS_ERROR("[GraphOpt][PostCubePss][RelinkOpEdges]  Fail to add edge between output and dst node[%s].",
                    GNodeGetName(post_cubeenhancenode.GetNode()).GetString());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

bool PostCubeUtils::IsNodeInPass(const std::vector<PostCubeNodeInfo> &fixednodeids, const ge::GNodePtr &input_node) const {
  bool found = (fixednodeids.end() != std::find(fixednodeids.begin(), fixednodeids.end(), input_node));
  OPS_LOG_D("PostCube", "IsNodeInPass is %u name = %s type = %s", found, GNodeGetName(input_node).GetString(),
          GNodeGetType(input_node).GetString());
  return found;
}

ge::graphStatus PostCubeUtils::AddInputs(ge::Graph &graph, const PostCubePassInfo &match_pass,
                              const ge::GNodePtr &post_cubenode, std::vector<ge::GNodePtr> &new_nodes) {
  for (uint32_t i = POST_CUBE_INPUT_2_INDEX; i < 10; i++) {
    PostCubeFunctionParamPtr funtcparam;
    OPS_MAKE_SHARED(funtcparam = std::make_shared<PostCubeFunctionParam>("Dummy", post_cubenode, i), return ge::GRAPH_FAILED);
    PostCubeAddInputPtr addinputptr = AddInputStrategy(match_pass, funtcparam);
    if (addinputptr != nullptr) {
      if (addinputptr->DoAddInput(graph, match_pass, funtcparam, new_nodes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::UpdateInputDesc(ge::GNode &cur_new_post_cubenode) const {
  ge::Shape shape{};
  ge::TensorDesc fakedesc(shape, ge::FORMAT_RESERVED, ge::DT_UNDEFINED);
  cur_new_post_cubenode.UpdateInputDesc(0, fakedesc);
  cur_new_post_cubenode.UpdateInputDesc(1, fakedesc);
  cur_new_post_cubenode.UpdateInputDesc(POST_CUBE_INPUT_2_INDEX, fakedesc);
  cur_new_post_cubenode.UpdateInputDesc(POST_CUBE_INPUT_3_INDEX, fakedesc);
  cur_new_post_cubenode.UpdateInputDesc(POST_CUBE_INPUT_4_INDEX, fakedesc);
  cur_new_post_cubenode.UpdateInputDesc(POST_CUBE_INPUT_5_INDEX, fakedesc);
  cur_new_post_cubenode.UpdateInputDesc(POST_CUBE_INPUT_6_INDEX, fakedesc);
  cur_new_post_cubenode.UpdateInputDesc(POST_CUBE_INPUT_7_INDEX, fakedesc);
  cur_new_post_cubenode.UpdateInputDesc(POST_CUBE_INPUT_8_INDEX, fakedesc);
  cur_new_post_cubenode.UpdateInputDesc(POST_CUBE_INPUT_9_INDEX, fakedesc);
  return ge::GRAPH_SUCCESS;
}

void PostCubeUtils::PrepareFusionAttributes(const std::vector<PostCubeNodeInfo> &fixednodeids,
                                            std::vector<ge::AscendString> &fuzedoptypes_asc,
                                            std::vector<ge::AscendString> &activatepost_cubeunits_asc,
                                            ge::AscendString &eltwise_mode_asc) const {
  std::vector<std::string> fuzedoptypes;
  std::vector<std::string> activatepost_cubeunits;
  std::string eltwiseops;
  for (const auto &tofuzednode : fixednodeids) {
    fuzedoptypes.push_back(GNodeGetType(tofuzednode.GetNode()).GetString());
    activatepost_cubeunits.push_back(tofuzednode.GetBelongUnitType());
    if (tofuzednode.GetBelongUnitType() == kPostEltwise) {
      if (GNodeGetType(tofuzednode.GetNode()) == kAscendEltwiseAsc) {
        eltwiseops = GetEltWiseType(tofuzednode);
      } else {
        eltwiseops = GNodeGetType(tofuzednode.GetNode()).GetString();
      }
    }
  }
  if (!fixednodeids.empty() &&
      GNodeGetType(fixednodeids[0].GetNode()) == kAscendDequantAsc &&
      fixednodeids[0].GetNode()->HasAttr(kAscendOffsetAsc)) {
    if (fixednodeids.size() > 1 && fixednodeids[1].GetBelongUnitType() == kPreAct) {
      fuzedoptypes.insert(fuzedoptypes.cbegin() + INSERTWOINDEX, ASCEND_QUANT);
    } else {
      fuzedoptypes.insert(fuzedoptypes.cbegin() + 1, ASCEND_QUANT);
    }
  }
  std::transform(eltwiseops.begin(), eltwiseops.end(), eltwiseops.begin(), ::toupper);
  std::string tmp_str1(eltwiseops.substr(0, ELTWISEOPSUBSTRINDEX));
  eltwise_mode_asc = ge::AscendString(tmp_str1.c_str());
  for (const auto &it : fuzedoptypes) {
    fuzedoptypes_asc.push_back(ge::AscendString(it.c_str()));
  }
  for (const auto &it : activatepost_cubeunits) {
    activatepost_cubeunits_asc.push_back(ge::AscendString(it.c_str()));
  }
}

ge::GNodePtr PostCubeUtils::CreatePostCubeNode(const PostCubePassInfo &match_pass, const PostCubeNodeInfo &head_node,
                                           ge::Graph &graph) const {
  OPS_LOG_D("PostCube", "CreatePostCubeNode 1 ");
  bool need_judge = true;
  std::string op_name = std::string(GNodeGetName(head_node.GetNode()).GetString()) + "_pass." + std::to_string(match_pass.pass_index);
  ge::GNode post_cubenode = ge::es::CompliantNodeBuilder(&graph)
                        .OpType(kPostCube.c_str())
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

  post_cubenode.SetAttr(kAscendNeedJudgeDtypeAsc, need_judge);
  std::vector<PostCubeNodeInfo> fixednodeids(match_pass.m_opnodes.begin() + 1, match_pass.m_opnodes.end());
  std::vector<ge::AscendString> fuzedoptypes_asc;
  std::vector<ge::AscendString> activatepost_cubeunits_asc;
  ge::AscendString eltwise_mode_asc;
  PrepareFusionAttributes(fixednodeids, fuzedoptypes_asc, activatepost_cubeunits_asc, eltwise_mode_asc);
  post_cubenode.SetAttr(kAscendEltwiseModeAsc, eltwise_mode_asc);
  post_cubenode.SetAttr(kAscendFusionOpListAsc, fuzedoptypes_asc);
  post_cubenode.SetAttr(kAscendUnitListAsc, activatepost_cubeunits_asc);
  UpdateInputDesc(post_cubenode);
  if (!PostCubeComm::CheckIsInVector(match_pass.m_opnodes)) {
    return nullptr;
  }
  PostCubeNodeInfo last_tofuzednode = match_pass.m_opnodes[match_pass.m_opnodes.size() - 1];
  ge::TensorDesc output_desc;
  if (last_tofuzednode.GetNode()->GetOutputDesc(0, output_desc) == ge::GRAPH_SUCCESS) {
    post_cubenode.UpdateOutputDesc(0, output_desc);
    OPS_LOG_D("PostCube", "CreatePostCubeNode node_name = %s.", op_name.c_str());
  }

  ge::GNodePtr post_cubenode_ptr = graph.FindNodeByName(op_name.c_str());
  return post_cubenode_ptr;
}

ge::graphStatus PostCubeUtils::DeleteToFusedNodeEdge(ge::Graph &graph, const PostCubePassInfo &match_pass,
                                          std::set<ge::GNodePtr> &todeletenode) const {
  std::vector<PostCubeNodeInfo> fixednodeids(match_pass.m_opnodes.begin() + 1, match_pass.m_opnodes.end());
  for (auto &tofuzednode : fixednodeids) {
    if (graph.FindNodeByName(GNodeGetName(tofuzednode.GetNode())) == nullptr) {
      continue;
    }
    OPS_LOG_D("PostCube", "DeleteToFusedNodeEdge 2 name = %s type = %s", GNodeGetName(tofuzednode.GetNode()).GetString(),
            GNodeGetType(tofuzednode.GetNode()).GetString());
    todeletenode.emplace(tofuzednode.GetNode());
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::DeleteNode(ge::Graph &graph, const std::set<ge::GNodePtr> &todeletenode) {
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

ge::graphStatus PostCubeUtils::FusionImpl(const string &pass_name, PostCubeNodeInfo &head_node, ge::Graph &graph,
                               std::vector<ge::GNodePtr> &new_nodes) {
  std::vector<ge::GNodePtr> post_cube_nodes;
  for (auto &pass : m_matchpasses_) {
    std::vector<ge::GNodePtr> original_nodes;
    std::unordered_set<ge::GNodePtr> original_nodes_set;
    for (auto &node : pass.m_opnodes) {
      /* 1. trans node is not original node
       * 2. cube node is not in the fusion node list, so we skip */
      const auto &node_ptr = node.GetNode();
      const std::string &op_type = GNodeGetType(node_ptr).GetString();
      bool is_cube_op = std::find(kSupportPostCubeCubeTypeVec.begin(), kSupportPostCubeCubeTypeVec.end(), op_type) !=
          kSupportPostCubeCubeTypeVec.end();
      if (op_type == CAST || op_type == TRANSDATA || is_cube_op) {
        continue;
      }
      original_nodes_set.emplace(node_ptr);
    }
    OPS_LOG_D("PostCube", "FusionImpl 1.1");
    ge::GNodePtr post_cubenode = CreatePostCubeNode(pass, head_node, graph);
    OPS_CHECK_NOTNULL(post_cubenode);
    OPS_LOG_D("PostCube", "FusionImpl 1.2");
    PostCubeNodeInfo post_cubeenhancenode(post_cubenode);
    OPS_LOG_D("PostCube", "FusionImpl 1.3");
    AddInputs(graph, pass, post_cubenode, new_nodes);
    OPS_LOG_D("PostCube", "FusionImpl 1.4");
    RelinkOpEdges(graph, head_node, pass, post_cubeenhancenode);
    OPS_LOG_D("PostCube", "FusionImpl 1.5");
    std::vector<ge::GNodePtr> fusion_nodes;
    fusion_nodes.push_back(post_cubenode);
    original_nodes.assign(original_nodes_set.begin(), original_nodes_set.end());
    OPS_LOG_D("PostCube", "FusionImpl 1.6");
    post_cube_nodes.push_back(post_cubenode);
  }
  OPS_LOG_D("PostCube", "PostCube_nodes size is %zu.", post_cube_nodes.size());
  std::set<ge::GNodePtr> todeletenode;
  for (auto &pass : m_matchpasses_) {
    DeleteToFusedNodeEdge(graph, pass, todeletenode);
  }
  DeleteNode(graph, todeletenode);
  if (post_cube_nodes.empty()) {
    return ge::GRAPH_FAILED;
  }
  for (const auto &node : post_cube_nodes) {
    SetPostCubeRealtiveNodeScopeId(node);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::InitInput2() {
  if (isa_arch_version_ != ISAArchVersion::EN_ISA_ARCH_V350) {
    PostCubeAddInputPtr strategy21ptr = nullptr;
    OPS_MAKE_SHARED(strategy21ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy21>()),
                   return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendQuant, strategy21ptr));
    OPS_LOG_D("PostCube", "strategy21 name = %s", ("quant_scale_0:" + kAscendQuant).c_str());
    PostCubeAddInputPtr strategy22ptr = nullptr;
    OPS_MAKE_SHARED(strategy22ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy22>()),
                   return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendRequant, strategy22ptr));
    OPS_LOG_D("PostCube", "strategy22 name = %s", ("quant_scale_0:" + kAscendRequant).c_str());
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant, strategy22ptr));
    OPS_LOG_D("PostCube", "strategy22 name = %s", ("quant_scale_0:" + kAscendDequant).c_str());
  } else {
    PostCubeAddInputPtr strategy23ptr = nullptr;
    OPS_MAKE_SHARED(strategy23ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy23>()),
                   return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant, strategy23ptr));
    OPS_LOG_D("PostCube", "Add strategy23 name = %s", ("quant_scale_0:" + kAscendDequant).c_str());
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + RELU, strategy23ptr));
    OPS_LOG_D("PostCube", "Add strategy23 name = %s", ("quant_scale_0:" + kAscendDequant + "_" + RELU).c_str());
    PostCubeAddInputPtr strategy24ptr = nullptr;
    OPS_MAKE_SHARED(strategy24ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy24>()),
                   return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + kLeakyRelu, strategy24ptr));
    OPS_LOG_D("PostCube", "Add strategy24 name = %s", ("quant_scale_0:" + kAscendDequant + "_" + kLeakyRelu).c_str());

    PostCubeAddInputPtr strategy25ptr = nullptr;
    OPS_MAKE_SHARED(strategy25ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy25>()),
                 return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + kPRelu, strategy25ptr));
    OPS_LOG_D("PostCube", "Add strategy25 name = %s", ("quant_scale_0:" + kAscendDequant + "_" + kPRelu).c_str());

    PostCubeAddInputPtr pattern_dequant_lut_ptr = nullptr;
    OPS_MAKE_SHARED(pattern_dequant_lut_ptr =
                       std::shared_ptr<PostCubeAddInputBase>(std::make_shared<AddInputStrategyDequntLut>()),
                   return ge::GRAPH_FAILED);
    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + kTanh, pattern_dequant_lut_ptr));
    OPS_LOG_D("PostCube", "Add strategy for pattern [%s]", ("quant_scale_0:" + kAscendDequant + "_" + kTanh).c_str());

    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + kElu, pattern_dequant_lut_ptr));
    OPS_LOG_D("PostCube", "Add strategy for pattern [%s]", ("quant_scale_0:" + kAscendDequant + "_" + kElu).c_str());

    m_opmaps_.emplace(make_pair("quant_scale_0:" + kAscendDequant + "_" + kSigmoid, pattern_dequant_lut_ptr));
    OPS_LOG_D("PostCube", "Add strategy for pattern [%s]", ("quant_scale_0:" + kAscendDequant + "_" + kSigmoid).c_str());
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::InitInput3() {
  if (isa_arch_version_ == ISAArchVersion::EN_ISA_ARCH_V350) {
    OPS_LOG_D("PostCube", "There is no need to init strategy for input3.");
    return ge::GRAPH_SUCCESS;
  }
  PostCubeAddInputPtr strategy31ptr = nullptr;
  OPS_MAKE_SHARED(strategy31ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy31>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendQuant + "_" + kPRelu, strategy31ptr));
  OPS_LOG_D("PostCube", "strategy31 name = %s", ("relu_weight_0:" + kAscendQuant + "_" + kPRelu).c_str());
  PostCubeAddInputPtr strategy32ptr = nullptr;
  OPS_MAKE_SHARED(strategy32ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy32>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendDequant + "_" + kPRelu, strategy32ptr));
  OPS_LOG_D("PostCube", "strategy32 name = %s", ("relu_weight_0:" + kAscendDequant + "_" + kPRelu).c_str());
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendRequant + "_" + kPRelu, strategy32ptr));
  OPS_LOG_D("PostCube", "strategy32 name = %s", ("relu_weight_0:" + kAscendRequant + "_" + kPRelu).c_str());
  PostCubeAddInputPtr strategy33ptr = nullptr;
  OPS_MAKE_SHARED(strategy33ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy33>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kPRelu, strategy33ptr));
  OPS_LOG_D("PostCube", "strategy33 name = %s", ("relu_weight_0:" + kPRelu).c_str());
  PostCubeAddInputPtr strategy34ptr = nullptr;
  OPS_MAKE_SHARED(strategy34ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy34>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendQuant + "_" + kLeakyRelu, strategy34ptr));
  OPS_LOG_D("PostCube", "strategy34 name = %s", ("relu_weight_0:" + kAscendQuant + "_" + kLeakyRelu).c_str());
  PostCubeAddInputPtr strategy35ptr = nullptr;
  OPS_MAKE_SHARED(strategy35ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy35>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendDequant + "_" + kLeakyRelu, strategy35ptr));
  OPS_LOG_D("PostCube", "strategy35 name = %s", ("relu_weight_0:" + kAscendDequant + "_" + kLeakyRelu).c_str());
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kAscendRequant + "_" + kLeakyRelu, strategy35ptr));
  OPS_LOG_D("PostCube", "strategy35 name = %s", ("relu_weight_0:" + kAscendRequant + "_" + kLeakyRelu).c_str());
  PostCubeAddInputPtr strategy36ptr = nullptr;
  OPS_MAKE_SHARED(strategy36ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy36>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + kLeakyRelu, strategy36ptr));
  OPS_LOG_D("PostCube", "strategy36 name = %s", ("relu_weight_0:" + kLeakyRelu).c_str());
  PostCubeAddInputPtr strategy37ptr = nullptr;
  OPS_MAKE_SHARED(strategy37ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy37>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + CAST + "_" + kPRelu, strategy37ptr));
  OPS_LOG_D("PostCube", "strategy37 name = %s", ("relu_weight_0:" + CAST + "_" + kPRelu).c_str());
  PostCubeAddInputPtr strategy38ptr = nullptr;
  OPS_MAKE_SHARED(strategy38ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy38>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_0:" + CAST + "_" + kLeakyRelu, strategy38ptr));
  OPS_LOG_D("PostCube", "strategy38 name = %s", ("relu_weight_0:" + CAST + "_" + kLeakyRelu).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::InitInput4() {
  PostCubeAddInputPtr strategy41ptr = nullptr;
  OPS_MAKE_SHARED(strategy41ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy41>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("clip_value_0:" + RELU6, strategy41ptr));
  OPS_LOG_D("PostCube", "strategy41 name = %s", ("clip_value_0:" + RELU6).c_str());
  PostCubeAddInputPtr strategy42ptr = nullptr;
  OPS_MAKE_SHARED(strategy42ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy42>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("clip_value_0:" + kAscendQuant + "_" + RELU6, strategy42ptr));
  OPS_LOG_D("PostCube", "strategy42 name = %s", ("clip_value_0:" + kAscendQuant + "_" + RELU6).c_str());
  PostCubeAddInputPtr strategy43ptr = nullptr;
  OPS_MAKE_SHARED(strategy43ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy43>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("clip_value_0:" + kAscendDequant + "_" + RELU6, strategy43ptr));
  OPS_LOG_D("PostCube", "strategy43 name = %s", ("clip_value_0:" + kAscendDequant + "_" + RELU6).c_str());
  m_opmaps_.emplace(make_pair("clip_value_0:" + kAscendRequant + "_" + RELU6, strategy43ptr));
  OPS_LOG_D("PostCube", "strategy43 name = %s", ("clip_value_0:" + kAscendRequant + "_" + RELU6).c_str());
  PostCubeAddInputPtr strategy44ptr = nullptr;
  OPS_MAKE_SHARED(strategy44ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy44>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("clip_value_0:" + CAST + "_" + RELU6, strategy44ptr));
  OPS_LOG_D("PostCube", "strategy44 name = %s", ("clip_value_0:" + CAST + "_" + RELU6).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::InitInput5() {
  PostCubeAddInputPtr strategy5xptr = nullptr;
  if (isa_arch_version_ != ISAArchVersion::EN_ISA_ARCH_V350) {
    OPS_MAKE_SHARED(strategy5xptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy51>()),
                   return ge::GRAPH_FAILED);
  } else {
    OPS_MAKE_SHARED(strategy5xptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy52>()),
                   return ge::GRAPH_FAILED);
  }
  m_opmaps_.emplace(make_pair("quant_scale_1:" + kAscendQuant, strategy5xptr));
  OPS_LOG_D("PostCube", "strategy5x name = %s", ("quant_scale_1:" + kAscendQuant).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::InitInput6() {
  PostCubeAddInputPtr strategy61ptr = nullptr;
  OPS_MAKE_SHARED(strategy61ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy61>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_1:" + kPRelu + "_" + kAscendQuant, strategy61ptr));
  OPS_LOG_D("PostCube", "strategy61 name = %s", ("relu_weight_1:" + kPRelu + "_" + kAscendQuant).c_str());

  PostCubeAddInputPtr strategy62ptr = nullptr;
  OPS_MAKE_SHARED(strategy62ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy62>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_1:" + kPRelu, strategy62ptr));
  OPS_LOG_D("PostCube", "strategy62 name = %s", ("relu_weight_1:" + kPRelu).c_str());

  PostCubeAddInputPtr strategy63ptr = nullptr;
  OPS_MAKE_SHARED(strategy63ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy63>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_1:" + kLeakyRelu + "_" + kAscendQuant, strategy63ptr));
  OPS_LOG_D("PostCube", "strategy63 name = %s", ("relu_weight_1:" + kLeakyRelu + "_" + kAscendQuant).c_str());
  PostCubeAddInputPtr strategy64ptr = nullptr;
  OPS_MAKE_SHARED(strategy64ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy64>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("relu_weight_1:" + kLeakyRelu, strategy64ptr));
  OPS_LOG_D("PostCube", "strategy64 name = %s", ("relu_weight_1:" + kLeakyRelu).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::InitInput7() {
  PostCubeAddInputPtr strategy71ptr = nullptr;
  OPS_MAKE_SHARED(strategy71ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy71>()),
                 return ge::GRAPH_FAILED);
  PostCubeAddInputPtr strategy72ptr = nullptr;
  OPS_MAKE_SHARED(strategy72ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy72>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("clip_value_1:" + RELU6, strategy71ptr));
  OPS_LOG_D("PostCube", "strategy71 name = %s", ("clip_value_1:" + RELU6).c_str());
  m_opmaps_.emplace(make_pair("clip_value_1:" + RELU6 + "_" + kAscendQuant, strategy72ptr));
  OPS_LOG_D("PostCube", "strategy72 name = %s", ("clip_value_1:" + RELU6 + "_" + kAscendQuant).c_str());
  return ge::GRAPH_SUCCESS;
}


ge::graphStatus PostCubeUtils::InitInput8() {
  PostCubeAddInputPtr strategy81ptr = nullptr;
  OPS_MAKE_SHARED(strategy81ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy81>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("anti_quant_scale:" + kAscendAntiQuant, strategy81ptr));
  OPS_LOG_D("PostCube", "strategy81 name = %s", ("anti_quant_scale:" + kAscendAntiQuant).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::InitInput9() {
  PostCubeAddInputPtr strategy91ptr = nullptr;
  OPS_MAKE_SHARED(strategy91ptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategy91>()),
                 return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("anti_quant_offset:" + kAscendAntiQuant, strategy91ptr));
  OPS_LOG_D("PostCube", "strategy91 name = %s", ("anti_quant_offset:" + kAscendAntiQuant).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::InitInputDefault() {
  PostCubeAddInputPtr strategydefaultptr = nullptr;
  OPS_MAKE_SHARED(
      strategydefaultptr = std::shared_ptr<PostCubeAddInputBase>(std::make_shared<PostCubeAddInputStrategyDefault>()),
      return ge::GRAPH_FAILED);
  m_opmaps_.emplace(make_pair("DEFAULT", strategydefaultptr));
  OPS_LOG_D("PostCube", "strategydefault name = DEFAULT");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::InitInput() {
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

void PostCubeUtils::ClearPasses() {
  m_matchpasses_.clear();
  m_toantiquantnodes_.clear();
  m_idxtonodetypes_.clear();
  unitmapindex_.clear();
  not_support_post_cube_fusion_nodes_.clear();
}

void PostCubeUtils::CollectSwitchMergeNodes(const ge::GNodePtr &cube_node,
                                          std::vector<ge::GNodePtr> &post_cube_nodes) const {
  post_cube_nodes.emplace_back(cube_node);
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
      post_cube_nodes.emplace_back(all_cube_in_node.first);
    }
    for (const auto &node : switch_nodes) {
      post_cube_nodes.emplace_back(node);
    }
  }
  return;
}

void PostCubeUtils::CollectPostCube(const ge::GNodePtr &cube_node, std::vector<ge::GNodePtr> &post_cube_nodes) const {
  if (cube_node == nullptr) {
    return;
  }
  for (size_t idx = 0; idx < cube_node->GetOutputsSize(); ++idx) {
    auto outputNodesPairs = cube_node->GetOutDataNodesAndPortIndexs(idx);
    for (const auto &outputPair : outputNodesPairs) {
      if (GNodeGetType(outputPair.first) == kAscendPostCubeAsc) {
        if (outputPair.second == 0) {
          post_cube_nodes.emplace_back(outputPair.first);
        }
      }
    }
  }
}

void PostCubeUtils::CommonCollectPostCubeRelativeNodes(const ge::GNodePtr &node,
                                                    std::vector<ge::GNodePtr> &post_cube_nodes) const {
  auto cube_node = node->GetInDataNodesAndPortIndexs(0);
  if (cube_node.first == nullptr) {
    return;
  }
  CollectSwitchMergeNodes(cube_node.first, post_cube_nodes);
  CollectPostCube(cube_node.first, post_cube_nodes);
}

void PostCubeUtils::SetPostCubeRealtiveNodeScopeId(const ge::GNodePtr &node) const {
  if (GNodeGetType(node) != kAscendPostCubeAsc) {
    return;
  }
  int64_t post_cube_scope_id = 0;
  GNodeGetAttr(node, kSamePostCubeNodeScope, post_cube_scope_id);
  if (post_cube_scope_id != 0) {
    return;
  }
  std::vector<ge::GNodePtr> post_cube_nodes;
  CommonCollectPostCubeRelativeNodes(node, post_cube_nodes);
  if (post_cube_nodes.empty()) {
    return;
  }
  post_cube_scope_id = GetPostCubeAtomicId();
  for (auto setnode : post_cube_nodes) {
    if (setnode == nullptr) {
      continue;
    }
    int64_t tmp_int = post_cube_scope_id;
    GNodeSetAttr(setnode, kSamePostCubeNodeScope, tmp_int);
  }
}

uint32_t PostCubeUtils::GetPostCubeAtomicId() const {
  static std::atomic<uint32_t> global_cmo_id(1);
  return global_cmo_id.fetch_add(1, std::memory_order_relaxed);
}

// api start
ge::graphStatus PostCubeUtils::GetPostCubeNodeList(ge::GNodePtr conv_node, const ge::CustomPassContext &context) {
  ClearPasses();
  if (!ReadConfig(context)) {
    return ge::GRAPH_FAILED;
  }
  OPS_CHECK_NOTNULL(conv_node);
  if (GNodeGetType(conv_node) == kAscendMergeAsc) {
    OPS_LOG_D("PostCube", "head node name = %s type = %s is merge not do with",
            GNodeGetName(conv_node).GetString(), GNodeGetType(conv_node).GetString());
    return ge::GRAPH_FAILED;
  }
  if (PostCubeComm::CheckPostCubeAbilityAttr(conv_node, PostCubeAbilityType::NodeCantAccess)) {
    OPS_LOG_D("PostCube", "Node[%s, %s] can not do post_cube fusion.", GNodeGetName(conv_node).GetString(),
            GNodeGetType(conv_node).GetString());
    return ge::GRAPH_FAILED;
  }
  ge::GNodePtr origin_conv_node = conv_node;
  OPS_LOG_D("PostCube", "convnode_name = %s type = %s", GNodeGetName(conv_node).GetString(), GNodeGetType(conv_node).GetString());
  ge::GNodePtr merge_node = PostCubeComm::GetMergeNodeByCube(conv_node);
  if (merge_node != nullptr) {
    conv_node = merge_node;
    OPS_LOG_D("PostCube", "head node has replaced with merge node name = %s type = %s", GNodeGetName(conv_node).GetString(),
          GNodeGetType(conv_node).GetString());
  }
  OPS_LOG_D("PostCube", "Fusion 2");
  auto cur_head_info = std::make_shared<PostCubeNodeInfo> (conv_node, origin_conv_node);
  cur_head_info->SetIsHeadNode(true);

  GenerateMatchedPasses(*cur_head_info);
  cur_head_info_ = cur_head_info;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::SelectPostCubeNodeList(bool firstround_cut) {
  ModfiyMatchedPasses(firstround_cut);
  for (auto &pass : m_matchpasses_) {
    OPS_LOG_D("PostCube", "Fusion 3 matchedpass passid = %d", pass.pass_index);
    for (auto &node : pass.m_opnodes) {
      OPS_LOG_D("PostCube", "Fusion 3 node_name = %s type = %s", GNodeGetName(node.GetNode()).GetString(),
              GNodeGetType(node.GetNode()).GetString());
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostCubeUtils::CreatePostCubeNode(const string &pass_name, ge::Graph &graph,
                                       std::vector<ge::GNodePtr> &new_nodes) {
  OPS_LOG_D("PostCube", "Fusion 4");
  if (InitInput() != ge::GRAPH_SUCCESS) {
    OPS_LOG_W("PostCube", "The initialization of the post_cube node construction strategy was not successful");
    return ge::GRAPH_FAILED;
  }
  ge::graphStatus res = FusionImpl(pass_name, *cur_head_info_, graph, new_nodes);
  for (auto &node : not_support_post_cube_fusion_nodes_) {
    bool not_support = false;
    GNodeSetAttr(node, kNotSupportPostCubeFusion, not_support);
  }
  return res;
}
// api end


PostCubeAddInputPtr PostCubeUtils::AddInputStrategy(const PostCubePassInfo &match_pass,
                                                 PostCubeFunctionParamPtr funtcparam) {
  switch (funtcparam->GetParaIndex()) {
    case POST_CUBE_INPUT_2_INDEX:  // quant_scale0
      return AddInput2Strategy(match_pass, funtcparam);
    case POST_CUBE_INPUT_3_INDEX:  // relu_weight_0
      return AddInput3Strategy(match_pass, funtcparam);
    case POST_CUBE_INPUT_4_INDEX:  // clip_value_0
      return AddInput4Strategy(match_pass, funtcparam);
    case POST_CUBE_INPUT_5_INDEX:  // quant_scale1
      return AddInput5Strategy(match_pass, funtcparam);
    case POST_CUBE_INPUT_6_INDEX:  // relu_weight_1
      return AddInput6Strategy(match_pass, funtcparam);
    case POST_CUBE_INPUT_7_INDEX:  // clip_value_1
      return AddInput7Strategy(match_pass, funtcparam);
    case POST_CUBE_INPUT_8_INDEX:  // eltwise+antiquant
      return AddInput8Strategy(match_pass, funtcparam);
    case POST_CUBE_INPUT_9_INDEX:  // eltwise+antiquant
      return AddInput9Strategy(match_pass, funtcparam);
    default:
      return nullptr;
  }
}

PostCubeAddInputPtr PostCubeUtils::AddInputSingleUnitStrategy(const PostCubePassInfo &match_pass,
                                                           PostCubeFunctionParamPtr funtcparam,
                                                           const std::string &first_unitname) {
  OPS_LOG_D("PostCube", "AddInputSingleUnitStrategy funtcparam->GetInputName = %s inputdex = %d first_unitname = %s",
          funtcparam->GetInputName().c_str(), funtcparam->GetParaIndex(), first_unitname.c_str());
  std::string strategy_key = "DEFAULT";
  for (size_t idx = 1; idx < match_pass.m_opnodes.size(); idx++) {
    auto tofuzednode = match_pass.m_opnodes[idx];
    if (tofuzednode.GetBelongUnitType() == first_unitname) {
      funtcparam->SetFirstIndex(idx);
      funtcparam->SetSecondIndex(idx);
      OPS_LOG_D("PostCube", "first_index = %d firstname = %s second_index = %d secondname = %s", funtcparam->GetFirstIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString(),
              funtcparam->GetSecondIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
      strategy_key = funtcparam->GetInputName() + ":" + std::string(GNodeGetType(tofuzednode.GetNode()).GetString());
      break;
    }
  }
  OPS_LOG_D("PostCube", "AddInputSingleUnitStrategy strategy key is [%s].", strategy_key.c_str());
  auto iter = m_opmaps_.find(strategy_key);
  if (iter != m_opmaps_.end()) {
    return iter->second;
  }
  OPS_LOG_D("PostCube", "AddInputSingleUnitStrategy default");
  return m_opmaps_["DEFAULT"];
}

PostCubeAddInputPtr PostCubeUtils::AddInputAntiStrategy(const PostCubePassInfo &match_pass,
                                                     PostCubeFunctionParamPtr funtcparam,
                                                     const std::string &first_unitname) {
  OPS_LOG_D("PostCube", "AddInputAntiStrategy funtcparam->GetInputName = %s inputdex = %d first_unitname = %s",
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
        REPORT_OPS_ERROR("[GraphOpt][PostCubePss][AddInputAntiStrategy] node [%s] ",
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
        OPS_LOG_W("PostCube", "Get scale attr of quant node[%s] failed!", GNodeGetName(inputfather_node).GetString());
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
      OPS_LOG_D("PostCube", "first_index = %d firstname = %s second_index = %d secondname = %s", funtcparam->GetFirstIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString(),
              funtcparam->GetSecondIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
      return m_opmaps_[funtcparam->GetInputName() + ":" + std::string(GNodeGetType(inputfather_node).GetString())];
    }
  }
  OPS_LOG_D("PostCube", "AddInputAntiStrategy default");
  return m_opmaps_["DEFAULT"];
}

PostCubeAddInputPtr PostCubeUtils::AddInputTwoUnitStrategy(const PostCubePassInfo &match_pass,
                                                        PostCubeFunctionParamPtr funtcparam,
                                                        const std::string &first_unitname,
                                                        const std::string &second_unitname) {
  OPS_LOG_D("PostCube", "AddInputTwoUnitStrategy input name = %s input indexdex = %d first_unitname = %s second_unitname = %s",
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
      OPS_LOG_D("PostCube", "1.first_index = %d second_index = %d firstname = %ssecondname = %s", funtcparam->GetFirstIndex(),
              funtcparam->GetSecondIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
      strategy_key = funtcparam->GetInputName() + ":" +
                     std::string(GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString()) + "_" +
                     std::string(GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
    } else {
      funtcparam->SetFirstIndex(funtcparam->GetSecondIndex());
      OPS_LOG_D("PostCube", "2.second_index = %d secondname = %s", funtcparam->GetSecondIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
      strategy_key = funtcparam->GetInputName() + ":" +
                     std::string(GNodeGetType(match_pass.m_opnodes[funtcparam->GetSecondIndex()].GetNode()).GetString());
    }
  } else {
    if (has_first_unit) {
      funtcparam->SetSecondIndex(funtcparam->GetFirstIndex());
      OPS_LOG_D("PostCube", "3.first_index = %d firstname = %s", funtcparam->GetFirstIndex(),
              GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString());
      strategy_key = funtcparam->GetInputName() + ":" +
                     std::string(GNodeGetType(match_pass.m_opnodes[funtcparam->GetFirstIndex()].GetNode()).GetString());
    }
  }
  OPS_LOG_D("PostCube", "AddInputTwoUnitStrategy strategy key is [%s]", strategy_key.c_str());
  auto iter = m_opmaps_.find(strategy_key);
  if (iter != m_opmaps_.end()) {
    return iter->second;
  }
  OPS_LOG_D("PostCube", "AddInputTwoUnitStrategy default");
  return m_opmaps_["DEFAULT"];
}

PostCubeAddInputPtr PostCubeUtils::AddInput2Strategy(const PostCubePassInfo &match_pass,
                                                  PostCubeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("quant_scale_0");
  funtcparam->SetDataType(ge::DT_UINT64);
  if (isa_arch_version_ == ISAArchVersion::EN_ISA_ARCH_V350) {
    return AddInputTwoUnitStrategy(match_pass, funtcparam, kPreConv, kPreAct);
  } else {
    return AddInputSingleUnitStrategy(match_pass, funtcparam, kPreConv);
  }
}

PostCubeAddInputPtr PostCubeUtils::AddInput3Strategy(const PostCubePassInfo &match_pass,
                                                  PostCubeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("relu_weight_0");
  funtcparam->SetDataType(ge::DT_FLOAT);
  return AddInputTwoUnitStrategy(match_pass, funtcparam, kPreConv, kPreAct);
}

PostCubeAddInputPtr PostCubeUtils::AddInput4Strategy(const PostCubePassInfo &match_pass,
                                                  PostCubeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("clip_value_0");
  return AddInputTwoUnitStrategy(match_pass, funtcparam, kPreConv, kPreAct);
}

PostCubeAddInputPtr PostCubeUtils::AddInput5Strategy(const PostCubePassInfo &match_pass,
                                                  PostCubeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("quant_scale_1");
  funtcparam->SetDataType(ge::DT_UINT64);
  return AddInputSingleUnitStrategy(match_pass, funtcparam, kPostQuant);
}

PostCubeAddInputPtr PostCubeUtils::AddInput6Strategy(const PostCubePassInfo &match_pass,
                                                  PostCubeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("relu_weight_1");
  funtcparam->SetDataType(ge::DT_FLOAT);
  return AddInputTwoUnitStrategy(match_pass, funtcparam, kPostAct, kPostQuant);
}

PostCubeAddInputPtr PostCubeUtils::AddInput7Strategy(const PostCubePassInfo &match_pass,
                                                  PostCubeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("clip_value_1");
  funtcparam->SetDataType(ge::DT_FLOAT16);
  return AddInputTwoUnitStrategy(match_pass, funtcparam, kPostAct, kPostQuant);
}

PostCubeAddInputPtr PostCubeUtils::AddInput8Strategy(const PostCubePassInfo &match_pass,
                                                  PostCubeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("anti_quant_scale");
  funtcparam->SetDataType(ge::DT_FLOAT16);
  return AddInputAntiStrategy(match_pass, funtcparam, kPostEltwise);
}

PostCubeAddInputPtr PostCubeUtils::AddInput9Strategy(const PostCubePassInfo &match_pass,
                                                  PostCubeFunctionParamPtr funtcparam) {
  funtcparam->SetInputName("anti_quant_offset");
  return AddInputAntiStrategy(match_pass, funtcparam, kPostEltwise);
}
}  // namespace ops
