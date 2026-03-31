/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_UTILS_H_
#define COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_UTILS_H_

#include <map>
#include <queue>
#include <stack>
#include <string>
#include <set>
#include <unordered_set>
#include <vector>

#include "cube_utils/cube_addinputstrategy.h"
#include "cube_utils/cube_common.h"

namespace ops {
class FixpipeUtils {
 public:
  ge::graphStatus GetFixpipeNodeList(ge::GNodePtr conv_node, const ge::CustomPassContext &context);
  ge::graphStatus SelectFixpipeNodeList(bool firstround_cut = true);
  ge::graphStatus CreateFixpipeNode(const string &pass_name, ge::Graph &graph, std::vector<ge::GNodePtr> &new_nodes);

  void ClearPasses();
  std::string GetIsaArchVersionStr(const ISAArchVersion isa_arch_version);
  void ParseIsaArchVersion(fe::PlatFormInfos &platform_infos);
  bool ReadConfig(const ge::CustomPassContext &context);
  ge::graphStatus GenerateMatchedPasses(FixPipeNodeInfo &conv_node);
  ge::graphStatus ModfiyMatchedPasses(bool firstround_cut = true);

  std::vector<FixPipePassInfo> m_matchpasses_;
  std::vector<ge::GNodePtr> not_support_fixpipe_fusion_nodes_;
  std::set<ge::GNodePtr> m_toantiquantnodes_;
  std::vector<FixPipeUnit> m_idxtonodetypes_;
  std::map<std::string, FixPipeAddInputPtr> m_opmaps_;
  std::map<std::string, uint32_t> unitmapindex_;
  // ge::CycleDetectorSharedPtr cycle_detector_ = nullptr;
  std::shared_ptr<FixPipeNodeInfo> cur_head_info_ = nullptr;
  ISAArchVersion isa_arch_version_ = ISAArchVersion::EN_ISA_ARCH_V100;

 private:
  bool JudgeCachePass(const FixPipeNodeInfo &node, std::stack<uint32_t> &index, uint32_t &ret_index) const;
  ge::graphStatus JudgeIsMatch(const FixPipeNodeInfo &node, std::stack<uint32_t> &cur_index, uint32_t &ret_index) const;
  void GenerateMatchedPassesFromStack(const std::stack<FixPipeNodeInfo> &cur_pass,
                                        const uint32_t &pass_index, const uint32_t &cur_index);
  bool IsInWhitelist(const FixPipeNodeInfo &node) const;
  bool GetNodeIndex(const FixPipeNodeInfo &node, const uint32_t &index) const;
  void GetFusionNodes(const std::stack<FixPipeNodeInfo> &cur_pass, std::vector<ge::GNodePtr> &fusion_nodes) const;
  std::string GetCubeType(const std::stack<FixPipeNodeInfo> &cur_pass) const;
  std::string GetMergeInputNodeType(const ge::GNodePtr &merge_node) const;
  void GenerateMatchedPassesImpl(FixPipeNodeInfo &node, FixpipeMatchParams &fp_mch_params);
  void ChangeOrInsertPass(FixPipePassInfo &tmp_pass);
  uint32_t GetUnitIndex(const FixPipePassInfo &fixpipe_pass, const FixPipeNodeInfo &fixpipe_node);
  ge::graphStatus ModfiyMatchedPasseSecRound();
  void GenerateSinglePass(const ge::GNodePtr &cube_node,
                            std::vector<std::vector<ge::GNodePtr>> &matched_pass,
                            std::unordered_set<ge::GNodePtr> &fixpipe_nodes) const;
  void GeneratePass(const ge::GNodePtr &cube_node, const ge::GNodePtr &branch_node,
                    std::vector<std::vector<ge::GNodePtr>> &matched_pass,
                    std::unordered_set<ge::GNodePtr> &fixpipe_nodes) const;
  void GenerateFixpipePasses(std::vector<std::vector<ge::GNodePtr>> &matched_pass,
                             std::map<ge::GNodePtr, FixPipeNodeInfo> &fixpipe_info);
  ge::graphStatus FiltrNodeStrategy(const FixPipeNodeInfo &node) const;
  ge::graphStatus FiltrNodeStrategyForTransData(const FixPipeNodeInfo &node) const;
  ge::graphStatus FiltrNodeStrategyForRelu(const FixPipeNodeInfo &node) const;
  ge::graphStatus FiltrNodeStrategyForCast(const FixPipeNodeInfo &node) const;
  ge::graphStatus FiltrNodeStrategyForEltWise(const FixPipeNodeInfo &node) const;
  ge::graphStatus FiltrNodeStrategyForQuant(const FixPipeNodeInfo &cur_node, const FixPipeNodeInfo &prenode) const;
  bool IsConfictWithSkipConfig(std::stack<uint32_t> index, const uint32_t &ret_index) const;
  bool IsConfictWithSkipConfig(const std::vector<uint32_t> &index, const uint32_t &ret_index) const;
  bool IsConfictWithSkipConfig(const FixPipePassInfo &cur_pass, const uint32_t &ret_index) const;
  std::string GetEltWiseType(const FixPipeNodeInfo &node) const;
  bool PreCachePass(const FixPipePassInfo &cur_pass, const FixPipeNodeInfo &node) const;
  bool PreMatchAcorrdingToPass(const FixPipePassInfo &cur_pass, const FixPipeNodeInfo &node) const;
  ge::graphStatus RelinkOpEdges(ge::Graph &graph, FixPipeNodeInfo &head_node,
                       const FixPipePassInfo &match_pass, FixPipeNodeInfo &fixpipeenhancenode);
  bool IsNodeInPass(const std::vector<FixPipeNodeInfo> &fixednodeids, const ge::GNodePtr &input_node) const;
  ge::GNodePtr CreateFixpipeNode(const FixPipePassInfo &match_pass, const FixPipeNodeInfo &head_node,
                                ge::Graph &graph) const;
  bool NeedToCutPass(FixPipePassInfo &m_pass) const;
  ge::graphStatus AddInputs(ge::Graph &graph, const FixPipePassInfo &match_pass,
                   const ge::GNodePtr &fixpipenode, std::vector<ge::GNodePtr> &new_nodes);
  ge::graphStatus UpdateInputDesc(ge::GNode &cur_new_fixpipenode) const;
  ge::graphStatus RelinkHeadEdges(ge::Graph &graph, FixPipeNodeInfo &head_node, FixPipeNodeInfo &fixpipeenhancenode) const;
  ge::graphStatus RelinkOutputEdges(ge::Graph &graph, const FixPipePassInfo &match_pass, FixPipeNodeInfo &fixpipeenhancenode) const;
  ge::graphStatus RelinkEltWiseEdges(ge::Graph &graph, const FixPipePassInfo &match_pass, FixPipeNodeInfo &fixpipeenhancenode);
  ge::graphStatus RelinkEltWiseEdgesImpl(ge::Graph &graph, const FixPipePassInfo &match_pass, FixPipeNodeInfo &tofuzednode,
                                  FixPipeNodeInfo &fixpipeenhancenode);
  ge::graphStatus RelinkOtherEltWiseEdges(ge::Graph &graph, const int32_t &src_port, const FixPipeNodeInfo &tofuzednode,
                                   FixPipeNodeInfo &fixpipeenhancenode, const ge::GNodePtr &inputfather_node) const;
  ge::graphStatus RelinkAntiEltWiseEdges(ge::Graph &graph, const ge::GNodePtr &inputfather_node, const FixPipeNodeInfo &tofuzednode,
                                  FixPipeNodeInfo &fixpipeenhancenode);
  ge::graphStatus DeleteToFusedNodeEdge(ge::Graph &graph, const FixPipePassInfo &match_pass,
                               std::set<ge::GNodePtr> &todeletenode) const;
  ge::graphStatus DeleteNode(ge::Graph &graph, const std::set<ge::GNodePtr> &todeletenode);
  ge::graphStatus InitInput();
  ge::graphStatus InitInput2();
  ge::graphStatus InitInput3();
  ge::graphStatus InitInput4();
  ge::graphStatus InitInput5();
  ge::graphStatus InitInput6();
  ge::graphStatus InitInput7();
  ge::graphStatus InitInput8();
  ge::graphStatus InitInput9();
  ge::graphStatus InitInputDefault();
  FixPipeAddInputPtr AddInputStrategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam);
  FixPipeAddInputPtr AddInput2Strategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam);
  FixPipeAddInputPtr AddInput3Strategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam);
  FixPipeAddInputPtr AddInput4Strategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam);
  FixPipeAddInputPtr AddInput5Strategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam);
  FixPipeAddInputPtr AddInput6Strategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam);
  FixPipeAddInputPtr AddInput7Strategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam);
  FixPipeAddInputPtr AddInput8Strategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam);
  FixPipeAddInputPtr AddInput9Strategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam);
  FixPipeAddInputPtr AddInputAntiStrategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam,
                                          const std::string &first_unitname);
  FixPipeAddInputPtr AddInputSingleUnitStrategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam,
                                                const std::string &first_unitname);
  FixPipeAddInputPtr AddInputTwoUnitStrategy(const FixPipePassInfo &match_pass, FixPipeFunctionParamPtr funtcparam,
                                             const std::string &first_unitname,
                                             const std::string &second_unitname);
  ge::graphStatus FusionImpl(const string &pass_name, FixPipeNodeInfo &head_node, ge::Graph &graph, std::vector<ge::GNodePtr> &new_nodes);
  void CollectSwitchMergeNodes(const ge::GNodePtr &cube_node, std::vector<ge::GNodePtr> &fixpipe_nodes) const;
  void CollectFixpipe(const ge::GNodePtr &cube_node, std::vector<ge::GNodePtr> &fixpipe_nodes) const;
  void CommonCollectFixpipeRelativeNodes(const ge::GNodePtr &node,
                                         std::vector<ge::GNodePtr> &fixpipe_nodes) const;
  void SetFixpipeRealtiveNodeScopeId(const ge::GNodePtr &node) const;

  template <typename... Args>
  void PrintNodeFilterReason(const FixPipeNodeInfo &node, const Args &...args) const;

  ge::graphStatus CheckEltWiseShapeIsSame(const FixPipeNodeInfo &node, const ge::TensorDesc &input_desc0,
                                 const ge::TensorDesc &input_desc1) const;
  uint32_t GetFixpipeAtomicId() const;
};
}  // namespace ops

#endif  // COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_UTILS_H_
