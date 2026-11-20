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
class PostCubeUtils {
public:
    ge::graphStatus GetPostCubeNodeList(ge::GNodePtr conv_node, const ge::CustomPassContext& context);
    ge::graphStatus SelectPostCubeNodeList(bool firstround_cut = true);
    ge::graphStatus CreatePostCubeNode(const string& pass_name, ge::Graph& graph, std::vector<ge::GNodePtr>& new_nodes);

    void ClearPasses();
    void ResetNotSupportPostCubeFusionAttr();
    std::string GetIsaArchVersionStr(const ISAArchVersion isa_arch_version);
    void ParseIsaArchVersion(fe::PlatFormInfos& platform_infos);
    bool ReadConfig(const ge::CustomPassContext& context);
    bool InitAndBuildUnits(const ge::CustomPassContext& context);
    void ResolveUnitDependencies();
    ge::graphStatus GenerateMatchedPasses(PostCubeNodeInfo& conv_node);
    ge::graphStatus ModfiyMatchedPasses(bool firstround_cut = true);

    std::vector<PostCubePassInfo> m_matchpasses_;
    std::vector<ge::GNodePtr> not_support_post_cube_fusion_nodes_;
    std::set<ge::GNodePtr> m_toantiquantnodes_;
    // producers feeding any created post cube node input, they must survive the folded
    // node removal of every matched pass, otherwise the graph keeps dangling input edges.
    // keyed by node name: the GNode wrappers returned by the neighbor traversal apis are
    // built on demand and never compare equal by pointer, a GNodePtr keyed set would
    // never hit and the keep alive would silently stop working
    std::set<std::string> m_pc_input_src_nodes_;
    std::vector<PostCubeUnit> m_idxtonodetypes_;
    std::map<std::string, PostCubeAddInputPtr> m_opmaps_;
    std::map<std::string, uint32_t> unitmapindex_;
    // ge::CycleDetectorSharedPtr cycle_detector_ = nullptr;
    std::shared_ptr<PostCubeNodeInfo> cur_head_info_ = nullptr;
    ISAArchVersion isa_arch_version_ = ISAArchVersion::EN_ISA_ARCH_V100;

private:
    bool JudgeCachePass(const PostCubeNodeInfo& node, std::stack<uint32_t>& index, uint32_t& ret_index) const;
    ge::graphStatus JudgeIsMatch(const PostCubeNodeInfo& node, std::stack<uint32_t>& cur_index,
                                 uint32_t& ret_index) const;
    void GenerateMatchedPassesFromStack(const std::stack<PostCubeNodeInfo>& cur_pass, const uint32_t& pass_index,
                                        const uint32_t& cur_index);
    bool IsInWhitelist(const PostCubeNodeInfo& node) const;
    bool GetNodeIndex(const PostCubeNodeInfo& node, const uint32_t& index) const;
    void GetFusionNodes(const std::stack<PostCubeNodeInfo>& cur_pass, std::vector<ge::GNodePtr>& fusion_nodes) const;
    // equals the HasDetectedCycle of the legacy shared cycle detector: folding the whole
    // matched chain into one post cube node must not create a cycle, the walk covers the
    // accumulated view of the previously fused heads implicitly on the live graph
    bool HasDetectedCycleOnChain(const std::vector<ge::GNodePtr>& fusion_nodes) const;
    // equals the GraphUtils::IsolateNode of the legacy framework: unlink every data edge
    // of the node before the removal
    ge::graphStatus IsolateNode(ge::Graph& graph, const ge::GNodePtr& node) const;
    // give every materialized ir input slot of the created post cube node a desc: the
    // builder materializes the slots lazily and slots below a wired slot exist without
    // descs, the legacy op desc always carried all ten descs and the downstream fixpipe
    // partition conversion dereferences every input desc it meets
    void FillMissingInputDescs(const ge::GNodePtr& post_cubenode) const;
    std::string GetCubeType(const std::stack<PostCubeNodeInfo>& cur_pass) const;
    std::string GetMergeInputNodeType(const ge::GNodePtr& merge_node) const;
    void GenerateMatchedPassesImpl(PostCubeNodeInfo& node, PostCubeMatchParams& fp_mch_params);
    bool CheckEltWiseCycle(const PostCubeNodeInfo& node, const PostCubeMatchParams& fp_mch_params, uint32_t cur_index);
    // the eltwise x2 father must not sit in the input basin of the head cube (reachable
    // upward from the cube inputs without crossing a cube or post cube node): folding
    // such an eltwise makes the same upstream tensor feed both the cube input chain and
    // the x2 input, after the subgraph partitioning the tensor enters the resulting
    // subgraph twice and the downstream tbe fixpipe buffer fusion can not consume the
    // fixpipe, it falls to a single op compilation and fails with E61001
    bool IsX2FatherInCubeInputBasin(const PostCubeNodeInfo& node, const std::stack<PostCubeNodeInfo>& cur_pass) const;
    void ProcessOutputNodesForPass(const PostCubeNodeInfo& node, PostCubeMatchParams& fp_mch_params);
    void ChangeOrInsertPass(PostCubePassInfo& tmp_pass);
    uint32_t GetUnitIndex(const PostCubePassInfo& post_cube_pass, const PostCubeNodeInfo& post_cube_node);
    ge::graphStatus ModfiyMatchedPasseSecRound();
    // node identity must be the node name: the sec round pass reconstruction walks the
    // graph with on demand built GNode wrappers, pointer keyed containers compare against
    // wrappers that never repeat, the map lookups then miss and return default built
    // infos with null nodes, the rebuilt passes lose their chain nodes and the folded
    // nodes survive as orphans that break the subgraph compilation
    void CollectPostCubeNodesAndInfo(std::set<std::string>& post_cube_nodes,
                                     std::map<std::string, PostCubeNodeInfo>& post_cube_info) const;
    ge::GNodePtr FindBranchNode(const ge::GNodePtr& cube_node, int32_t min_length,
                                const std::set<std::string>& post_cube_nodes) const;
    void GenerateSinglePass(const ge::GNodePtr& cube_node, std::vector<std::vector<ge::GNodePtr>>& matched_pass,
                            std::set<std::string>& post_cube_nodes) const;
    void GeneratePass(const ge::GNodePtr& cube_node, const ge::GNodePtr& branch_node,
                      std::vector<std::vector<ge::GNodePtr>>& matched_pass,
                      std::set<std::string>& post_cube_nodes) const;
    void ExtendPassWithPostCubeNodes(std::vector<ge::GNodePtr>& pass, const ge::GNodePtr& start_node,
                                     const std::set<std::string>& post_cube_nodes) const;
    void GeneratePostCubePasses(std::vector<std::vector<ge::GNodePtr>>& matched_pass,
                                std::map<std::string, PostCubeNodeInfo>& post_cube_info);
    ge::graphStatus FiltrNodeStrategy(const PostCubeNodeInfo& node) const;
    ge::graphStatus FiltrNodeStrategyForTransData(const PostCubeNodeInfo& node) const;
    ge::graphStatus FiltrNodeStrategyForCast(const PostCubeNodeInfo& node) const;
    ge::graphStatus FiltrNodeStrategyForRelu(const PostCubeNodeInfo& node) const;
    ge::graphStatus FiltrNodeStrategyForEltWise(const PostCubeNodeInfo& node) const;
    ge::graphStatus FiltrNodeStrategyForQuant(const PostCubeNodeInfo& cur_node, const PostCubeNodeInfo& prenode) const;
    bool IsConfictWithSkipConfig(std::stack<uint32_t> index, const uint32_t& ret_index) const;
    bool IsConfictWithSkipConfig(const std::vector<uint32_t>& index, const uint32_t& ret_index) const;
    bool IsConfictWithSkipConfig(const PostCubePassInfo& cur_pass, const uint32_t& ret_index) const;
    std::string GetEltWiseType(const PostCubeNodeInfo& node) const;
    bool PreCachePass(const PostCubePassInfo& cur_pass, const PostCubeNodeInfo& node) const;
    bool PreMatchAcorrdingToPass(const PostCubePassInfo& cur_pass, const PostCubeNodeInfo& node) const;
    ge::graphStatus RelinkOpEdges(ge::Graph& graph, PostCubeNodeInfo& head_node, const PostCubePassInfo& match_pass,
                                  PostCubeNodeInfo& post_cubeenhancenode);
    bool IsNodeInPass(const std::vector<PostCubeNodeInfo>& fixednodeids, const ge::GNodePtr& input_node) const;
    ge::GNodePtr CreatePostCubeNode(const PostCubePassInfo& match_pass, const PostCubeNodeInfo& head_node,
                                    ge::Graph& graph) const;
    void PrepareFusionAttributes(const std::vector<PostCubeNodeInfo>& fixednodeids,
                                 std::vector<ge::AscendString>& fuzedoptypes_asc,
                                 std::vector<ge::AscendString>& activatepost_cubeunits_asc,
                                 ge::AscendString& eltwise_mode_asc) const;
    bool NeedToCutPass(PostCubePassInfo& m_pass) const;
    ge::graphStatus AddInputs(ge::Graph& graph, const PostCubePassInfo& match_pass, const ge::GNodePtr& post_cubenode,
                              std::vector<ge::GNodePtr>& new_nodes);
    ge::graphStatus UpdateInputDesc(ge::GNode& cur_new_post_cubenode) const;
    ge::graphStatus RelinkHeadEdges(ge::Graph& graph, PostCubeNodeInfo& head_node,
                                    PostCubeNodeInfo& post_cubeenhancenode) const;
    ge::graphStatus RelinkOutputEdges(ge::Graph& graph, const PostCubePassInfo& match_pass,
                                      PostCubeNodeInfo& post_cubeenhancenode) const;
    ge::graphStatus RelinkEltWiseEdges(ge::Graph& graph, const PostCubePassInfo& match_pass,
                                       PostCubeNodeInfo& post_cubeenhancenode);
    ge::graphStatus RelinkEltWiseEdgesImpl(ge::Graph& graph, const PostCubePassInfo& match_pass,
                                           PostCubeNodeInfo& tofuzednode, PostCubeNodeInfo& post_cubeenhancenode);
    ge::graphStatus RelinkOtherEltWiseEdges(ge::Graph& graph, const int32_t& src_port,
                                            const PostCubeNodeInfo& tofuzednode, PostCubeNodeInfo& post_cubeenhancenode,
                                            const ge::GNodePtr& inputfather_node) const;
    ge::graphStatus RelinkAntiEltWiseEdges(ge::Graph& graph, const ge::GNodePtr& inputfather_node,
                                           const PostCubeNodeInfo& tofuzednode, PostCubeNodeInfo& post_cubeenhancenode);
    ge::graphStatus DeleteToFusedNodeEdge(ge::Graph& graph, const PostCubePassInfo& match_pass,
                                          std::set<ge::GNodePtr>& todeletenode) const;
    ge::graphStatus DeleteNode(ge::Graph& graph, const std::set<ge::GNodePtr>& todeletenode);
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
    PostCubeAddInputPtr AddInputStrategy(const PostCubePassInfo& match_pass, PostCubeFunctionParamPtr funtcparam);
    PostCubeAddInputPtr AddInput2Strategy(const PostCubePassInfo& match_pass, PostCubeFunctionParamPtr funtcparam);
    PostCubeAddInputPtr AddInput3Strategy(const PostCubePassInfo& match_pass, PostCubeFunctionParamPtr funtcparam);
    PostCubeAddInputPtr AddInput4Strategy(const PostCubePassInfo& match_pass, PostCubeFunctionParamPtr funtcparam);
    PostCubeAddInputPtr AddInput5Strategy(const PostCubePassInfo& match_pass, PostCubeFunctionParamPtr funtcparam);
    PostCubeAddInputPtr AddInput6Strategy(const PostCubePassInfo& match_pass, PostCubeFunctionParamPtr funtcparam);
    PostCubeAddInputPtr AddInput7Strategy(const PostCubePassInfo& match_pass, PostCubeFunctionParamPtr funtcparam);
    PostCubeAddInputPtr AddInput8Strategy(const PostCubePassInfo& match_pass, PostCubeFunctionParamPtr funtcparam);
    PostCubeAddInputPtr AddInput9Strategy(const PostCubePassInfo& match_pass, PostCubeFunctionParamPtr funtcparam);
    PostCubeAddInputPtr AddInputAntiStrategy(const PostCubePassInfo& match_pass, PostCubeFunctionParamPtr funtcparam,
                                             const std::string& first_unitname);
    PostCubeAddInputPtr AddInputSingleUnitStrategy(const PostCubePassInfo& match_pass,
                                                   PostCubeFunctionParamPtr funtcparam,
                                                   const std::string& first_unitname);
    PostCubeAddInputPtr AddInputTwoUnitStrategy(const PostCubePassInfo& match_pass, PostCubeFunctionParamPtr funtcparam,
                                                const std::string& first_unitname, const std::string& second_unitname);
    ge::graphStatus FusionImpl(const string& pass_name, PostCubeNodeInfo& head_node, ge::Graph& graph,
                               std::vector<ge::GNodePtr>& new_nodes);
    void CollectSwitchMergeNodes(const ge::GNodePtr& cube_node, std::vector<ge::GNodePtr>& post_cube_nodes) const;
    void CollectPostCube(const ge::GNodePtr& cube_node, std::vector<ge::GNodePtr>& post_cube_nodes) const;
    void CommonCollectPostCubeRelativeNodes(const ge::GNodePtr& node, std::vector<ge::GNodePtr>& post_cube_nodes) const;
    void SetPostCubeRealtiveNodeScopeId(const ge::GNodePtr& node) const;

    template <typename... Args>
    void PrintNodeFilterReason(const PostCubeNodeInfo& node, const Args&... args) const;

    ge::graphStatus CheckEltWiseShapeIsSame(const PostCubeNodeInfo& node, const ge::TensorDesc& input_desc0,
                                            const ge::TensorDesc& input_desc1) const;
    uint32_t GetPostCubeAtomicId() const;
};
} // namespace ops

#endif // COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_UTILS_H_
