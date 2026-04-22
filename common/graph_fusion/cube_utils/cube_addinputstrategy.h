/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_ADDINPUTSTRATEGY_H_
#define COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_ADDINPUTSTRATEGY_H_

#include <map>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "cube_utils/cube_common.h"

namespace ops {
using std::queue;
using std::set;
using std::unordered_set;
class PostCubeAddInputBase {
 public:
  PostCubeAddInputBase(){};
  virtual ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                             const PostCubeFunctionParamPtr &functpara, std::vector<ge::GNodePtr> &new_nodes) const {
    (void)graph;
    (void)match_pass;
    (void)functpara;
    (void)new_nodes;
    return ge::GRAPH_SUCCESS;
  }
  template <typename T>
  ge::graphStatus UpdateSalarInput(ge::TensorDesc tensor_desc, T value, ge::Tensor tensornode,
                          const ge::DataType &data_type) const;
  template <typename T>
  ge::graphStatus CreateAndUpdateScalarInput(ge::Graph &graph, const PostCubeFunctionParamPtr &functpara, T value,
                                   const ge::DataType &data_type, std::vector<ge::GNodePtr> &new_nodes) const;
  ge::graphStatus CreateScalarInputNode(ge::Graph &graph, const PostCubeFunctionParamPtr &functpara,
                                const ge::GNodePtr &first_node, const int64_t input_strategy) const;
  void SetClipValue6(ge::Graph &graph, const PostCubeFunctionParamPtr &functpara,
                     ge::DataType dst_datatype, std::vector<ge::GNodePtr> &new_nodes) const;
  ge::graphStatus CloneVectorInput(ge::Graph &graph,
                          const PostCubeNodeInfo &tofuzednode,
                          const PostCubeFunctionParamPtr &functpara,
                          std::vector<ge::GNodePtr> &new_nodes) const;
  ge::graphStatus CreateAndUpdateVectorMulsInput(ge::Graph &graph, const PostCubeFunctionParamPtr &functpara,
                                        const PostCubeNodeInfo &postfuzednode, const PostCubeNodeInfo &tofuzednode,
                                        std::vector<ge::GNodePtr> &new_nodes) const;
  ge::graphStatus CreateNewDataNodeDirect(ge::Graph &graph, ge::TensorDesc tensor_desc, ge::Tensor tensornode,
                                 const PostCubeFunctionParamPtr &functpara) const;
  ge::GNode CreateNewDataNodeOnly(ge::Graph &graph, ge::TensorDesc tensor_desc,
                                    ge::Tensor tensornode, const std::string &op_name) const;
  ge::GNodePtr GetQuantScaleOffset(const PostCubePassInfo &match_pass,
                                  const uint32_t &index,
                                  float &scale, float &offset) const;
  ge::GNodePtr GetFirstNodeOutputDesc(const PostCubePassInfo &match_pass,
                                      const PostCubeFunctionParamPtr &functpara,
                                      ge::TensorDesc &output_desc) const;
  ge::GNodePtr GetNodeOutputDescByIndex(const std::vector<PostCubeNodeInfo> &opnodes,
                                        uint32_t index,
                                        ge::TensorDesc &output_desc) const;
  ge::graphStatus CreateAndRelinkCastNode(ge::Graph &graph,
                                 const ge::GNodePtr &inputnode,
                                 const ge::GNodePtr &outputnode,
                                 const int &input_index,
                                 std::vector<ge::GNodePtr> &new_nodes) const;
  template <typename T>
  ge::graphStatus CreateAndUpdateVectorMulScalarInput(ge::Graph &graph, const PostCubeFunctionParamPtr &functpara,
                                             const PostCubeNodeInfo &prefuzednode, const T &value,
                                             std::vector<ge::GNodePtr> &new_nodes) const;
  static bool IsScalar(const ge::Shape &origin_shape);

  static ge::GNode CreateQuantScaleOpDesc(ge::Graph &graph, const std::string &op_name, const ge::GNodePtr &pre_op_desc,
                                              const ge::GNodePtr &post_op_desc, const ge::GNodePtr &input2_op_desc);
  ge::GNode CreateVectorMulsOpDesc(ge::Graph &graph, const std::string &op_name, const ge::GNodePtr &pre_op_desc,
                                       const ge::GNodePtr &post_op_desc) const;
  static ge::GNode CreateVectorMulScalarOpDesc(ge::Graph &graph, const std::string &op_name, const ge::GNodePtr &pre_op_desc,
                                                   const ge::GNodePtr &post_op_desc, const ge::DataType &data_type);
  ge::graphStatus DoWithClipReluInputWithSingleRelu6(ge::Graph &graph, const PostCubePassInfo &match_pass,
                                           const PostCubeFunctionParamPtr &functpara,
                                           std::vector<ge::GNodePtr> &new_nodes) const;
  static void GetShapeAfterExpandDims(const ge::GNode &node, const ge::TensorDesc &tensor_desc, ge::Shape &shape);
  static bool GetShapeByFormat(const ge::Format old_format, const ge::Format new_format, const ge::DataType data_type,
                 const ge::Shape &old_shape, ge::Shape &new_shape);
  static bool CanShapeBroadcast(const ge::Shape &shape1, const ge::Shape &shape2);
  static bool CanBroadcast(const ge::TensorDesc &tensor_desc1,
                           const ge::TensorDesc &tensor_desc2);
  static ge::Shape GetBroadcastShape(const ge::Shape &shape1, const ge::Shape &shape2);
  static uint32_t IsSameShape(const ge::TensorDesc &tensor_desc1, const ge::TensorDesc &tensor_desc2);
  ge::graphStatus UpdateVectorMulsOutputTensorDesc(const ge::TensorDesc &prenode_inputdesc,
                                          const ge::TensorDesc &postnode_inputdesc,
                                          ge::TensorDesc &out_tensor_desc) const;
  virtual ~PostCubeAddInputBase() {}
};

using PostCubeAddInputPtr = std::shared_ptr<PostCubeAddInputBase>;

inline int32_t GetPrimaryFormat(int32_t format) {
  return static_cast<int32_t>(static_cast<uint32_t>(format) & 0xff);
}

class PostCubeAddInputStrategy21 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy22 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy23 : public PostCubeAddInputBase {
public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy24 : public PostCubeAddInputBase {
public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy25 : public PostCubeAddInputBase {
public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
  static void UpdatePostNodeShape(const ge::GNode &pre_node, const ge::GNode &post_node);
  static void UpdateQuantScaleNodeShape(const ge::GNode &quant_scale_node);
private:
  ge::graphStatus CreateAndLinkQuantScaleNodes(ge::Graph &graph,
                                              const PostCubeFunctionParamPtr &functpara,
                                              const ge::GNodePtr &pre_fuze_node,
                                              const ge::GNodePtr &post_fuze_node,
                                              const ge::GNodePtr &post_cube_node) const;
};

class AddInputStrategyDequntLut : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy31 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy32 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy33 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy34 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy35 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy36 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy37 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy38 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy41 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy42 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy43 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy44 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy51 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy52 : public PostCubeAddInputBase {
public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy61 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy62 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy63 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy64 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy71 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy72 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy81 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategy91 : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};

class PostCubeAddInputStrategyDefault : public PostCubeAddInputBase {
 public:
  ge::graphStatus DoAddInput(ge::Graph &graph, const PostCubePassInfo &match_pass,
                    const PostCubeFunctionParamPtr &functpara,
                    std::vector<ge::GNodePtr> &new_nodes) const override;
};
}  // namespace ops
#endif  // COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_ADDINPUTSTRATEGY_H_
