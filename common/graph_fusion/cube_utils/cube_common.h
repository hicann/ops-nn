/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_COMMON_H_
#define COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_COMMON_H_
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>
#include <memory>

#include "graph/gnode.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "graph/operator.h"
#include "graph/ascend_string.h"
#include "error_util.h"
#include "graph/ge_error_codes.h"
#include "graph/utils/type_utils.h"
#include "cube_utils/cube_fp16_t.h"
#include "register/register_custom_pass.h"

#include "platform/platform_info.h"
#include "ge/compliant_node_builder.h"

namespace ops {
using std::queue;
using std::set;
using std::unordered_set;
extern const std::string kPostCubePassName;
extern const std::string kPatternConv;
extern const std::string kPatternRelu;
extern const std::string kPatternQuant;
extern const std::string kConv2DBackpropInputD;
extern const std::string kConv2DBackpropFilterD;
extern const std::string kConv3D;
extern const std::string kConv3DBackpropInputD;
extern const std::string kConv3DTransposeD;
extern const std::string kDepthwiseConv2D;
extern const std::string kMatMul;
extern const std::string kMatMulV2;
extern const std::string kBatchMatMulV2;
extern const std::string kDepthwiseConv2DBackpropFilterD;
extern const std::string kDepthwiseConv2DBackpropInputD;
extern const std::string kMerge;
extern const std::string kAscendDequant;
extern const std::string kAscendQuant;
extern const std::string kAscendAntiQuant;
extern const std::string kAdd;
extern const std::string kSub;
extern const std::string kLeakyRelu;
extern const std::string kPRelu;
extern const std::string kSigmoid;
extern const std::string kElu;
extern const std::string kTanh;
extern const std::string kPostCube;
extern const std::string kFusionOpList;
extern const std::string kUnitList;
extern const std::string kEltwiseMode;
extern const std::string kCubeUnit;
extern const std::string kPreConv;
extern const std::string kPreAct;
extern const std::string kPostEltwise;
extern const std::string kPostAct;
extern const std::string kPostQuant;
extern const std::string kPostTransform;
extern const std::string kAttrScale;
extern const std::string kAttrNeedJudgeDtype;
extern const std::string kAttrEltwiseMode;
extern const std::string kAttrFakeCubeNode;
extern const std::string kAttrNameWeight;
extern const std::string kAttrNameOriginalInput;
extern const std::string kAttrNameReshapeTypeMask;
extern const std::string kAttrDumpAble;

extern const std::string CONV2D;
extern const std::string DECONVOLUTION;
extern const std::string MATMULV2OP;
extern const std::string DEPTHWISECONV2D;
extern const std::string kFullyConnection;
extern const std::string kConv2DTranspose;
extern const std::string kConv2DTransposeD;
extern const std::string kBatchMatMul;
extern const std::string kConv2DCompress;
extern const std::string kFullyConnectionCompress;
extern const std::string kMatMulV2Compress;
extern const std::string kConv2DTransposeDCompress;
extern const std::string kBatchMatMulCompress;
extern const std::string CAST;
extern const std::string TRANSDATA;
extern const std::string CONSTANT;
extern const std::string RELU;
extern const std::string RELU6;
extern const std::string LEAKY_RELU;
extern const std::string ASCEND_QUANT;
extern const std::string ASCEND_DEQUANT;
extern const std::string kAscendRequant;
extern const std::string kInfNan;
extern const std::string SWITCH;
extern const std::string ATTR_NEGATIVE_SLOPE;
extern const std::string ELTWISE;
extern const std::string kSupportPostCubeAbility;
extern const std::string kNotSupportPostCubeFusion;
extern const std::string TRANSDATA_INPUT_NAME;
extern const std::string TRANSDATA_OUTPUT_NAME;
extern const std::string UNSQUEEZE_V2_INPUT_NAME;
extern const std::string UNSQUEEZE_V2_OUTPUT_NAME;
extern const std::string ATTR_NAME_SRC_FORMAT;
extern const std::string ATTR_NAME_DST_FORMAT;
extern const std::string UNSQUEEZE_V2;
extern const std::string AXIS_ATTR_NAME;
extern const std::string ATTR_SCALE; 
extern const std::string ATTR_OFFSET;
extern const size_t kPostCubeNodeLimited;
extern const size_t kPassMinSize;
extern const size_t kMaxOpNmaLen;
extern const uint32_t kPassFlag;
extern const uint32_t kMaxDepth;
extern const uint32_t kEltMaxDepth;
extern const uint32_t kNumber0;
extern const uint32_t kNumber1;
extern const uint32_t kNumber2;
extern const int64_t SHAPE_NUMBER_16;
extern const float kRelu6Value;
extern const std::vector<std::string> kSupportPostCubeCubeTypeVec;
extern const std::unordered_set<std::string> kSupportMultipleOutputCubeSet;
extern const std::unordered_set<std::string> kMatMulSet;
extern const std::unordered_set<std::string> kAtomicWriteCubeSet;
extern const std::unordered_set<std::string> kCandidateReluTypes;
extern const std::unordered_set<ge::Format> kMatmulNotSupportPostCubeFormatSet;
extern const std::map<std::string, std::string> kPostCubeInstruction2OpTypeMap;

// AscendString 常量定义（避免重复构造，用于比较操作）
// 注意：这些常量需要在 .cc 文件中定义，这里只是声明
// 命名规则：添加 Asc 后缀以区分 std::string 版本
extern const ge::AscendString kAscendEltwiseAsc;
extern const ge::AscendString kAscendAddAsc;
extern const ge::AscendString kAscendConv2DAsc;
extern const ge::AscendString kAscendMergeAsc;
extern const ge::AscendString kAscendPostCubeAsc;
extern const ge::AscendString kAscendLeakyReluAsc;
extern const ge::AscendString kAscendPReluAsc;
extern const ge::AscendString kAscendRelu6Asc;
extern const ge::AscendString kAscendQuantAsc;
extern const ge::AscendString kAscendDequantAsc;
extern const ge::AscendString kAscendAntiQuantAsc;
extern const ge::AscendString kAscendRequantAsc;
extern const ge::AscendString kAscendConstantAsc;
extern const ge::AscendString kAscendTransdataAsc;
extern const ge::AscendString kAscendSwitchAsc;
extern const ge::AscendString kAscendOffsetAsc;
extern const ge::AscendString kAscendNegativeSlopeAsc;
extern const ge::AscendString kAscendReshapeTypeMaskAsc;
extern const ge::AscendString kAscendInfNanAsc;
extern const ge::AscendString kAscendNameWeightAsc;
extern const ge::AscendString kAscendOriginalInputAsc;
extern const ge::AscendString kAscendScaleAsc;
extern const ge::AscendString kAscendNeedJudgeDtypeAsc;
extern const ge::AscendString kAscendFusionOpListAsc;
extern const ge::AscendString kAscendUnitListAsc;
extern const ge::AscendString kAscendEltwiseModeAsc;

#define GET_DEQUANT_SCALE_DEQ(dequant_scale_date) (((dequant_scale_date) & 0x00000000ffffffff))

#ifndef REPORT_OPS_ERROR
#define REPORT_OPS_ERROR(fmt, ...) OPS_LOG_E("PostCube", fmt, ##__VA_ARGS__)
#endif

#ifndef OPS_MAKE_SHARED
#define OPS_MAKE_SHARED(exec_expr0, exec_expr1)   \
  do {                                            \
    try {                                         \
      exec_expr0;                                 \
    } catch (...) {                               \
      OPS_LOG_E("PostCube", "Make shared failed"); \
      exec_expr1;                                 \
    }                                             \
  } while (0)
#endif

#ifndef OPS_CHECK_NOTNULL
#define OPS_CHECK_NOTNULL(val)              \
do {                                        \
  if ((val) == nullptr) {                   \
    OPS_LOG_E("PostCube", #val " is null");  \
    return ge::GRAPH_FAILED;                \
  }                                         \
} while (0)
#endif

enum class ISAArchVersion { EN_ISA_ARCH_V100 = 0, EN_ISA_ARCH_V200, EN_ISA_ARCH_V220, EN_ISA_ARCH_V300,
                            EN_ISA_ARCH_V350 };
struct Configtype2Datatype {
  bool has_output_dtype;
  ge::DataType input_dtype;
  ge::DataType output_dtype;
};
using CONFIGDTYPE = Configtype2Datatype;

enum class PostCubeCubeType {
  NotCube = 0,
  Cube = 1,
  CubeMerge = 2
};

enum class PostCubeAbilityType {
  SupportPostEltwiseBroadcast = 0,
  SupportMultipleOutput = 1,
  UseGmAtomicAdd = 2,
  NodeCantAccess = 3,
  PostCubeAbilityTypeBottom
};

using PostCubeAbilityAttr = int64_t;
const PostCubeAbilityAttr kNoPostCubeAbility = 0x00UL;
const PostCubeAbilityAttr kSupportPostEltwiseBroadcast = 0x01UL;
const PostCubeAbilityAttr kSupportMultipleOutput = 0x02UL;
const PostCubeAbilityAttr kUseGmAtomicAdd = 0x04UL;
const PostCubeAbilityAttr kNodeCantAccess = 0x08UL;

extern const std::array<PostCubeAbilityAttr,
    static_cast<size_t>(PostCubeAbilityType::PostCubeAbilityTypeBottom)> kPostCubeAbilityAttrs;

class PostCubeUnit {
 public:
  PostCubeUnit(const std::string &m_unitname, const std::map<std::string, std::vector<CONFIGDTYPE>> &m_optypes)
      : unitname_(m_unitname), opnodes_(m_optypes) {}
  const std::map<std::string, std::vector<CONFIGDTYPE>> &GetNode() const { return opnodes_; }
  const std::string &GetName() const { return unitname_; }
  const std::vector<std::string> &GetDependsUnits() const { return dependunits_; }
  void SetDependUnits(const std::vector<std::string> &depenunits) {
    dependunits_.assign(depenunits.begin(), depenunits.end());
  }
  void SetDependUnitsIndex(const std::vector<uint32_t> &index) { dependunitsindex_.assign(index.begin(), index.end()); }
  const std::vector<uint32_t> &GetDependsUnitsIndex() const { return dependunitsindex_; }
  ~PostCubeUnit();
 private:
  std::string unitname_;
  std::vector<std::string> dependunits_;
  std::vector<uint32_t> dependunitsindex_;
  std::map<std::string, std::vector<CONFIGDTYPE>> opnodes_;
};

class PostCubeNodeInfo {
 public:
  PostCubeNodeInfo() {}
  explicit PostCubeNodeInfo(const ge::GNodePtr &node)
      : op_kernel_(node),
        belong_unit_type_(""),
        nodeInfo_post_cubeability_(0),
        isheadnode_(false),
        cube_node_(),
        belong_unit_index_(0) {}
  PostCubeNodeInfo(const ge::GNodePtr &node, const ge::GNodePtr &cube_node)
      : op_kernel_(node),
        belong_unit_type_(""),
        nodeInfo_post_cubeability_(0),
        isheadnode_(false),
        cube_node_(cube_node),
        belong_unit_index_(0) {}
  const ge::GNodePtr &GetNode() const { return op_kernel_; }
  const std::string &GetBelongUnitType() const { return belong_unit_type_; }
  char GetNodePostCubeability() const { return nodeInfo_post_cubeability_; }
  bool GetIsHeadNode() const { return isheadnode_; }
  const ge::GNodePtr &GetCubeNode() const { return cube_node_; }
  void SetBelongUnitType(const std::string &belong_unit_type) { belong_unit_type_ = belong_unit_type; }
  void SetNodePostCubeability(char nodeInfo_post_cubeability) { nodeInfo_post_cubeability_ = nodeInfo_post_cubeability; }
  void SetIsHeadNode(bool isheadnode) { isheadnode_ = isheadnode; }
  void SetBelongUnitIndex(uint32_t belong_unit_index) { belong_unit_index_ = belong_unit_index; }
  uint32_t GetBelongUnitIndex() const { return belong_unit_index_; }
  bool operator==(const ge::GNodePtr &input_node) const {
    return (this->GetNode() == input_node);
  }
  ~PostCubeNodeInfo();

 private:
  ge::GNodePtr op_kernel_;
  std::string belong_unit_type_;
  char nodeInfo_post_cubeability_;
  bool isheadnode_;
  ge::GNodePtr cube_node_;
  uint32_t belong_unit_index_;
  std::vector<ge::GNodePtr> node_topost_cubelist_;
};

struct PostCubeMatchParams {
  std::stack<PostCubeNodeInfo> cur_pass;
  std::stack<uint32_t> cur_index;
  uint64_t post_cube_index;
  explicit PostCubeMatchParams()
      : post_cube_index(0) {}
};

struct PostCubePassInfoData {
  std::vector<PostCubeNodeInfo> m_opnodes;
  uint32_t pass_index;
  uint32_t m_flag;
  uint32_t unit_index;
};
using PostCubePassInfo = PostCubePassInfoData;

class PostCubeNodePair {
 public:
  PostCubeNodePair(const ge::GNodePtr &first, const ge::GNodePtr &second) : parent_(first), child_(second) {}
  const ge::GNodePtr &GetParent() const { return parent_; }
  const ge::GNodePtr &GetChild() const { return child_; }
  ~PostCubeNodePair() {}

 private:
  ge::GNodePtr parent_;
  ge::GNodePtr child_;
};

class PostCubeFunctionParam {
 public:
  PostCubeFunctionParam(const std::string &inputname, const ge::GNodePtr &post_cubenode, const uint32_t &input_index)
      : inputname_(inputname),
        post_cubenode_(post_cubenode),
        inputindex_(input_index),
        firstnodeindex_(0),
        secondnodeindex_(0),
        datatype_(ge::DT_UNDEFINED),
        srcconstnode_() {}
  void SetInputName(const std::string &inputname) { inputname_ = inputname; }
  void SetPostCubeNode(const ge::GNodePtr &post_cubenode) { post_cubenode_ = post_cubenode; }
  void SetInputindex(uint32_t input_index) { inputindex_ = input_index; }
  const std::string &GetInputName() const { return inputname_; }
  const ge::GNodePtr &GetPostCubeNode() const { return post_cubenode_; }
  uint32_t GetParaIndex() const { return inputindex_; }
  void SetFirstIndex(uint32_t firstnodeindex) { firstnodeindex_ = firstnodeindex; }
  uint32_t GetFirstIndex() const { return firstnodeindex_; }
  void SetSecondIndex(uint32_t secondnodeindex) { secondnodeindex_ = secondnodeindex; }
  uint32_t GetSecondIndex() const { return secondnodeindex_; }
  void SetDataType(const ge::DataType datatype) { datatype_ = datatype; }
  ge::DataType GetDataType() const { return datatype_; }
  void SetSrcConstNode(const ge::GNodePtr &srcconstnode) { srcconstnode_ = srcconstnode; }
  const ge::GNodePtr &GetSrcConstNode() const { return srcconstnode_; }
  ~PostCubeFunctionParam() {}

 private:
  std::string inputname_;
  ge::GNodePtr post_cubenode_;
  uint32_t inputindex_;
  uint32_t firstnodeindex_;
  uint32_t secondnodeindex_;
  ge::DataType datatype_;
  ge::GNodePtr srcconstnode_;
};
using PostCubeFunctionParamPtr = std::shared_ptr<PostCubeFunctionParam>;

class PostCubeComm {
 public:
  PostCubeComm() = default;;
  ~PostCubeComm() = default;;
  static bool ReadPlatFormConfig(const ge::CustomPassContext &context, const bool &skip_trans, std::vector<std::string> &unit_list,
                                 std::map<std::string, std::vector<std::string>> &depends_list,
                                 std::map<std::string, std::map<std::string, std::vector<CONFIGDTYPE>>> &post_cube_map);
  static ge::graphStatus CheckPeerOutNode(const ge::GNodePtr &vectornode, const uint32_t &input_index);
  static bool CheckConstValueData(const ge::GNodePtr &vectornode);
  static bool CheckIsInVector(const std::vector<PostCubeNodeInfo> &m_opnodes, const uint32_t &index = 0);
  static PostCubeCubeType GetPostCubeCubeType(const ge::GNodePtr &node_ptr);
  static ge::GNodePtr GetMergeNodeByCube(const ge::GNodePtr &node_ptr);
  static void SetPostCubeAbilityAttr(const ge::GNodePtr &node_ptr, const PostCubeAbilityType &post_cube_ability_type);
  static void UnSetPostCubeAbilityAttr(const ge::GNodePtr &node_ptr, const PostCubeAbilityType &post_cube_ability_type);
  static bool CheckPostCubeAbilityAttr(const ge::GNodePtr &node_ptr, const PostCubeAbilityType &post_cube_ability_type);
  static bool HasControlEdge(const ge::GNodePtr &src_node_ptr, const ge::GNodePtr &dst_node_ptr);
  static std::string GetStrByDataTypeVec(const std::vector<ge::DataType>& data_type_vec);
  // Conv2D Process
  static bool GetSupportDN2NZSoc(bool &supportOut2L1Dn2Nz);
  static bool GetConv2DOpType(const ge::GNodePtr &head_node);
  static ge::graphStatus GetConv2DReluSwapForbiddenFlag(const ge::GNodePtr &head_node);
  static ge::graphStatus GetPostCubeUtilFuncSupportFlag(const ge::GNodePtr &head_node);
  static uint32_t GetOutDataNodesSize(const ge::GNodePtr &node);
  static bool IsShapeEqual(const ge::Shape &shape1, const ge::Shape &shape2);
  static std::string ShapeToString(const ge::Shape &shape);

  static std::vector<string> Split(const string &str, const string &pattern);
 private:
  static ge::DataType TranferString(const std::string &configstr);
  static CONFIGDTYPE TransFerConfig2Dtype(const std::string &configstr);
  static ge::GNodePtr GetConstNode(const ge::GNodePtr &vectornode, uint32_t &depth);
  static ge::GNodePtr GetConstNode(const ge::GNodePtr &vectornode);
  static bool JudegedataUInt64(uint8_t *data, const size_t &data_size);
  static bool JudgedataFp16(uint8_t *data, const size_t &data_size);
  template <typename T>
  static bool JudgedataImpl(uint8_t *data, const size_t &data_size);
  static bool Judgedata(uint8_t *data, const size_t &data_size, const ge::DataType &data_type);
  static ge::GNodePtr GetMergeNode(const ge::GNodePtr &node_ptr);
  static bool CheckMergeInput(const ge::GNodePtr &merge_node);
  static bool IsEltwiseNode(const ge::GNodePtr &node);
};

// GNode 适配函数：返回 AscendString，避免 string 构造
inline ge::AscendString GNodeGetName(const ge::GNodePtr &node) {
    ge::AscendString name;
    if (node != nullptr) {
        node->GetName(name);
    }
    return name;
}

inline ge::AscendString GNodeGetType(const ge::GNodePtr &node) {
    ge::AscendString type;
    if (node != nullptr) {
        node->GetType(type);
    }
    return type;
}

// 新增重载版本（接受 GNode&）
inline ge::AscendString GNodeGetName(const ge::GNode &node) {
    ge::AscendString name;
    node.GetName(name);
    return name;
}

inline ge::AscendString GNodeGetType(const ge::GNode &node) {
    ge::AscendString type;
    node.GetType(type);
    return type;
}

// GNode GetAttr/SetAttr 适配函数 - 将 std::string 转换为 AscendString
template<typename T>
inline ge::graphStatus GNodeGetAttr(const ge::GNodePtr &node, const std::string &name, T &value) {
    if (node == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::AscendString attr_name(name.c_str());
    return node->GetAttr(attr_name, value);
}

template<typename T>
inline ge::graphStatus GNodeSetAttr(const ge::GNodePtr &node, const std::string &name, T &value) {
    if (node == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::AscendString attr_name(name.c_str());
    return node->SetAttr(attr_name, value);
}

}  // namespace ops
#endif  // COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_COMMON_H_
