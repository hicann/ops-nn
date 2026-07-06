/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_CONV_FUSION_UTILS_PASS_H
#define NN_CONV_FUSION_UTILS_PASS_H

#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "ge/compliant_node_builder.h"
#include "ge/es_tensor_holder.h"
#include "ge/fusion/subgraph_boundary.h"
#include "graph/operator.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace ConvFusionUtils {
const ge::AscendString ASCEND_DEQUANT = "AscendDequant";
const ge::AscendString ASCEND_QUANT = "AscendQuant";
const ge::AscendString CONV2D = "Conv2D";
const ge::AscendString CONV3D = "Conv3D";
const ge::AscendString DEPTHWISE_CONV2D = "DepthwiseConv2D";
const ge::AscendString POST_CUBE_OP = "FixPipe";
const ge::AscendString TRANS_DATA_OP = "TransData";
const ge::AscendString IFMR_OP = "IFMR";
const ge::AscendString STRIDES = "strides";
const ge::AscendString PADS = "pads";
const ge::AscendString DILATIONS = "dilations";
const ge::AscendString GROUPS = "groups";
const ge::AscendString OFFSET_X = "offset_x";
const ge::AscendString DATA_FORMAT = "data_format";
const ge::AscendString PADDING = "padding";
const ge::AscendString AUTO_PAD = "auto_pad";
const ge::AscendString OP_IMPL_MODE_ENUM = "_op_impl_mode_enum";
const ge::AscendString PAD_MODE = "pad_mode";
const ge::AscendString ENABLE_HF32 = "enable_hf32";
const std::string UTIL_NAME = "ConvFusionUtilsPass";

constexpr size_t REQUIRED_INPUT_NUMS = 2;
constexpr size_t CONV_COUNT_PARAMS_BIAS = 3; // [fmap, filter, bias]

constexpr int32_t INPUT_FMAP_INDEX = 0;
constexpr int32_t INPUT_FILTER_INDEX = 1;
constexpr int32_t INPUT_BIAS_INDEX = 2;
constexpr int32_t OUTPUT_INDEX = 0;

const std::set<ge::AscendString> SPECIFIC_PAD_LIST = {"NOTSET", "EXPLICIT"};
const std::vector<int64_t> HF32_PRECISION_MODES_INT = {0x1, 0x2, 0x40};

#define FUSION_PASS_CHECK(condition, log_func, return_expr)                                                      \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
    do {                                                                                                         \
        if (condition) {                                                                                         \
            log_func;                                                                                            \
            return_expr;                                                                                         \
        }                                                                                                        \
    } while (0)

#define FUSION_PASS_CHECK_NOLOG(condition, return_expr)                                                          \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
    do {                                                                                                         \
        if (condition) {                                                                                         \
            return_expr;                                                                                         \
        }                                                                                                        \
    } while (0)

std::string GeFormatToString(const ge::Format& geFormat);
std::string GeDtypeToString(const ge::DataType& geDtype);

inline std::string SocListToString(const std::map<std::string, NpuArch>& socList)
{
    std::string result = "[";
    for (auto it = socList.begin(); it != socList.end(); ++it) {
        result += it->first;
        if (std::next(it) != socList.end()) {
            result += ", ";
        }
    }
    result += "]";
    return result;
}

inline std::string VectorToString(const std::vector<ge::DataType>& dtypeVec)
{
    std::string result = "[";
    for (size_t i = 0; i < dtypeVec.size(); ++i) {
        result += GeDtypeToString(dtypeVec[i]);
        if (i < dtypeVec.size() - 1) {
            result += ",";
        }
    }
    result += "]";
    return result;
}

inline std::string VectorsToString(const std::vector<std::vector<ge::DataType>>& dtypeVecs)
{
    std::string result = "[";
    for (size_t j = 0; j < dtypeVecs.size(); ++j) {
        result += VectorToString(dtypeVecs[j]);
        if (j < dtypeVecs.size() - 1) {
            result += ",";
        }
    }
    result += "]";
    return result;
}

struct ConvBaseAttrs {
    std::vector<int64_t> strides = {};
    std::vector<int64_t> pads = {};
    std::vector<int64_t> dilations = {};
    int64_t groups = 0;
    int64_t offsetX = 0;
    int64_t opImplModeEnum = 0;
    ge::AscendString dataFormat = "";
    ge::AscendString padMode = "";
    ge::AscendString padding = "";
    bool enableHf32 = false;
};

struct ConvDescInfo {
    ge::TensorDesc fmapDesc;
    ge::TensorDesc filterDesc;
    ge::TensorDesc biasDesc;
    ge::TensorDesc outputDesc;

    ge::DataType fmapDtype = ge::DataType::DT_MAX;
    ge::DataType filterDtype = ge::DataType::DT_MAX;
    ge::DataType biasDtype = ge::DataType::DT_MAX;
    ge::DataType outputDtype = ge::DataType::DT_MAX;

    ge::Format fmapFormat = ge::Format::FORMAT_NULL;
    ge::Format filterFormat = ge::Format::FORMAT_NULL;
    ge::Format biasFormat = ge::Format::FORMAT_NULL;
    ge::Format outputFormat = ge::Format::FORMAT_NULL;

    ge::AscendString nodeName = "";
    std::string nodeNameStr = "";
    bool hasBias = false;
};

class ConvFusionUtilsPass {
public:
    static bool AddSubgraphInput(std::unique_ptr<ge::fusion::SubgraphBoundary>& boundary, const ge::GNode& node,
                                 const int64_t subgraphIndex, const int64_t boundaryIndex);
    static bool AddSubgraphOutput(std::unique_ptr<ge::fusion::SubgraphBoundary>& boundary, const ge::GNode& node,
                                  const int64_t subgraphIndex, const int64_t boundaryIndex);
    template <typename T>
    static bool CheckSupportList(const std::vector<std::vector<T>>& supportLists, const std::vector<T>& curList);
    static bool CheckSocList(const std::map<std::string, NpuArch>& socList, NpuArch& npuArch);
    static bool GetConvBaseAttr(const ge::GNode& convNode, ConvBaseAttrs& baseAttrs, const ConvDescInfo& convDescInfo);
    static bool GetConvDescInfo(const ge::GNode& convNode, ConvDescInfo& convDescInfo);
    static bool GetMatchedNodes(const ge::GraphPtr& graph, std::vector<ge::GNode>& matchedNodes,
                                const ge::AscendString& nodeType);
    static ge::GNodePtr GetNodePtr(const ge::GNode& node, const ConvDescInfo& convDescInfo);
    static bool IsUnknownShape(const ge::TensorDesc& tensorDesc);
    static ge::AscendString ListToAscendString(const std::vector<ge::AscendString>& strList);
    static void PrintConvDescInfo(const ConvDescInfo& convDescInfo);
    static bool UpdateInputDesc(ge::GNode* convNode, const ConvDescInfo& convDescInfo);
    static bool BuildConv2dNode(ge::Graph* graph, const std::string& nodeName,
                                const std::vector<ge::es::EsTensorHolder>& inputs, ge::GNode& conv2dNode);
};

template <typename T>
bool ConvFusionUtilsPass::CheckSupportList(const std::vector<std::vector<T>>& supportLists,
                                           const std::vector<T>& curList)
{
    size_t compareSize = curList.size();

    bool isSupported = false;
    for (size_t listIndex = 0; listIndex < supportLists.size(); ++listIndex) {
        if (supportLists[listIndex].size() < compareSize) {
            continue;
        }

        auto supportList = supportLists[listIndex];
        isSupported = true;
        for (size_t index = 0; index < compareSize; ++index) {
            if (curList[index] != supportList[index]) {
                isSupported = false;
                break;
            }
        }

        if (isSupported) {
            break;
        }
    }

    return isSupported;
}

} // namespace ConvFusionUtils
} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // NN_CONV_FUSION_UTILS_PASS_H