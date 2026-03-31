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
#include <vector>

#include "ge/fusion/subgraph_boundary.h"
#include "platform/soc_spec.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace ConvFusionUtils {
using ge::AscendString;
using ge::DataType;
using ge::GNodePtr;
using ge::GNode;
using ge::Format;
using ge::TensorDesc;
using ge::GraphPtr;

const AscendString ASCEND_DEQUANT = "AscendDequant";
const AscendString ASCEND_QUANT = "AscendQuant";
const AscendString CONV2D = "Conv2D";
const AscendString CONV3D = "Conv3D";
const AscendString FIXPIPE = "FixPipe";
const AscendString STRIDES = "strides";
const AscendString PADS = "pads";
const AscendString DILATIONS = "dilations";
const AscendString GROUPS = "groups";
const AscendString OFFSET_X = "offset_x";
const AscendString DATA_FORMAT = "data_format";
const AscendString PADDING = "padding";
const AscendString AUTO_PAD = "auto_pad";
const AscendString OP_IMPL_MODE_ENUM = "_op_impl_mode_enum";
const AscendString PAD_MODE = "pad_mode";
const AscendString ENABLE_HF32 = "enable_hf32";
const std::string UTIL_NAME = "ConvFusionUtilsPass";

constexpr size_t REQUIRED_INPUT_NUMS = 2;
constexpr size_t CONV_COUNT_PARAMS_BIAS = 3; // [fmap, filter, bias]

constexpr int32_t INPUT_FMAP_INDEX = 0;
constexpr int32_t INPUT_FILTER_INDEX = 1;
constexpr int32_t INPUT_BIAS_INDEX = 2;
constexpr int32_t OUTPUT_INDEX = 0;

const std::set<AscendString> SPECIFIC_PAD_LIST = {"NOTSET", "EXPLICIT"};
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

struct ConvBaseAttrs {
    std::vector<int64_t> strides = {};
    std::vector<int64_t> pads = {};
    std::vector<int64_t> dilations = {};
    int64_t groups = 0;
    int64_t offsetX = 0;
    int64_t opImplModeEnum = 0;
    AscendString dataFormat = "";
    AscendString padMode = "";
    bool enableHf32 = false;
};

struct ConvDescInfo {
    TensorDesc fmapDesc;
    TensorDesc filterDesc;
    TensorDesc biasDesc;
    TensorDesc outputDesc;

    DataType fmapDtype = DataType::DT_MAX;
    DataType filterDtype = DataType::DT_MAX;
    DataType biasDtype = DataType::DT_MAX;
    DataType outputDtype = DataType::DT_MAX;

    Format fmapFormat = Format::FORMAT_NULL;
    Format filterFormat = Format::FORMAT_NULL;
    Format biasFormat = Format::FORMAT_NULL;
    Format outputFormat = Format::FORMAT_NULL;

    AscendString nodeName = "";
    std::string nodeNameStr = "";
    bool hasBias = false;
};

class ConvFusionUtilsPass {
public:
    static bool AddSubgraphInput(std::unique_ptr<ge::fusion::SubgraphBoundary> &boundary, const GNode &node,
        const int64_t subgraphIndex, const int64_t boundaryIndex);
    static bool AddSubgraphOutput(std::unique_ptr<ge::fusion::SubgraphBoundary> &boundary, const GNode &node,
        const int64_t subgraphIndex, const int64_t boundaryIndex);
    template <typename T>
    static bool CheckSupportList(const std::vector<std::vector<T>> &supportLists,
        const std::vector<T> &curList);
    static bool CheckSocSupport(const std::map<std::string, NpuArch> &supportSocList, NpuArch &npuArch);
    static bool GetConvBaseAttr(const GNode &convNode, ConvBaseAttrs &baseAttrs,
        const ConvDescInfo &convDescInfo);
    static bool GetConvDescInfo(const GNode &convNode, ConvDescInfo &convDescInfo);
    static bool GetMatchedNodes(const GraphPtr &graph, std::vector<GNode> &matchedNodes,
        const AscendString &nodeType);
    static GNodePtr GetNodePtr(const GNode &node, const ConvDescInfo &convDescInfo);
    static AscendString ListToAscendString(const std::vector<AscendString> &strList);
    static void PrintConvDescInfo(const ConvDescInfo &convDescInfo);
    static bool UpdateInputDesc(GNode *convNode, const ConvDescInfo &convDescInfo);
};

template <typename T>
bool ConvFusionUtilsPass::CheckSupportList(const std::vector<std::vector<T>> &supportLists,
    const std::vector<T> &curList)
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