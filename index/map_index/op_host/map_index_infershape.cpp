/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file map_index_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"
#include "op_common/op_host/util/op_const_def.h"
#include "acl/acl_platform.h"
#include <cstdlib>

using namespace ge;
namespace ops {

constexpr size_t X_INDEX = 0;
constexpr size_t DATA_SEQ_INDEX = 1;
constexpr size_t LEVEL_INDEX_INDEX = 2;
constexpr size_t Y_INDEX = 0;

// NpuArch 3510/9201/9202 为 arch35 家族, 上限 24000/256; 其余架构 400/100
static constexpr bool IsArch35Family(int32_t arch)
{
    return arch == static_cast<int32_t>(Ops::Base::DAV_3510) || arch == static_cast<int32_t>(Ops::Base::DAV_9201) ||
           arch == static_cast<int32_t>(Ops::Base::DAV_9202);
}

static ge::graphStatus InferShape4MapIndex(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MapIndexInferShape");
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* dataSeqShape = context->GetInputShape(DATA_SEQ_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dataSeqShape);
    const gert::Shape* levelIndexShape = context->GetOptionalInputShape(LEVEL_INDEX_INDEX);

    OP_CHECK_IF(xShape->GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "x", std::to_string(xShape->GetDimNum()), "the rank of x must be 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dataSeqShape->GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "data_seq",
                                                         std::to_string(dataSeqShape->GetDimNum()),
                                                         "the rank of data_seq must be 1"),
                return ge::GRAPH_FAILED);
    const bool hasLevelIndex = levelIndexShape != nullptr && levelIndexShape->GetDimNum() != 0;

    const int64_t xLength = xShape->GetDim(0);
    const int64_t dataSeqLength = dataSeqShape->GetDim(0);
    // x/data_seq/level_index 不支持动态 shape(未知维-1/未知秩-2)
    OP_CHECK_IF(Ops::Base::IsUnknownShape(*xShape) || Ops::Base::IsUnknownRank(*xShape),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", std::to_string(xLength),
                                                      "dynamic shape (unknown dim -1 / unknown rank -2) is not "
                                                      "supported"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(Ops::Base::IsUnknownShape(*dataSeqShape) || Ops::Base::IsUnknownRank(*dataSeqShape),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "data_seq", std::to_string(dataSeqLength),
                                                      "dynamic shape (unknown dim -1 / unknown rank -2) is not "
                                                      "supported"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        hasLevelIndex && (Ops::Base::IsUnknownShape(*levelIndexShape) || Ops::Base::IsUnknownRank(*levelIndexShape)),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "level_index",
                                              std::to_string(levelIndexShape->GetDim(0)),
                                              "dynamic shape (unknown dim -1 / unknown rank -2) is not supported"),
        return ge::GRAPH_FAILED);

    // arch 由 aclplatformGetDeviceInfo 运行时 API 获取
    char archBuf[64] = {0};
    aclError archRet = aclplatformGetDeviceInfo(ACL_PLATFORM_NPU_ARCH, archBuf, sizeof(archBuf));
    OP_CHECK_IF(archRet != ACL_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "npu_arch", "runtime_api",
                                                      "aclplatformGetDeviceInfo failed"),
                return ge::GRAPH_FAILED);
    const bool isArch35Family = IsArch35Family(std::atoi(archBuf));
    const int64_t maxLen = isArch35Family ? 24000 : 400;
    const int64_t maxGear = isArch35Family ? 256 : 100;
    // x<=0: 0 为空张量, 负数为非法维
    OP_CHECK_IF(xLength <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(xLength),
                                                      "the length of x should be greater than 0"),
                return ge::GRAPH_FAILED);
    // data_seq<=0: 负整倍数可穿过整除/档位检查(负商恒不超上限), 须显式拦截
    OP_CHECK_IF(dataSeqLength <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "data_seq", std::to_string(dataSeqLength),
                                                      "the length of data_seq should be greater than 0"),
                return ge::GRAPH_FAILED);
    // 整除前置: 保证后续除法均为精确商
    OP_CHECK_IF(dataSeqLength % xLength != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "data_seq", std::to_string(dataSeqLength),
                    "the length of data_seq must be a multiple of the length of x, and the length of x is " +
                        std::to_string(xLength)),
                return ge::GRAPH_FAILED);
    if (hasLevelIndex) {
        const int64_t levelIndexLength = levelIndexShape->GetDim(0);
        OP_CHECK_IF(levelIndexLength != (dataSeqLength / xLength),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "level_index", std::to_string(levelIndexLength),
                        "the length of level_index must be equal to the length of data_seq divided by the length "
                        "of x, expected " +
                            std::to_string(dataSeqLength / xLength)),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(xLength > maxLen,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "x", std::to_string(xLength),
                    "the length of x should be less than or equal to " + std::to_string(maxLen)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dataSeqLength / xLength > maxGear,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "data_seq", std::to_string(dataSeqLength / xLength),
                    "data_seq_length / x_length should be less than or equal to " + std::to_string(maxGear)),
                return ge::GRAPH_FAILED);
    gert::Shape* outputShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    outputShape->SetScalar();
    OP_LOGD(context->GetNodeName(), "End of MapIndexInferShape");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MapIndex).InferShape(InferShape4MapIndex);
} // namespace ops
