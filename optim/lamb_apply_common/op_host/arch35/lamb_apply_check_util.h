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
 * \file lamb_apply_check_util.h
 * \brief lamb_apply_optimizer_assign / lamb_apply_weight_assign 复用的输入校验(dtype 一致性、ref 输出形状)。
 */
#ifndef OPS_OPTIM_LAMB_APPLY_COMMON_CHECK_UTIL_H
#define OPS_OPTIM_LAMB_APPLY_COMMON_CHECK_UTIL_H

#include <cstddef>
#include <string>
#include <vector>
#include "exe_graph/runtime/tiling_context.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "infershape_broadcast_util.h"
#include "log/log.h"

namespace optiling {

inline ge::graphStatus CheckLambApplyDtypeConsistency(gert::TilingContext* context, int32_t inputNum,
                                                      const char* const inputNames[], int32_t outputNum,
                                                      const char* const outputNames[])
{
    auto input0Desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    for (int32_t inputIdx = 1; inputIdx < inputNum; inputIdx++) {
        auto inputDesc = context->GetInputDesc(inputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
        if (inputDesc->GetDataType() != input0DType) {
            std::string paramNames = std::string(inputNames[inputIdx]) + " and input0";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context->GetNodeName(), paramNames.c_str(),
                (Ops::Base::ToString(inputDesc->GetDataType()) + " and " + Ops::Base::ToString(input0DType)).c_str(),
                "Their dtypes should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    for (int32_t outputIdx = 0; outputIdx < outputNum; outputIdx++) {
        auto outputDesc = context->GetOutputDesc(outputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
        if (outputDesc->GetDataType() != input0DType) {
            std::string paramNames = std::string(outputNames[outputIdx]) + " and input0";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context->GetNodeName(), paramNames.c_str(),
                (Ops::Base::ToString(outputDesc->GetDataType()) + " and " + Ops::Base::ToString(input0DType)).c_str(),
                "Their dtypes should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ref(原地写回)输出所绑定的输入, 其形状必须恰好等于全部输入广播的结果:
// 内核按广播后的完整网格计算并写回该输入的 buffer, 形状不等就会越过它的显存边界。
inline ge::graphStatus CheckLambApplyBroadcastIntoRef(gert::TilingContext* context, int32_t inputNum, int32_t refIdx,
                                                      const char* refName)
{
    std::vector<const gert::Shape*> inShapes;
    inShapes.reserve(static_cast<size_t>(inputNum));
    for (int32_t i = 0; i < inputNum; i++) {
        auto inShape = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
        inShapes.push_back(&Ops::Base::EnsureNotScalar(inShape->GetStorageShape()));
    }
    // 本地折叠广播, 不依赖 Ops::Base::BroadcastShape —— 该符号由 libops_base.so 导出, 但自定义
    // vendor 包的 tiling 库不链它, 装包后 so 带未解析符号(ldd -r 可见)。带未解析符号的 so 早期
    // dlopen 会失败、被退到内置之后加载, 导致本算子的 tiling 模板抢不到注册槽位(实测 tilingKey
    // 拿到的是内置 ATVOSS 的值)。改为本地实现后 so 无未解析符号。
    gert::Shape bcShape = *inShapes[0];
    for (size_t i = 1; i < inShapes.size(); i++) {
        const gert::Shape& rhs = *inShapes[i];
        size_t lr = bcShape.GetDimNum();
        size_t rr = rhs.GetDimNum();
        size_t rank = (lr > rr) ? lr : rr;
        gert::Shape tmpShape;
        tmpShape.SetDimNum(rank);
        for (size_t d = 0; d < rank; d++) {
            // 右对齐: 缺的高位维按 1 处理
            int64_t a = (d + lr >= rank) ? bcShape.GetDim(d + lr - rank) : 1;
            int64_t b = (d + rr >= rank) ? rhs.GetDim(d + rr - rank) : 1;
            if (a != b && a != 1 && b != 1) {
                OP_LOGE(context->GetNodeName(), "input shapes cannot broadcast together");
                return ge::GRAPH_FAILED;
            }
            // 取"非 1 的那个", 不能取 max —— 空 Tensor 场景下 0 与 1 广播应得 0。
            tmpShape.SetDim(d, (a == 1) ? b : a);
        }
        bcShape = tmpShape;
    }
    const gert::Shape& refShape = *inShapes[refIdx];
    if (!(bcShape == refShape)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), refName, Ops::Base::ToString(refShape).c_str(),
            "it is an in-place(ref) output, so the broadcast shape of all inputs must equal it");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

#endif // OPS_OPTIM_LAMB_APPLY_COMMON_CHECK_UTIL_H
