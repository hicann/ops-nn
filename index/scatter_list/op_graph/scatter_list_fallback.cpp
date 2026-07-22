/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
static const size_t VAR_INDEX = 0;
static const size_t INDICES_INDEX = 1;
static const size_t UPDATA_INDEX = 2;
static const size_t MASK_INDEX = 3;
static const size_t MAX_TENSOR_NUM = 256;

static graphStatus ScatterListHostExecuteFunc(OpExecuteContext* hostApiCtx)
{
    OP_CHECK_IF(hostApiCtx == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "hostApiCtx is null"), return GRAPH_FAILED);
    OP_LOGD(hostApiCtx->GetNodeName(), "ScatterListHostExecuteFunc in ");

    std::vector<const gert::Tensor*> geTenserListVar;
    uint32_t idx = 0;
    do {
        auto var = hostApiCtx->GetDynamicInputTensor(VAR_INDEX, idx);
        if (var == nullptr || idx > MAX_TENSOR_NUM) {
            break;
        } else {
            idx++;
            geTenserListVar.push_back(var);
        }
    } while (true);

    auto indices = hostApiCtx->GetInputTensor(idx++);
    OP_CHECK_IF(indices == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "indices is null"), return GRAPH_FAILED);

    auto update = hostApiCtx->GetInputTensor(idx++);
    OP_CHECK_IF(update == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "update is null"), return GRAPH_FAILED);

    auto mask = hostApiCtx->GetOptionalInputTensor(MASK_INDEX);
    auto output = hostApiCtx->GetOutputTensor(0);
    OP_CHECK_IF(output == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "output is null"), return GRAPH_FAILED);

    auto attrs = hostApiCtx->GetAttrs();
    const char* reduce = attrs->GetAttrPointer<char>(0);
    const int64_t* axis = attrs->GetAttrPointer<int64_t>(1);

    // execute opapi
    auto host_api_ctx = hostApiCtx;
    auto apiRet = EXEC_OPAPI_CMD(aclnnScatterList, geTenserListVar, indices, update, mask, reduce, *axis);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE(hostApiCtx->GetNodeName(), "apiRet faild:%d", apiRet),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(ScatterList).OpExecuteFunc(ScatterListHostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
