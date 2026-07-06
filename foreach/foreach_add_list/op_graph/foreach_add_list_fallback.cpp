/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
static const size_t FIRST_INPUT = 0;
static const size_t SECOND_INPUT = 1;
static const size_t THIRD_INPUT = 2;
static const size_t FIRST_OUTPUT = 0;

graphStatus PrepareInputTensorListForeachAddList(OpExecuteContext* hostApiCtx,
                                                 std::vector<const gert::Tensor*>& tensorList, size_t index,
                                                 size_t& num)
{
    while (1) {
        auto inputGe = hostApiCtx->GetDynamicInputTensor(index, num);
        if (inputGe == nullptr) {
            break;
        }
        tensorList.push_back(inputGe);
        num++;
    }
    return GRAPH_SUCCESS;
}

graphStatus PrepareOutputTensorListForeachAddList(OpExecuteContext* hostApiCtx,
                                                  std::vector<const gert::Tensor*>& tensorList, size_t index,
                                                  size_t& num)
{
    while (1) {
        auto outputGe = hostApiCtx->GetDynamicOutputTensor(index, num);
        if (outputGe == nullptr) {
            break;
        }
        tensorList.push_back(outputGe);
        num++;
    }
    return GRAPH_SUCCESS;
}

graphStatus ForeachAddListExecuteFunc(OpExecuteContext* hostApiCtx)
{
    OP_CHECK_IF(hostApiCtx == nullptr, OP_LOGE("aclnnfallback", "hostApiCtx nullptr"), return GRAPH_FAILED);

    auto input_num = hostApiCtx->GetComputeNodeInputNum();
    OP_CHECK_IF(input_num <= 1, OP_LOGE("aclnnfallback", "input_num <=1"), return GRAPH_FAILED);

    auto x1 = hostApiCtx->GetDynamicInputTensor(FIRST_INPUT, 0);
    OP_CHECK_IF(x1 == nullptr, OP_LOGE("aclnnfallback", "first dynamic input is null"), return GRAPH_FAILED);

    auto x2 = hostApiCtx->GetDynamicInputTensor(SECOND_INPUT, 0);
    OP_CHECK_IF(x2 == nullptr, OP_LOGE("aclnnfallback", "second dynamic input is null"), return GRAPH_FAILED);

    auto out = hostApiCtx->GetDynamicOutputTensor(FIRST_OUTPUT, 0);
    OP_CHECK_IF(out == nullptr, OP_LOGE("aclnnfallback", "dynamic output is null"), return GRAPH_FAILED);

    std::vector<const gert::Tensor*> geTensorListX1;
    size_t numGeX1 = 0;
    PrepareInputTensorListForeachAddList(hostApiCtx, geTensorListX1, FIRST_INPUT, numGeX1);

    std::vector<const gert::Tensor*> geTensorListX2;
    size_t numGeX2 = 0;
    PrepareInputTensorListForeachAddList(hostApiCtx, geTensorListX2, SECOND_INPUT, numGeX2);

    auto geT = hostApiCtx->GetInputTensor(THIRD_INPUT);
    OP_CHECK_IF(geT == nullptr, OP_LOGE("aclnnfallback", "scalar is nullptr"), return GRAPH_FAILED);

    const ge::DataType x1Type = x1->GetDataType();

    aclScalar* alpha = nullptr;
    switch (x1Type) {
        case DT_FLOAT16:
        case DT_FLOAT:
        case DT_BF16: {
            const float* alpha_data = geT->GetData<float>();
            alpha = ConvertScalarType(static_cast<double>(*alpha_data));
            break;
        }
        case DT_INT32: {
            const int64_t* alpha_data = geT->GetData<int64_t>();
            alpha = ConvertScalarType(*alpha_data);
            break;
        }
        default:
            OP_CHECK_IF(true, OP_LOGE("aclnnfallback", "unsupported tensor dtype"), return GRAPH_FAILED);
    }

    std::vector<const gert::Tensor*> geTenserListOut;
    size_t numGeOut = 0;
    PrepareOutputTensorListForeachAddList(hostApiCtx, geTenserListOut, FIRST_OUTPUT, numGeOut);

    // execute opapi
    auto apiRet = EXEC_OPAPI_CMD(aclnnForeachAddList, geTensorListX1, geTensorListX2, alpha, geTenserListOut);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "apiRet faild:%d", apiRet), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(ForeachAddList).OpExecuteFunc(ForeachAddListExecuteFunc).HostInputs({THIRD_INPUT});

} // namespace fallback

#ifdef __cplusplus
}
#endif
