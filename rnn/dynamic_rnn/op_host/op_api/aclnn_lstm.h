/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_LSTM_H_
#define OP_API_INC_LEVEL2_ACLNN_LSTM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnLSTMGetWorkspaceSize(
    const aclTensor *input,
    const aclTensorList *params,
    const aclTensorList *hx,
    const aclTensor *batchSizes,
    bool has_biases,
    int64_t numLayers,
    double droupout,
    bool train,
    bool bidirectional,        
    bool batch_first,
    aclTensor *output,
    aclTensor *hy,
    aclTensor *cy,
    aclTensorList *iOut,  
    aclTensorList *jOut, 
    aclTensorList *fOut,
    aclTensorList *oOut,
    aclTensorList *hOut,
    aclTensorList *cOut,
    aclTensorList *tanhCOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
/**
 * @brief A second interface of aclnnLSTMGetWorkspaceSize, used to perform calculation.
 * @param [in] workspace: start address of the workspace memory allocated on the NPU device.
 * @param [in] workspaceSize: size of the workspace applied on the NPU device, which is obtained by calling the first segment interface aclnnMaxPool3dWithArgmaxGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: acl stream.
 * @return aclnnStatus: returned status code
 */

ACLNN_API aclnnStatus aclnnLSTM(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_MAX_POOL3D_WITH_ARGMAX_H_