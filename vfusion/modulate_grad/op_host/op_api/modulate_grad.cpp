/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "modulate_grad.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include <tuple>

namespace l0op{
    OP_TYPE_REGISTER(ModulateGrad);
    std::tuple<const aclTensor*,const aclTensor*,const aclTensor*> ModulateGrad(const aclTensor *grad_output, const aclTensor *input, const aclTensor *scale, 
                                    const aclTensor * shift, aclOpExecutor *executor){
                                        L0_DFX(ModulateGrad,grad_output, input, scale, shift);

                                        auto grad_inputShape = grad_output->GetViewShape();
                                        const aclTensor* grad_scale = nullptr;
                                        const aclTensor* grad_shift = nullptr;
                                        if (scale != nullptr){
                                            auto grad_scaleShape = scale->GetViewShape();
                                            grad_scale = executor->AllocTensor(grad_scaleShape, scale->GetDataType());
                                        }
                                        if (shift != nullptr){
                                            auto grad_shiftShape = shift->GetViewShape();
                                            grad_shift = executor->AllocTensor(grad_shiftShape, shift->GetDataType());
                                        }
                                        const aclTensor* grad_input = executor->AllocTensor(grad_inputShape, grad_output->GetDataType());
                                        ADD_TO_LAUNCHER_LIST_AICORE(ModulateGrad,
                                                                OP_INPUT(grad_output,input,scale,shift),
                                                                OP_OUTPUT(grad_input,grad_scale,grad_shift)
                                                                );
                                        return std::make_tuple(grad_input,grad_scale,grad_shift);
    }
}
//namespace l0op