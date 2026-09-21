/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_CLA_GATE_QUANT_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_CLA_GATE_QUANT_H_

#include <tuple>
#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief ClaGateQuant的L0接口，推导4个输出的shape、申请输出tensor并将算子任务下发到执行器。
 *
 * @param globalAttn: Global/CLA分支Attention输出，shape为[T, N, D]。
 * @param localAttn: Local/SWA分支Attention输出，shape同globalAttn。
 * @param globalGateLogits: Global gate的Sigmoid前值，shape为[T, N]。
 * @param localGateLogits: Local gate的Sigmoid前值，shape同globalGateLogits。
 * @param roundMode: 量化数据转换的舍入模式，传入空指针时默认"rint"。
 * @param scaleAlg: 缩放因子的计算方法，0为OCP实现，1为cuBLAS向上取整算法（仅FP8）。
 * @param dstType: 量化数据的数据类型，取值35/36/40/41。
 * @param inputAttnLayout: 输入Attention的排布格式，传入空指针时默认"TND"。
 * @param dualAxisFlag: 是否双轴量化，false时colData、colScale为空tensor（shape为[0]）。
 * @param executor: op执行器，包含了算子计算流程。
 * @return std::tuple，依次为rowData、rowScale、colData、colScale；申请输出tensor或算子下发失败时返回空指针。
 */
std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*> ClaGateQuant(
    const aclTensor* globalAttn, const aclTensor* localAttn, const aclTensor* globalGateLogits,
    const aclTensor* localGateLogits, const char* roundMode, int64_t scaleAlg, int64_t dstType,
    const char* inputAttnLayout, bool dualAxisFlag, aclOpExecutor* executor);

} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_CLA_GATE_QUANT_H_
