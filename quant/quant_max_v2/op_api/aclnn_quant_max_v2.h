/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_QUANT_MAX_V2_H_
#define OP_API_INC_LEVEL2_ACLNN_QUANT_MAX_V2_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 两段式调用第一步：校验参数、构建执行计划并计算 workspace 大小。
// 参数约定：
//   x                  量化输入，2~8 维，FLOAT/FLOAT16/BFLOAT16。
//   xScaleOptional     量化比例（乘性，=常规量化 scale 的倒数），FLOAT [num_groups]；
//                      传 nullptr 表示动态模式，由 amax 推导 y_scale。
//   groupListOptional  [num_groups] INT64。groupListOptional[g] 表示第 g 组沿第 0 维的累计
//                      结束行号。边界必须单调不减、位于 [0, x.shape[0]]，最后一个边界等于
//                      x.shape[0]；相邻边界相等表示空组。与 V1 一致，ACLNN 由调用方保证这些
//                      数值约束，违反约束的结果未定义；Torch 入口通过缓存的 D2H 检查约束。
//                      传 nullptr 表示单张量场景（等价 G=1）。
//   roundMode          舍入模式字符串：HIFLOAT8 支持 "round"/"hybrid"，FP8 仅支持 "rint"。
//   dstType            目标类型：34=HIFLOAT8，35=FLOAT8_E5M2，36=FLOAT8_E4M3FN。
//   dstTypeMax         量化满量程，有限值且 [0, 32768]；仅对 HIFLOAT8 生效（E5M2/E4M3FN 在
//                      tiling 内覆盖为类型最大值 57344/448）。
//   yOut               量化输出，形状同 x，dtype 由 dstType 决定；支持非连续（内部 Contiguous
//                      + ViewCopy 回写）。
//   amaxOut            必选输出，[num_groups]，dtype 同 x，记录各组 |x| 最大值。
//   yScaleOutOptional  动态模式必传（[num_groups] FLOAT，= amax/dstTypeMax，供反量化用），
//                      支持非连续视图（内部 Contiguous 计算 + ViewCopy 按原 strides 回写）；
//                      静态模式必须传 nullptr（不输出 y_scale，非空会被拒绝）。
//   workspaceSize/executor  出参：workspace 字节数与执行计划句柄。
ACLNN_API aclnnStatus aclnnQuantMaxV2GetWorkspaceSize(const aclTensor* x, const aclTensor* xScaleOptional,
                                                      const aclTensor* groupListOptional, const char* roundMode,
                                                      int64_t dstType, float dstTypeMax, const aclTensor* yOut,
                                                      const aclTensor* amaxOut, const aclTensor* yScaleOutOptional,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor);

// 两段式调用第二步：在指定 stream 上执行 GetWorkspaceSize 生成的执行计划。
// workspace 由调用方按 workspaceSize 申请；executor 与 stream 须来自同一上下文。
ACLNN_API aclnnStatus aclnnQuantMaxV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_QUANT_MAX_V2_H_
