/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_QUANT_MAX_V2_H_
#define OP_API_INC_LEVEL0_QUANT_MAX_V2_H_

#include "opdev/op_executor.h"

namespace l0op {
constexpr size_t QUANT_MAX_V2_OUT_NUM = 3;

// QuantMaxV2 L0 算子接口：按张量或按组做 max 量化（y = x*scale 后取整到目标类型）。
// 参数约定：
//   x                量化输入，2~8 维，FLOAT/FLOAT16/BFLOAT16。
//   scale            量化比例（乘性，即常规量化 scale 的倒数）；nullptr 表示动态模式，
//                    由 amax 现场推导 y_scale。
//   groupList        [num_groups] INT64，各组沿第 0 维的累计结束行号；nullptr 表示单张量
//                    （等价 G=1）。数值约束（单调不减、末值=rows）由调用方保证。
//   roundMode        舍入模式：HIFLOAT8 支持 "round"/"hybrid"，FP8 仅支持 "rint"。
//   dstType          目标类型：34=HIFLOAT8，35=FLOAT8_E5M2，36=FLOAT8_E4M3FN。
//   dstTypeMax       量化满量程，取值 [0, 32768]；仅对 HIFLOAT8 生效（FP8 在 tiling 内
//                    覆盖为类型最大值）。
//   y/amax/yScaleOutOptional  三个输出张量：amax 必选（[num_groups]，dtype 同 x）；
//                    yScale 仅动态模式写入（[num_groups] FLOAT），静态模式可传 nullptr 或
//                    非空张量（不写值）。
// 返回：与输入张量一一对应的输出指针数组 {y, amax, yScale}，失败时为全 nullptr。
const std::array<aclTensor*, QUANT_MAX_V2_OUT_NUM> QuantMaxV2(const aclTensor* x, const aclTensor* scale,
                                                              const aclTensor* groupList, const char* roundMode,
                                                              int64_t dstType, float dstTypeMax, const aclTensor* y,
                                                              const aclTensor* amax, const aclTensor* yScaleOutOptional,
                                                              aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_QUANT_MAX_V2_H_
