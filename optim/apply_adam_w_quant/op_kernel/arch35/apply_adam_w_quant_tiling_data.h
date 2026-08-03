/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file apply_adam_w_quant_tiling_data.h
 * \brief ApplyAdamWQuant arch35 (Ascend950) plain tiling-data struct.
 *
 * arch35 走 regbase 出包路径(host GetTilingData<T> 直写裸 buffer + kernel
 * GET_TILING_DATA_WITH_STRUCT 直读),故用 plain POD,字段名/顺序/类型与 A2 的
 * framework 版 ApplyAdamWQuantTilingData 一一对应,使 arch35 的计算头(从 A2 拷贝)
 * 无需改动成员访问即可复用。参考 deep_norm/apply_adam_w_v2 的 arch35 tiling_data 布局。
 */

#ifndef APPLY_ADAM_W_QUANT_ARCH35_TILING_DATA_H
#define APPLY_ADAM_W_QUANT_ARCH35_TILING_DATA_H

#include <cstdint>

struct ApplyAdamWQuantRegbaseTilingData {
    uint64_t use_num_core = 0;               // 总共使用的核数
    uint64_t last_pre_core_row_work = 0;     // 尾核一个核循环的个数
    uint64_t not_last_core_num = 0;          // 非尾核的个数
    uint64_t not_last_pre_core_row_work = 0; // 非尾核一个核循环的个数
    uint64_t last_core_last_block = 0;       // 最后一个核最后一次循环的 block 个数
    float lr = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float weight_decay = 0.0f;
    float eps = 0.0f;
    float gnorm_scale = 0.0f;
    int64_t block_size = 0;
    uint64_t one_core_do_block_num_per_row = 0;
    uint64_t tiling_key = 0;
    uint64_t last_block_size = 0; // 最后一个量化 block 的有效元素数
};

#endif // APPLY_ADAM_W_QUANT_ARCH35_TILING_DATA_H
