/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file quantize_tpl_def.h
 * \brief
 */

#ifndef OPS_ACTIVATION_QUANTIZE_OP_KERNEL_V35_QUANTIZE_TPL_DEF_H_
#define OPS_ACTIVATION_QUANTIZE_OP_KERNEL_V35_QUANTIZE_TPL_DEF_H_

#define TPL_PER_TENSOR 0
#define TPL_PER_CHANNEL 1
#define TPL_PER_HEAD 2
#define TPL_PER_CHANNEL_NDDMA 3

#define TPL_NONE 0
#define TPL_INT8 1
#define TPL_UINT8 2
#define TPL_INT32 3
#define TPL_BF16 4
#define TPL_HIFLOAT8 5
#define TPL_FP8_E4M3FN 6
#define TPL_FP8_E5M2 7
#define TPL_INT4 8
#define TPL_FLOAT 9

#define TPL_DIV_MODE_DIV 1
#define TPL_DIV_MODE_MUL 2

#define TPL_ROUND_MODE_NONE 0
#define TPL_ROUND_MODE_ROUND 1
#define TPL_ROUND_MODE_FLOOR 2
#define TPL_ROUND_MODE_CEIL 3
#define TPL_ROUND_MODE_TRUNC 4
#define TPL_ROUND_MODE_RINT 5
#define TPL_ROUND_MODE_HYBRID 6

#define TPL_NO_SQRT_MODE 0
#define TPL_SQRT_MODE 1

#endif // _OPS_ACTIVATION_QUANTIZE_OP_KERNEL_V35_QUANTIZE_TPL_DEF_H_