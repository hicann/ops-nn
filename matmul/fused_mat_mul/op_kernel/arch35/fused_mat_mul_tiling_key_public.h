/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_mat_mul_tiling_key_public.h
 * \brief
 */
#ifndef FUSED_MAT_MUL_TILING_KEY_PUBLIC_H
#define FUSED_MAT_MUL_TILING_KEY_PUBLIC_H
#if defined(__CCE_AICORE__)
#include "../../mat_mul_v3/arch35/mat_mul_v3_tiling_key_public.h"
#else
#include "matmul/mat_mul_v3/op_kernel/arch35/mat_mul_v3_tiling_key_public.h"
#endif

#define F_NO_TRANS 0
#define F_A_TRANS 1
#define F_B_TRANS 2
#define F_AB_TRANS 3

#define F_OPTYPE_NONE 0
#define F_OPTYPE_ADD 1
#define F_OPTYPE_MUL 2
#define F_OPTYPE_GELU_ERF 3
#define F_OPTYPE_GELU_TANH 4
#define F_OPTYPE_RELU 5
#endif
