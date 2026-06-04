/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mat_mul_nz2nd.h
 * \brief NZ2ND process utilities for AIV-based NZ to ND format conversion.
 *        Provides low-level utility functions that can be reused by different kernel templates.
 *        Refer to mat_mul_nd2nz.h for the ND2NZ counterpart pattern.
 */
#ifndef __OP_KERNEL_MATMUL_V3_NZ2ND_H__
#define __OP_KERNEL_MATMUL_V3_NZ2ND_H__

#include "mat_mul_nz2nd_util.h"
#include "mat_mul_base_block.h"

using namespace AscendC;
using namespace matmul;

const uint8_t AIV_DB_SYNC_FLAG = 0x2;

#endif
