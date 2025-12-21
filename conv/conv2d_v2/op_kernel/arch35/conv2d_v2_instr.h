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
 * \file conv2d_v2_instr.h
 * \brief
 */

#ifndef CONV2D_V2_INSTR_H
#define CONV2D_V2_INSTR_H

#if (__CCE_AICORE__ > 300)
    #include "conv2d_v2_instr_impl.h"
    #include "conv2d_v2_instr_c04_impl.h"
    #include "conv2d_v2_instr_dma_impl.h"
    #include "conv2d_v2_instr_weight_ub_trans_impl.h"
    #include "conv2d_v2_instr_fz_impl.h"
#endif

#endif // CONV2D_V2_INSTR_H