/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_grad_arch35.h
 * \brief GRU 反向 Kernel（dav_3510）—— 基础 API（裸 Mmad/Fixpipe）实现，GM 解耦形态。
 * 聚合 include：类声明（gru_grad_kernel_decl_arch35.h）与全部成员定义（各实现片段）。
 * 架构要点与同步控制面说明见 gru_grad_kernel_decl_arch35.h 及 gru_grad_sync.h。
 */
#ifndef _GRU_GRAD_ARCH35_KERNEL_H_
#define _GRU_GRAD_ARCH35_KERNEL_H_

#include "gru_grad_kernel_decl_arch35.h"
#include "gru_grad_vf_arch35.h"
#include "gru_grad_init_arch35.h"
#include "gru_grad_dgate_arch35.h"
#include "gru_grad_aic_arch35.h"
#include "gru_grad_io_arch35.h"
#include "gru_grad_gates_arch35.h"
#include "gru_grad_band_arch35.h"
#include "gru_grad_transpose_arch35.h"

#endif // _GRU_GRAD_ARCH35_KERNEL_H_
