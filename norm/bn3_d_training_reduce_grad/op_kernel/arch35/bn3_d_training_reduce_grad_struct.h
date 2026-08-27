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
 * \file bn3_d_training_reduce_grad_struct.h
 * \brief BN3DTrainingReduceGrad 的 ASCENDC_TPL 模板参数声明
 */

#ifndef BN3_D_TRAINING_REDUCE_GRAD_STRUCT_H_
#define BN3_D_TRAINING_REDUCE_GRAD_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h" // ASCENDC_TPL macros

#define BN3_D_TRAINING_REDUCE_GRAD_RANK_4 4
#define BN3_D_TRAINING_REDUCE_GRAD_RANK_8 8

ASCENDC_TPL_ARGS_DECL(BN3DTrainingReduceGrad,
                      ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST, BN3_D_TRAINING_REDUCE_GRAD_RANK_4,
                                            BN3_D_TRAINING_REDUCE_GRAD_RANK_8));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, BN3_D_TRAINING_REDUCE_GRAD_RANK_4)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, BN3_D_TRAINING_REDUCE_GRAD_RANK_8)));

#endif
