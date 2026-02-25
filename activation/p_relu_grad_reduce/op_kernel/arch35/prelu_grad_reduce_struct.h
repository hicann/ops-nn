/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the 'License').
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prelu_grad_reduce_struct.h
 * \brief prelu_grad_reduce_struct tiling key
 */

#ifndef _PRELU_GRAD_REDUCE_STRUCT_H_
#define _PRELU_GRAD_REDUCE_STRUCT_H_

#include "atvoss/reduce/reduce_tiling_key_decl.h"
#include "atvoss/reduce/reduce_tiling_key_sel.h"

ASCENDC_TPL_ARGS_DECL(PreluGradReduce, REDUCE_TPL_KEY_DECL());

#endif