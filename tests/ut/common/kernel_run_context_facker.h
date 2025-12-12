/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/


#ifndef NN_TESTS_UT_COMMON_KERNEL_RUN_CONTEXT_FACKER_H_
#define NN_TESTS_UT_COMMON_KERNEL_RUN_CONTEXT_FACKER_H_

#define private public
#define protected public

#include "kernel_run_context_holder.h"
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shaperange_context_faker.h"
#include "tiling_context_faker.h"
#include "tiling_parse_context_faker.h"
#include "op_impl_registry.h"

using std::vector;
using std::string;

#endif  //NN_TESTS_UT_COMMON_KERNEL_RUN_CONTEXT_FACKER_H_
