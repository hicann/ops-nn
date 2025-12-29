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
 * \file embedding_hash_table_lookup_or_insert.cpp
 * \brief embedding_hash_table_lookup_or_insert
 */

#include "./arch35/embedding_hash_table_lookup_or_insert.h"

using namespace Hashtbl;

extern "C" __global__ __aicore__ void embedding_hash_table_lookup_or_insert(GM_ADDR tableHandle, GM_ADDR keys, GM_ADDR values, GM_ADDR workspace,
                                                     GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if (workspace == nullptr) {
    return;
  }
  SetSysWorkspace(workspace);
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }
  GET_TILING_DATA(tilingData, tiling);

  KernelLookupOrInsert op;
  op.Init(tableHandle, keys, values, tilingData);
  op.Process();
}
