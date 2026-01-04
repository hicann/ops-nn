/*
 * Copyright (c) 2025 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SILU_MUL_TILING_DEF_H
#define SILU_MUL_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define __CCE_UT_TEST__

#define __aicore__

struct SiluMulTilingData {
  int32_t lastDimSize = 4;
  int32_t batchSize = 2;
  int32_t PPMaxCalNum = 6144;
  uint32_t needCoreNum = 1;
};

inline void ISiluMulTilingData(uint8_t* tiling, SiluMulTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(SiluMulTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
  SiluMulTilingData tilingData;                           \
  ISiluMulTilingData(tilingPointer, &tilingData)
#endif