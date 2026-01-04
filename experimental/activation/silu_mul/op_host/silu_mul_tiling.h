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

/*!
 * \file silu_mul_tiling_def.h
 * \brief silu_mul_tiling_def
 */

#ifndef SILU_MUL_TILING_DEF_H
#define SILU_MUL_TILING_DEF_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct SiluMulCompileInfo {};

BEGIN_TILING_DATA_DEF(SiluMulTilingData)

TILING_DATA_FIELD_DEF(int32_t, lastDimSize);
TILING_DATA_FIELD_DEF(int32_t, batchSize);
TILING_DATA_FIELD_DEF(int32_t, PPMaxCalNum);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SiluMul, SiluMulTilingData)
} // namespace optiling

#endif