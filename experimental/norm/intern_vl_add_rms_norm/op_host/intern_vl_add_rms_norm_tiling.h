/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
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
#ifndef INTERNVL_ADD_RMS_NORM_TILING_H_
#define INTERNVL_ADD_RMS_NORM_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(InternVlAddRmsNormTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, hidden_size);
    TILING_DATA_FIELD_DEF(uint32_t, tile_size);
    TILING_DATA_FIELD_DEF(uint32_t, tile_num);
    TILING_DATA_FIELD_DEF(uint32_t, dtype_size);
    TILING_DATA_FIELD_DEF(uint64_t, total_size);
    TILING_DATA_FIELD_DEF(float, eps);
    TILING_DATA_FIELD_DEF(uint32_t, block_num);
    TILING_DATA_FIELD_DEF(uint32_t, tokens_per_core);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InternVlAddRmsNorm, InternVlAddRmsNormTilingData)

} // namespace optiling

#endif
