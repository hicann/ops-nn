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
 * \file silu_mul.cpp
 * \brief silu_mul kernel
 */

#include "silu_mul.h"

using namespace AscendC;

using namespace SiluMul;

extern "C" __global__ __aicore__ void silu_mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        GM_ADDR userWs = nullptr;

        SiluMulND<DTYPE_X> op;

        op.Init(x, y, z, userWs, &tilingData);
        op.Process();
    }
}