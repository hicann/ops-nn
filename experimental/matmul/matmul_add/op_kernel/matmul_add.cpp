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
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "matmul_add_tiling.h"

using namespace AscendC;


template <typename T>
class KernelMatmulAdd {
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, T>;

    static constexpr MatmulApiStaticTiling MATMUL_ADD_CFG =
        GetMatmulApiTiling<AType, BType, CType, BiasType>(CFG_NORM);

public:
    __aicore__ inline KernelMatmulAdd() {}

    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR b, GM_ADDR bias,
        GM_ADDR y, GM_ADDR workspace,
        GM_ADDR tilingGm)
    {
        auto* header =
            reinterpret_cast<__gm__ optiling::MatmulAddTilingData*>(tilingGm);
        hasBias = (header->has_bias != 0);
        cubeTilingGm = reinterpret_cast<__gm__ TCubeTiling*>(&(header->cubeTiling));

        a_gm.SetGlobalBuffer((__gm__ T*)a);
        b_gm.SetGlobalBuffer((__gm__ T*)b);
        if (hasBias) {
            bias_gm.SetGlobalBuffer((__gm__ T*)bias);
        }
        y_gm.SetGlobalBuffer((__gm__ T*)y);
        mm.Init(cubeTilingGm);
    }

    __aicore__ inline void Process()
    {
        mm.SetTensorA(a_gm);
        mm.SetTensorB(b_gm);
        if (hasBias) {
            mm.SetBias(bias_gm);
        }
        mm.IterateAll(y_gm);
        mm.End();
    }

private:
    GlobalTensor<T> a_gm;
    GlobalTensor<T> b_gm;
    GlobalTensor<T> bias_gm;
    GlobalTensor<T> y_gm;
    Matmul<AType, BType, CType, BiasType, MATMUL_ADD_CFG> mm;
    __gm__ TCubeTiling* cubeTilingGm;
    bool hasBias;
};

extern "C" __global__ __aicore__ void matmul_add(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(10)) {
        KernelMatmulAdd<half> op;
        op.Init(a, b, bias, y, workspace, tiling);
        op.Process();
    } else if (TILING_KEY_IS(30)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        KernelMatmulAdd<bfloat16_t> op;
        op.Init(a, b, bias, y, workspace, tiling);
        op.Process();
#endif
    }
}
