/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "relu_v3_tiling_data.h"
#include "relu_v3_tiling_key.h"

using namespace AscendC;

template <typename T, typename... Ts>
struct is_one_of : std::false_type {};

template <typename T, typename U, typename... Ts>
struct is_one_of<T, U, Ts...> : std::conditional_t<std::is_same_v<T, U>, std::true_type, is_one_of<T, Ts...>> {};

template <typename T, typename... Ts>
constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;

template <typename T>
__aicore__ inline constexpr T ceil_div(T x, T y)
{
    return (x - 1) / y + 1;
}

template <typename T>
__aicore__ inline constexpr T ceil_round(T x, T y)
{
    return ceil_div(x, y) * y;
}

#define RELU_V3_SETUP()                                                                                  \
    int block_index = GetBlockIdx();                                                                     \
    int block_dim = GetBlockNum();                                                                       \
    constexpr int MTE_BLOCK_SIZE = 512 / sizeof(uint8_t);                                                \
    int compute_blocks = ceil_div(tiling.size, MTE_BLOCK_SIZE);                                          \
    int compute_start = compute_blocks * block_index / block_dim * MTE_BLOCK_SIZE;                       \
    int compute_end = min(compute_blocks * (block_index + 1) / block_dim * MTE_BLOCK_SIZE, tiling.size); \
                                                                                                         \
    GlobalTensor<T> x_global_tensor, y_global_tensor;                                                    \
    GlobalTensor<uint8_t> mask_global_tensor;                                                            \
    x_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));                                     \
    y_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));                                     \
    mask_global_tensor.SetGlobalBuffer(mask);                                                            \
                                                                                                         \
    TPipe t_pipe;                                                                                        \
    TQue<TPosition::VECIN, 1> x_t_que;                                                                   \
    TQue<TPosition::VECOUT, 1> y_t_que, mask_t_que;                                                      \
                                                                                                         \
    constexpr int MAX_TILE_SIZE = (30 << 10) / sizeof(U);                                                \
    t_pipe.InitBuffer(x_t_que, 2, MAX_TILE_SIZE * sizeof(U));                                            \
    t_pipe.InitBuffer(y_t_que, 2, MAX_TILE_SIZE * sizeof(U));                                            \
    t_pipe.InitBuffer(mask_t_que, 2, MAX_TILE_SIZE * sizeof(U));

template <typename T, typename U, typename V>
__aicore__ inline std::enable_if_t<std::is_same_v<T, half>> ReluV3(LocalTensor<U>& y, LocalTensor<uint8_t>& mask,
                                                                   LocalTensor<U>& x, int _)
{
    LocalTensor<V> mask_V = mask.ReinterpretCast<V>();
    Relu(y, x, _);
    Duplicate(mask_V, static_cast<V>(0), _);
    Compare(mask, x, mask_V, CMPMODE::GT, ceil_round(_, static_cast<int>(256 / sizeof(V))));
}

template <typename T, typename U, typename V>
__aicore__ inline std::enable_if_t<std::is_same_v<T, float>> ReluV3(LocalTensor<U>& y, LocalTensor<uint8_t>& mask,
                                                                    LocalTensor<U>& x, int _)
{
    LocalTensor<V> mask_V = mask.ReinterpretCast<V>();
    Maxs(y, x, 0.0f, _);
    Duplicate(mask_V, 0.0f, _);
    Compare(mask, x, mask_V, CMPMODE::GT, ceil_round(_, static_cast<int>(256 / sizeof(V))));
}

template <typename T, typename U, typename V>
__aicore__ inline std::enable_if_t<std::is_same_v<T, int>> ReluV3(LocalTensor<U>& y, LocalTensor<uint8_t>& mask,
                                                                  LocalTensor<U>& x, int _)
{
    LocalTensor<V> mask_V = mask.ReinterpretCast<V>();
    LocalTensor<V> x_V = x.template ReinterpretCast<V>();
    Maxs(y, x, 0, _);
    Cast(x_V, x, RoundMode::CAST_RINT, _);
    Duplicate(mask_V, 0.0f, _);
    Compare(mask, x_V, mask_V, CMPMODE::GT, ceil_round(_, static_cast<int>(256 / sizeof(V))));
}

template <typename T, typename U, typename V>
__aicore__ inline std::enable_if_t<is_one_of_v<T, int8_t, uint8_t>> ReluV3(LocalTensor<U>& y,
                                                                           LocalTensor<uint8_t>& mask,
                                                                           LocalTensor<U>& x, int _)
{
    LocalTensor<U> mask_U = mask.ReinterpretCast<U>();
    Cast(mask_U, x.template ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
    Relu(y, mask_U, _);
    Cast(y.template ReinterpretCast<T>(), y, RoundMode::CAST_TRUNC, _);
    Duplicate(x, static_cast<V>(0), _);
    Compare(mask, mask_U, x, CMPMODE::GT, ceil_round(_, static_cast<int>(256 / sizeof(V))));
}

template <typename T, typename U, typename V>
__aicore__ inline std::enable_if_t<std::is_same_v<T, bfloat16_t>> ReluV3(LocalTensor<U>& y, LocalTensor<uint8_t>& mask,
                                                                         LocalTensor<U>& x, int _)
{
    LocalTensor<U> mask_U = mask.ReinterpretCast<U>();
    Cast(mask_U, x.template ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
    Maxs(y, mask_U, static_cast<U>(0), _);
    Cast(y.template ReinterpretCast<T>(), y, RoundMode::CAST_RINT, _);
    Duplicate(x, static_cast<V>(0), _);
    Compare(mask, mask_U, x, CMPMODE::GT, ceil_round(_, static_cast<int>(256 / sizeof(V))));
}

template <typename T, typename U, typename V>
__aicore__ inline void relu_v3(GM_ADDR x, GM_ADDR y, GM_ADDR mask, const ReluV3TilingData& tiling)
{
    RELU_V3_SETUP()

    for (int i = compute_start; i < compute_end; i += MAX_TILE_SIZE) {
        int _ = min(compute_end - i, MAX_TILE_SIZE);
        {
            LocalTensor<T> x = x_t_que.AllocTensor<T>();
            int tile_size;
            if (_ < MAX_TILE_SIZE)
                tile_size = ceil_round(_, static_cast<int>(32 / sizeof(T)));
            else
                tile_size = _;
            DataCopy(x, x_global_tensor[i], tile_size);
            x_t_que.EnQue(x);
        }
        {
            LocalTensor<U> x = x_t_que.DeQue<U>();
            LocalTensor<U> y = y_t_que.AllocTensor<U>();
            LocalTensor<uint8_t> mask = mask_t_que.AllocTensor<uint8_t>();
            ReluV3<T, U, V>(y, mask, x, _);
            x_t_que.FreeTensor(x);
            y_t_que.EnQue(y);
            mask_t_que.EnQue(mask);
        }
        {
            LocalTensor<T> y = y_t_que.DeQue<T>();
            LocalTensor<uint8_t> mask = mask_t_que.DeQue<uint8_t>();
            int y_tile_size, mask_tile_size;
            if (_ < MAX_TILE_SIZE) {
                y_tile_size = ceil_round(_, static_cast<int>(32 / sizeof(T)));
                mask_tile_size = ceil_round(ceil_div(_, 8), static_cast<int>(32 / sizeof(uint8_t)));
            } else {
                y_tile_size = _;
                mask_tile_size = _ / 8;
            }
            DataCopy(y_global_tensor[i], y, y_tile_size);
            DataCopy(mask_global_tensor[i / 8], mask, mask_tile_size);
            mask_t_que.FreeTensor(mask);
            y_t_que.FreeTensor(y);
        }
    }
}

template <typename T>
__aicore__ inline void relu_v3(GM_ADDR x, GM_ADDR y, GM_ADDR mask, const ReluV3TilingData& tiling)
{
    if constexpr (is_one_of_v<T, half, float>)
        relu_v3<T, T, T>(x, y, mask, tiling);
    else if constexpr (std::is_same_v<T, int>)
        relu_v3<T, T, float>(x, y, mask, tiling);
    else if constexpr (is_one_of_v<T, int8_t, uint8_t>)
        relu_v3<T, half, half>(x, y, mask, tiling);
    else if constexpr (std::is_same_v<T, bfloat16_t>)
        relu_v3<T, float, float>(x, y, mask, tiling);
}
