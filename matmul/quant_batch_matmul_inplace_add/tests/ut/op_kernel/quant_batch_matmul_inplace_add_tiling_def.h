/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BATCH_MATMUL_INPLACE_ADD_TILING_DEF_H
#define QUANT_BATCH_MATMUL_INPLACE_ADD_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "arch35/quant_batch_matmul_inplace_add_tiling_data.h"

#if defined(ASCENDC_CPU_DEBUG) && defined(__CCE_KT_TEST__)
// Define DT_* before the tiling-key header evaluates its preprocessor guards.
#include "kernel_operator.h"
#include "stub_fun.h"

#ifndef __biasbuf__
#define __biasbuf__
#endif

#ifndef ASCENDC_POS_CONSTANTS_DEFINED
#include "blaze_kernel_stub.h"
#endif

// Tensor API uses Bisheng type names; tikicpulib exposes the same CPU types
// under fp4/fp8 names. Keep these aliases local to this operator's Kernel UT.
using float8_e4m3_t = fp8_e4m3fn_t;
using float8_e5m2_t = fp8_e5m2_t;
using float8_e8m0_t = fp8_e8m0_t;
using float4_e1m2x2_t = fp4x2_e1m2_t;
using float4_e2m1x2_t = fp4x2_e2m1_t;

// The C API supplies an extra cache hint (currently zero). CPU simulation has
// no cache policy, but must still perform the real GM-to-L1 copy, not a no-op.
inline void copy_gm_to_cbuf_v2(void* dst, void* src, uint8_t sid, uint32_t nBurst, uint32_t lenBurst, uint8_t padMode,
                               uint8_t /* cacheMode */, uint64_t srcStride, uint32_t dstStride)
{
    copy_gm_to_cbuf_v2(dst, src, sid, nBurst, lenBurst, padMode, srcStride, dstStride);
}

// Install the CPU overloads before other Tensor API headers bind make_mem_ptr.
// Device offsets are not host pointers; use the SDK's per-space CPU buffers.
#include "tensor_api/tensor/pointer.h"
namespace asc::te {
template <typename Location>
inline uint8_t* QbmmiaCpuMemoryBase()
{
    auto& buffers = AscendC::ConstDefiner::Instance();
    if constexpr (Std::is_same_v<Location, location::l0scalea>) {
        return buffers.cpuL0AMx;
    } else if constexpr (Std::is_same_v<Location, location::l0scaleb>) {
        return buffers.cpuL0BMx;
    } else {
        constexpr auto hardware = [] {
            if constexpr (Std::is_same_v<Location, location::l1>)
                return AscendC::Hardware::L1;
            else if constexpr (Std::is_same_v<Location, location::l0a>)
                return AscendC::Hardware::L0A;
            else if constexpr (Std::is_same_v<Location, location::l0b>)
                return AscendC::Hardware::L0B;
            else if constexpr (Std::is_same_v<Location, location::l0c>)
                return AscendC::Hardware::L0C;
            else if constexpr (Std::is_same_v<Location, location::bias>)
                return AscendC::Hardware::BIAS;
            else if constexpr (Std::is_same_v<Location, location::fixbuf>)
                return AscendC::Hardware::FIXBUF;
            else {
                static_assert(Std::is_same_v<Location, location::ub>, "Unsupported CPU memory space");
                return AscendC::Hardware::UB;
            }
        }();
        return buffers.GetHardwareBaseAddr(hardware);
    }
}

template <typename Location, typename DataType = uint16_t>
inline auto make_mem_ptr(uint64_t offset)
{
    using pointer = typename location_attr<DataType>::location_map::template get<Location>;
    if constexpr (Std::is_same_v<Location, location::gm>) {
        return make_location_mem_ptr<Location>(reinterpret_cast<pointer>(offset));
    } else {
        return make_location_mem_ptr<Location>(reinterpret_cast<pointer>(QbmmiaCpuMemoryBase<Location>() + offset));
    }
}

template <typename Location, typename DataType = uint16_t>
inline auto make_mem_ptr(uint32_t offset)
{
    return make_mem_ptr<Location, DataType>(static_cast<uint64_t>(offset));
}
} // namespace asc::te
#endif

template <typename T>
inline void InitQbmmiaTilingData(uint8_t* tiling, T* constData)
{
    memcpy(constData, tiling, sizeof(T));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    QMMIA::QbmmiaTilingData tiling_data;         \
    InitQbmmiaTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitQbmmiaTilingData(tiling_arg, &tiling_data)

#endif // QUANT_BATCH_MATMUL_INPLACE_ADD_TILING_DEF_H
