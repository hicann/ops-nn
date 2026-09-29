/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <type_traits>

#include "tikicpulib.h"

// The shared Blaze stub's nine-argument overload is empty. Numerical tests must
// execute the copy, so suppress it and forward to the simulator's legacy form.
#ifndef COPY_GM_TO_CBUF_V2_WITH_CACHE_MODE
#define COPY_GM_TO_CBUF_V2_WITH_CACHE_MODE
#endif
#include "blaze_kernel_stub.h"

template <typename Dst, typename Src, typename CacheMode>
inline auto copy_gm_to_cbuf_v2(Dst* dst, Src* src, uint8_t sid, uint32_t nBurst, uint32_t lenBurst, uint8_t padFuncMode,
                               CacheMode cacheMode, uint64_t srcStride,
                               uint32_t dstStride) -> decltype(copy_gm_to_cbuf_v2(dst, src, sid, nBurst, lenBurst,
                                                                                  padFuncMode, srcStride, dstStride))
{
    (void)cacheMode;
    return copy_gm_to_cbuf_v2(dst, src, sid, nBurst, lenBurst, padFuncMode, srcStride, dstStride);
}

// Parse SDK headers before redirecting the production code's logical addresses.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"

namespace QMMAQ::CpuDebug {

template <typename Location>
inline uint8_t* GetBufferBaseAddress()
{
    auto& buffers = AscendC::ConstDefiner::Instance();
    using namespace asc::te;
    if constexpr (std::is_same_v<Location, location::ub>) {
        return buffers.GetHardwareBaseAddr(AscendC::Hardware::UB);
    } else if constexpr (std::is_same_v<Location, location::l1>) {
        return buffers.GetHardwareBaseAddr(AscendC::Hardware::L1);
    } else if constexpr (std::is_same_v<Location, location::l0a>) {
        return buffers.GetHardwareBaseAddr(AscendC::Hardware::L0A);
    } else if constexpr (std::is_same_v<Location, location::l0b>) {
        return buffers.GetHardwareBaseAddr(AscendC::Hardware::L0B);
    } else if constexpr (std::is_same_v<Location, location::l0c>) {
        return buffers.GetHardwareBaseAddr(AscendC::Hardware::L0C);
    } else if constexpr (std::is_same_v<Location, location::bias>) {
        return buffers.GetHardwareBaseAddr(AscendC::Hardware::BIAS);
    } else if constexpr (std::is_same_v<Location, location::l0scalea>) {
        // MX scale storage is separate from L0A/L0B in the CPU simulator, as in
        // AscendC's LoadData2DL12L0A/B implementation for Ascend950.
        return buffers.cpuL0AMx;
    } else if constexpr (std::is_same_v<Location, location::l0scaleb>) {
        return buffers.cpuL0BMx;
    } else {
        static_assert(!std::is_same_v<Location, Location>, "Unsupported kernel UT memory location");
    }
}

inline uint64_t GetUbAddress(uint64_t byteOffset)
{
    return reinterpret_cast<uintptr_t>(GetBufferBaseAddress<asc::te::location::ub>()) + byteOffset;
}

} // namespace QMMAQ::CpuDebug

namespace asc::te {

template <typename Location, typename DataType, typename Addr, enable_make_ptr_by_trait<Location, Addr> = 0>
inline auto QmmaqMakeMemPtr(Addr byteOffset)
{
    auto* physicalAddress = reinterpret_cast<DataType*>(QMMAQ::CpuDebug::GetBufferBaseAddress<Location>() + byteOffset);
    return make_mem_ptr<Location>(physicalAddress);
}

template <typename Location, typename Iterator, enable_make_hardware_ptr<Location, Iterator> = 0>
inline constexpr auto QmmaqMakeMemPtr(Iterator iterator)
{
    return make_mem_ptr<Location>(iterator);
}

template <typename Iterator, enable_make_ptr_by_iter<Iterator> = 0>
inline constexpr auto QmmaqMakeMemPtr(Iterator iterator)
{
    return make_mem_ptr(iterator);
}

} // namespace asc::te

// The test source undefines these immediately after including the real kernel.
// Raw addresses in the activation epilogues are UB addresses; Tensor API calls
// above retain their explicit Cube/Vector memory location.
#define make_mem_ptr QmmaqMakeMemPtr
#define asc_get_phy_buf_addr(byteOffset) QMMAQ::CpuDebug::GetUbAddress(byteOffset)
