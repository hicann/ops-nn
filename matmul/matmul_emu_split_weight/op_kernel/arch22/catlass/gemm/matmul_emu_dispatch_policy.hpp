/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_CATLASS_GEMM_DISPATCH_POLICY_HPP
#define MATMUL_EMU_CATLASS_GEMM_DISPATCH_POLICY_HPP

#include "../matmul_emu_catlass.hpp"
#include "../arch/matmul_emu_arch.hpp"

namespace Catlass::Gemm {

template <class ArchTag_, bool ASYNC_ = false>
struct MmadBase {
    using ArchTag = ArchTag_;
    static constexpr uint32_t ASYNC = ASYNC_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
          bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2,
          uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadPingpong : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

} // namespace Catlass::Gemm

#endif // MATMUL_EMU_CATLASS_GEMM_DISPATCH_POLICY_HPP
