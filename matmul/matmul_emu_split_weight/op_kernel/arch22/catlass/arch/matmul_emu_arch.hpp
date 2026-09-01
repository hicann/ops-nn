/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_CATLASS_ARCH_ARCH_HPP
#define MATMUL_EMU_CATLASS_ARCH_ARCH_HPP

#include "../matmul_emu_catlass.hpp"

namespace Catlass::Arch {

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7 * 1024;
    static constexpr uint32_t UB_SIZE = 192 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

template <AscendC::TPosition POS>
using PositionType = std::integral_constant<AscendC::TPosition, POS>;

using PositionGM = PositionType<AscendC::TPosition::GM>;
using PositionL1 = PositionType<AscendC::TPosition::A1>;
using PositionL0A = PositionType<AscendC::TPosition::A2>;
using PositionL0B = PositionType<AscendC::TPosition::B2>;
using PositionL0C = PositionType<AscendC::TPosition::CO1>;
using PositionBias = PositionType<AscendC::TPosition::C2>;
using PositionUB = PositionType<AscendC::TPosition::VECCALC>;

struct LocalTensorBufferBase {
public:
    template <class Element = half>
    CATLASS_DEVICE AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const
    {
        return tensor[offset].template ReinterpretCast<Element>();
    }

protected:
    CATLASS_DEVICE
    LocalTensorBufferBase() = default;

    AscendC::LocalTensor<uint8_t> tensor;
};

template <class ArchTag, AscendC::TPosition Position>
struct LocalTensorBuffer {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported local tensor buffer, can not find the specialization.");
};

/// Partial specialization for TPosition::A1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A1> tbufA1;
        GetTPipePtr()->InitBuffer(tbufA1, ArchTag::L1_SIZE);
        tensor = tbufA1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::A2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A2> tbufA2;
        GetTPipePtr()->InitBuffer(tbufA2, ArchTag::L0A_SIZE);
        tensor = tbufA2.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::B2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::B2> tbufB2;
        GetTPipePtr()->InitBuffer(tbufB2, ArchTag::L0B_SIZE);
        tensor = tbufB2.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::C2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2> tbufC2;
        GetTPipePtr()->InitBuffer(tbufC2, ArchTag::BIAS_SIZE);
        tensor = tbufC2.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::CO1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::CO1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::CO1> tbufCO1;
        GetTPipePtr()->InitBuffer(tbufCO1, ArchTag::L0C_SIZE);
        tensor = tbufCO1.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::C2PIPE2GM
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::C2PIPE2GM> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C2PIPE2GM;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2PIPE2GM> tbufC2PIPE2GM;
        GetTPipePtr()->InitBuffer(tbufC2PIPE2GM, ArchTag::FIXBUF_SIZE);
        tensor = tbufC2PIPE2GM.Get<uint8_t>();
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECCALC
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECCALC> tbufVECCALC;
        GetTPipePtr()->InitBuffer(tbufVECCALC, ArchTag::UB_SIZE);
        tensor = tbufVECCALC.Get<uint8_t>();
    }
};

template <class ArchTag>
struct Resource {
public:
    AscendC::TPipe pipe;

    LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> btBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2PIPE2GM> fpBuf;

    CATLASS_DEVICE
    Resource()
    {
        // The initialization of AscendC::Tpipe will insert some synchronization interfaces,
        // which may conflict with the usage by users. Therefore, the "destroy" interface is used for releasing.
        pipe.Destroy();
    }
};

// A certain cross core flag can be continuously set up to 15 times without waiting. In scenarios where there is only
// one-way synchronization between cores and the synchronization operation is executed multiple times, if the core
// performing the set operation runs faster, it may result in the flag being set more than 15 times consecutively,
// leading to a system freeze. To prevent such a freeze, we need to wait for the waiting cores to perform a reverse
// synchronization operation after executing the set operation MAX_REVERSE_DEPTH times.
constexpr uint32_t MAX_REVERSE_DEPTH = 15;

using FlagID = uint16_t;
constexpr FlagID AIV_INTER_BLOCK_BARRIER = 8;
constexpr FlagID AIC_INTER_BLOCK_BARRIER = 9;
constexpr FlagID AIV_INTER_SUBBLOCK_BARRIER = 10;
constexpr FlagID FFTS_MAX_FLAG = 7;

struct CrossCoreFlag {
    CATLASS_DEVICE
    CrossCoreFlag() : id(0) {}

    CATLASS_DEVICE
    CrossCoreFlag(FlagID id) : id(id) {}

    FlagID id;
};

template <uint32_t REVERSE_DEPTH_ = MAX_REVERSE_DEPTH>
struct CrossCoreFlagWithReverse {
    CATLASS_DEVICE
    CrossCoreFlagWithReverse() : id(0), reverseId(0) {}

    CATLASS_DEVICE
    CrossCoreFlagWithReverse(FlagID id, FlagID reverseId) : id(id), reverseId(reverseId) {}

    FlagID id;
    FlagID reverseId;
    uint32_t count{0};
};

template <uint8_t MODE, pipe_t PIPE>
CATLASS_DEVICE void CrossCoreSetFlag(CrossCoreFlag& flag)
{
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
}

CATLASS_DEVICE
void CrossCoreWaitFlag(CrossCoreFlag& flag) { AscendC::CrossCoreWaitFlag(flag.id); }

template <uint8_t MODE, pipe_t PIPE>
CATLASS_DEVICE void CrossCoreWaitFlag(CrossCoreFlag& flag)
{
    AscendC::CrossCoreWaitFlag<MODE, PIPE>(flag.id);
}

template <uint8_t MODE, pipe_t PIPE, uint32_t REVERSE_DEPTH>
CATLASS_DEVICE void CrossCoreSetFlagWithReverse(CrossCoreFlagWithReverse<REVERSE_DEPTH>& flag)
{
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
    if (++flag.count >= REVERSE_DEPTH) {
        AscendC::CrossCoreWaitFlag(flag.reverseId);
        flag.count = 0;
    }
}

template <uint8_t MODE, pipe_t PIPE, uint32_t REVERSE_DEPTH>
CATLASS_DEVICE void CrossCoreWaitFlagWithReverse(CrossCoreFlagWithReverse<REVERSE_DEPTH>& flag)
{
    AscendC::CrossCoreWaitFlag(flag.id);
    if (++flag.count >= REVERSE_DEPTH) {
        AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.reverseId);
        flag.count = 0;
    }
}

} // namespace Catlass::Arch

#endif // MATMUL_EMU_CATLASS_ARCH_ARCH_HPP
