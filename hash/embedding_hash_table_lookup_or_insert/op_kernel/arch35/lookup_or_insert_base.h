/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file lookup_or_insert_base.h
 * \brief lookup_or_insert_base
 */
#ifndef __LOOKUP_OR_INSERT_BASE_H__
#define __LOOKUP_OR_INSERT_BASE_H__

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "../../inc/hashtable_common.h"

namespace Hashtbl {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 128;
#else
constexpr uint32_t THREAD_NUM = 512;
#endif
constexpr uint32_t UB_BLOCK_SIZE = Ops::Base::GetUbBlockSize();
constexpr uint32_t REG_WIDTH = Ops::Base::GetVRegSize();
constexpr uint32_t VL_FOR_B64 = static_cast<uint32_t>(REG_WIDTH / sizeof(int64_t));

constexpr size_t HANDLE_SIZE_ALL_OFFSET = 2;
constexpr size_t HANDLE_SIZE_ALL_NOEXPORT_OFFSET = 4;
constexpr size_t TABLE_FLAG_OFFSET_FOR_B32 = 20;
constexpr size_t TABLE_STATE_OFFSET = 16;
constexpr size_t COUNTER_OFFSET = 1 * sizeof(int64_t);
constexpr size_t VALUES_OFFSET = 3 * sizeof(int64_t);
constexpr int32_t BIG_ENDIAN_ONE = 16777216; // 0x 01,00,00,00
constexpr int32_t EVICTED_FLAG_MASK = 1 << 3;

template <HardEvent event>
__aicore__ inline void SetWaitFlag()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

__simd_vf__ __aicore__ void ComputeInplaceReduceSumB64(
    __ubuf__ int64_t* inAddr, uint32_t vlForElem, uint32_t elemNum, uint16_t vfLoopNum)
{
    using namespace AscendC::MicroAPI;

    RegTensor<int64_t> inReg;
    RegTensor<int64_t> sumReg, zeroReg;
    Duplicate<int64_t>(sumReg, 0LL);
    Duplicate<int64_t>(zeroReg, 0LL);

    MaskReg maskLoop;
    MaskReg maskALL = CreateMask<int64_t, MaskPattern::ALL>();
    MaskReg maskVL1 = CreateMask<int64_t, MaskPattern::VL1>();
    for (uint16_t i = 0; i < vfLoopNum; ++i) {
        maskLoop = UpdateMask<int64_t>(elemNum);
        LoadAlign<int64_t>(inReg, inAddr + i * vlForElem);
        Add<int64_t>(inReg, inReg, zeroReg, maskLoop); // 用maskLoop把尾Reg的尾部脏数据清0
        Add<int64_t>(sumReg, sumReg, inReg, maskALL);  // 然后sum就可以求和整个Reg的元素
    }
    // 最后ReduceSum一个总和值
    Reduce<AscendC::MicroAPI::ReduceType::SUM, int64_t>(sumReg, sumReg, maskALL);
    StoreAlign<int64_t>(inAddr, sumReg, maskVL1); // 写进第一个位置
}

class KernelLookupOrInsertBase {
public:
    __aicore__ KernelLookupOrInsertBase(TPipe* pipe) : pipe_{pipe}
    {
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
    }

    __aicore__ void Init(GM_ADDR tableHandles, GM_ADDR keys, GM_ADDR values, const LookupOrInsertTilingData data)
    {
        int64_t handleAddr = *reinterpret_cast<__gm__ int64_t*>(
            tableHandles); // tableHandles[0]是当前tableHandle的地址，存储的是64位整型
        pTableHandle_ = reinterpret_cast<__gm__ int64_t*>(handleAddr); // 再把int64_t的值本身转为指针
        int64_t tableAddr = *pTableHandle_;                            // 表本身的地址值
        pTable_ = reinterpret_cast<__gm__ uint8_t*>(tableAddr);
        pKeys_ = reinterpret_cast<__gm__ int64_t*>(keys);
        pValues_ = reinterpret_cast<__gm__ float*>(values);

        tableSize_ = data.size;
        embeddingDim_ = data.embeddingDim;
        filterFreq_ = data.filterFreq;
        keyNum_ = data.keyNum;
        filterMode_ = data.filterMode;
        defaultKeyOrValue_ = data.defaultKeyOrValue;
        defaultKey_ = data.defaultKey;
        defaultValue_ = data.defaultValue;
        filterKeyFlag_ = data.filterKeyFlag;
        filterKey_ = data.filterKey;
        threadXNum_ = data.threadXNum;
        threadYNum_ = data.threadYNum;
        bucketSize_ = RoundUpTo8<size_t>(static_cast<size_t>(VALUES_OFFSET + embeddingDim_ * sizeof(float)));

        uint32_t bufSize = Ops::Base::CeilAlign<uint32_t>(threadYNum_ * sizeof(int64_t), UB_BLOCK_SIZE);
        pipe_->InitBuffer(threadInsertCountsBuf_, bufSize);
    }

    __aicore__ void Process()
    {}

protected:
    uint32_t blockIdx_{0};
    uint32_t blockNum_{0};

    TPipe* pipe_{nullptr};
    TBuf<TPosition::VECCALC> threadInsertCountsBuf_;

    __gm__ int64_t* pTableHandle_{nullptr};
    __gm__ uint8_t* pTable_{nullptr};
    __gm__ int64_t* pKeys_{nullptr};
    __gm__ float* pValues_{nullptr};

    // TilingData
    int64_t tableSize_;
    int64_t embeddingDim_;
    int64_t filterFreq_;
    int64_t keyNum_;
    uint32_t filterMode_;
    uint32_t defaultKeyOrValue_;
    int64_t defaultKey_;
    float defaultValue_;
    uint32_t filterKeyFlag_;
    int64_t filterKey_;
    uint32_t threadXNum_;
    uint32_t threadYNum_;
    size_t bucketSize_;
};
} // namespace Hashtbl

#endif //__LOOKUP_OR_INSERT_BASE_H__