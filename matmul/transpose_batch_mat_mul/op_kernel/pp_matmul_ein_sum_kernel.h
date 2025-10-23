/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pp_matmul_ein_sum_kernel.h
 * \brief
 */

#ifndef __BATCH_MAT_MUL_V4_EIN_SUM_KERNEL_H__
#define __BATCH_MAT_MUL_V4_EIN_SUM_KERNEL_H__

#ifdef __CCE_KT_TEST__
using __bf16 = bfloat16_t;
#include "stub_def.h"
#include "stub_fun.h"
#else
#define __aicore__ [aicore]
#endif
#include "utils/common_func.h"
#include "utils/common.h"
#include "utils/hardware.h"
#include "utils/iterator.h"
#include "utils/utils.h"
#include "pp_matmul_common.h"

namespace PpMatMulNS {

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
template <uint32_t SwizzleDirect, bool TA, bool TB, typename InDtype = half, typename OutDtype = half, DataFormat FormatB = DataFormat::ND>
class PpMatmulEinSum {
    using LocalTensor = AscendC::LocalTensor<InDtype>;
    template <DataFormat srcFormat = DataFormat::ND, DataFormat dstFormat = DataFormat::ND>
    using CopyGmToCbuf = gm_to_l1<ArchType::ASCEND_V220, InDtype, srcFormat, dstFormat>;
    using LoadCbufToCa = l1_to_l0_a<ArchType::ASCEND_V220, InDtype, TA, DataFormat::ZN, DataFormat::ZZ>;
    using LoadCbufToCb = l1_to_l0_b<ArchType::ASCEND_V220, InDtype, TB, DataFormat::ZN, DataFormat::NZ>;
    using CopyCcToGm = l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, OutDtype, float>;
public:
    __aicore__ explicit PpMatmulEinSum(){};

    __aicore__ FORCE_INLINE void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, const PpMatmulTilingData* tilingData, AscendC::TPipe* tPipe)
    {
        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(a));
        gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(b));
        gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(c));
        batch_size = tilingData->batch;
        m = tilingData->m;
        k = tilingData->k;
        n = tilingData->n;
        m0 = tilingData->m0;
        k0 = tilingData->k0;
        n0 = tilingData->n0;
        tdim.m = tilingData->mLoop;
        tdim.k = tilingData->kLoop;
        tdim.n = tilingData->nLoop;
        core_loop = tilingData->coreLoop;
        swizzle_cnt = tilingData->swizzlCount;
        en_shuffle_k = tilingData->enShuffleK;

        AsdopsBuffer<ArchType::ASCEND_V220> buf(tPipe);
        l1_base_a = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
        l1_base_b = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(RoundUp<CONST_256>(m0 * k0 * sizeof(InDtype)));
        l0a_base = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
        l0b_base = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
        num_core = AscendC::GetBlockNum();
        core_idx = AscendC::GetBlockIdx();
        ping_flag = 1;
    }

    __aicore__ FORCE_INLINE void GetBlockIdx(uint64_t index, MatCoord &tidx)
    {
        uint64_t in_batch_idx = index % (tdim.m * tdim.n);
        if constexpr (SwizzleDirect == 0) { // Zn
            uint64_t tile_block_loop = (tdim.m + swizzle_cnt - 1) / swizzle_cnt;
            uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * tdim.n);
            uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * tdim.n);

            uint64_t n_row = swizzle_cnt;
            if (tile_block_idx == tile_block_loop - 1) {
                n_row = tdim.m - swizzle_cnt * tile_block_idx;
            }
            tidx.m = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_row;
            tidx.n = in_tile_block_idx / n_row;
            if (tile_block_idx % 2 != 0) {
                tidx.n = tdim.n - tidx.n - 1;
            }
        } else if constexpr (SwizzleDirect == 1) { // Nz
            uint64_t tile_block_loop = (tdim.n + swizzle_cnt - 1) / swizzle_cnt;
            uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * tdim.m);
            uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * tdim.m);

            uint64_t n_col = swizzle_cnt;
            if (tile_block_idx == tile_block_loop - 1) {
                n_col = tdim.n - swizzle_cnt * tile_block_idx;
            }
            tidx.m = in_tile_block_idx / n_col;
            tidx.n = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_col;
            if (tile_block_idx % 2 != 0) {
                tidx.m = tdim.m - tidx.m - 1;
            }
        }
    }

    __aicore__ FORCE_INLINE void Process()
    {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);

        constexpr uint32_t K_ROUND_CONST = (std::is_same_v<InDtype, float>) ? CONST_8 : CONST_16;
        uint32_t L0_PINGPONG_BUFFER_SIZE = L0_PINGPONG_BUFFER_LEN / sizeof(InDtype);
        uint32_t L1_PINGPONG_BUFFER_SIZE = L1_PINGPONG_BUFFER_LEN / sizeof(InDtype);
        uint8_t padList[4] = {0, 0, 0, 0};
        for (uint64_t loop_idx = core_idx; loop_idx < core_loop; loop_idx += num_core) {
            uint64_t batch_idx = loop_idx / tdim.n / tdim.m;
            MatCoord tidx{0};
            GetBlockIdx(loop_idx, tidx);
            uint64_t offset_a = 0, offset_b = 0, offset_a_next = 0, offset_b_next = 0;
            uint64_t offset_c = tidx.m * m0 * batch_size * n + batch_idx * n + tidx.n * n0;
            uint64_t m_actual = (tidx.m == (tdim.m - 1)) ? (m - tidx.m * m0) : m0;
            uint64_t n_actual = (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
            uint64_t m_round = RoundUp16(m_actual);
            uint64_t n_round = RoundUp16(n_actual);
            uint64_t mn_max = m_round > n_round ? m_round : n_round;
            uint64_t k_part_len = L0_PINGPONG_BUFFER_SIZE / mn_max / CONST_16 * CONST_16;
            uint64_t shuffle_k = en_shuffle_k ? (core_idx % tdim.k) : 0;
            if (TA) {
                offset_a = shuffle_k * k0 * m * batch_size + batch_idx * m + tidx.m * m0;
            } else {
                offset_a = tidx.m * m0 * batch_size * k + batch_idx * k + shuffle_k * k0;
            }

            if (TB) {
                if constexpr (FormatB != DataFormat::NZ) {
                    offset_b = batch_idx * k * n + tidx.n * n0 * k + shuffle_k * k0;
                } else {
                    offset_b = batch_idx * RoundUp16(k) * RoundUp16(n) + tidx.n * n0 * BLOCK_SIZE_16 + shuffle_k * k0 * RoundUp16(n);
                }
            } else {
                if constexpr (FormatB != DataFormat::NZ) {
                    offset_b = batch_idx * k * n + shuffle_k * k0 * n + tidx.n * n0;
                } else {
                    offset_b = batch_idx * RoundUp16(k) * RoundUp16(n) + shuffle_k * k0 * BLOCK_SIZE_16 + tidx.n * n0 * RoundUp16(k);
                }
            }

            uint64_t k_actual = (shuffle_k == tdim.k - 1) ? k - shuffle_k * k0 : k0;
            uint64_t k_round = RoundUp<K_ROUND_CONST>(k_actual);

            LocalTensor l1_buf_a = ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_SIZE];
            LocalTensor l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_SIZE];
            LocalTensor l0a_buf = ping_flag ? l0a_base : l0a_base[L0_PINGPONG_BUFFER_SIZE];
            LocalTensor l0b_buf = ping_flag ? l0b_base : l0b_base[L0_PINGPONG_BUFFER_SIZE];
            event_t event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            if (loop_idx == core_idx) {
                WAIT_FLAG(MTE1, MTE2, event_id);
                // *** load matrix A to L1
                if ((m == 1) || (m_actual == 1 && !TA)) {
                    CopyGmToCbuf<DataFormat::ND, DataFormat::ND>(l1_buf_a,       // dst
                                 gm_a[offset_a], // src
                                 1,              // nTileActual
                                 16,             // nTileCeil
                                 1,              // nVal
                                 k_actual,       // kTileActual
                                 k_round,        // kTileCeil
                                 k);             // dVal
                } else {
                    if (TA) {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a,        // dst
                                          gm_a[offset_a],  // src
                                          k_actual,        // nTileActual
                                          k_round,         // nTileCeil
                                          k,               // nVal
                                          m_actual,        // dTileActual
                                          m_round,         // dTileCeil
                                          m * batch_size); // dVal
                    } else {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a,        // dst
                                          gm_a[offset_a],  // src
                                          m_actual,        // nTileActual
                                          m_round,         // nTileCeil
                                          m,               // nVal
                                          k_actual,        // dTileActual
                                          k_round,         // dTileCeil
                                          k * batch_size); // dVal
                    }
                }
                SET_FLAG(MTE2, MTE1, event_id);
                // *** load matrix B to L1
                WaitFlag<HardEvent::MTE1_MTE2>(event_id + 2);
                if constexpr (FormatB != DataFormat::NZ) {
                    if (TB) {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b,       // dst
                                        gm_b[offset_b], // src
                                        n_actual,       // nTileActual
                                        n_round,        // nTileCeil
                                        n,              // nVal
                                        k_actual,       // dTileActual
                                        k_round,        // dTileCeil
                                        k);             // dVal
                    } else {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b,       // dst
                                        gm_b[offset_b], // src
                                        k_actual,       // nTileActual
                                        k_round,        // nTileCeil
                                        k,              // nVal
                                        n_actual,       // dTileActual
                                        n_round,        // dTileCeil
                                        n);             // dVal
                    }
                } else {
                    if (TB) {
                        CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b,       // dst
                                        gm_b[offset_b], // src
                                        n_actual,       // nTileActual
                                        n_round,        // nTileCeil
                                        RoundUp16(n),              // nVal
                                        k_actual,       // dTileActual
                                        k_round,        // dTileCeil
                                        RoundUp16(k));             // dVal
                    } else {
                        CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b,       // dst
                                        gm_b[offset_b], // src
                                        k_actual,       // nTileActual
                                        k_round,        // nTileCeil
                                        RoundUp16(k),              // nVal
                                        n_actual,       // dTileActual
                                        n_round,        // dTileCeil
                                        RoundUp16(n));             // dVal
                    }
                }
                SET_FLAG(MTE2, MTE1, event_id + 2);
            }

            for (tidx.k = 0; tidx.k < tdim.k; ++tidx.k) {
                shuffle_k = en_shuffle_k ? (tidx.k + core_idx) % tdim.k : tidx.k;
                uint64_t k_actual = (shuffle_k == (tdim.k - 1)) ? (k - shuffle_k * k0) : k0;
                uint64_t k_round = RoundUp<K_ROUND_CONST>(k_actual);
                fdim.k = (k_actual + k_part_len - 1) / k_part_len;

                LocalTensor l1_buf_a = ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_SIZE];
                LocalTensor l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_SIZE];
                auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

                if (tidx.k < tdim.k - 1) {
                    uint64_t shuffle_k_next = en_shuffle_k ? (core_idx + tidx.k + 1) % tdim.k : (tidx.k + 1);
                    if (TA) {
                        offset_a_next = shuffle_k_next * k0 * m * batch_size + batch_idx * m + tidx.m * m0;
                    } else {
                        offset_a_next = tidx.m * m0 * batch_size * k + batch_idx * k + shuffle_k_next * k0;
                    }

                    if (TB) {
                        if constexpr (FormatB != DataFormat::NZ) {
                            offset_b_next = batch_idx * k * n + tidx.n * n0 * k + shuffle_k_next * k0;
                        } else {
                            offset_b_next = batch_idx * RoundUp16(k) * RoundUp16(n) + tidx.n * n0 * BLOCK_SIZE_16 + shuffle_k_next * k0 * RoundUp16(n);
                        }
                    } else {
                        if constexpr (FormatB != DataFormat::NZ) {
                            offset_b_next = batch_idx * k * n + shuffle_k_next * k0 * n + tidx.n * n0;
                        } else {
                            offset_b_next = batch_idx * RoundUp16(k) * RoundUp16(n) + shuffle_k_next * k0 * BLOCK_SIZE_16 + tidx.n * n0 * RoundUp16(k);
                        }
                    }

                    uint64_t k_actual_next = (shuffle_k_next == (tdim.k - 1)) ? (k - shuffle_k_next * k0) : k0;
                    uint64_t k_round_next = RoundUp<K_ROUND_CONST>(k_actual_next);

                    LocalTensor l1_buf_a_next = (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_SIZE];
                    LocalTensor l1_buf_b_next = (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_SIZE];
                    event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                    WAIT_FLAG(MTE1, MTE2, event_id_next);
                    // *** load matrix A to L1
                    if ((m == 1) || (m_actual == 1 && !TA)) {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::ND>(l1_buf_a_next,       // dst
                                     gm_a[offset_a_next], // src
                                     m_actual,            // nTileActual
                                     m_round,             // nTileCeil
                                     m,                   // nVal
                                     k_actual_next,       // kTileActual
                                     k_round_next,        // kTileCeil
                                     k);                  // dVal
                    } else {
                        if (TA) {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a_next,       // dst
                                              gm_a[offset_a_next], // src
                                              k_actual_next,       // nTileActual
                                              k_round_next,        // nTileCeil
                                              k,                   // nVal
                                              m_actual,            // dTileActual
                                              m_round,             // dTileCeil
                                              m * batch_size);     // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a_next,       // dst
                                              gm_a[offset_a_next], // src
                                              m_actual,            // nTileActual
                                              m_round,             // nTileCeil
                                              m,                   // nVal
                                              k_actual_next,       // dTileActual
                                              k_round_next,        // dTileCeil
                                              k * batch_size);     // dVal
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next);

                    // *** load matrix B to L1
                    WaitFlag<HardEvent::MTE1_MTE2>(event_id_next + 2);
                    if constexpr (FormatB != DataFormat::NZ) {
                        if (TB) {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b_next,       // dst
                                            gm_b[offset_b_next], // src
                                            n_actual,            // nTileActual
                                            n_round,             // nTileCeil
                                            n,                   // nVal
                                            k_actual_next,       // dTileActual
                                            k_round_next,        // dTileCeil
                                            k);                  // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b_next,       // dst
                                            gm_b[offset_b_next], // src
                                            k_actual_next,       // nTileActual
                                            k_round_next,        // nTileCeil
                                            k,                   // nVal
                                            n_actual,            // dTileActual
                                            n_round,             // dTileCeil
                                            n);                  // dVal
                        }
                    } else {
                        if (TB) {
                            CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b_next,       // dst
                                            gm_b[offset_b_next], // src
                                            n_actual,            // nTileActual
                                            n_round,             // nTileCeil
                                            RoundUp16(n),                   // nVal
                                            k_actual_next,       // dTileActual
                                            k_round_next,        // dTileCeil
                                            RoundUp16(k));                  // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b_next,       // dst
                                            gm_b[offset_b_next], // src
                                            k_actual_next,       // nTileActual
                                            k_round_next,        // nTileCeil
                                            RoundUp16(k),                   // nVal
                                            n_actual,            // dTileActual
                                            n_round,             // dTileCeil
                                            RoundUp16(n));                  // dVal
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next + 2);
                }

                if (tidx.k == tdim.k - 1 && loop_idx + num_core < core_loop) {
                    uint64_t b_idx_next = (loop_idx + num_core) / tdim.n / tdim.m;
                    MatCoord tidx{0};
                    GetBlockIdx(loop_idx + num_core, tidx);
                    uint64_t shuffle_k_next = en_shuffle_k ? (core_idx % tdim.k) : 0;
                    uint64_t m_actual_next = (tidx.m == (tdim.m - 1)) ? (m - tidx.m * m0) : m0;
                    uint64_t n_actual_next = (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
                    uint64_t m_round_next = (m_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                    uint64_t n_round_next = (n_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                    uint64_t k_actual_next = (shuffle_k_next == (tdim.k - 1)) ? (k - shuffle_k_next * k0) : k0;
                    uint64_t k_round_next = RoundUp<K_ROUND_CONST>(k_actual_next);
                    if (TA) {
                        offset_a_next = shuffle_k_next * k0 * m * batch_size + b_idx_next * m + tidx.m * m0;
                    } else {
                        offset_a_next = tidx.m * m0 * batch_size * k + b_idx_next * k + shuffle_k_next * k0;
                    }

                    if (TB) {
                        if constexpr (FormatB != DataFormat::NZ) {
                            offset_b_next = b_idx_next * k * n + tidx.n * n0 * k + shuffle_k_next * k0;
                        } else {
                            offset_b_next = b_idx_next * RoundUp16(k) * RoundUp16(n) + tidx.n * n0 * BLOCK_SIZE_16 + shuffle_k_next * k0 * RoundUp16(n);
                        }
                    } else {
                        if constexpr (FormatB != DataFormat::NZ) {
                            offset_b_next = b_idx_next * k * n + shuffle_k_next * k0 * n + tidx.n * n0;
                        } else {
                            offset_b_next = b_idx_next * RoundUp16(k) * RoundUp16(n) + shuffle_k_next * k0 * BLOCK_SIZE_16 + tidx.n * n0 * RoundUp16(k);
                        }
                    }

                    LocalTensor l1_buf_a_next = (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_SIZE];
                    LocalTensor l1_buf_b_next = (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_SIZE];
                    event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                    WAIT_FLAG(MTE1, MTE2, event_id_next);
                    // *** load matrix A to L1
                    if (m == 1 || m_actual_next == 1 && !TA) {
                        CopyGmToCbuf<DataFormat::ND, DataFormat::ND>(l1_buf_a_next,       // dst
                                     gm_a[offset_a_next], // src
                                     m_actual_next,       // nTileActual
                                     m_round_next,        // nTileCeil
                                     m,                   // nVal
                                     k_actual_next,       // kTileActual
                                     k_round_next,        // kTileCeil
                                     k);                  // dVal
                    } else {
                        if (TA) {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a_next,       // dst
                                              gm_a[offset_a_next], // src
                                              k_actual_next,       // nTileActual
                                              k_round_next,        // nTileCeil
                                              k,                   // nVal
                                              m_actual_next,       // dTileActual
                                              m_round_next,        // dTileCeil
                                              m * batch_size);     // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_a_next,       // dst
                                              gm_a[offset_a_next], // src
                                              m_actual_next,       // nTileActual
                                              m_round_next,        // nTileCeil
                                              m,                   // nVal
                                              k_actual_next,       // dTileActual
                                              k_round_next,        // dTileCeil
                                              k * batch_size);     // dVal
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next);

                    // *** load matrix B to L1
                    WaitFlag<HardEvent::MTE1_MTE2>(event_id_next + 2);
                    if constexpr (FormatB != DataFormat::NZ) {
                        if (TB) {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b_next,       // dst
                                            gm_b[offset_b_next], // src
                                            n_actual_next,       // nTileActual
                                            n_round_next,        // nTileCeil
                                            n,                   // nVal
                                            k_actual_next,       // dTileActual
                                            k_round_next,        // dTileCeil
                                            k);                  // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::ND, DataFormat::NZ>(l1_buf_b_next,       // dst
                                            gm_b[offset_b_next], // src
                                            k_actual_next,       // nTileActual
                                            k_round_next,        // nTileCeil
                                            k,                   // nVal
                                            n_actual_next,       // dTileActual
                                            n_round_next,        // dTileCeil
                                            n);                  // dVal
                        }
                    } else {
                        if (TB) {
                            CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b_next,       // dst
                                            gm_b[offset_b_next], // src
                                            n_actual_next,       // nTileActual
                                            n_round_next,        // nTileCeil
                                            RoundUp16(n),                   // nVal
                                            k_actual_next,       // dTileActual
                                            k_round_next,        // dTileCeil
                                            RoundUp16(k));                  // dVal
                        } else {
                            CopyGmToCbuf<DataFormat::NZ, DataFormat::NZ>(l1_buf_b_next,       // dst
                                            gm_b[offset_b_next], // src
                                            k_actual_next,       // nTileActual
                                            k_round_next,        // nTileCeil
                                            RoundUp16(k),                   // nVal
                                            n_actual_next,       // dTileActual
                                            n_round_next,        // dTileCeil
                                            RoundUp16(n));                  // dVal
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next + 2);
                }

                MatCoord fidx{0};
                for (fidx.k = 0; fidx.k < fdim.k; ++fidx.k) {
                    uint32_t k0_round = (fidx.k < fdim.k - 1) ? k_part_len : k_round - fidx.k * k_part_len;
                    uint32_t k0_actual = (fidx.k < fdim.k - 1) ? k_part_len : k_actual - fidx.k * k_part_len;

                    auto mte1_mad_ping_flag = 1 - fidx.k % 2;
                    auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                    auto l0a_buf = l0a_base[(fidx.k % 2) * L0_PINGPONG_BUFFER_SIZE];
                    auto l0b_buf = l0b_base[(fidx.k % 2) * L0_PINGPONG_BUFFER_SIZE];

                    // *** load matrix A from L1 to L0A
                    if (fidx.k == 0) {
                        WAIT_FLAG(MTE2, MTE1, event_id);
                    }
                    WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                    if ((m == 1) || (m_actual == 1 && !TA)) {
                        l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0a_buf,                                         // dst
                            l1_buf_a[fidx.k * k_part_len],                   // src
                            0,                                               // mTileCeil
                            CeilDiv<CONST_512 / sizeof(InDtype)>(k0_round),  // kPartCeil
                            0,                                               // mSrcStride
                            1,                                               // kSrcStride
                            0,                                               // mDstStride
                            0);                                              // kDstStride
                    } else {
                        if (TA) {
                            LoadCbufToCa(l0a_buf,                                  // l0Tensor
                                         l1_buf_a[fidx.k * k_part_len * CONST_16], // l1Tensor
                                         m_round,                                  // mTileCeil
                                         k0_round,                                 // kPartCeil
                                         k_round / CONST_16,                       // mSrcStride
                                         1,                                        // kSrcStride
                                         k0_round / CONST_16,                      // mDstStride
                                         1);                                       // kDstStride
                        } else {
                            LoadCbufToCa(l0a_buf,                                 // l0Tensor
                                         l1_buf_a[fidx.k * k_part_len * m_round], // l1Tensor
                                         m_round,                                 // mTileCeil
                                         k0_round,                                // kPartCeil
                                         1,                                       // mSrcStride
                                         m_round / CONST_16,                      // kSrcStride
                                         CeilDiv<K_ROUND_CONST>(k0_round),        // mDstStride
                                         1);                                      // kDstStride
                        }
                    }
                    if (fidx.k == fdim.k - 1) {
                        SET_FLAG(MTE1, MTE2, event_id);
                    }

                    // *** load matrix B from L1 to L0B
                    if (fidx.k == 0) {
                        WAIT_FLAG(MTE2, MTE1, event_id + 2);
                    }
                    if (TB) {
                        LoadCbufToCb(l0b_buf,                                 // l0Tensor
                                     l1_buf_b[fidx.k * k_part_len * n_round], // l1Tensor
                                     n_round,                                 // nTileCeil
                                     k0_round,                                // kPartCeil
                                     1,                                       // nSrcStride
                                     n_round / CONST_16,                      // kSrcStride
                                     1,                                       // nDstStride
                                     k0_round / CONST_16);                    // kDstStride
                    } else {
                        if (!(std::is_same_v<InDtype, float>)) {
                            LoadCbufToCb(l0b_buf,                                  // l0Tensor
                                         l1_buf_b[fidx.k * k_part_len * CONST_16], // l1Tensor
                                         n_round,                                  // nTileCeil
                                         k0_round,                                 // kPartCeil
                                         k_round / CONST_16,                       // nSrcStride
                                         1,                                        // kSrcStride
                                         1,                                        // nDstStride
                                         n_round / CONST_16);                      // kDstStride
                        } else {
                            AscendC::Load3DSetFMatrixCal(1, RoundUp<K_ROUND_CONST>(k_round), padList);
                            uint64_t extConfig = ((uint64_t)RoundUp<K_ROUND_CONST>(k0_round) << 16) | (uint64_t)RoundUp16(n_round);
                            AscendC::LoadData(l0b_buf, l1_buf_b[fidx.k * k_part_len * CONST_8], AscendC::LoadData3DParamsV2Pro(
                                    RoundUp16(n_round), // channelSize
                                    false,              // enTranspose(Default)
                                    false,              // enSmallK(Default)
                                    false,              // filterSizeW(Default)
                                    false,              // filterSizeH(Default)
                                    false,              // fMatrixCtrl(Default)
                                    extConfig,          // extConfig
                                    0X10101010101));    // filterConfig(Default)
                        }
                    }
                    if (fidx.k == fdim.k - 1) {
                        SET_FLAG(MTE1, MTE2, event_id + 2);
                    }

                    SET_FLAG(MTE1, M, mte1_mad_event_id);
                    WAIT_FLAG(MTE1, M, mte1_mad_event_id);

                    bool init_c = (tidx.k == 0 && fidx.k == 0);
                    if (init_c) {
                        WAIT_FLAG(FIX, M, EVENT_ID0);
                    }

                    if (m != 1 && m_actual == 1 && TA) {
                        AscendC::Mmad(l0c_buf,                       // C
                                      l0a_buf,                       // A
                                      l0b_buf,                       // B
                                      AscendC::MmadParams(CONST_16,  // m
                                                          n_actual,  // n
                                                          k0_actual, // k
                                                          0,         // unitFlag
                                                          false,     // cmatrixSource
                                                          init_c));  // cmatrixInitVal
                    } else {
                        AscendC::Mmad(l0c_buf,                       // C
                                      l0a_buf,                       // A
                                      l0b_buf,                       // B
                                      AscendC::MmadParams(m_actual,  // m
                                                          n_actual,  // n
                                                          k0_actual, // k
                                                          0,         // unitFlag
                                                          false,     // cmatrixSource
                                                          init_c));  // cmatrixInitVal
                    }

                    PIPE_BARRIER(M);
                    SET_FLAG(M, MTE1, mte1_mad_event_id);
                }

                ping_flag = 1 - ping_flag;
            }

            SET_FLAG(M, FIX, EVENT_ID0);
            WAIT_FLAG(M, FIX, EVENT_ID0);

            // copy from L0C to gm
            CopyCcToGm(gm_c[offset_c],  // dst
                       l0c_buf,         // src
                       m_actual,        // mTileActual
                       n_actual,        // nTileActual
                       m_round,         // mTileCeil
                       n * batch_size); // nActual
            SET_FLAG(FIX, M, EVENT_ID0);
        }

        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(FIX, M, EVENT_ID0);
        PIPE_BARRIER(ALL);
    }

private:
    AscendC::GlobalTensor<InDtype> gm_a;
    AscendC::GlobalTensor<InDtype> gm_b;
    AscendC::GlobalTensor<OutDtype> gm_c;
    AscendC::LocalTensor<InDtype> l1_base_a;
    AscendC::LocalTensor<InDtype> l1_base_b;
    AscendC::LocalTensor<InDtype> l0a_base;
    AscendC::LocalTensor<InDtype> l0b_base;
    AscendC::LocalTensor<float> l0c_buf;

    uint32_t num_core{0};
    uint32_t batch_size{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    MatCoord tdim{0};
    MatCoord fdim{0};
    uint32_t core_loop{0};
    uint32_t swizzle_cnt{1};
    uint32_t core_idx{0};
    uint32_t en_shuffle_k{0};
    uint32_t ping_flag{0};
};

#endif
}
#endif
