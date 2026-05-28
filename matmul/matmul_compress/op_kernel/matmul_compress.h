/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_compress.h
 * \brief
 */
#ifndef MATMUL_COMPRESS_H
#define MATMUL_COMPRESS_H

#include "matmul_compress_tiling_data.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../transpose_batch_mat_mul/utils/common.h"
#include "../transpose_batch_mat_mul/utils/common_func.h"
#include "../transpose_batch_mat_mul/utils/hardware.h"
#include "../transpose_batch_mat_mul/utils/iterator.h"
#include "../transpose_batch_mat_mul/utils/mma.h"
#include "../transpose_batch_mat_mul/utils/mem.h"
#include "../transpose_batch_mat_mul/utils/simd.h"
#include "../transpose_batch_mat_mul/utils/utils.h"

using namespace PpMatMulNS;
namespace MatmulCompressSpace {

constexpr uint64_t CONST_4 = 4;
constexpr uint64_t CONST_8 = 8;
constexpr uint32_t CONST_16 = 16;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t CONST_62 = 62;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t CONST_512 = 512;
constexpr uint32_t CONST_768 = 768;
constexpr uint32_t CONST_1024 = 1024;
constexpr uint32_t L1_DOUBLE_BUFFER_SIZE = 131072;
constexpr uint32_t L0AB_DOUBLE_BUFFER_LEN_INT8 = 32768; 
constexpr uint32_t BLOCK_SIZE_16 = 16;
constexpr uint32_t BLOCK_SIZE_32 = 32;
constexpr uint32_t CUBE_BLOCK_SIZE_INT8 = 512;
constexpr uint32_t BLOCK_SIZE_K = 32;
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024;

#if defined(__DAV_C100__) || defined(__DAV_M200__)

template <uint32_t SWIZZLE_DIR, bool TRANSPOSE_A, bool TRANSPOSE_B, bool SPLIT_K = false, typename IN_DTYPE = int8_t,
          typename DESCALE_TYPE = uint64_t, typename BIAS_TYPE = int32_t, typename MMAD_TYPE = int32_t,
          bool ENABLE_BIAS = true, bool ENABLE_SCALE = false>
class MatmulCompress {
public:
    __aicore__ explicit MatmulCompress()
    {
        SetPadding<uint64_t>((uint64_t)0x0);
        SetAtomicnone();
        set_ctrl(sbitset1(get_ctrl(), CONST_62));
        SetMasknorm();
    };

    __aicore__ FORCE_INLINE void Init(GM_ADDR A, GM_ADDR B,
                        GM_ADDR bias, GM_ADDR compress_index, GM_ADDR C,
                        const MatmulCompressTilingDataArch20* tilingData)
    {
        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(A));
        gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(B));
        gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(C));
        gm_compress_index.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(compress_index));

        b_ = tilingData->batch;
        m_ = tilingData->m;
        k_ = tilingData->k;
        n_ = tilingData->n;
        m0_ = tilingData->m0;
        k0_ = tilingData->k0;
        n0_ = tilingData->n0;
        m_loop_ = tilingData->mLoop;
        k_loop_ = tilingData->kLoop;
        n_loop_ = tilingData->nLoop;
        core_loop_ = tilingData->coreLoop;
        swizzle_cnt_ = tilingData->swizzleCount;
        copress_tiling_k_ = tilingData->tilingK;
        copress_tiling_n_ = tilingData->tilingN;
        compress_overlap_n_ = tilingData->compressOverlapN;

        align_k = CONST_32 / sizeof(IN_DTYPE);
        block_size_k = BLOCK_SIZE_32 / sizeof(IN_DTYPE);
        l0ab_pingpong_buffer_len = L0AB_PINGPONG_BUFFER_LEN / sizeof(IN_DTYPE);

        uint32_t scale_space = 0;
        if constexpr (ENABLE_SCALE) {
           scale_space = CONST_8 * CONST_1024;
        }
        
        n_compress_num = CeilDiv(n_, copress_tiling_n_ * CONST_16);
        k_compress_num = CeilDiv<uint32_t>(k_, copress_tiling_k_ * align_k);
        compress_size = copress_tiling_k_ * copress_tiling_n_ * CUBE_BLOCK_SIZE_INT8;
        l1_a_ping = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
        l1_a_pong = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(CONST_256 * CONST_1024);
        l1_b_ping = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(CONST_512 * CONST_1024);
        l1_b_pong = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(CONST_768 * CONST_1024);
        l0_a_ping = buf.template GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(0);
        l0_a_pong = buf.template GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(L0AB_DOUBLE_BUFFER_LEN_INT8);
        l0_b_ping = buf.template GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(0);
        l0_b_pong = buf.template GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(L0AB_DOUBLE_BUFFER_LEN_INT8);
        ub_c = buf.template GetBuffer<BufferType::ASCEND_UB, half>(0 + scale_space);
        if constexpr (ENABLE_BIAS) {
            gm_bias.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_TYPE *>(bias));
            ub_bias = buf.template GetBuffer<BufferType::ASCEND_UB, BIAS_TYPE>(L1_DOUBLE_BUFFER_SIZE + scale_space);
        }
    }

    __aicore__ FORCE_INLINE void GetIdx(uint32_t loop_idx, uint32_t &m_idx, uint32_t &n_idx)
    {
        uint32_t in_batch_idx = loop_idx % (m_loop_ * n_loop_);
        if constexpr (SWIZZLE_DIR == 0) { // Zn
            int32_t tile_block_loop = (m_loop_ + swizzle_cnt_ - 1) / swizzle_cnt_;
            int32_t tile_block_idx = in_batch_idx / (swizzle_cnt_ * n_loop_);
            int32_t in_tile_block_idx = in_batch_idx % (swizzle_cnt_ * n_loop_);

            int32_t n_row = swizzle_cnt_;
            if (tile_block_idx == tile_block_loop - 1) {
                n_row = m_loop_ - swizzle_cnt_ * tile_block_idx;
            }
            m_idx = tile_block_idx * swizzle_cnt_ + in_tile_block_idx % n_row;
            n_idx = in_tile_block_idx / n_row;
            if (tile_block_idx % CONST_2 != 0) {
                n_idx = n_loop_ - n_idx - 1;
            }
        } else if constexpr (SWIZZLE_DIR == 1) { // Nz
            int32_t tile_block_loop = (n_loop_ + swizzle_cnt_ - 1) / swizzle_cnt_;
            int32_t tile_block_idx = in_batch_idx / (swizzle_cnt_ * m_loop_);
            int32_t in_tile_block_idx = in_batch_idx % (swizzle_cnt_ * m_loop_);

            int32_t n_col = swizzle_cnt_;
            if (tile_block_idx == tile_block_loop - 1) {
                n_col = n_loop_ - swizzle_cnt_ * tile_block_idx;
            }
            m_idx = in_tile_block_idx / n_col;
            n_idx = tile_block_idx * swizzle_cnt_ + in_tile_block_idx % n_col;
            if (tile_block_idx % CONST_2 != 0) {
                m_idx = m_loop_ - m_idx - 1;
            }
        }
    }

    __aicore__ FORCE_INLINE void Process()
    {
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(V, M, EVENT_ID0);
        SET_FLAG(MTE3, V, EVENT_ID0);
        uint32_t curr_core_loop = 0;
        uint32_t l1_ping_pong = 1;
        for (uint32_t loop_idx = 0; loop_idx < core_loop_; ++loop_idx) {
            if (loop_idx % block_num != block_idx) {
                continue;
            }
            uint64_t b_idx = loop_idx / (m_loop_ * n_loop_);
            uint32_t m_idx = 0, n_idx = 0;
            GetIdx(loop_idx, m_idx, n_idx);
            uint32_t m_actual = (m_idx == (m_loop_ - 1)) ? (m_ - m_idx * m0_) : m0_;
            uint32_t n_actual = (n_idx == (n_loop_ - 1)) ? (n_ - n_idx * n0_) : n0_;
            uint32_t m_round = RoundUp<BLOCK_SIZE_16>(m_actual);
            uint32_t n_round = RoundUp<BLOCK_SIZE_16>(n_actual);
            uint32_t k_actual = (k_loop_ == 1) ? k_ : k0_;
            uint32_t k_round = RoundUp(k_actual, block_size_k);
            uint64_t src_offset = 0, dst_offset = 0;
            uint64_t src_offset_index = 0, src_offset_compress_L1_size = 0;
            uint32_t mn_max = m_round > n_round ? m_round : n_round;
            uint32_t k_part_len = l0ab_pingpong_buffer_len / mn_max / align_k * align_k;
            uint32_t repeat_n = copress_tiling_n_;
            uint32_t m_org_up = RoundUp<BLOCK_SIZE_16>(m_);
            uint32_t n_org_up = RoundUp<BLOCK_SIZE_16>(n_);
            uint32_t k_org_up = RoundUp(k_, block_size_k);

            uint32_t index_k_all = CeilDiv(k_, copress_tiling_k_ * block_size_k);
            src_offset_index = (b_idx * n_compress_num * k_compress_num +
                                n_idx * CeilDiv(n0_, copress_tiling_n_ * BLOCK_SIZE_16) * k_compress_num) *
                               CONST_8;
            load_unzip_index_from_gm(((__gm__ int8_t *)(gm_compress_index.GetPhyAddr() + src_offset_index)),
                                     (uint64_t)index_k_all);
            for (uint32_t k_idx = 0; k_idx < k_loop_; ++k_idx) {
                uint32_t k_actual = k_idx == k_loop_ - 1 ? k_ - k_idx * k0_ : k0_;
                uint32_t k_round = RoundUp(k_actual, align_k);

                AscendC::LocalTensor<IN_DTYPE> l1_a = l1_ping_pong ? l1_a_ping : l1_a_pong;
                AscendC::LocalTensor<IN_DTYPE> l1_b = l1_ping_pong ? l1_b_ping : l1_b_pong;

                event_t l1_a_event = l1_ping_pong ? EVENT_ID0 : EVENT_ID1;
                event_t l1_b_event = l1_ping_pong ? EVENT_ID2 : EVENT_ID3;
                uint32_t index_num_n = CeilDiv(n_round, copress_tiling_n_ * BLOCK_SIZE_16);
                uint32_t index_num_k = CeilDiv(k_round, copress_tiling_k_ * block_size_k);
                WAIT_FLAG(MTE1, MTE2, l1_b_event);
                for (uint32_t _n_idx = 0; _n_idx < index_num_n; _n_idx++) {
                    if (k_idx == k_loop_ - 1) {
                        uint32_t copress_k_tile =
                            (k_round - (index_num_k - 1) * (copress_tiling_k_ * block_size_k)) / block_size_k;
                        uint32_t compress_tile_size = copress_k_tile * copress_tiling_n_ * CUBE_BLOCK_SIZE_INT8;
                        src_offset_compress_L1_size = _n_idx * ((index_num_k - 1) * compress_size + compress_tile_size);
                    } else {
                        src_offset_compress_L1_size = _n_idx * index_num_k * compress_size;
                    }

                    for (uint32_t _k_idx = 0; _k_idx < index_num_k; _k_idx++) {
                        load_gm_to_cbuf_unzip((__cbuf__ half *)(((__cbuf__ int8_t *)l1_b.GetPhyAddr() +
                                                                 src_offset_compress_L1_size + _k_idx * compress_size)),
                                              (__gm__ half *)(((__gm__ int8_t *)gm_b.GetPhyAddr())));
                    }
                }
                SET_FLAG(MTE2, MTE1, l1_b_event);
                WAIT_FLAG(MTE1, MTE2, l1_a_event);

                src_offset = b_idx * m_org_up * k_org_up + k_idx * k0_ * m_org_up + m_idx * m0_ * block_size_k;
                gm_to_l1<ArchType::ASCEND_V200, IN_DTYPE, DataFormat::NZ, DataFormat::NZ>(
                    l1_a, gm_a[src_offset], m_actual, m_round, m_org_up, k_actual, k_round, k_org_up);

                SET_FLAG(MTE2, MTE1, l1_a_event);
                AscendC::PipeBarrier<PIPE_MTE2>();

                uint32_t k_part_loop = CeilDiv(k_actual, k_part_len);
                for (uint32_t k_part_idx = 0; k_part_idx < k_part_loop; ++k_part_idx) {
                    uint32_t k0_round = k_part_idx < k_part_loop - 1 ? k_part_len : k_round - k_part_idx * k_part_len;
                    uint32_t k0_actual = k_part_idx < k_part_loop - 1 ? k_part_len : k_actual - k_part_idx * k_part_len;
                    uint32_t l0_ping_pong = 1 - k_part_idx % 2;
                    event_t l0_event = l0_ping_pong ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<IN_DTYPE> l0_a = l0_ping_pong ? l0_a_ping : l0_a_pong;
                    AscendC::LocalTensor<IN_DTYPE> l0_b = l0_ping_pong ? l0_b_ping : l0_b_pong;
                    if (k_part_idx == 0) {
                        WAIT_FLAG(MTE2, MTE1, l1_a_event);
                    }
                    WAIT_FLAG(M, MTE1, l0_event);
                    l1_to_l0_a<ArchType::ASCEND_V200, IN_DTYPE, false, DataFormat::ZN, DataFormat::ZZ>(
                        l0_a, l1_a[k_part_idx * k_part_len * m_round], m_round, k0_round, 1, m_round / CONST_16,
                        k0_round / align_k, 1);
                    if (k_part_idx == k_part_loop - 1) {
                        SET_FLAG(MTE1, MTE2, l1_a_event);
                    }
                    if (k_part_idx == 0) {
                        WAIT_FLAG(MTE2, MTE1, l1_b_event);
                    }

                    if (n_idx == (n_loop_ - 1) && (compress_overlap_n_ > 0)) {
                        //对于L1的解压缩数据为N方向连续，因此偏移的基块为compress_overlap_n_ * 512
                        for (uint32_t i = 0; i < k0_round / align_k; i++) {
                            dst_offset = n_round * i * align_k;
                            src_offset = k_part_idx * k_part_len * n0_ + i * n0_ * align_k  + compress_overlap_n_ * CONST_512;
                            l1_to_l0_b<ArchType::ASCEND_V200, IN_DTYPE, true, DataFormat::ZN, DataFormat::NZ>(
                                l0_b[dst_offset], // dst
                                l1_b[src_offset], // src
                                n_round, block_size_k, 1, n0_ / CONST_16, 1, n_round / CONST_16);
                        }
                    } else {
                        src_offset = k_part_idx * k_part_len * n_round;
                        l1_to_l0_b<ArchType::ASCEND_V200, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0_b,             // dst
                            l1_b[src_offset], // src
                            0,
                            k0_round * n_round * sizeof(IN_DTYPE) / CUBE_BLOCK_SIZE_INT8, // repeat
                            0,
                            1, // srcStride
                            0,
                            0 // dstStride
                        );
                    }
                    if (k_part_idx == k_part_loop - 1) {
                        SET_FLAG(MTE1, MTE2, l1_b_event);
                    }
                    SET_FLAG(MTE1, M, l0_event);
                    WAIT_FLAG(MTE1, M, l0_event);
                    bool cmatrix_init_val = (k_idx == 0 && k_part_idx == 0);
                    if (k_idx == 0 && k_part_idx == 0) {
                        if constexpr (ENABLE_BIAS) {
                            cmatrix_init_val = false;
                            AscendC::PipeBarrier<PIPE_MTE2>();
                            gm_to_ub<ArchType::ASCEND_V200, BIAS_TYPE>(ub_bias, gm_bias[n_idx * n0_],
                                                                    0,           // sid
                                                                    1,           // nBurst
                                                                    n_round / CONST_8, // lenBurst(unit: 32B)
                                                                    0,           // srcStride
                                                                    0            // dstStride
                            );
                            SET_FLAG(MTE2, V, EVENT_ID0);
                            WAIT_FLAG(MTE2, V, EVENT_ID0);
                            for (uint32_t i = 0; i < n_round / BLOCK_SIZE_16; i++) {
                                for (uint32_t j = 0; j < m_round / BLOCK_SIZE_16; j++) {
                                    AscendC::BroadCastVecToMM(
                                        l0c[i * m_round * BLOCK_SIZE_16 + j * BLOCK_SIZE_16 * BLOCK_SIZE_16],
                                        ub_bias[i * BLOCK_SIZE_16], 1, 1, 0, 0);
                                }
                            }
                            SET_FLAG(V, M, EVENT_ID1);
                            WAIT_FLAG(V, M, EVENT_ID1);
                        }
                        WAIT_FLAG(V, M, EVENT_ID0);
                    }
                    uint32_t m_mad_actual =
                        (m_actual == 1)
                            ? 2
                            : m_actual; // m_actual == 1 时，mad 指令会走 gemv 向量 × 矩阵模式，此时设为 2，避免走 gemv
                    AscendC::PipeBarrier<PIPE_M>();
                    mmad<ArchType::ASCEND_V200, IN_DTYPE, IN_DTYPE, MMAD_TYPE, false>(l0c, l0_a,
                                                                                    l0_b,
                                                                                    m_mad_actual, // m
                                                                                    n_actual,     // n
                                                                                    k0_actual,    // k
                                                                                    cmatrix_init_val  // cmatrixInitVal
                    );
                    SET_FLAG(M, MTE1, l0_event);
                }
                l1_ping_pong = 1 - l1_ping_pong;
            }
            curr_core_loop = curr_core_loop + 1;
            AscendC::PipeBarrier<PIPE_MTE2>();
            if constexpr (ENABLE_SCALE) {
                gm_to_ub<ArchType::ASCEND_V200, DESCALE_TYPE>(ub_scale, gm_scale[n_idx * n0_],
                                                              0,           // sid
                                                              1,           // nBurst
                                                              n_round / CONST_4, // lenBurst(unit: 32B)
                                                              0,           // srcStride
                                                              0            // dstStride
                );
            }
            
            SET_FLAG(M, V, EVENT_ID0);
            WAIT_FLAG(M, V, EVENT_ID0);
            WAIT_FLAG(MTE3, V, EVENT_ID0);
            SET_FLAG(MTE2, V, EVENT_ID1);
            WAIT_FLAG(MTE2, V, EVENT_ID1);
            l0c_to_ub<ArchType::ASCEND_V200, MMAD_TYPE, half>(ub_c, l0c,
                                                            (uint16_t)(n_round / BLOCK_SIZE_16), // nBurst
                                                            (uint16_t)(m_round / BLOCK_SIZE_16), // lenBurst
                                                            (uint16_t)0,                         // srcStride
                                                            (uint16_t)0                          // dstStride
            );
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (ENABLE_BIAS) {
                if (m_actual == 1) { // 对随路加 bias 产生的脏数据进行清理
                    SetVectorMask<int8_t>((uint64_t)0x0, (uint64_t)0xffff);
                    half zero = 0;
                    for (uint32_t i = 0; i < n_round / CONST_16; i++) {
                        uint64_t curr_offset_c = i * m_round * BLOCK_SIZE_16 + m_actual * BLOCK_SIZE_16;
                        muls_v<ArchType::ASCEND_V200, half>(ub_c[curr_offset_c], ub_c[curr_offset_c],
                                                            zero, // src1
                                                            1,    // repeat
                                                            1,    // dstBlockStride
                                                            1,    // srcBlockStride
                                                            CONST_2,    // dstRepeatStride
                                                            CONST_2);   // srcRepeatStride
                    }
                }
            }
            SET_FLAG(V, M, EVENT_ID0);
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            dst_offset = b_idx * m_org_up * n_org_up + n_idx * n0_ * m_org_up + m_idx * m0_ * CONST_16;
            ub_to_gm<ArchType::ASCEND_V200, half, DataFormat::NZ, DataFormat::NZ>(
                gm_c[dst_offset], ub_c, m_round, m_round, m_org_up, n_round, n_round, n_org_up);
            SET_FLAG(MTE3, V, EVENT_ID0);
        }
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(V, M, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

public:
    OnChipBuffer<ArchType::ASCEND_V200> buf;

protected:
    AscendC::GlobalTensor<IN_DTYPE> gm_a;
    AscendC::GlobalTensor<int8_t> gm_b;
    AscendC::GlobalTensor<half> gm_c;
    AscendC::GlobalTensor<BIAS_TYPE> gm_bias;
    AscendC::GlobalTensor<DESCALE_TYPE> gm_scale;
    AscendC::GlobalTensor<int8_t> gm_compress_index;
    AscendC::LocalTensor<IN_DTYPE> l1_a_ping = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
    AscendC::LocalTensor<IN_DTYPE> l1_a_pong = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
    AscendC::LocalTensor<IN_DTYPE> l1_b_ping = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
    AscendC::LocalTensor<IN_DTYPE> l1_b_pong = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
    AscendC::LocalTensor<IN_DTYPE> l0_a_ping = buf.template GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(0);
    AscendC::LocalTensor<IN_DTYPE> l0_a_pong = buf.template GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(0);
    AscendC::LocalTensor<IN_DTYPE> l0_b_ping = buf.template GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(0);
    AscendC::LocalTensor<IN_DTYPE> l0_b_pong = buf.template GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(0);
    AscendC::LocalTensor<MMAD_TYPE> l0c = buf.template GetBuffer<BufferType::ASCEND_L0C, MMAD_TYPE>(0);
    AscendC::LocalTensor<half> ub_c = buf.template GetBuffer<BufferType::ASCEND_UB, half>(0);
    AscendC::LocalTensor<DESCALE_TYPE> ub_scale = buf.template GetBuffer<BufferType::ASCEND_UB, DESCALE_TYPE>(0);
    AscendC::LocalTensor<BIAS_TYPE> ub_bias = buf.template GetBuffer<BufferType::ASCEND_UB, BIAS_TYPE>(0);

    uint32_t b_{0};
    uint32_t m_{0};
    uint32_t k_{0};
    uint32_t n_{0};
    uint32_t m0_{0};
    uint32_t k0_{0};
    uint32_t n0_{0};
    uint32_t m_loop_{0};
    uint32_t k_loop_{0};
    uint32_t n_loop_{0};
    uint32_t m_org_{0};
    uint32_t n_org_{0};
    uint32_t core_loop_{0};
    uint32_t swizzle_cnt_{0};
    uint32_t copress_tiling_k_{0};
    uint32_t copress_tiling_n_{0};
    uint32_t compress_size{0};
    uint32_t compress_overlap_n_{0};
    uint32_t n_compress_num{0};
    uint32_t k_compress_num{0};

    uint32_t align_k{0};
    uint32_t block_size_k{0};
    uint32_t l0ab_pingpong_buffer_len{0};
};

#endif
}
#endif