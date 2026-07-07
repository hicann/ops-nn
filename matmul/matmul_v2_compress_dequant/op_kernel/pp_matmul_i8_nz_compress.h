/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PP_MATMUL_I8_NZ_COMPRESS_H
#define PP_MATMUL_I8_NZ_COMPRESS_H

#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/mem.h"
#include "kernels/utils/kernel/iterator.h"
#include "kernels/utils/kernel/mma.h"
#include "kernels/utils/kernel/utils.h"
#include "kernels/utils/kernel/simd.h"

constexpr uint32_t BLOCK_SIZE_32 = 32;
constexpr uint32_t CUBE_BLOCK_SIZE_INT8 = 512;
constexpr uint32_t BLOCK_SIZE_16 = 16;
constexpr uint32_t BLOCK_SIZE_K = 32;
constexpr uint32_t M_TILE_LIMIT = 64;
constexpr uint32_t K_TILE_LIMIT = 256;
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024;
constexpr uint32_t kSparseCtrlBitPos = 62;
constexpr uint32_t kByteSizeBits = 8;
constexpr uint32_t kHalfBlkSize = 4;

__aicore__ FORCE_INLINE uint32_t CeilDivKernel(const uint32_t dividend, const uint32_t divisor)
{
    if (divisor == 0) {
        return UINT32_MAX;
    }
    return (dividend + divisor - 1) / divisor;
}

__aicore__ FORCE_INLINE uint32_t RoundUpKernel(const uint32_t val, const uint32_t align)
{
    if (align == 0) {
        return 0;
    }
    return (val + align - 1) / align * align;
}

__aicore__ FORCE_INLINE uint32_t MinKernel(const uint32_t a, const uint32_t b)
{
    return a < b ? a : b;
}

#if defined(__DAV_C100__) || defined(__DAV_M200__)

template <uint32_t SWIZZL_DIR, bool TRANSPOSE_A, bool TRANSPOSE_B, bool SPLIT_K = false,
          typename IN_DTYPE = int8_t, typename DESCALE_TYPE = uint64_t, typename BIAS_TYPE = int32_t>
class PpMatmulI8NzCompress {
public:
    __aicore__ explicit PpMatmulI8NzCompress()
    {
        SetPadding<uint64_t>((uint64_t)0x0);
        SetAtomicnone();
        set_ctrl(sbitset1(get_ctrl(), kSparseCtrlBitPos));
        SetMasknorm();
    }

    __aicore__ FORCE_INLINE void Init(
        __gm__ uint8_t *__restrict__ A, __gm__ uint8_t *__restrict__ B,
        __gm__ uint8_t *__restrict__ bias, __gm__ uint8_t *__restrict__ scale,
        __gm__ uint8_t *__restrict__ compress_index, __gm__ uint8_t *__restrict__ C,
        int32_t b, int32_t m, int32_t k, int32_t n,
        int32_t m0, int32_t k0, int32_t n0,
        int32_t m_loop, int32_t k_loop, int32_t n_loop,
        int32_t core_loop, int32_t swizzl_cnt,
        int32_t copress_tiling_k, int32_t copress_tiling_n, int32_t compress_overlap_n)
    {
        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(A));
        gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(B));
        gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(C));
        gm_bias.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_TYPE *>(bias));
        gm_scale.SetGlobalBuffer(reinterpret_cast<__gm__ DESCALE_TYPE *>(scale));
        gm_compress_index.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(compress_index));
        b_ = b;
        m_ = m;
        k_ = k;
        n_ = n;
        m0_ = m0;
        k0_ = m0 > M_TILE_LIMIT ? MinKernel(k0, K_TILE_LIMIT) : k0;
        n0_ = n0;
        m_org_ = m;
        n_org_ = n;
        m_loop_ = m_loop;
        k_loop_ = CeilDivKernel(k, k0_);
        n_loop_ = n_loop;
        core_loop_ = core_loop;
        swizzl_cnt_ = swizzl_cnt;
        copress_tiling_k_ = copress_tiling_k;
        copress_tiling_n_ = copress_tiling_n;
        compress_overlap_n_ = compress_overlap_n;
        n_compress_num = CeilDivKernel(n_, copress_tiling_n_ * 16);
        k_compress_num = CeilDivKernel(k_, copress_tiling_k_ * 32);
        compress_size = copress_tiling_k_ * copress_tiling_n_ * CUBE_BLOCK_SIZE_INT8;

        l1_a_ping = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
        l1_a_pong = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(256 * 1024);
        l1_b_ping = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(512 * 1024);
        l1_b_pong = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(768 * 1024);
        l0_a_ping = buf.template GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(0);
        l0_a_pong = buf.template GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(32768);
        l0_b_ping = buf.template GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(0);
        l0_b_pong = buf.template GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(32768);
        ub_c = buf.template GetBuffer<BufferType::ASCEND_UB, half>(0 + 8 * 1024);
        ub_bias = buf.template GetBuffer<BufferType::ASCEND_UB, BIAS_TYPE>(131072 + 8 * 1024);
        ub_scale = buf.template GetBuffer<BufferType::ASCEND_UB, DESCALE_TYPE>(0);
    }

    __aicore__ FORCE_INLINE void GetIdx(uint32_t loop_idx, uint32_t &m_idx, uint32_t &n_idx)
    {
        uint32_t in_batch_idx = loop_idx % (m_loop_ * n_loop_);
        if constexpr (SWIZZL_DIR == 0) {
            int32_t tile_block_loop = (m_loop_ + swizzl_cnt_ - 1) / swizzl_cnt_;
            int32_t tile_block_idx = in_batch_idx / (swizzl_cnt_ * n_loop_);
            int32_t in_tile_block_idx = in_batch_idx % (swizzl_cnt_ * n_loop_);

            int32_t n_row = swizzl_cnt_;
            if (tile_block_idx == tile_block_loop - 1) {
                n_row = m_loop_ - swizzl_cnt_ * tile_block_idx;
            }
            m_idx = tile_block_idx * swizzl_cnt_ + in_tile_block_idx % n_row;
            n_idx = in_tile_block_idx / n_row;
            if (tile_block_idx % 2 != 0) {
                n_idx = n_loop_ - n_idx - 1;
            }
        } else if constexpr (SWIZZL_DIR == 1) {
            int32_t tile_block_loop = (n_loop_ + swizzl_cnt_ - 1) / swizzl_cnt_;
            int32_t tile_block_idx = in_batch_idx / (swizzl_cnt_ * m_loop_);
            int32_t in_tile_block_idx = in_batch_idx % (swizzl_cnt_ * m_loop_);

            int32_t n_col = swizzl_cnt_;
            if (tile_block_idx == tile_block_loop - 1) {
                n_col = n_loop_ - swizzl_cnt_ * tile_block_idx;
            }
            m_idx = in_tile_block_idx / n_col;
            n_idx = tile_block_idx * swizzl_cnt_ + in_tile_block_idx % n_col;
            if (tile_block_idx % 2 != 0) {
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
            uint32_t m_idx = 0;
            uint32_t n_idx = 0;
            GetIdx(loop_idx, m_idx, n_idx);
            uint32_t m_actual = (m_idx == (m_loop_ - 1)) ? (m_ - m_idx * m0_) : m0_;
            uint32_t n_actual = (n_idx == (n_loop_ - 1)) ? (n_ - n_idx * n0_) : n0_;
            uint32_t m_round = RoundUpKernel(m_actual, 16);
            uint32_t n_round = RoundUpKernel(n_actual, 16);
            uint32_t k_actual = (k_loop_ == 1) ? k_ : k0_;
            uint32_t k_round = RoundUpKernel(k_actual, BLOCK_SIZE_32);
            uint64_t src_offset = 0;
            uint64_t dst_offset = 0;
            uint64_t src_offset_index = 0;
            uint64_t src_offset_compress_L1_size = 0;
            uint32_t mn_max = m_round > n_round ? m_round : n_round;
            uint32_t k_part_len = L0AB_PINGPONG_BUFFER_LEN / mn_max / 32 * 32;
            uint32_t repeat_n = copress_tiling_n_;
            uint32_t m_org_up = RoundUpKernel(m_, 16);
            uint32_t n_org_up = RoundUpKernel(n_, 16);
            uint32_t k_org_up = RoundUpKernel(k_, 32);

            uint32_t index_k_all = CeilDivKernel(k_, copress_tiling_k_ * BLOCK_SIZE_32);
            src_offset_index = (b_idx * n_compress_num * k_compress_num +
                                n_idx * CeilDivKernel(n0_, copress_tiling_n_ * BLOCK_SIZE_16) * k_compress_num) *
                               kByteSizeBits;
            load_unzip_index_from_gm(((__gm__ IN_DTYPE *)(gm_compress_index.GetPhyAddr() + src_offset_index)),
                                     (uint64_t)index_k_all);
            for (uint32_t k_idx = 0; k_idx < k_loop_; ++k_idx) {
                uint32_t k_actual = k_idx == k_loop_ - 1 ? k_ - k_idx * k0_ : k0_;
                uint32_t k_round = RoundUpKernel(k_actual, BLOCK_SIZE_K);

                AscendC::LocalTensor<int8_t> l1_a = l1_ping_pong ? l1_a_ping : l1_a_pong;
                AscendC::LocalTensor<int8_t> l1_b = l1_ping_pong ? l1_b_ping : l1_b_pong;

                event_t l1_a_event = l1_ping_pong ? EVENT_ID0 : EVENT_ID1;
                event_t l1_b_event = l1_ping_pong ? EVENT_ID2 : EVENT_ID3;
                uint32_t index_num_n = CeilDivKernel(n_round, copress_tiling_n_ * BLOCK_SIZE_16);
                uint32_t index_num_k = CeilDivKernel(k_round, copress_tiling_k_ * BLOCK_SIZE_32);
                WAIT_FLAG(MTE1, MTE2, l1_b_event);
                for (uint32_t _n_idx = 0; _n_idx < index_num_n; _n_idx++) {
                    if (k_idx == k_loop_ - 1) {
                        uint32_t copress_k_tile =
                            (k_round - (index_num_k - 1) * (copress_tiling_k_ * BLOCK_SIZE_32)) / BLOCK_SIZE_32;
                        uint32_t compress_tile_size = copress_k_tile * copress_tiling_n_ * CUBE_BLOCK_SIZE_INT8;
                        src_offset_compress_L1_size = _n_idx * ((index_num_k - 1) * compress_size + compress_tile_size);
                    } else {
                        src_offset_compress_L1_size = _n_idx * index_num_k * compress_size;
                    }

                    for (uint32_t _k_idx = 0; _k_idx < index_num_k; _k_idx++) {
                        load_gm_to_cbuf_unzip((__cbuf__ half *)(((__cbuf__ IN_DTYPE *)l1_b.GetPhyAddr() +
                                                                 src_offset_compress_L1_size + _k_idx * compress_size)),
                                              (__gm__ half *)(((__gm__ IN_DTYPE *)gm_b.GetPhyAddr())));
                    }
                }
                SET_FLAG(MTE2, MTE1, l1_b_event);
                WAIT_FLAG(MTE1, MTE2, l1_a_event);

                src_offset = b_idx * m_org_up * k_org_up + k_idx * k0_ * m_org_up + m_idx * m0_ * BLOCK_SIZE_32;
                gm_to_l1<ArchType::ASCEND_V200, IN_DTYPE, DataFormat::NZ, DataFormat::NZ>(
                    l1_a, gm_a[src_offset], m_actual, m_round, m_org_up, k_actual, k_round, k_org_up);

                SET_FLAG(MTE2, MTE1, l1_a_event);
                AscendC::PipeBarrier<PIPE_MTE2>();

                uint32_t k_part_loop = CeilDivKernel(k_actual, k_part_len);
                for (uint32_t k_part_idx = 0; k_part_idx < k_part_loop; ++k_part_idx) {
                    uint32_t k0_round = k_part_idx < k_part_loop - 1
                                            ? k_part_len
                                            : k_round - k_part_idx * k_part_len;
                    uint32_t k0_actual = k_part_idx < k_part_loop - 1
                                             ? k_part_len
                                             : k_actual - k_part_idx * k_part_len;
                    uint32_t l0_ping_pong = 1 - k_part_idx % 2;
                    event_t l0_event = l0_ping_pong ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<IN_DTYPE> l0_a = l0_ping_pong ? l0_a_ping : l0_a_pong;
                    AscendC::LocalTensor<IN_DTYPE> l0_b = l0_ping_pong ? l0_b_ping : l0_b_pong;
                    if (k_part_idx == 0) {
                        WAIT_FLAG(MTE2, MTE1, l1_a_event);
                    }
                    WAIT_FLAG(M, MTE1, l0_event);
                    l1_to_l0_a<ArchType::ASCEND_V200, IN_DTYPE, false, DataFormat::ZN, DataFormat::ZZ>(
                        l0_a, l1_a[k_part_idx * k_part_len * m_round], m_round, k0_round, 1, m_round / BLOCK_SIZE_16,
                        k0_round / BLOCK_SIZE_32, 1);
                    if (k_part_idx == k_part_loop - 1) {
                        SET_FLAG(MTE1, MTE2, l1_a_event);
                    }
                    if (k_part_idx == 0) {
                        WAIT_FLAG(MTE2, MTE1, l1_b_event);
                    }

                    if (n_idx == (n_loop_ - 1) && (compress_overlap_n_ > 0)) {
                        for (uint32_t i = 0; i < k0_round / BLOCK_SIZE_32; i++) {
                            dst_offset = n_round * i * BLOCK_SIZE_32;
                            src_offset = k_part_idx * k_part_len * n0_ + i * n0_ * BLOCK_SIZE_32 + compress_overlap_n_ * CUBE_BLOCK_SIZE_INT8;
                            l1_to_l0_b<ArchType::ASCEND_V200, IN_DTYPE, true, DataFormat::ZN, DataFormat::NZ>(
                                l0_b[dst_offset],
                                l1_b[src_offset],
                                n_round, BLOCK_SIZE_32, 1, n0_ / 16, 1, n_round / BLOCK_SIZE_16);
                        }
                    } else {
                        src_offset = k_part_idx * k_part_len * n_round;
                        l1_to_l0_b<ArchType::ASCEND_V200, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0_b,
                            l1_b[src_offset],
                            0,
                            k0_round * n_round / CUBE_BLOCK_SIZE_INT8,
                            0,
                            1,
                            0,
                            0);
                    }
                    if (k_part_idx == k_part_loop - 1) {
                        SET_FLAG(MTE1, MTE2, l1_b_event);
                    }
                    SET_FLAG(MTE1, M, l0_event);
                    WAIT_FLAG(MTE1, M, l0_event);
                    if (k_idx == 0 && k_part_idx == 0) {
                        AscendC::PipeBarrier<PIPE_MTE2>();
                        gm_to_ub<ArchType::ASCEND_V200, BIAS_TYPE>(ub_bias, gm_bias[n_idx * n0_],
                                                                   0,
                                                                   1,
                                                                   n_round / kByteSizeBits,
                                                                   0,
                                                                   0);
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
                        WAIT_FLAG(V, M, EVENT_ID0);
                    }
                    uint32_t m_mad_actual = (m_actual == 1) ? 2 : m_actual;
                    AscendC::PipeBarrier<PIPE_M>();
                    mmad<ArchType::ASCEND_V200, IN_DTYPE, IN_DTYPE, int32_t, false>(
                        l0c, l0_a, l0_b,
                        m_mad_actual,
                        n_actual,
                        k0_actual,
                        0);
                    SET_FLAG(M, MTE1, l0_event);
                }
                l1_ping_pong = 1 - l1_ping_pong;
            }
            curr_core_loop = curr_core_loop + 1;
            AscendC::PipeBarrier<PIPE_MTE2>();
            gm_to_ub<ArchType::ASCEND_V200, DESCALE_TYPE>(ub_scale, gm_scale[n_idx * n0_],
                                                          0,
                                                          1,
                                                          n_round / kHalfBlkSize,
                                                          0,
                                                          0);
            SET_FLAG(M, V, EVENT_ID0);
            WAIT_FLAG(M, V, EVENT_ID0);
            WAIT_FLAG(MTE3, V, EVENT_ID0);
            SET_FLAG(MTE2, V, EVENT_ID1);
            WAIT_FLAG(MTE2, V, EVENT_ID1);
            l0c_to_ub<ArchType::ASCEND_V200, int32_t, half>(ub_c, l0c,
                                                            (uint16_t)(n_round / BLOCK_SIZE_16),
                                                            (uint16_t)(m_round / BLOCK_SIZE_16),
                                                            (uint16_t)0,
                                                            (uint16_t)0);
            AscendC::PipeBarrier<PIPE_V>();
            if (m_actual == 1) {
                SetVectorMask<int8_t>((uint64_t)0x0, (uint64_t)0xffff);
                half zero = 0;
                for (uint32_t i = 0; i < n_round / BLOCK_SIZE_16; i++) {
                    uint64_t curr_offset_c = i * m_round * BLOCK_SIZE_16 + m_actual * BLOCK_SIZE_16;
                    muls_v<ArchType::ASCEND_V200, half>(ub_c[curr_offset_c], ub_c[curr_offset_c],
                                                        zero,
                                                        1,
                                                        1,
                                                        1,
                                                        2,
                                                        2);
                }
            }
            SET_FLAG(V, M, EVENT_ID0);
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            dst_offset = b_idx * m_org_up * n_org_up + n_idx * n0_ * m_org_up + m_idx * m0_ * BLOCK_SIZE_16;
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
    AsdopsBuffer<ArchType::ASCEND_V200> buf;

private:
    AscendC::GlobalTensor<IN_DTYPE> gm_a;
    AscendC::GlobalTensor<IN_DTYPE> gm_b;
    AscendC::GlobalTensor<half> gm_c;
    AscendC::GlobalTensor<BIAS_TYPE> gm_bias;
    AscendC::GlobalTensor<DESCALE_TYPE> gm_scale;
    AscendC::GlobalTensor<IN_DTYPE> gm_compress_index;
    AscendC::LocalTensor<IN_DTYPE> l1_a_ping;
    AscendC::LocalTensor<IN_DTYPE> l1_a_pong;
    AscendC::LocalTensor<IN_DTYPE> l1_b_ping;
    AscendC::LocalTensor<IN_DTYPE> l1_b_pong;
    AscendC::LocalTensor<IN_DTYPE> l0_a_ping;
    AscendC::LocalTensor<IN_DTYPE> l0_a_pong;
    AscendC::LocalTensor<IN_DTYPE> l0_b_ping;
    AscendC::LocalTensor<IN_DTYPE> l0_b_pong;
    AscendC::LocalTensor<int32_t> l0c = buf.template GetBuffer<BufferType::ASCEND_L0C, int32_t>(0);
    AscendC::LocalTensor<half> ub_c;
    AscendC::LocalTensor<DESCALE_TYPE> ub_scale;
    AscendC::LocalTensor<BIAS_TYPE> ub_bias;

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
    uint32_t swizzl_cnt_{0};
    uint32_t copress_tiling_k_{0};
    uint32_t copress_tiling_n_{0};
    uint32_t compress_size{0};
    uint32_t compress_overlap_n_{0};
    uint32_t n_compress_num{0};
    uint32_t k_compress_num{0};
};

#endif // __DAV_C100__ || __DAV_M200__

#endif // PP_MATMUL_I8_NZ_COMPRESS_H
