/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file bn_training_reduce_grad_merged.h
 * \brief BNTrainingReduceGrad arch35 kernel 的 R==1/R==2 合并路径实现
 *        （BNTrainingReduceGradKernel 的类外定义；仅由 bn_training_reduce_grad.h 末尾、
 *         命名空间 BNTrainingReduceGradOps 闭合前包含，不单独使用/包含）
 */

// R==1 合并 tile：单 DMA 搬入 extent 个连续 plane（extent==元素数，channel 与元素一一
// 对应），计算后单 DMA 写回；grads/x/y 三路双缓冲流水与逐 plane 路径同构
template <typename T>
__aicore__ inline void BNTrainingReduceGradKernel<T>::ProcessMergedR1(int64_t p0, int64_t extent)
{
    // innerSize==1 时 rStart_==0，GM 基址即 plane 号
    DataCopyExtParams cpIn{1, static_cast<uint32_t>(extent * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padIn{false, 0, 0, 0};

    LocalTensor<T> gradsUb = gradsQue_.AllocTensor<T>();
    DataCopyPad(gradsUb, gradsGm_[p0], cpIn, padIn);
    gradsQue_.EnQue(gradsUb);
    gradsUb = gradsQue_.DeQue<T>();

    LocalTensor<T> xUb = xQue_.AllocTensor<T>();
    DataCopyPad(xUb, xGm_[p0], cpIn, padIn);
    xQue_.EnQue(xUb);
    xUb = xQue_.DeQue<T>();

    LocalTensor<T> yUb = yQue_.AllocTensor<T>();
    ComputeTileMergedR1((__ubuf__ T*)gradsUb.GetPhyAddr(), (__ubuf__ T*)xUb.GetPhyAddr(), (__ubuf__ T*)yUb.GetPhyAddr(),
                        extent);

    yQue_.EnQue(yUb);
    yUb = yQue_.DeQue<T>();
    DataCopyPad(yGm_[p0], yUb, cpIn);
    yQue_.FreeTensor(yUb);
    xQue_.FreeTensor(xUb);
    gradsQue_.FreeTensor(gradsUb);
}

// R==1 合并 tile 主计算：系数即 per-element 向量（channel c 的元素就在位置 c），
// 每个 VL 直接 DIST_NORM 加载三系数对应切片（无需广播），满块 for + 尾块 0/1 次 for；
// 运算序与逐 plane 路径完全一致（Mul→Add→Add→Mul，同一预计算系数），位级一致
template <typename T>
__aicore__ inline void BNTrainingReduceGradKernel<T>::ComputeTileMergedR1(__ubuf__ T* gradsAddr, __ubuf__ T* xAddr,
                                                                          __ubuf__ T* yAddr, int64_t extent)
{
    __ubuf__ float* multiplierAddr = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr();
    __ubuf__ float* addendAddr = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr();
    __ubuf__ float* mulScaleAddr = (__ubuf__ float*)mulScaleBuf_.Get<float>().GetPhyAddr();
    uint16_t fullLoops = static_cast<uint16_t>(extent / static_cast<int64_t>(VL_FP32));
    uint16_t totalLoops = static_cast<uint16_t>((extent + static_cast<int64_t>(VL_FP32) - 1) /
                                                static_cast<int64_t>(VL_FP32));
    uint32_t tailCount = static_cast<uint32_t>(extent) - fullLoops * VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> multReg, addReg, mulScaleReg, gradsReg, xReg, yReg;
        MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t i = 0; i < fullLoops; i++) {
            uint32_t offset = i * VL_FP32;
            LoadToFp32(gradsAddr, gradsReg, fullMask, offset);
            LoadToFp32(xAddr, xReg, fullMask, offset);
            DataCopy<float, LoadDist::DIST_NORM>(multReg, multiplierAddr + offset);
            DataCopy<float, LoadDist::DIST_NORM>(addReg, addendAddr + offset);
            DataCopy<float, LoadDist::DIST_NORM>(mulScaleReg, mulScaleAddr + offset);
            Mul(yReg, xReg, multReg, fullMask);
            Add(yReg, gradsReg, yReg, fullMask);
            Add(yReg, yReg, addReg, fullMask);
            Mul(yReg, yReg, mulScaleReg, fullMask);
            StoreYFromFp32(yAddr, yReg, fullMask, offset);
        }
        for (uint16_t i = fullLoops; i < totalLoops; i++) { // 尾块 0 或 1 次，无 if
            uint32_t tail = tailCount;
            MaskReg tailMask = UpdateMask<float>(tail);
            uint32_t offset = i * VL_FP32;
            LoadToFp32(gradsAddr, gradsReg, tailMask, offset);
            LoadToFp32(xAddr, xReg, tailMask, offset);
            // 系数无掩码整行加载（缓冲 256 项恒不越界），无效 lane 为陈旧值，
            // 经 tailMask 的 Mul/Add(ZEROING)/store 全部屏蔽，不进输出
            DataCopy<float, LoadDist::DIST_NORM>(multReg, multiplierAddr + offset);
            DataCopy<float, LoadDist::DIST_NORM>(addReg, addendAddr + offset);
            DataCopy<float, LoadDist::DIST_NORM>(mulScaleReg, mulScaleAddr + offset);
            Mul(yReg, xReg, multReg, tailMask);
            Add(yReg, gradsReg, yReg, tailMask);
            Add(yReg, yReg, addReg, tailMask);
            Mul(yReg, yReg, mulScaleReg, tailMask);
            StoreYFromFp32(yAddr, yReg, tailMask, offset);
        }
    }
}

// R==2 合并 tile（仅满 chunk 调用）：grads/x 各单 DMA 搬入 64 plane = 128 连续元素，
// 计算后单 DMA 写回；grads/x/y 三路双缓冲流水与逐 plane 路径同构
template <typename T>
__aicore__ inline void BNTrainingReduceGradKernel<T>::ProcessMergedR2(int64_t p0)
{
    constexpr int64_t MERGED_ELEMS = CHUNK_CHANNELS * 2; // 64 plane × R=2
    int64_t gmBase = p0 * 2;                             // innerSize==2 时 rStart_==0
    DataCopyExtParams cpIn{1, static_cast<uint32_t>(MERGED_ELEMS * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padIn{false, 0, 0, 0};

    LocalTensor<T> gradsUb = gradsQue_.AllocTensor<T>();
    DataCopyPad(gradsUb, gradsGm_[gmBase], cpIn, padIn);
    gradsQue_.EnQue(gradsUb);
    gradsUb = gradsQue_.DeQue<T>();

    LocalTensor<T> xUb = xQue_.AllocTensor<T>();
    DataCopyPad(xUb, xGm_[gmBase], cpIn, padIn);
    xQue_.EnQue(xUb);
    xUb = xQue_.DeQue<T>();

    LocalTensor<T> yUb = yQue_.AllocTensor<T>();
    ComputeTileMergedR2((__ubuf__ T*)gradsUb.GetPhyAddr(), (__ubuf__ T*)xUb.GetPhyAddr(),
                        (__ubuf__ T*)yUb.GetPhyAddr());

    yQue_.EnQue(yUb);
    yUb = yQue_.DeQue<T>();
    DataCopyPad(yGm_[gmBase], yUb, cpIn);
    yQue_.FreeTensor(yUb);
    xQue_.FreeTensor(xUb);
    gradsQue_.FreeTensor(gradsUb);
}

// R==2 合并 tile 主计算：128 连续元素 = 64 plane 交错排布（plane p 占位置 2p/2p+1）。
// grads/x 各自两个 64 元素寄存器 DeInterleave 成 r=0/r=1 两路后，两路共用同一组
// per-plane 系数（直接 DIST_NORM 加载，无需系数展开），算完 Interleave 还原交错布局
// 写回；运算序与逐 plane 路径一致（Mul→Add→Add→Mul，同一预计算系数），位级一致
template <typename T>
__aicore__ inline void BNTrainingReduceGradKernel<T>::ComputeTileMergedR2(__ubuf__ T* gradsAddr, __ubuf__ T* xAddr,
                                                                          __ubuf__ T* yAddr)
{
    __ubuf__ float* multiplierAddr = (__ubuf__ float*)multiplierBuf_.Get<float>().GetPhyAddr();
    __ubuf__ float* addendAddr = (__ubuf__ float*)addendBuf_.Get<float>().GetPhyAddr();
    __ubuf__ float* mulScaleAddr = (__ubuf__ float*)mulScaleBuf_.Get<float>().GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> multReg, addReg, mulScaleReg;
        RegTensor<float> xReg0, xReg1, xEven, xOdd, gReg0, gReg1, gEven, gOdd;
        RegTensor<float> yEven, yOdd, yReg0, yReg1;
        MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
        LoadToFp32(xAddr, xReg0, fullMask, 0);
        LoadToFp32(xAddr, xReg1, fullMask, VL_FP32);
        AscendC::MicroAPI::DeInterleave<float>(xEven, xOdd, xReg0, xReg1);
        LoadToFp32(gradsAddr, gReg0, fullMask, 0);
        LoadToFp32(gradsAddr, gReg1, fullMask, VL_FP32);
        AscendC::MicroAPI::DeInterleave<float>(gEven, gOdd, gReg0, gReg1);
        DataCopy<float, LoadDist::DIST_NORM>(multReg, multiplierAddr);
        DataCopy<float, LoadDist::DIST_NORM>(addReg, addendAddr);
        DataCopy<float, LoadDist::DIST_NORM>(mulScaleReg, mulScaleAddr);
        Mul(yEven, xEven, multReg, fullMask);
        Add(yEven, gEven, yEven, fullMask);
        Add(yEven, yEven, addReg, fullMask);
        Mul(yEven, yEven, mulScaleReg, fullMask);
        Mul(yOdd, xOdd, multReg, fullMask);
        Add(yOdd, gOdd, yOdd, fullMask);
        Add(yOdd, yOdd, addReg, fullMask);
        Mul(yOdd, yOdd, mulScaleReg, fullMask);
        AscendC::MicroAPI::Interleave<float>(yReg0, yReg1, yEven, yOdd);
        StoreYFromFp32(yAddr, yReg0, fullMask, 0);
        StoreYFromFp32(yAddr, yReg1, fullMask, VL_FP32);
    }
}
