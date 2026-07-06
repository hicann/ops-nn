/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {
namespace LayerNormStride {
#include <iostream>
#include <stdio.h>
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"
#include "dtype_convert.h"

using namespace AscendC;

constexpr int64_t UB_MAX_BYTES = 184 * 1024;
constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t WORK_LOCAL_SIZE = 64;

template <typename T>
class LayerNormStride {
public:
    __aicore__ inline LayerNormStride() {}

    __aicore__ inline void Init(const int64_t gbH, const int64_t gbW, const int64_t stride, const float epsilon,
                                GM_ADDR in, GM_ADDR gamma, GM_ADDR beta, GM_ADDR out)
    {
        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        gbH_ = gbH;
        gbW_ = gbW;
        stride_ = stride;
        epsilon_ = epsilon;
        invert_ = (float)(1.0) / gbW;

        bkH_ = 1;
        bkW_ = gbW_;
        bkAlignW_ = AlignUp(bkW_, 64);
        bkLoop_ = (int64_t)(gbH_ / blockNum_);
        if (gbH_ % blockNum_ != 0) {
            bkLoop_ += 1;
        }

        inGm_.SetGlobalBuffer((__gm__ T*)in, gbH_ * gbW_);
        gammaGm_.SetGlobalBuffer((__gm__ T*)gamma, gbW_);
        betaGm_.SetGlobalBuffer((__gm__ T*)beta, gbW_);
        outGm_.SetGlobalBuffer((__gm__ T*)out, gbH_ * gbW_);

        pipe_.InitBuffer(inQueIn_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(T));
        pipe_.InitBuffer(inQueGamma_, BUFFER_NUM, bkAlignW_ * sizeof(T) * 2);
        pipe_.InitBuffer(inQueBeta_, BUFFER_NUM, bkAlignW_ * sizeof(T) * 2);
        pipe_.InitBuffer(outQueOut_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(T) * 2);

        pipe_.InitBuffer(bufQueWork_, WORK_LOCAL_SIZE * sizeof(float));
        pipe_.InitBuffer(bufQueSum_, 16 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<T> gammaLocal = inQueGamma_.AllocTensor<T>();
        LocalTensor<T> betaLocal = inQueBeta_.AllocTensor<T>();

        DataCopyParams copyPas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
        DataCopyPadParams padPas;
        DataCopyPad(gammaLocal[bkAlignW_], gammaGm_, copyPas, padPas);
        DataCopyPad(betaLocal[bkAlignW_], betaGm_, copyPas, padPas);

        inQueGamma_.EnQue(gammaLocal);
        inQueBeta_.EnQue(betaLocal);
        gammaLm_ = inQueGamma_.DeQue<T>();
        betaLm_ = inQueBeta_.DeQue<T>();

        gammaFloatLm_ = gammaLm_.template ReinterpretCast<float>();
        betaFloatLm_ = betaLm_.template ReinterpretCast<float>();

        Cast(gammaFloatLm_, gammaLm_[bkAlignW_], RoundMode::CAST_NONE, bkAlignW_);
        Cast(betaFloatLm_, betaLm_[bkAlignW_], RoundMode::CAST_NONE, bkAlignW_);

        for (int64_t i = 0; i < bkLoop_; i++) {
            if (i * blockNum_ + blockIdx_ < gbH_) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }

        inQueGamma_.FreeTensor(gammaLocal);
        inQueBeta_.FreeTensor(betaLocal);
    }

private:
    __aicore__ inline void CopyIn(int64_t process)
    {
        LocalTensor<T> inLocal = inQueIn_.AllocTensor<T>();

        int64_t offset = (process * blockNum_ + blockIdx_) * stride_;

        DataCopyParams copyPas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
        DataCopyPadParams padPas;
        DataCopyPad(inLocal, inGm_[offset], copyPas, padPas);
        inQueIn_.EnQue(inLocal);
    }

    __aicore__ inline void Compute(int64_t progress)
    {
        LocalTensor<T> inLocal = inQueIn_.DeQue<T>();

        if (bkAlignW_ - AlignUp(bkW_, 16) != 0) {
            Duplicate(inLocal[AlignUp(bkW_, 16)], (T)0.0, bkAlignW_ - AlignUp(bkW_, 16));
        }

        LocalTensor<T> outLocal = outQueOut_.AllocTensor<T>();
        LocalTensor<float> inFloatLocal = outLocal.template ReinterpretCast<float>();

        LocalTensor<float> sumLocal = bufQueSum_.Get<float>();
        LocalTensor<float> workLocal = bufQueWork_.Get<float>();

        Duplicate<float>(inFloatLocal, (float)0.0, bkAlignW_);
        Cast(inFloatLocal, inLocal, RoundMode::CAST_NONE, bkAlignW_);
        {
            Duplicate<float>(workLocal, (float)0.0, 64);
            BinaryRepeatParams byrtPas = {1, 1, 1, 0, 8, 0};
            Add(workLocal, inFloatLocal, workLocal, 64, bkAlignW_ / 64, byrtPas);
            WholeReduceSum(sumLocal, workLocal, 64, 1, 1, 1, 8);
        }

        Muls(sumLocal, sumLocal, (float)invert_, 1);
        Muls(sumLocal, sumLocal, (float)(-1.0), 1);
        float cur_mean = sumLocal.GetValue(0);

        Cast(inFloatLocal, inLocal, RoundMode::CAST_NONE, bkAlignW_);
        Mul(inFloatLocal, inFloatLocal, inFloatLocal, bkW_);
        {
            Duplicate<float>(workLocal, (float)0.0, 64);
            BinaryRepeatParams byrtPas = {1, 1, 1, 0, 8, 0};
            Add(workLocal, inFloatLocal, workLocal, 64, bkAlignW_ / 64, byrtPas);
            WholeReduceSum(sumLocal, workLocal, 64, 1, 1, 1, 8);
        }

        Muls(sumLocal, sumLocal, (float)invert_, 1);
        Adds(sumLocal, sumLocal, (float)epsilon_, 1);
        Rsqrt(sumLocal, sumLocal, 1);
        float cur_var = sumLocal.GetValue(0);

        Duplicate<float>(inFloatLocal, (float)0.0, bkAlignW_);
        Cast(inFloatLocal, inLocal, RoundMode::CAST_NONE, bkW_);
        Muls(inFloatLocal, inFloatLocal, cur_var, bkW_);
        Mul(inFloatLocal, inFloatLocal, gammaFloatLm_, bkW_);
        Add(inFloatLocal, inFloatLocal, betaFloatLm_, bkW_);
        Cast(outLocal, inFloatLocal, RoundMode::CAST_ROUND, bkW_);

        inQueIn_.FreeTensor(inLocal);

        outQueOut_.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress)
    {
        LocalTensor<T> outLocal = outQueOut_.DeQue<T>();

        int64_t offset = (progress * blockNum_ + blockIdx_) * stride_;

        DataCopyParams copyPas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
        DataCopyPad(outGm_[offset], outLocal, copyPas);

        outQueOut_.FreeTensor(outLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueIn_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueGamma_, inQueBeta_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueOut_;

    GlobalTensor<T> inGm_, gammaGm_, betaGm_;
    GlobalTensor<T> outGm_;

    LocalTensor<T> gammaLm_, betaLm_;
    LocalTensor<float> gammaFloatLm_, betaFloatLm_;

    TBuf<QuePosition::VECCALC> bufQueWork_;
    TBuf<QuePosition::VECCALC> bufQueSum_;

    int64_t blockNum_, blockIdx_;
    int64_t gbH_, gbW_;

    int64_t bkH_;
    int64_t bkW_, bkAlignW_;
    int64_t bkLoop_;

    float epsilon_ = (float)0.0f;
    float invert_ = (float)0.0f;

    int64_t stride_;
};

extern "C" __global__ __aicore__ void ComputeLayernormStride(const int64_t gbH, const int64_t gbW, const int64_t stride,
                                                             const float epsilon, GM_ADDR in, GM_ADDR gamma,
                                                             GM_ADDR beta, GM_ADDR out, int32_t dtype)
{
#if __NPU_ARCH__ == 2201
    TYPE_SWITCH(dtype, T, {
        LayerNormStride<T> op;
        op.Init(gbH, gbW, stride, epsilon, in, gamma, beta, out);
        op.Process();
    });
#endif
}

void LayerNormStrideKernelLaunch(int64_t blockDim, void* stream, const int64_t gbH, const int64_t gbW,
                                 const int64_t stride, const float epsilon, uint8_t* in, uint8_t* gamma, uint8_t* beta,
                                 uint8_t* out, int32_t dtype)
{
    if (gbH < blockDim) {
        blockDim = gbH;
    }

    ComputeLayernormStride<<<blockDim, nullptr, stream>>>(gbH, gbW, stride, epsilon, in, gamma, beta, out, dtype);
}

inline int64_t CalculateAlignUp(const int64_t number, const int64_t alignSize)
{
    if (number % alignSize == 0) {
        return number;
    }

    return ((number / alignSize + 1) * alignSize);
}

int JudgeLayerNormStrideLaunch(const int64_t gbW, const int64_t ubSize)
{
    constexpr int64_t BUFFER_NUM = 1;
    constexpr int64_t WORK_LOCAL_SIZE = 64;

    int64_t gbW_;

    int64_t bkH_;
    int64_t bkW_, bkAlignW_;

    gbW_ = gbW;

    bkH_ = 1;
    bkW_ = gbW_;
    bkAlignW_ = CalculateAlignUp(bkW_, 64);

    float useByte = BUFFER_NUM * bkH_ * bkAlignW_ * sizeof(half) * 2;
    useByte += BUFFER_NUM * bkAlignW_ * sizeof(half) * 2;
    useByte += (WORK_LOCAL_SIZE + 16 + 2 * bkAlignW_) *
               sizeof(float); // bf16 增加 2* bkAlignW_ * sizeof(bfloat16_t) tbuf

    if (useByte > ubSize) {
        std::cout << __FUNCTION__ << ": "
                  << "bkW_,bkAlignW_,UB = " << bkW_ << "," << bkAlignW_ << "," << useByte / 1024 << " KB" << std::endl;
        return 1;
    }

    return 0;
}

int LayerNormStrideLaunch(int64_t blockDim, void* stream, const int64_t gbH, const int64_t gbW, const int64_t stride,
                          const float epsilon, uint8_t* in, uint8_t* gamma, uint8_t* beta, uint8_t* out, int32_t dtype)
{
    int64_t ubSize = 184 * 1024;

    int ret = JudgeLayerNormStrideLaunch(gbW, ubSize);
    if (ret == 0) {
        LayerNormStrideKernelLaunch(blockDim, stream, gbH, gbW, stride, epsilon, in, gamma, beta, out, dtype);
        return 0;
    }

    std::cout << __FUNCTION__ << ": "
              << "UB size is limited, please check!" << std::endl;
    return 1;
}

int64_t LayerNormStrideNpu(int64_t blockDim, const int64_t hiddenDim, const double epsilon, torch::Tensor& in,
                           torch::Tensor& gamma, torch::Tensor& beta, torch::Tensor& out)
{
    TORCH_CHECK(torch_npu::utils::is_npu(in), "input tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(gamma), "gama tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(beta), "beta tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(out), "output tensor must be on NPU device");
    TORCH_CHECK(in.scalar_type() == at::kBFloat16 || in.scalar_type() == at::kHalf,
                "dtype of input tensor is invalid, only BF16 or FP16 is supported.");
    TORCH_CHECK(gamma.scalar_type() == at::kBFloat16 || gamma.scalar_type() == at::kHalf,
                "dtype of gamma tensor is invalid, only BF16 or FP16 is supported.");
    TORCH_CHECK(beta.scalar_type() == at::kBFloat16 || beta.scalar_type() == at::kHalf,
                "dtype of beta tensor is invalid, only BF16 or FP16 is supported.");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kHalf,
                "dtype of output tensor is invalid, only BF16 or FP16 is supported.");
    TORCH_CHECK(blockDim > 0, "blockDim must be greater than 0");
    TORCH_CHECK(hiddenDim > 0, "hiddenDim must be greater than 0");

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int launchStatus = 0;
    auto aclCall = [=, &launchStatus]() -> int {
        launchStatus = LayerNormStrideLaunch(blockDim, stream, in.size(0) * in.size(1), hiddenDim, in.size(2), epsilon,
                                             (uint8_t*)in.data_ptr(), (uint8_t*)gamma.data_ptr(),
                                             (uint8_t*)beta.data_ptr(), (uint8_t*)out.data_ptr(),
                                             in.scalar_type() == at::kHalf ? 1 : 27);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("layerNormStride", aclCall);

    return launchStatus;
}

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("layernorm_stride(int blockDim, int hiddenDim, float epsilon, Tensor input, Tensor gamma, Tensor beta, "
          "Tensor output) -> int");
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m) { m.impl("layernorm_stride", LayerNormStrideNpu); }
} // namespace LayerNormStride
} // namespace ascend_ops