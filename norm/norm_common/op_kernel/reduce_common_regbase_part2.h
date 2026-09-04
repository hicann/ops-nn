/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Internal implementation section of reduce_common_regbase.h. Include only from reduce_common_regbase.h. */

namespace NormCommonRegbase {

template <typename T, LoadDist FLOAT_LOAD_DIST = LoadDist::DIST_NORM,
          LoadDist NON_FLOAT_LOAD_DIST = LoadDist::DIST_UNPACK_B16>
__aicore__ inline void LoadRegForDtype(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, FLOAT_LOAD_DIST>(dst, src + offset);
    } else {
        RegTensor<T> srcReg;
        DataCopy<T, NON_FLOAT_LOAD_DIST>(srcReg, src + offset);
        Cast<float, T, castTraitB162B32>(dst, srcReg, preg);
    }
}

template <typename T, StoreDist FLOAT_STORE_DIST = StoreDist::DIST_NORM,
          StoreDist NON_FLOAT_STORE_DIST = StoreDist::DIST_PACK_B32>
__aicore__ inline void StoreRegForDtype(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, FLOAT_STORE_DIST>(dst + offset, src, preg);
    } else {
        RegTensor<T> dstReg;
        Cast<T, float, castTraitB322B16>(dstReg, src, preg);
        DataCopy<T, NON_FLOAT_STORE_DIST>(dst + offset, dstReg, preg);
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSumLessThanVL(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                          uint16_t rows, uint32_t rowStride, uint32_t reduceNum)
{
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> sumReg;
        MaskReg pregLoop = UpdateMask<float>(reduceNum);
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        for (uint16_t i = 0; i < rows; ++i) {
            LoadRegForDtype<T>(xPtr, xReg, pregLoop, static_cast<uint32_t>(i) * rowStride);
            Mul(xReg, xReg, xReg, pregLoop);
            ReduceSum(sumReg, xReg, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, sumReg, pregOne);
        }
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                             uint16_t rows, uint32_t rowStride, uint32_t reduceNum)
{
    uint32_t tailLen = reduceNum - V_LENGTH;
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregTail = UpdateMask<float>(tailLen);
        for (uint16_t i = 0; i < rows; ++i) {
            uint32_t baseOffset = static_cast<uint32_t>(i) * rowStride;
            LoadRegForDtype<T>(xPtr, xReg, pregFull, baseOffset);
            LoadRegForDtype<T>(xPtr + V_LENGTH, xFoldReg, pregTail, baseOffset);
            Mul(xReg, xReg, xReg, pregFull);
            Mul(xFoldReg, xFoldReg, xFoldReg, pregTail);
            ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                       pregTail);
            Add(sumReg, xReg, xFoldReg, pregFull);
            ReduceSum(reduceReg, sumReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
        }
    }
}

template <typename T, int32_t LAST_LOOP_NUMS>
__aicore__ inline void CalculateSquareReduceSumCommon(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                      __local_mem__ float* tmpPtr, uint16_t rows, uint32_t rowStride,
                                                      uint32_t reduceNum, uint32_t foldPoint, uint32_t tmpStride)
{
    uint16_t foldLoops = static_cast<uint16_t>((foldPoint + V_LENGTH - 1) / V_LENGTH);
    uint32_t lastNum = foldPoint / V_LENGTH;
    uint32_t tail = (reduceNum > foldPoint) ? reduceNum - foldPoint : 0;
    uint16_t tailCeilLoops = static_cast<uint16_t>((tail + V_LENGTH - 1) / V_LENGTH);
    uint16_t firstFlodWithOutAddLoops = static_cast<uint16_t>(foldLoops - tailCeilLoops);

    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t i = 0; i < rows; ++i) {
            uint32_t baseOffset = static_cast<uint32_t>(i) * rowStride;
            uint32_t tmpOffset = static_cast<uint32_t>(i) * tmpStride;
            uint32_t sregTail = tail;
            for (uint16_t j = 0; j < tailCeilLoops; ++j) {
                pregLoop = UpdateMask<float>(sregTail);
                uint32_t offset = static_cast<uint32_t>(j) * V_LENGTH + baseOffset;
                LoadRegForDtype<T>(xPtr, xReg, pregFull, offset);
                Mul(xReg, xReg, xReg, pregFull);
                LoadRegForDtype<T>(xPtr + foldPoint, xFoldReg, pregFull, offset);
                Mul(xFoldReg, xFoldReg, xFoldReg, pregLoop);
                Add(sumReg, xReg, xFoldReg, pregFull);
                ReduceSum(reduceReg, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + tmpOffset + j, reduceReg, pregOne);
            }
            for (uint16_t j = 0; j < firstFlodWithOutAddLoops; ++j) {
                uint32_t offset = static_cast<uint32_t>(tailCeilLoops + j) * V_LENGTH + baseOffset;
                LoadRegForDtype<T>(xPtr, xReg, pregFull, offset);
                Mul(xReg, xReg, xReg, pregFull);
                ReduceSum(reduceReg, xReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + tmpOffset + tailCeilLoops + j, reduceReg,
                                                                   pregOne);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        if constexpr (LAST_LOOP_NUMS == 1) {
            MaskReg pregLast = UpdateMask<float>(lastNum);
            for (uint16_t i = 0; i < rows; ++i) {
                DataCopy(xReg, tmpPtr + static_cast<uint32_t>(i) * tmpStride);
                ReduceSum(reduceReg, xReg, pregLast);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
            }
        } else if constexpr (LAST_LOOP_NUMS == DICHOTOMY_ADD_COEFF) {
            lastNum -= V_LENGTH;
            MaskReg pregLast = UpdateMask<float>(lastNum);
            for (uint16_t i = 0; i < rows; ++i) {
                uint32_t tmpOffset = static_cast<uint32_t>(i) * tmpStride;
                DataCopy(xReg, tmpPtr + tmpOffset);
                DataCopy(xFoldReg, tmpPtr + tmpOffset + V_LENGTH);
                ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                           pregLast);
                Add(sumReg, xReg, xFoldReg, pregFull);
                ReduceSum(reduceReg, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
            }
        }
    }
}

template <typename T>
// Squares input values inside this function, then reduces each row.
__aicore__ inline void CalculateSquareReduceSum(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                __local_mem__ float* tmpPtr, uint16_t rows, uint32_t rowStride,
                                                uint32_t reduceNum, uint32_t foldPoint, uint32_t tmpStride,
                                                uint32_t branchNum = 0)
{
    uint32_t reduceBranchNum = branchNum == 0 ? reduceNum : branchNum;
    if (reduceBranchNum <= V_LENGTH) {
        CalculateSquareReduceSumLessThanVL<T>(xPtr, dstPtr, rows, rowStride, reduceNum);
    } else if (reduceBranchNum <= V_LENGTH + V_LENGTH) {
        CalculateSquareReduceSumLessThanTwoVL<T>(xPtr, dstPtr, rows, rowStride, reduceNum);
    } else if (reduceBranchNum <= V_LENGTH * V_LENGTH * DICHOTOMY_ADD_COEFF) {
        CalculateSquareReduceSumCommon<T, 1>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum, foldPoint, tmpStride);
    } else {
        CalculateSquareReduceSumCommon<T, DICHOTOMY_ADD_COEFF>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum,
                                                               foldPoint, tmpStride);
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSum(LocalTensor<T>& xLocal, LocalTensor<float>& dstLocal,
                                                LocalTensor<float>& tmpLocal, uint16_t rows, uint32_t rowStride,
                                                uint32_t reduceNum, uint32_t foldPoint, uint32_t blockAlign,
                                                uint32_t branchNum = 0)
{
    __local_mem__ T* xPtr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ float* dstPtr = (__local_mem__ float*)dstLocal.GetPhyAddr();
    __local_mem__ float* tmpPtr = (__local_mem__ float*)tmpLocal.GetPhyAddr();
    uint32_t foldLoops = (foldPoint + V_LENGTH - 1) / V_LENGTH;
    uint32_t tmpStride = (foldLoops + blockAlign - 1) / blockAlign * blockAlign;
    CalculateSquareReduceSum<T>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum, foldPoint, tmpStride, branchNum);
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSum(LocalTensor<T>& xLocal, LocalTensor<float>& dstLocal,
                                                TBuf<TPosition::VECCALC>& tmpBuf, uint16_t rows, uint32_t rowStride,
                                                uint32_t reduceNum, uint32_t foldPoint, uint32_t blockAlign,
                                                uint32_t branchNum = 0)
{
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    CalculateSquareReduceSum<T>(xLocal, dstLocal, tmpLocal, rows, rowStride, reduceNum, foldPoint, blockAlign,
                                branchNum);
}

__aicore__ inline void CalculateReduceSumLessThanVL(__local_mem__ float* xPtr, __local_mem__ float* dstPtr,
                                                    uint32_t reduceNum)
{
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> sumReg;
        MaskReg pregLoop = UpdateMask<float>(reduceNum);
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        DataCopy<float, LoadDist::DIST_NORM>(xReg, xPtr);
        ReduceSum(sumReg, xReg, pregLoop);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, sumReg, pregOne);
    }
}

__aicore__ inline void CalculateReduceSumLessThanTwoVL(__local_mem__ float* xPtr, __local_mem__ float* dstPtr,
                                                       uint32_t reduceNum)
{
    uint32_t tailLen = reduceNum - V_LENGTH;
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(tailLen);
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        DataCopy<float, LoadDist::DIST_NORM>(xReg, xPtr);
        DataCopy<float, LoadDist::DIST_NORM>(xFoldReg, xPtr + V_LENGTH);
        ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0), pregTail);
        Add(sumReg, xReg, xFoldReg, pregFull);
        ReduceSum(reduceReg, sumReg, pregFull);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, reduceReg, pregOne);
    }
}

template <int32_t LAST_LOOP_NUMS>
__aicore__ inline void CalculateReduceSumCommon(__local_mem__ float* xPtr, __local_mem__ float* dstPtr,
                                                __local_mem__ float* tmpPtr, uint32_t reduceNum, uint32_t foldPoint)
{
    uint16_t foldLoops = static_cast<uint16_t>((foldPoint + V_LENGTH - 1) / V_LENGTH);
    uint32_t lastNum = foldPoint / V_LENGTH;
    uint32_t tail = reduceNum - foldPoint;
    uint16_t tailCeilLoops = static_cast<uint16_t>((tail + V_LENGTH - 1) / V_LENGTH);
    uint16_t tailFullLoops = static_cast<uint16_t>(tail / V_LENGTH);

    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t r = 0; r < tailFullLoops; ++r) {
            uint32_t offset = static_cast<uint32_t>(r) * V_LENGTH;
            DataCopy<float, LoadDist::DIST_NORM>(xReg, xPtr + offset);
            DataCopy<float, LoadDist::DIST_NORM>(xFoldReg, xPtr + foldPoint + offset);
            Add(sumReg, xReg, xFoldReg, pregFull);
            ReduceSum(reduceReg, sumReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + r, reduceReg, pregOne);
        }
        uint32_t tailRemain = tail - static_cast<uint32_t>(tailFullLoops) * V_LENGTH;
        if (tailRemain != 0) {
            pregLoop = UpdateMask<float>(tailRemain);
            uint32_t offset = static_cast<uint32_t>(tailFullLoops) * V_LENGTH;
            DataCopy<float, LoadDist::DIST_NORM>(xReg, xPtr + offset);
            DataCopy<float, LoadDist::DIST_NORM>(xFoldReg, xPtr + foldPoint + offset);
            ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                       pregLoop);
            Add(sumReg, xReg, xFoldReg, pregFull);
            ReduceSum(reduceReg, sumReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + tailFullLoops, reduceReg, pregOne);
        }
        // Fix the original local implementations' fixed-offset bug in the remaining reduce blocks.
        for (uint16_t r = 0; r < static_cast<uint16_t>(foldLoops - tailCeilLoops); ++r) {
            uint32_t offset = static_cast<uint32_t>(tailCeilLoops + r);
            DataCopy<float, LoadDist::DIST_NORM>(xReg, xPtr + offset * V_LENGTH);
            ReduceSum(reduceReg, xReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + offset, reduceReg, pregOne);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        if constexpr (LAST_LOOP_NUMS == 1) {
            MaskReg pregLast = UpdateMask<float>(lastNum);
            DataCopy<float, LoadDist::DIST_NORM>(xReg, tmpPtr);
            ReduceSum(reduceReg, xReg, pregLast);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, reduceReg, pregOne);
        } else if constexpr (LAST_LOOP_NUMS == DICHOTOMY_ADD_COEFF) {
            lastNum -= V_LENGTH;
            MaskReg pregLast = UpdateMask<float>(lastNum);
            DataCopy<float, LoadDist::DIST_NORM>(xReg, tmpPtr);
            DataCopy<float, LoadDist::DIST_NORM>(xFoldReg, tmpPtr + V_LENGTH);
            ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                       pregLast);
            Add(sumReg, xReg, xFoldReg, pregFull);
            ReduceSum(reduceReg, sumReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, reduceReg, pregOne);
        }
    }
}

// Reduces an fp32 buffer whose values have already been squared by the caller.
__aicore__ inline void CalculateReduceSum(__local_mem__ float* xPtr, __local_mem__ float* dstPtr,
                                          __local_mem__ float* tmpPtr, uint32_t reduceNum, uint32_t foldPoint)
{
    if (reduceNum <= V_LENGTH) {
        CalculateReduceSumLessThanVL(xPtr, dstPtr, reduceNum);
    } else if (reduceNum <= V_LENGTH + V_LENGTH) {
        CalculateReduceSumLessThanTwoVL(xPtr, dstPtr, reduceNum);
    } else if (reduceNum <= V_LENGTH * V_LENGTH * DICHOTOMY_ADD_COEFF) {
        CalculateReduceSumCommon<1>(xPtr, dstPtr, tmpPtr, reduceNum, foldPoint);
    } else {
        CalculateReduceSumCommon<DICHOTOMY_ADD_COEFF>(xPtr, dstPtr, tmpPtr, reduceNum, foldPoint);
    }
}

__aicore__ inline void CalculateReduceSum(LocalTensor<float>& xLocal, LocalTensor<float>& dstLocal,
                                          LocalTensor<float>& tmpLocal, uint32_t reduceNum, uint32_t foldPoint)
{
    __local_mem__ float* xPtr = (__local_mem__ float*)xLocal.GetPhyAddr();
    __local_mem__ float* dstPtr = (__local_mem__ float*)dstLocal.GetPhyAddr();
    __local_mem__ float* tmpPtr = (__local_mem__ float*)tmpLocal.GetPhyAddr();
    CalculateReduceSum(xPtr, dstPtr, tmpPtr, reduceNum, foldPoint);
}

__aicore__ inline void CalculateReduceSum(LocalTensor<float>& xLocal, LocalTensor<float>& dstLocal,
                                          TBuf<TPosition::VECCALC>& tmpBuf, uint32_t reduceNum, uint32_t foldPoint)
{
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    CalculateReduceSum(xLocal, dstLocal, tmpLocal, reduceNum, foldPoint);
}
} // namespace NormCommonRegbase
