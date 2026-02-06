//  /**
//    * Copyright (c) 2026 Huawei Technologies Co., Ltd.
//    * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//    * CANN Open Software License Agreement Version 2.0 (the "License").
//    * Please refer to the License for details. You may not use this file except in compliance with the License.
//    * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//    * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//    * See LICENSE in the root of the software repository for the full text of the License.
//    */

// #include <array>
// #include <vector>
// #include <iostream>
// #include <string>
// #include <cstdint>
// #include <limits>
// #include <float.h>
// #include "gtest/gtest.h"
// #include "tikicpulib.h"
// #include "max_pool_grad_with_argmax_tiling_def.h"
// #include "data_utils.h"

// #include <cstdint>
// using namespace std;

// extern "C" __global__ __aicore__ void max_pool_grad_with_argmax(
//     GM_ADDR x, GM_ADDR y, GM_ADDR grad, GM_ADDR argmax,GM_ADDR workspace, GM_ADDR tiling);

// class max_pool_grad_with_argmax_test : public testing::Test {
// protected:
//     static void SetUpTestCase()
//     {
//         cout << "max_pool_grad_with_argmax_test SetUp\n" << endl;
//     }
//     static void TearDownTestCase()
//     {
//         cout << "max_pool_grad_with_argmax_test TearDown\n" << endl;
//     }
// };


// TEST_F(max_pool_grad_with_argmax_test, test_case_nchw_simt_float32)
// {
//     AscendC:: SetKernelMode(KernelMode::AIV_MODE);
//     uint32_t blockDim = 24;
//     size_t inputByteSize = 2 * 2 * 3911 * 2 * sizeof(float);
//     size_t outputByteSize = 2 * 2 * 3911 * 2 * sizeof(float);
//     size_t gradByteSize = 2 * 2 * 1955 * 1 * sizeof(float);
//     size_t argmaxByteSize = 2 * 2 * 1955 * 1 * sizeof(int32_t);
//     size_t tilingDataSize = sizeof(MaxPoolGradWithArgmaxSimtTilingCommonData);

//     uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* out = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
//     uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     MaxPoolGradWithArgmaxSimtTilingCommonData* tilingDatafromBin =
//         reinterpret_cast<MaxPoolGradWithArgmaxSimtTilingCommonData*>(tiling);
//     tilingDatafromBin->nDim      = 2;
//     tilingDatafromBin->cDim      = 2;
//     tilingDatafromBin->hInDim    = 3911;
//     tilingDatafromBin->wInDim    = 2;
//     tilingDatafromBin->hOutDim   = 1955;  // floor((3911 - 2) / 2) + 1 = 1955
//     tilingDatafromBin->wOutDim   = 1;     // floor((2 - 2) / 366) + 1 = 1 (VALID)

//     tilingDatafromBin->kSizeH    = 2;     // from ksize[2]
//     tilingDatafromBin->kSizeW    = 2;     // from ksize[3]
//     tilingDatafromBin->stridesH  = 2;     // from strides[2]
//     tilingDatafromBin->stridesW  = 366;   // from strides[3]

//     tilingDatafromBin->padH      = 0;     // "VALID" padding
//     tilingDatafromBin->padW      = 0;

//     ICPU_SET_TILING_KEY(801UL);

//     ICPU_RUN_KF(max_pool_grad_with_argmax, blockDim, input, out, grad, argmax, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(input);
//     AscendC::GmFree(out);
//     AscendC::GmFree(grad);
//     AscendC::GmFree(argmax);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(max_pool_grad_with_argmax_test, test_case_nchw_simt_float32_large_803)
// {
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     uint32_t blockDim = 24;

//     // Input shape: N=2, C=2, H=268435457, W=2 (NCHW)
//     // Total elements = 2*2*268435457*2 = 2,147,483,656 > MAX_INT32
//     size_t inputByteSize = 2ULL * 2 * 268435457 * 2 * sizeof(float);
//     size_t outputByteSize = inputByteSize;

//     size_t gradByteSize = 2 * 2 * 134217728 * 1 * sizeof(float);
//     size_t argmaxByteSize = 2 * 2 * 134217728 * 1 * sizeof(int32_t);

//     size_t tilingDataSize = sizeof(MaxPoolGradWithArgmaxSimtTilingCommonData);

//     uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* out = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
//     uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 32);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

//     MaxPoolGradWithArgmaxSimtTilingCommonData* tilingDatafromBin =
//         reinterpret_cast<MaxPoolGradWithArgmaxSimtTilingCommonData*>(tiling);
//     // Fill tiling data for 803 case (NCHW, large)
//     tilingDatafromBin->nDim      = 2;
//     tilingDatafromBin->cDim      = 2;
//     tilingDatafromBin->hInDim    = 268435457;
//     tilingDatafromBin->wInDim    = 2;
//     tilingDatafromBin->hOutDim   = 134217728;
//     tilingDatafromBin->wOutDim   = 1;
//     tilingDatafromBin->kSizeH    = 2;
//     tilingDatafromBin->kSizeW    = 2;
//     tilingDatafromBin->stridesH  = 2;
//     tilingDatafromBin->stridesW  = 366;

//     tilingDatafromBin->padH      = 0;
//     tilingDatafromBin->padW      = 0;
//     ICPU_SET_TILING_KEY(803UL);

//     ICPU_RUN_KF(max_pool_grad_with_argmax, blockDim, input, grad, argmax, out, workspace, (uint8_t*)(tilingDatafromBin));

//     // Cleanup
//     AscendC::GmFree(input);
//     AscendC::GmFree(out);
//     AscendC::GmFree(grad);
//     AscendC::GmFree(argmax);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
// }

// TEST_F(max_pool_grad_with_argmax_test, test_case_nhwc_simt_float32_802)
// {
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     uint32_t blockDim = 24;

//     // Input shape: NHWC = [3, 3, 1870, 6]
//     size_t inputByteSize = 3 * 3 * 1870 * 6 * sizeof(float);
//     size_t outputByteSize = inputByteSize;

//     // Grad/argmax shape: [3, 2, 935, 6]
//     size_t gradByteSize = 3 * 2 * 935 * 6 * sizeof(float);
//     size_t argmaxByteSize = 3 * 2 * 935 * 6 * sizeof(int32_t);

//     size_t tilingDataSize = sizeof(MaxPoolGradWithArgmaxSimtTilingCommonData);

//     uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* out = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
//     uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16); // 16MB
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

//     MaxPoolGradWithArgmaxSimtTilingCommonData* tilingDatafromBin =
//         reinterpret_cast<MaxPoolGradWithArgmaxSimtTilingCommonData*>(tiling);

//     // Fill tiling data for NHWC 802 case
//     tilingDatafromBin->nDim      = 3;
//     tilingDatafromBin->cDim      = 6;
//     tilingDatafromBin->hInDim    = 3;
//     tilingDatafromBin->wInDim    = 1870;
//     tilingDatafromBin->hOutDim   = 2;
//     tilingDatafromBin->wOutDim   = 935;

//     tilingDatafromBin->kSizeH    = 4377;
//     tilingDatafromBin->kSizeW    = 2;
//     tilingDatafromBin->stridesH  = 2;
//     tilingDatafromBin->stridesW  = 2;

//     tilingDatafromBin->padH      = 0;
//     tilingDatafromBin->padW      = 0;

//     ICPU_SET_TILING_KEY(802UL);

//     ICPU_RUN_KF(max_pool_grad_with_argmax, blockDim, input, grad, argmax, out, workspace, (uint8_t*)(tilingDatafromBin));

//     // Cleanup
//     AscendC::GmFree(input);
//     AscendC::GmFree(out);
//     AscendC::GmFree(grad);
//     AscendC::GmFree(argmax);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
// }

// TEST_F(max_pool_grad_with_argmax_test, test_case_nhwc_simt_float32_large_804)
// {
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     uint32_t blockDim = 36;

//     size_t inputByteSize = 2 * 10000 * 10000 * 11 * sizeof(float);
//     size_t outputByteSize = inputByteSize;

//     size_t gradByteSize = 2 * 5000 * 5000 * 11 * sizeof(float);
//     size_t argmaxByteSize = 2 * 5000 * 5000 * 11 * sizeof(int32_t);

//     size_t tilingDataSize = sizeof(MaxPoolGradWithArgmaxSimtTilingCommonData);

//     uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* out = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
//     uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 64);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

//     MaxPoolGradWithArgmaxSimtTilingCommonData* tilingDatafromBin =
//         reinterpret_cast<MaxPoolGradWithArgmaxSimtTilingCommonData*>(tiling);

//     tilingDatafromBin->nDim      = 2;
//     tilingDatafromBin->cDim      = 11;
//     tilingDatafromBin->hInDim    = 10000;
//     tilingDatafromBin->wInDim    = 10000;
//     tilingDatafromBin->hOutDim   = 5000;
//     tilingDatafromBin->wOutDim   = 5000;

//     tilingDatafromBin->kSizeH    = 4377;
//     tilingDatafromBin->kSizeW    = 2;
//     tilingDatafromBin->stridesH  = 2;
//     tilingDatafromBin->stridesW  = 2;

//     tilingDatafromBin->padH      = 0;
//     tilingDatafromBin->padW      = 0;
//     ICPU_SET_TILING_KEY(804UL);

//     ICPU_RUN_KF(max_pool_grad_with_argmax, blockDim, input, grad, argmax, out, workspace, (uint8_t*)(tilingDatafromBin));

//     // Cleanup
//     AscendC::GmFree(input);
//     AscendC::GmFree(out);
//     AscendC::GmFree(grad);
//     AscendC::GmFree(argmax);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
// }

// TEST_F(max_pool_grad_with_argmax_test, test_case_nhwc_merge_hwc_)
// {
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     uint32_t blockDim = 36;

//     size_t inputByteSize = 2 * 10000 * 4 * 3 * sizeof(float);
//     size_t outputByteSize = inputByteSize; 

//     size_t gradByteSize = 2 * 5000 * 2 * 11 * sizeof(float);
//     size_t argmaxByteSize = 2 * 5000 * 2 * 11 * sizeof(int32_t);

//     size_t tilingDataSize = sizeof(MaxPoolGradWithArgmaxSimtTilingCommonData);

//     uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* out = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
//     uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 64); 
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

//     MaxPoolGradWithArgmaxSimtTilingCommonData* tilingDatafromBin =
//         reinterpret_cast<MaxPoolGradWithArgmaxSimtTilingCommonData*>(tiling);

//     tilingDatafromBin->nDim      = 2;
//     tilingDatafromBin->cDim      = 3;       
//     tilingDatafromBin->hInDim    = 10000;    
//     tilingDatafromBin->wInDim    = 4;   
//     tilingDatafromBin->hOutDim   = 5000;
//     tilingDatafromBin->wOutDim   = 2;

//     tilingDatafromBin->kSizeH    = 2;     
//     tilingDatafromBin->kSizeW    = 2;        
//     tilingDatafromBin->stridesH  = 2;       
//     tilingDatafromBin->stridesW  = 2;       

//     tilingDatafromBin->padH      = 0;        
//     tilingDatafromBin->padW      = 0;
//     ICPU_SET_TILING_KEY(501UL);

//     ICPU_RUN_KF(max_pool_grad_with_argmax, blockDim, input, grad, argmax, out, workspace, (uint8_t*)(tilingDatafromBin));

//     // Cleanup
//     AscendC::GmFree(input);
//     AscendC::GmFree(out);
//     AscendC::GmFree(grad);
//     AscendC::GmFree(argmax);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
// }

// TEST_F(max_pool_grad_with_argmax_test, test_case_nhwc_merge_hwc_int64)
// {
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     uint32_t blockDim = 36;

//     size_t inputByteSize = 2 * 10000 * 4 * 3 * sizeof(float);
//     size_t outputByteSize = inputByteSize; 

//     size_t gradByteSize = 2 * 5000 * 2 * 11 * sizeof(float);
//     size_t argmaxByteSize = 2 * 5000 * 2 * 11 * sizeof(int32_t);

//     size_t tilingDataSize = sizeof(MaxPoolGradWithArgmaxSimtTilingCommonData);

//     uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* out = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
//     uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 64); 
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

//     MaxPoolGradWithArgmaxSimtTilingCommonData* tilingDatafromBin =
//         reinterpret_cast<MaxPoolGradWithArgmaxSimtTilingCommonData*>(tiling);

//     tilingDatafromBin->nDim      = 2;
//     tilingDatafromBin->cDim      = 3;       
//     tilingDatafromBin->hInDim    = 10000;    
//     tilingDatafromBin->wInDim    = 4;   
//     tilingDatafromBin->hOutDim   = 5000;
//     tilingDatafromBin->wOutDim   = 2;

//     tilingDatafromBin->kSizeH    = 2;     
//     tilingDatafromBin->kSizeW    = 2;        
//     tilingDatafromBin->stridesH  = 2;       
//     tilingDatafromBin->stridesW  = 2;       

//     tilingDatafromBin->padH      = 0;        
//     tilingDatafromBin->padW      = 0;
//     ICPU_SET_TILING_KEY(511UL);

//     ICPU_RUN_KF(max_pool_grad_with_argmax, blockDim, input, grad, argmax, out, workspace, (uint8_t*)(tilingDatafromBin));

//     // Cleanup
//     AscendC::GmFree(input);
//     AscendC::GmFree(out);
//     AscendC::GmFree(grad);
//     AscendC::GmFree(argmax);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
// }

// TEST_F(max_pool_grad_with_argmax_test, test_case_nhwc_wc_601)
// {
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     uint32_t blockDim = 34;
//     size_t inputByteSize = 4 * 17 * 17 * 8 * sizeof(half);
//     size_t outputByteSize = inputByteSize;
//     size_t gradByteSize = 4 * 17 * 17 * 8 * sizeof(half);
//     size_t argmaxByteSize = 4 * 17 * 17 * 8 * sizeof(int64_t);
//     size_t tilingDataSize = sizeof(MaxPoolGradWithArgmaxNHWCTilingCommonData);

//     uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* out = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
//     uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     MaxPoolGradWithArgmaxNHWCTilingCommonData* tilingDatafromBin =
//         reinterpret_cast<MaxPoolGradWithArgmaxNHWCTilingCommonData*>(tiling);
//     // Fill tiling data for 600 case (NHWC, bigc,no check)
//     tilingDatafromBin->hArgmax = 17;
//     tilingDatafromBin->wArgmax = 17;
//     tilingDatafromBin->cOutput = 8;
//     tilingDatafromBin->hOutput = 17;
//     tilingDatafromBin->wOutput = 17;
//     tilingDatafromBin->hKernel = 3;
//     tilingDatafromBin->wKernel = 4;
//     tilingDatafromBin->hStride = 1;
//     tilingDatafromBin->wStride = 1;
//     tilingDatafromBin->padH = 1;
//     tilingDatafromBin->padW = 1;
//     tilingDatafromBin->nOutputInner = 1;
//     tilingDatafromBin->nOutputTail = 1;
//     tilingDatafromBin->nOutputOuter = 4;
//     tilingDatafromBin->hOutputInner = 1;
//     tilingDatafromBin->hOutputTail = 1;
//     tilingDatafromBin->hOutputOuter = 17;
//     tilingDatafromBin->wOutputInner = 17;
//     tilingDatafromBin->wOutputTail = 17;
//     tilingDatafromBin->wOutputOuter = 1;
//     tilingDatafromBin->cOutputInner = 8;
//     tilingDatafromBin->cOutputTail = 8;
//     tilingDatafromBin->cOutputOuter = 1;
//     tilingDatafromBin->normalCoreProcessNum = 2;
//     tilingDatafromBin->tailCoreProcessNum = 2;
//     tilingDatafromBin->usedCoreNum = 34;
//     tilingDatafromBin->outputBufferSize = 1088;
//     tilingDatafromBin->gradBufferSize = 2176;
//     tilingDatafromBin->argmaxBufferSize = 4096;
//     tilingDatafromBin->hProBatchSize = 3;
//     tilingDatafromBin->wProBatchSize = 4;
//     tilingDatafromBin->tilingKey = 601;
//     ICPU_SET_TILING_KEY(601UL);

//     ICPU_RUN_KF(
//         max_pool_grad_with_argmax, blockDim, input, out, grad, argmax, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(input);
//     AscendC::GmFree(out);
//     AscendC::GmFree(grad);
//     AscendC::GmFree(argmax);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(max_pool_grad_with_argmax_test, test_case_nhwc_bigc_600)
// {
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     uint32_t blockDim = 35;

//     size_t inputByteSize = 16 * 18 * 23 * 16 * sizeof(float);
//     size_t outputByteSize = inputByteSize;

//     size_t gradByteSize = 16 * 3 * 4 * 16 * sizeof(float);
//     size_t argmaxByteSize = 16 * 3 * 4 * 16 * sizeof(int64_t);

//     size_t tilingDataSize = sizeof(MaxPoolGradWithArgmaxNHWCTilingCommonData);

//     uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* out = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
//     uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 32);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

//     MaxPoolGradWithArgmaxNHWCTilingCommonData* tilingDatafromBin =
//         reinterpret_cast<MaxPoolGradWithArgmaxNHWCTilingCommonData*>(tiling);
//     // Fill tiling data for 600 case (NHWC, bigc)
//     tilingDatafromBin->hArgmax = 3;
//     tilingDatafromBin->wArgmax = 4;
//     tilingDatafromBin->cOutput = 16;
//     tilingDatafromBin->hOutput = 18;
//     tilingDatafromBin->wOutput = 23;
//     tilingDatafromBin->hKernel = 6;
//     tilingDatafromBin->wKernel = 5;
//     tilingDatafromBin->hStride = 6;
//     tilingDatafromBin->wStride = 5;
//     tilingDatafromBin->padH = 0;
//     tilingDatafromBin->padW = 0;
//     tilingDatafromBin->nOutputInner = 1;
//     tilingDatafromBin->nOutputTail = 1;
//     tilingDatafromBin->nOutputOuter = 16;
//     tilingDatafromBin->hOutputInner = 6;
//     tilingDatafromBin->hOutputTail = 6;
//     tilingDatafromBin->hOutputOuter = 3;
//     tilingDatafromBin->wOutputInner = 15;
//     tilingDatafromBin->wOutputTail = 8;
//     tilingDatafromBin->wOutputOuter = 2;
//     tilingDatafromBin->cOutputInner = 16;
//     tilingDatafromBin->cOutputTail = 16;
//     tilingDatafromBin->cOutputOuter = 1;
//     tilingDatafromBin->normalCoreProcessNum = 2;
//     tilingDatafromBin->tailCoreProcessNum = 2;
//     tilingDatafromBin->usedCoreNum = 48;
//     tilingDatafromBin->outputBufferSize = 5760;
//     tilingDatafromBin->gradBufferSize = 768;
//     tilingDatafromBin->argmaxBufferSize = 1280;
//     tilingDatafromBin->hProBatchSize = 1;
//     tilingDatafromBin->wProBatchSize = 1;
//     tilingDatafromBin->tilingKey = 600;
//     ICPU_SET_TILING_KEY(600UL);

//     ICPU_RUN_KF(
//         max_pool_grad_with_argmax, blockDim, input, grad, argmax, out, workspace, (uint8_t*)(tilingDatafromBin));

//     // Cleanup
//     AscendC::GmFree(input);
//     AscendC::GmFree(out);
//     AscendC::GmFree(grad);
//     AscendC::GmFree(argmax);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
// }

// TEST_F(max_pool_grad_with_argmax_test, test_case_nhwc_bigc_700)
// {
//     AscendC:: SetKernelMode(KernelMode::AIV_MODE);
//     uint32_t blockDim = 41;
//     size_t inputByteSize = 81 * 29 * 11 * 17 * sizeof(half);
//     size_t outputByteSize = inputByteSize;
//     size_t gradByteSize = 81 * 3 * 3 * 7 * sizeof(half);
//     size_t argmaxByteSize = 81 * 3 * 3 * 7 *  sizeof(int64_t);
//     size_t tilingDataSize = sizeof(MaxPoolGradWithArgmaxNHWCTilingCommonData);

//     uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* out = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
//     uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     MaxPoolGradWithArgmaxNHWCTilingCommonData* tilingDatafromBin =
//         reinterpret_cast<MaxPoolGradWithArgmaxNHWCTilingCommonData*>(tiling);
//     // Fill tiling data for 700 case (NHWC, bigc,no check)
//     tilingDatafromBin-> hArgmax = 3;
//     tilingDatafromBin-> wArgmax = 3;
//     tilingDatafromBin-> cOutput = 17;
//     tilingDatafromBin-> hOutput = 29;
//     tilingDatafromBin-> wOutput = 11;
//     tilingDatafromBin-> hKernel = 6;
//     tilingDatafromBin-> wKernel = 4;
//     tilingDatafromBin-> hStride = 11;
//     tilingDatafromBin-> wStride = 4;
//     tilingDatafromBin-> padH = 0;
//     tilingDatafromBin-> padW = 0;
//     tilingDatafromBin-> nOutputInner = 1;
//     tilingDatafromBin-> nOutputTail = 1;
//     tilingDatafromBin-> nOutputOuter = 81;
//     tilingDatafromBin-> hOutputInner = 29;
//     tilingDatafromBin-> hOutputTail = 29;
//     tilingDatafromBin-> hOutputOuter = 1;
//     tilingDatafromBin-> wOutputInner = 11;
//     tilingDatafromBin-> wOutputTail = 11;
//     tilingDatafromBin-> wOutputOuter = 1;
//     tilingDatafromBin-> cOutputInner = 17;
//     tilingDatafromBin-> cOutputTail = 17;
//     tilingDatafromBin-> cOutputOuter = 1;
//     tilingDatafromBin-> normalCoreProcessNum = 2;
//     tilingDatafromBin-> tailCoreProcessNum = 1;
//     tilingDatafromBin-> usedCoreNum = 41;
//     tilingDatafromBin-> outputBufferSize = 40832;
//     tilingDatafromBin-> gradBufferSize = 1280;
//     tilingDatafromBin-> argmaxBufferSize = 4352;
//     tilingDatafromBin-> hProBatchSize = 1;
//     tilingDatafromBin-> wProBatchSize = 1;
//     tilingDatafromBin-> tilingKey = 700;
//     ICPU_SET_TILING_KEY(700UL);

//     ICPU_RUN_KF(max_pool_grad_with_argmax, blockDim, input, out, grad, argmax, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(input);
//     AscendC::GmFree(out);
//     AscendC::GmFree(grad);
//     AscendC::GmFree(argmax);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(max_pool_grad_with_argmax_test, test_case_nhwc_bigc_701)
// {
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     uint32_t blockDim = 35;

//     size_t inputByteSize = 23ULL * 29 * 31 * 36 * sizeof(half);
//     size_t outputByteSize = inputByteSize;

//     size_t gradByteSize = 23 * 8 * 6 * 36 * sizeof(half);
//     size_t argmaxByteSize = 23 * 8 * 6 * 36 * sizeof(int64_t);

//     size_t tilingDataSize = sizeof(MaxPoolGradWithArgmaxNHWCTilingCommonData);

//     uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* out = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
//     uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 32);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

//     MaxPoolGradWithArgmaxNHWCTilingCommonData* tilingDatafromBin =
//         reinterpret_cast<MaxPoolGradWithArgmaxNHWCTilingCommonData*>(tiling);
//     // Fill tiling data for 701 case (NHWC, bigc)
//     tilingDatafromBin-> hArgmax = 8;
//     tilingDatafromBin-> wArgmax = 6;
//     tilingDatafromBin-> cOutput = 36;
//     tilingDatafromBin-> hOutput = 29;
//     tilingDatafromBin-> wOutput = 31;
//     tilingDatafromBin-> hKernel = 4;
//     tilingDatafromBin-> wKernel = 5;
//     tilingDatafromBin-> hStride = 4;
//     tilingDatafromBin-> wStride = 6;
//     tilingDatafromBin-> padH = 1;
//     tilingDatafromBin-> padW = 2;
//     tilingDatafromBin-> nOutputInner = 1;
//     tilingDatafromBin-> nOutputTail = 1;
//     tilingDatafromBin-> nOutputOuter = 23;
//     tilingDatafromBin-> hOutputInner = 10;
//     tilingDatafromBin-> hOutputTail = 9;
//     tilingDatafromBin-> hOutputOuter = 3;
//     tilingDatafromBin-> wOutputInner = 31;
//     tilingDatafromBin-> wOutputTail = 31;
//     tilingDatafromBin-> wOutputOuter = 1;
//     tilingDatafromBin-> cOutputInner = 36;
//     tilingDatafromBin-> cOutputTail = 36;
//     tilingDatafromBin-> cOutputOuter = 1;
//     tilingDatafromBin-> normalCoreProcessNum = 2;
//     tilingDatafromBin-> tailCoreProcessNum = 1;
//     tilingDatafromBin-> usedCoreNum = 35;
//     tilingDatafromBin-> outputBufferSize = 59520;
//     tilingDatafromBin-> gradBufferSize = 2560;
//     tilingDatafromBin-> argmaxBufferSize = 9472;
//     tilingDatafromBin-> hProBatchSize = 1;
//     tilingDatafromBin-> wProBatchSize = 1;
//     tilingDatafromBin-> tilingKey = 701;
//     ICPU_SET_TILING_KEY(701UL);

//     ICPU_RUN_KF(max_pool_grad_with_argmax, blockDim, input, grad, argmax, out, workspace, (uint8_t*)(tilingDatafromBin));

//     // Cleanup
//     AscendC::GmFree(input);
//     AscendC::GmFree(out);
//     AscendC::GmFree(grad);
//     AscendC::GmFree(argmax);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
// }