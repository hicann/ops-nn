#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "dequant_bias_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;


extern "C" __global__ __aicore__ void dequant_bias(GM_ADDR x, GM_ADDR weight_scale, GM_ADDR activate_scale,
                                                   GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
class dequant_bias_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "dequant_bias SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "dequant_bias TearDown\n" << endl;
  }
};

TEST_F(dequant_bias_test, test_dequant_bias_0)
{
    size_t M = 40;
    size_t N = 256;
    
    size_t xFileSize = M * N * sizeof(int32_t);

    size_t weightScaleFileSize = N * sizeof(int32_t);
    size_t activateScaleFileSize = M * sizeof(int32_t);
    size_t biasFileSize = N* sizeof(half);

    size_t yFileSize = M * N * sizeof(half);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);
    uint8_t *weight_scale = (uint8_t *)AscendC::GmAlloc(weightScaleFileSize);
    uint8_t *activate_scale = (uint8_t *)AscendC::GmAlloc(activateScaleFileSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 10111;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantBiasTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_bias, blockDim, x, weight_scale, activate_scale, bias,
                y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weight_scale);
    AscendC::GmFree((void *)activate_scale);
    AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_bias_test, test_dequant_bias_1)
{
    size_t M = 40;
    size_t N = 256;
    
    size_t xFileSize = M * N * sizeof(int32_t);

    size_t weightScaleFileSize = N * sizeof(int32_t);
    size_t activateScaleFileSize = M * sizeof(int32_t);
    size_t biasFileSize = N* sizeof(half);

    size_t yFileSize = M * N * sizeof(half);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);
    uint8_t *weight_scale = (uint8_t *)AscendC::GmAlloc(weightScaleFileSize);
    uint8_t *activate_scale = (uint8_t *)AscendC::GmAlloc(activateScaleFileSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 10110;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantBiasTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_bias, blockDim, x, weight_scale, activate_scale, bias,
                y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weight_scale);
    AscendC::GmFree((void *)activate_scale);
    AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_bias_test, test_dequant_bias_2)
{
    size_t M = 40;
    size_t N = 256;
    
    size_t xFileSize = M * N * sizeof(int32_t);

    size_t weightScaleFileSize = N * sizeof(int32_t);
    size_t activateScaleFileSize = M * sizeof(int32_t);
    size_t biasFileSize = N* sizeof(half);

    size_t yFileSize = M * N * sizeof(half);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);
    uint8_t *weight_scale = (uint8_t *)AscendC::GmAlloc(weightScaleFileSize);
    uint8_t *activate_scale = (uint8_t *)AscendC::GmAlloc(activateScaleFileSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 10112;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantBiasTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_bias, blockDim, x, weight_scale, activate_scale, bias,
                y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weight_scale);
    AscendC::GmFree((void *)activate_scale);
    AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_bias_test, test_dequant_bias_3)
{
    size_t M = 40;
    size_t N = 256;
    
    size_t xFileSize = M * N * sizeof(int32_t);

    size_t weightScaleFileSize = N * sizeof(int32_t);
    size_t activateScaleFileSize = M * sizeof(int32_t);
    size_t biasFileSize = N* sizeof(half);

    size_t yFileSize = M * N * sizeof(half);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);
    uint8_t *weight_scale = (uint8_t *)AscendC::GmAlloc(weightScaleFileSize);
    uint8_t *activate_scale = (uint8_t *)AscendC::GmAlloc(activateScaleFileSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 10113;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantBiasTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_bias, blockDim, x, weight_scale, activate_scale, bias,
                y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weight_scale);
    AscendC::GmFree((void *)activate_scale);
    AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_bias_test, test_dequant_bias_4)
{
    size_t M = 40;
    size_t N = 8193;
    
    size_t xFileSize = M * N * sizeof(int32_t);

    size_t weightScaleFileSize = N * sizeof(int32_t);
    size_t activateScaleFileSize = M * sizeof(int32_t);
    size_t biasFileSize = N* sizeof(half);

    size_t yFileSize = M * N * sizeof(half);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);
    uint8_t *weight_scale = (uint8_t *)AscendC::GmAlloc(weightScaleFileSize);
    uint8_t *activate_scale = (uint8_t *)AscendC::GmAlloc(activateScaleFileSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 10111;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantBiasTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    DequantBiasTilingData* tilingDatafromBin = reinterpret_cast<DequantBiasTilingData*>(tiling);
    tilingDatafromBin->N = 8193;
    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_bias, blockDim, x, weight_scale, activate_scale, bias,
                y, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weight_scale);
    AscendC::GmFree((void *)activate_scale);
    AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_bias_test, test_dequant_bias_5)
{
    size_t M = 40;
    size_t N = 8193;
    
    size_t xFileSize = M * N * sizeof(int32_t);

    size_t weightScaleFileSize = N * sizeof(int32_t);
    size_t activateScaleFileSize = M * sizeof(int32_t);
    size_t biasFileSize = N* sizeof(half);

    size_t yFileSize = M * N * sizeof(half);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);
    uint8_t *weight_scale = (uint8_t *)AscendC::GmAlloc(weightScaleFileSize);
    uint8_t *activate_scale = (uint8_t *)AscendC::GmAlloc(activateScaleFileSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 10110;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantBiasTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    DequantBiasTilingData* tilingDatafromBin = reinterpret_cast<DequantBiasTilingData*>(tiling);
    tilingDatafromBin->N = 8193;
    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_bias, blockDim, x, weight_scale, activate_scale, bias,
                y, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weight_scale);
    AscendC::GmFree((void *)activate_scale);
    AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_bias_test, test_dequant_bias_6)
{
    size_t M = 40;
    size_t N = 8193;
    
    size_t xFileSize = M * N * sizeof(int32_t);

    size_t weightScaleFileSize = N * sizeof(int32_t);
    size_t activateScaleFileSize = M * sizeof(int32_t);
    size_t biasFileSize = N* sizeof(half);

    size_t yFileSize = M * N * sizeof(half);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);
    uint8_t *weight_scale = (uint8_t *)AscendC::GmAlloc(weightScaleFileSize);
    uint8_t *activate_scale = (uint8_t *)AscendC::GmAlloc(activateScaleFileSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 10112;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantBiasTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    DequantBiasTilingData* tilingDatafromBin = reinterpret_cast<DequantBiasTilingData*>(tiling);
    tilingDatafromBin->N = 8193;
    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_bias, blockDim, x, weight_scale, activate_scale, bias,
                y, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weight_scale);
    AscendC::GmFree((void *)activate_scale);
    AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_bias_test, test_dequant_bias_7)
{
    size_t M = 40;
    size_t N = 8193;
    
    size_t xFileSize = M * N * sizeof(int32_t);

    size_t weightScaleFileSize = N * sizeof(int32_t);
    size_t activateScaleFileSize = M * sizeof(int32_t);
    size_t biasFileSize = N* sizeof(half);

    size_t yFileSize = M * N * sizeof(half);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);
    uint8_t *weight_scale = (uint8_t *)AscendC::GmAlloc(weightScaleFileSize);
    uint8_t *activate_scale = (uint8_t *)AscendC::GmAlloc(activateScaleFileSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 10113;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantBiasTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    DequantBiasTilingData* tilingDatafromBin = reinterpret_cast<DequantBiasTilingData*>(tiling);
    tilingDatafromBin->N = 8193;
    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_bias, blockDim, x, weight_scale, activate_scale, bias,
                y, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weight_scale);
    AscendC::GmFree((void *)activate_scale);
    AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}
