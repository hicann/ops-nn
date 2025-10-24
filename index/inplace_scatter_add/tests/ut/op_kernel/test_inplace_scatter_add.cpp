#include <unistd.h>

#include <array>
#include <vector>

#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"

#include <iostream>
#include <string>

#include "data_utils.h"
#include "inplace_scatter_add_tiling_def.h"
#include "string.h"
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

using namespace std;

extern "C" __global__ __aicore__ void inplace_scatter_add(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                                          GM_ADDR var_ref, GM_ADDR workspace, GM_ADDR tiling);


class inplace_scatter_add_test : public testing::Test {
   protected:
    static void SetUpTestCase() { cout << "inplace_scatter_add_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "inplace_scatter_add_test TearDown\n" << endl; }
};

TEST_F(inplace_scatter_add_test, inplace_scatter_add_kernel_test_000) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    // M =2 N = 32 K = 2
    size_t shape_var = 2 * 32 * sizeof(float);
    size_t shape_indices = 2 * sizeof(int32_t);
    size_t shape_updates = 2 * 32 * sizeof(float);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(InplaceScatterAddTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *varGM = (uint8_t *)AscendC::GmAlloc(shape_var);
    uint8_t *indicesGM = (uint8_t *)AscendC::GmAlloc(shape_indices);
    uint8_t *updatesGM = (uint8_t *)AscendC::GmAlloc(shape_updates);
    
    memset(varGM, 0, shape_var);
    memset(indicesGM, 0, shape_indices);
    memset(updatesGM, 0, shape_updates);

    InplaceScatterAddTilingData *tilingData = reinterpret_cast<InplaceScatterAddTilingData*>(tiling);
    tilingData->M = 2;
    tilingData->N = 32;
    tilingData->K = 2;
    tilingData->NAligned = 32;
    tilingData->frontCoreNum = 0;
    tilingData->tailCoreNum = 2;
    tilingData->frontCoreIndicesNum = 0;
    tilingData->tailCoreIndicesNum = 1;
    tilingData->ubSize = 49000;

    ICPU_SET_TILING_KEY(0b000);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(inplace_scatter_add, 2, varGM, indicesGM, updatesGM, varGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)varGM);
    AscendC::GmFree((void*)indicesGM);
    AscendC::GmFree((void*)updatesGM);
}

TEST_F(inplace_scatter_add_test, inplace_scatter_add_kernel_test_100) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    // M =2 N = 32 K = 2
    size_t shape_var = 2 * 32 * sizeof(int32_t);
    size_t shape_indices = 2 * sizeof(int32_t);
    size_t shape_updates = 2 * 32 * sizeof(int32_t);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(InplaceScatterAddTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *varGM = (uint8_t *)AscendC::GmAlloc(shape_var);
    uint8_t *indicesGM = (uint8_t *)AscendC::GmAlloc(shape_indices);
    uint8_t *updatesGM = (uint8_t *)AscendC::GmAlloc(shape_updates);
    
    memset(varGM, 0, shape_var);
    memset(indicesGM, 0, shape_indices);
    memset(updatesGM, 0, shape_updates);

    InplaceScatterAddTilingData *tilingData = reinterpret_cast<InplaceScatterAddTilingData*>(tiling);
    tilingData->M = 2;
    tilingData->N = 32;
    tilingData->K = 2;
    tilingData->NAligned = 32;
    tilingData->frontCoreNum = 0;
    tilingData->tailCoreNum = 2;
    tilingData->frontCoreIndicesNum = 0;
    tilingData->tailCoreIndicesNum = 1;
    tilingData->ubSize = 49000;

    ICPU_SET_TILING_KEY(0b100);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(inplace_scatter_add, 2, varGM, indicesGM, updatesGM, varGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)varGM);
    AscendC::GmFree((void*)indicesGM);
    AscendC::GmFree((void*)updatesGM);
}

TEST_F(inplace_scatter_add_test, inplace_scatter_add_kernel_test_010) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    // M =2 N = 32 K = 2
    size_t shape_var = 2 * 32 * sizeof(float);
    size_t shape_indices = 2 * sizeof(int64_t);
    size_t shape_updates = 2 * 32 * sizeof(float);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(InplaceScatterAddTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *varGM = (uint8_t *)AscendC::GmAlloc(shape_var);
    uint8_t *indicesGM = (uint8_t *)AscendC::GmAlloc(shape_indices);
    uint8_t *updatesGM = (uint8_t *)AscendC::GmAlloc(shape_updates);
    
    memset(varGM, 0, shape_var);
    memset(indicesGM, 0, shape_indices);
    memset(updatesGM, 0, shape_updates);

    InplaceScatterAddTilingData *tilingData = reinterpret_cast<InplaceScatterAddTilingData*>(tiling);
    tilingData->M = 2;
    tilingData->N = 32;
    tilingData->K = 2;
    tilingData->NAligned = 32;
    tilingData->frontCoreNum = 0;
    tilingData->tailCoreNum = 2;
    tilingData->frontCoreIndicesNum = 0;
    tilingData->tailCoreIndicesNum = 1;
    tilingData->ubSize = 49000;

    ICPU_SET_TILING_KEY(0b010);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(inplace_scatter_add, 2, varGM, indicesGM, updatesGM, varGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)varGM);
    AscendC::GmFree((void*)indicesGM);
    AscendC::GmFree((void*)updatesGM);
}

TEST_F(inplace_scatter_add_test, inplace_scatter_add_kernel_test_001) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    // M =2 N = 256 K = 2
    size_t shape_var = 2 * 256 * sizeof(float);
    size_t shape_indices = 2 * sizeof(int32_t);
    size_t shape_updates = 2 * 256 * sizeof(float);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(InplaceScatterAddTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *varGM = (uint8_t *)AscendC::GmAlloc(shape_var);
    uint8_t *indicesGM = (uint8_t *)AscendC::GmAlloc(shape_indices);
    uint8_t *updatesGM = (uint8_t *)AscendC::GmAlloc(shape_updates);
    
    memset(varGM, 0, shape_var);
    memset(indicesGM, 0, shape_indices);
    memset(updatesGM, 0, shape_updates);

    InplaceScatterAddTilingData *tilingData = reinterpret_cast<InplaceScatterAddTilingData*>(tiling);
    tilingData->M = 2;
    tilingData->N = 256;
    tilingData->K = 2;
    tilingData->NAligned = 256;
    tilingData->frontCoreNum = 0;
    tilingData->tailCoreNum = 2;
    tilingData->frontCoreIndicesNum = 0;
    tilingData->tailCoreIndicesNum = 1;
    tilingData->ubSize = 49000;

    ICPU_SET_TILING_KEY(0b001);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(inplace_scatter_add, 2, varGM, indicesGM, updatesGM, varGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)varGM);
    AscendC::GmFree((void*)indicesGM);
    AscendC::GmFree((void*)updatesGM);
}

TEST_F(inplace_scatter_add_test, inplace_scatter_add_kernel_test_110) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    // M =2 N = 32 K = 2
    size_t shape_var = 2 * 32 * sizeof(int32_t);
    size_t shape_indices = 2 * sizeof(int64_t);
    size_t shape_updates = 2 * 32 * sizeof(int32_t);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(InplaceScatterAddTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *varGM = (uint8_t *)AscendC::GmAlloc(shape_var);
    uint8_t *indicesGM = (uint8_t *)AscendC::GmAlloc(shape_indices);
    uint8_t *updatesGM = (uint8_t *)AscendC::GmAlloc(shape_updates);
    
    memset(varGM, 0, shape_var);
    memset(indicesGM, 0, shape_indices);
    memset(updatesGM, 0, shape_updates);

    InplaceScatterAddTilingData *tilingData = reinterpret_cast<InplaceScatterAddTilingData*>(tiling);
    tilingData->M = 2;
    tilingData->N = 32;
    tilingData->K = 2;
    tilingData->NAligned = 32;
    tilingData->frontCoreNum = 0;
    tilingData->tailCoreNum = 2;
    tilingData->frontCoreIndicesNum = 0;
    tilingData->tailCoreIndicesNum = 1;
    tilingData->ubSize = 49000;

    ICPU_SET_TILING_KEY(0b110);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(inplace_scatter_add, 2, varGM, indicesGM, updatesGM, varGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)varGM);
    AscendC::GmFree((void*)indicesGM);
    AscendC::GmFree((void*)updatesGM);
}

TEST_F(inplace_scatter_add_test, inplace_scatter_add_kernel_test_101) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    // M =2 N = 256 K = 2
    size_t shape_var = 2 * 256 * sizeof(int32_t);
    size_t shape_indices = 2 * sizeof(int32_t);
    size_t shape_updates = 2 * 256 * sizeof(int32_t);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(InplaceScatterAddTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *varGM = (uint8_t *)AscendC::GmAlloc(shape_var);
    uint8_t *indicesGM = (uint8_t *)AscendC::GmAlloc(shape_indices);
    uint8_t *updatesGM = (uint8_t *)AscendC::GmAlloc(shape_updates);
    
    memset(varGM, 0, shape_var);
    memset(indicesGM, 0, shape_indices);
    memset(updatesGM, 0, shape_updates);

    InplaceScatterAddTilingData *tilingData = reinterpret_cast<InplaceScatterAddTilingData*>(tiling);
    tilingData->M = 2;
    tilingData->N = 256;
    tilingData->K = 2;
    tilingData->NAligned = 256;
    tilingData->frontCoreNum = 0;
    tilingData->tailCoreNum = 2;
    tilingData->frontCoreIndicesNum = 0;
    tilingData->tailCoreIndicesNum = 1;
    tilingData->ubSize = 49000;

    ICPU_SET_TILING_KEY(0b101);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(inplace_scatter_add, 2, varGM, indicesGM, updatesGM, varGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)varGM);
    AscendC::GmFree((void*)indicesGM);
    AscendC::GmFree((void*)updatesGM);
}

TEST_F(inplace_scatter_add_test, inplace_scatter_add_kernel_test_011) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    // M =2 N = 256 K = 2
    size_t shape_var = 2 * 256 * sizeof(float);
    size_t shape_indices = 2 * sizeof(int64_t);
    size_t shape_updates = 2 * 256 * sizeof(float);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(InplaceScatterAddTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *varGM = (uint8_t *)AscendC::GmAlloc(shape_var);
    uint8_t *indicesGM = (uint8_t *)AscendC::GmAlloc(shape_indices);
    uint8_t *updatesGM = (uint8_t *)AscendC::GmAlloc(shape_updates);
    
    memset(varGM, 0, shape_var);
    memset(indicesGM, 0, shape_indices);
    memset(updatesGM, 0, shape_updates);

    InplaceScatterAddTilingData *tilingData = reinterpret_cast<InplaceScatterAddTilingData*>(tiling);
    tilingData->M = 2;
    tilingData->N = 256;
    tilingData->K = 2;
    tilingData->NAligned = 256;
    tilingData->frontCoreNum = 0;
    tilingData->tailCoreNum = 2;
    tilingData->frontCoreIndicesNum = 0;
    tilingData->tailCoreIndicesNum = 1;
    tilingData->ubSize = 49000;

    ICPU_SET_TILING_KEY(0b011);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(inplace_scatter_add, 2, varGM, indicesGM, updatesGM, varGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)varGM);
    AscendC::GmFree((void*)indicesGM);
    AscendC::GmFree((void*)updatesGM);
}

TEST_F(inplace_scatter_add_test, inplace_scatter_add_kernel_test_111) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    // M =2 N = 256 K = 2
    size_t shape_var = 2 * 256 * sizeof(int32_t);
    size_t shape_indices = 2 * sizeof(int64_t);
    size_t shape_updates = 2 * 256 * sizeof(int32_t);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(InplaceScatterAddTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *varGM = (uint8_t *)AscendC::GmAlloc(shape_var);
    uint8_t *indicesGM = (uint8_t *)AscendC::GmAlloc(shape_indices);
    uint8_t *updatesGM = (uint8_t *)AscendC::GmAlloc(shape_updates);
    
    memset(varGM, 0, shape_var);
    memset(indicesGM, 0, shape_indices);
    memset(updatesGM, 0, shape_updates);

    InplaceScatterAddTilingData *tilingData = reinterpret_cast<InplaceScatterAddTilingData*>(tiling);
    tilingData->M = 2;
    tilingData->N = 256;
    tilingData->K = 2;
    tilingData->NAligned = 256;
    tilingData->frontCoreNum = 0;
    tilingData->tailCoreNum = 2;
    tilingData->frontCoreIndicesNum = 0;
    tilingData->tailCoreIndicesNum = 1;
    tilingData->ubSize = 49000;

    ICPU_SET_TILING_KEY(0b111);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(inplace_scatter_add, 2, varGM, indicesGM, updatesGM, varGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)varGM);
    AscendC::GmFree((void*)indicesGM);
    AscendC::GmFree((void*)updatesGM);
}