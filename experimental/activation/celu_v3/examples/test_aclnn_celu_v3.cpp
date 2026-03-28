/**
 * @file test_aclnn_celu_v3.cpp
 * @brief CeluV3 算子调用示例
 *
 * 演示如何通过 aclnn 接口在 NPU 上调用 CeluV3 算子。
 *
 * 公式: CELU(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
 *
 * 编译方法:
 *   1. 先编译并安装算子包:
 *      cd ops/celu_v3 && bash build.sh --pkg --soc=ascend910b
 *      cd build && ./custom_opp_ubuntu_aarch64.run
 *
 *   2. 编译本示例:
 *      g++ -o test_aclnn_celu_v3 test_aclnn_celu_v3.cpp \
 *          -I${ASCEND_HOME_PATH}/include \
 *          -L${ASCEND_HOME_PATH}/lib64 \
 *          -lascendcl -lnnopbase -lopapi -lacl_op_compiler
 *
 * 运行方法:
 *   ./test_aclnn_celu_v3
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#include "acl/acl.h"
#include "aclnn_celu_v3.h"

#define CHECK_ACL(expr)                                                     \
    do {                                                                    \
        auto _ret = (expr);                                                 \
        if (_ret != ACL_SUCCESS) {                                          \
            std::cerr << "ACL Error: " << #expr << " returned " << _ret    \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            return 1;                                                       \
        }                                                                   \
    } while (0)

int main()
{
    // ========================================================================
    // 1. 参数设置
    // ========================================================================
    constexpr int64_t ELEM_COUNT = 8;
    const int64_t shape[] = {ELEM_COUNT};
    const int64_t strides[] = {1};
    constexpr int64_t ndim = 1;
    constexpr double alpha = 1.0;

    // 输入数据: 包含正值、负值、零
    float hostInput[ELEM_COUNT] = {-3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};

    // ========================================================================
    // 2. ACL 初始化
    // ========================================================================
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // ========================================================================
    // 3. 设备内存分配 & 数据拷贝
    // ========================================================================
    size_t dataBytes = ELEM_COUNT * sizeof(float);

    void *devInput = nullptr;
    void *devOutput = nullptr;
    CHECK_ACL(aclrtMalloc(&devInput, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devOutput, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(devOutput, dataBytes, 0, dataBytes));
    CHECK_ACL(aclrtMemcpy(devInput, dataBytes, hostInput, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ========================================================================
    // 4. 创建 aclTensor
    // ========================================================================
    aclTensor *selfTensor = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                            ACL_FORMAT_ND, shape, ndim, devInput);
    aclTensor *outTensor = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                           ACL_FORMAT_ND, shape, ndim, devOutput);

    // ========================================================================
    // 5. 调用 aclnnCeluV3
    // ========================================================================
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    CHECK_ACL(aclnnCeluV3GetWorkspaceSize(selfTensor, alpha, outTensor,
                                          &workspaceSize, &executor));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    CHECK_ACL(aclnnCeluV3(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // ========================================================================
    // 6. 读取输出 & 打印结果
    // ========================================================================
    float hostOutput[ELEM_COUNT] = {};
    CHECK_ACL(aclrtMemcpy(hostOutput, dataBytes, devOutput, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    std::cout << "CeluV3 Example (alpha=" << alpha << ")" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "  Index |     Input |    Output |  Expected" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    for (int i = 0; i < ELEM_COUNT; ++i) {
        float x = hostInput[i];
        float expected = std::max(0.0f, x) +
                         std::min(0.0f, static_cast<float>(alpha) *
                                        (expf(x / static_cast<float>(alpha)) - 1.0f));
        printf("  %5d | %9.4f | %9.4f | %9.4f\n", i, x, hostOutput[i], expected);
    }
    std::cout << "-------------------------------------------" << std::endl;

    // ========================================================================
    // 7. 资源释放
    // ========================================================================
    aclDestroyTensor(selfTensor);
    aclDestroyTensor(outTensor);
    if (workspace) aclrtFree(workspace);
    aclrtFree(devInput);
    aclrtFree(devOutput);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::cout << "CeluV3 example finished successfully." << std::endl;
    return 0;
}
