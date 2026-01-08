#include "opdev/op_executor.h"
#include "opdev/aicpu/aicpu_task.h"

aclnnStatus CreatAiCoreKernelLauncher([[maybe_unused]] const char *l0Name, uint32_t opType,
    aclOpExecutor *executor, op::OpArgContext *args)
{
    return ACLNN_SUCCESS;
}

namespace op::internal {
aclnnStatus CreatAicpuKernelLauncher(uint32_t opType, op::internal::AicpuTaskSpace &space,
    aclOpExecutor *executor, const op::FVector<std::string> &attrNames, op::OpArgContext *args)
{
    return ACLNN_SUCCESS;
}
}