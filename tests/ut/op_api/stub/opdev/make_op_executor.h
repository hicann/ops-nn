#ifndef __MAKE_OP_EXECUTOR_H__
#define __MAKE_OP_EXECUTOR_H__
#include <set>
#include "opdev/platform.h"
#include "opdev/op_arg_def.h"
#include "opdev/op_executor.h"

using namespace std;
#define CREATE_EXECUTOR() UniqueExecutor(__func__)

#define INFER_SHAPE(KERNEL_NAME, op_args...) ACLNN_SUCCESS


#define ADD_TO_LAUNCHER_LIST_AICORE(KERNEL_NAME, op_args...)   ACLNN_SUCCESS

#define ADD_TO_LAUNCHER_LIST_DSA(KERNEL_NAME, op_args...) \
  do { \
    op::OpArgContext *opArgCtx = GetOpArgContext(op_args); \
    CreatDSAKernelLauncher(#KERNEL_NAME, KERNEL_NAME##OpTypeId(), KERNEL_NAME##TaskType,\
                            executor, opArgCtx); \
  } while(0)

#endif
