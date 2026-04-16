/**
 * @file test_geir_fast_gelu_v2.cpp
 * @brief GE IR (Graph Engine) mode example for FastGeluV2 custom operator.
 *
 * NOTE: FastGeluV2 currently does NOT support GE IR graph mode.
 * Supported calling method: ACLNN single-operator mode only.
 *
 * This file serves as a placeholder to explain why graph mode is not
 * available and provides a reference for future graph-mode integration
 * if needed.
 *
 * For the working ACLNN single-operator example, see:
 *   test_aclnn_fast_gelu_v2.cpp
 *
 * Build & run:
 *   cd examples && bash run.sh --geir
 */

#include <iostream>

// ============================================================================
// Graph mode is not supported for FastGeluV2
// ============================================================================
//
// Reason:
//   FastGeluV2 is a user-defined custom activation function intended for
//   single-operator (ACLNN) invocation only. The following graph-mode
//   calling methods are NOT supported:
//
//   - torch_npu single-operator dispatch
//   - torch.compile graph capture
//   - GE static-shape graph mode
//   - GE dynamic-shape graph mode
//
// To use this operator, call the ACLNN two-phase interface directly:
//
//   1. aclnnFastGeluV2GetWorkspaceSize(x, out, &workspaceSize, &executor)
//   2. aclnnFastGeluV2(workspace, workspaceSize, executor, stream)
//
// See test_aclnn_fast_gelu_v2.cpp for a complete working example.
//
// If graph-mode support is required in the future, the following steps
// would be needed:
//   1. Define GE IR proto for FastGeluV2 (op type, inputs, outputs, attrs)
//   2. Register the operator with GE graph engine via op_proto
//   3. Implement GE graph-mode plugin (framework adapter)
//   4. Validate through graph compilation and execution pipeline
//
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "FastGeluV2 GE IR Graph Mode Example" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "FastGeluV2 does NOT support GE IR graph mode." << std::endl;
    std::cout << std::endl;
    std::cout << "Supported calling method:" << std::endl;
    std::cout << "  - ACLNN single-operator mode (two-phase interface)" << std::endl;
    std::cout << std::endl;
    std::cout << "Please refer to test_aclnn_fast_gelu_v2.cpp for a" << std::endl;
    std::cout << "complete working example of ACLNN invocation." << std::endl;
    std::cout << std::endl;
    std::cout << "Run the ACLNN example:" << std::endl;
    std::cout << "  cd examples && bash run.sh --eager" << std::endl;
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Result: PASS (informational only)" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
