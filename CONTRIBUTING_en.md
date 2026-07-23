# Contribution Guide

This project welcomes developers to experience and participate in contributions. Before participating in community contributions, please see [cann-community](https://gitcode.com/cann/community) to understand the code of conduct, sign the CLA agreement, and understand the contribution process of the source code repository.

Developers need to pay attention to the following points when preparing local code and submitting PRs:

1. When submitting a PR, please carefully fill in the business background, purpose, solution, and other information of this PR according to the PR template.
2. If your modification is not a simple bug fix, but involves adding new features, new interfaces, new configuration parameters, or modifying code flow, please be sure to discuss the solution through an Issue first to avoid your code being rejected. If you are not sure whether this modification can be classified as a "simple bug fix", you can also discuss the solution by submitting an Issue.

Developer contribution scenarios mainly include:

## I. Contribute New Operators

The operator development contribution process is as follows:

<!--[Operator Development Contribution Process](./docs/en/figures/operator_development_contribution_process.png "Operator Development Contribution Process Diagram")-->

If you have a brand new operator that you want to design and implement based on NPU, we welcome you to propose your ideas and design solutions in an Issue. The complete contribution process is as follows:

### 1. Create Issue Requirement

Create a new `Requirement|Feature Request` type Issue and clarify the design solution of the new operator. The Issue generally needs to include the following content:

- **Background Information**
- **Value/Function**
- **Design Solution**

Please comment `/assign @yourself` in the submitted Issue to claim this task.

### 2. Requirement Review

The Sig group will assign a Committer to review the Issue you submitted and provide feedback on modification opinions. After completing the modification, please @ the corresponding Committer in the Issue.

If the requirement is accepted, [sig members](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-nn/sig-info.yaml) will assign you a suitable operator classification path (such as `experimental/activation`). Please submit the contributed operator to the corresponding operator classification directory under `experimental`.

### 3. PR Submission

The minimum deliverables for ecosystem operators are as follows:

```text
${op_class}                                          # operator classification
├── ${op_name}                                       # operator name
│   ├── ${op_name}.cpp                               # operator Kernel implementation file
│   └── tests
│   │   ├── test_${op_name}.py                       # operator test file
│   ├── CMakeLists.txt                               # operator compilation configuration file
│   ├── README.md                                    # operator README document
```

PR submission requirements:

- Code Deliverables: Need to provide operator Kernel implementation and operator test files. For the development process, refer to [fast_kernel_launch_example](examples/fast_kernel_launch_example/README.md).
- Document Deliverables: Operator README document is required. Other documents can be provided as needed. For document writing templates and standards, refer to [Document Contribution Guide](docs/CONTRIBUTING_DOCS_en.md).
- Compliance Check:
  - Whether the code conforms to "[C++ Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)"
  - Whether the code compiles successfully
  - Whether Markdown document syntax conforms to standards
- Contribution Directory: Submit to the specified directory `experimental/${op_class}` according to sig member opinions. You can refer to the existing operator file placement rules.
- PR Submission: Submit the target branch PR through the `git` command. Check whether the PR title is clear, whether the PR description is standardized (specify the change content and reason, whether it is associated with the corresponding Issue), and whether the CLA is signed.

If you want to contribute project standard operators, their deliverables and development process are more complex than ecosystem operators, including Kernel, Tiling implementation, and so on. For specific contribution guidance, refer to [Appendix](#appendix).

### 4. CI Gate

Trigger the open-source repository gate by commenting the `compile` instruction, and make modifications according to the CI detection results. Currently, the CI gate includes the following check items:

- Code compilation
- Static check (if codecheck false positives are involved, please submit them to sig members for shielding)
- UT test
- Smoke test

After the gate passes, please @ the assigned Committer in the associated Issue.

### 5. Committer Review

After the Committer reviews, feedback will be provided on review opinions. Please modify according to the opinions, and then @ the assigned Committer after completion.

### 6. Maintainer Merge

After the Committer review passes, mark the `/lgtm` label. The Maintainer will conduct a final review within 1 day. After confirming that there are no problems, the `/approve` label will be marked to merge the PR.

## II. Operator Bug Fix

If you discover certain operator bugs in this project and want to fix them, we welcome you to create a new Issue for feedback and tracking.

You can create a new `Bug-Report|Bug Report` type Issue according to the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to describe the bug, and then enter "/assign" or "/assign @yourself" in the comment box to assign this Issue to you for processing.

## III. Operator Optimization

If you have generalization enhancement/performance optimization ideas for certain operator implementations in this project and want to implement these optimization points, we welcome you to contribute operator optimizations.

You can create a new `Requirement|Feature Request` type Issue according to the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to explain the optimization points and provide your design solution, and then enter "/assign" or "/assign @yourself" in the comment box to assign this Issue to you for tracking optimization.

## IV. Document Correction

If you discover certain operator document description errors in this project, we welcome you to create a new Issue for feedback and correction. For document standards, refer to [Document Contribution Guide](docs/CONTRIBUTING_DOCS_en.md).

You can create a new `Documentation|Documentation Feedback` type Issue according to the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to point out the problems in the corresponding document, and then enter "/assign" or "/assign @yourself" in the comment box to assign this Issue to you to correct the corresponding document description.

## V. Help Solve Others' Issues

If you have suitable solutions for problems encountered by others in the community, we welcome you to comment and communicate in the Issue to help others solve problems and pain points, and jointly optimize usability.

If the corresponding Issue requires code modification, you can enter "/assign" or "/assign @yourself" in the Issue comment box to assign this Issue to you for tracking and assisting in solving the problem.

## Appendix

The project standard operator deliverables are as follows:

```text
${op_class}                                          # operator classification
├── ${op_name}                                       # operator name
│   ├── op_host                                      # operator definition, Tiling related implementation
│   │   ├── ${op_name}_def.cpp                       # operator definition file
│   │   ├── ${op_name}_tiling.cpp                    # operator Tiling implementation file
│   │   └── CMakeLists.txt
│   ├── op_kernel                                    # operator Kernel directory
│   │   ├── ${op_name}.cpp                           # Kernel entry file, containing main function and scheduling logic
│   │   ├── ${op_name}.h                             # Kernel implementation file, defining Kernel header file, containing function description, structure definition, logic implementation
│   │   ├── ${op_name}_tiling_data.h                 # TilingData file, storing Tiling strategy related configuration information
│   │   └── ${op_name}_tiling_key.h                  # TilingKey file, defining the key of Tiling strategy, identifying different division methods
│   ├── CMakeLists.txt                               # operator compilation configuration file, keep the original file
│   └── README.md                                    # operator description document
│   └── tests                                        # operator test file
│   │   ├── ut                                       # operator UT test file
```

PR submission requirements:

- Code Deliverables: Need to provide op_host operator Tiling implementation, op_kernel operator Kernel implementation, operator UT test files. For the development process, please refer to [Operator Development Guide](docs/en/develop/aicore_develop_guide.md).
- Document Deliverables: Operator README document is required. Other documents can be provided as needed. For document writing templates and standards, please see [Document Contribution Guide](docs/CONTRIBUTING_DOCS_en.md).
- Compliance Check:
  - Whether the code conforms to "[C++ Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)", whether it conforms to standard operator basic programming standards
  - Whether the code compiles successfully
  - Whether Markdown document syntax conforms to standards
- Contribution Directory: Submit to the specified directory `experimental/${op_class}` according to sig member opinions. You can refer to the existing operator file placement rules.
- PR Submission: Submit the target branch PR through the `git` command. Check whether the PR title is clear, whether the PR description is standardized (specify the change content and reason, whether it is associated with the corresponding Issue), and whether the CLA is signed.
