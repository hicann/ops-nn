# Contribution Guide

This project welcomes all developers to try it out and participate in contributions. Before you participate in community contributions, see [cann-community](https://gitcode.com/cann/community) to learn the code of conduct, sign the CLA agreement, and become familiar with the contribution process of the source code repository.

## Preparation

When you prepare local code and submit PRs, pay attention to the following items:

1. When you submit a PR, carefully fill in information such as the business background, purpose, and solution of this PR according to the PR template.
2. If your modification involves adding features, adding APIs, adding configuration parameters, or changing code flows (not a simple bug fix), be sure to discuss the solution through an Issue first to prevent your code from being rejected. If you are not sure whether your modification is a "simple bug fix", submit an Issue for discussion.

## Contribution Scenarios

Developer contribution scenarios mainly include:

- [Contribute a new operator](#1-contribute-a-new-operator)
- [Operator bug fix](#2-operator-bug-fix)
- [Operator optimization](#3-operator-optimization)
- [Documentation correction](#4-documentation-correction)
- [Help others resolve Issues](#5-help-others-resolve-issues)

---

## 1. Contribute a New Operator

The contribution process for operator development is as follows:

![Operator development contribution flow](./docs/en/figures/op_dev_and_contribute_flow.png "Operator development contribution flowchart")

If you want to design and implement a brand-new operator based on the NPU, feel free to share your ideas and design solutions in an Issue. The complete contribution process is as follows:

### 1. Create a Requirement Issue

Create a `Requirement|需求建议` type Issue and describe the design solution of the new operator. An Issue generally needs to include the following content:

- **Background information**
- **Value/function**
- **Design solution**

For detailed operations, see: [Issue operation guide](https://gitcode.com/cann/community/blob/master/contributor/issue-operation.md).

### 2. Requirement Review

After you create a requirement Issue, you need to register a SIG topic and attend the Ops-nn SIG regular meeting at the specified time for the review.

- [Meeting time](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-nn)
- [Topic application](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-nn)
- [Meeting link](https://meeting.osinfra.cn/cann?sig=sig-ops-nn)

**Handling urgent requirements:**

If a requirement is urgent, you can apply for an ad-hoc SIG meeting:

1. Fill in the topic application: [Topic application](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-nn)
2. Send an email to the [maintainer](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-nn/README.md). Recommended email subject: Apply for an ad-hoc SIG topic, plus the topic content

**Requirement acceptance:**

If your requirement is accepted, [SIG members](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-nn/sig-info.yaml) will assign you a suitable operator classification path (for example, `experimental/activation`). Submit your contributed operator to the corresponding operator classification directory under `experimental`.

### 3. PR Submission

#### Minimum deliverables for an ecosystem operator

```text
${op_class}                    # Operator classification
├── ${op_name}                 # Operator name
│   ├── ${op_name}.cpp         # Operator Kernel implementation file
│   ├── tests                  # Test directory
│   │   └── test_${op_name}.py # Operator test file
│   ├── CMakeLists.txt         # Operator compilation configuration file
│   └── README.md              # Operator README document
```

#### Notes on PR submission

**Preparation before submission:**

- It is recommended that you submit the PR after the requirement review passes.
- Before you submit a PR, complete development environment preparation, carefully learn the project-specific development standards and copyright statement requirements (if you reference open-source code snippets, see the snippet reference guide), and make sure your contribution complies with the project standards. Sign the CLA. For detailed operations, see: [PR operation guide](https://gitcode.com/cann/community/blob/master/contributor/pull_request_operation.md).

**Deliverable requirements:**

| Type | Requirement | Reference |
|------|------|---------|
| Code deliverables | Provide the operator Kernel implementation and operator test files | [fast_kernel_launch_example](examples/fast_kernel_launch_example/README_en.md) |
| Documentation deliverables | The operator README document is mandatory; provide other documents as needed | [Documentation contribution guide](docs/CONTRIBUTING_DOCS.md) |
| Accuracy requirements | Newly contributed operators must meet the accuracy standards | [Open-source accuracy standard for ecosystem operators](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |

**Compliance check:**

- [ ] Does the code comply with the [C++ coding standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)?
- [ ] Does the code compile successfully?
- [ ] Does the Markdown document syntax comply with the standards?

**Submission standards:**

- **Contribution directory**: Submit to the specified directory `experimental/${op_class}` as advised by SIG members. You can refer to the placement rules of existing operator files.
- **PR submission**: Use the `git` command to submit a PR to the target branch. Check whether the PR title is clear and whether the PR description is standard (state the changed content and reasons, and link the related Issue).

> **NOTE**: If you want to contribute a project-standard operator, its deliverables and development process are more complex than those of an ecosystem operator, including the Kernel and Tiling implementations. For the detailed contribution guide, see [Appendix](#appendix).

### 4. CI Gating

Trigger the open-source repository gate by commenting with the `compile` command, and make modifications according to the CI check results. Currently, the CI gate includes the following checks:

- Code compilation
- Static check (if codecheck reports false positives, submit them to SIG members for suppression)
- UT tests
- Smoke tests

After the gate passes, @ Committer in the related Issue.

### 5. Committer Review

The Committer provides review comments after the review. Modify the code according to the comments, and then @ Committer again.

### 6. Maintainer Merge

After the Committer review passes, the Committer adds the `/lgtm` label. The Maintainer performs the final review and, after confirming that no problems exist, adds the `/approve` label to merge the PR.

---

## 2. Operator Bug Fix

If you find bugs in some operators in this project and want to fix them, feel free to create an Issue for feedback and tracking.

Follow the [Issue submission/Issue handling task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Bug-Report|缺陷反馈` type Issue and describe the bug.

---

## 3. Operator Optimization

If you have ideas for generalization enhancement or performance optimization of some operator implementations in this project and want to implement these optimization points, feel free to contribute operator optimizations.

Follow the [Issue submission/Issue handling task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Requirement|需求建议` type Issue, describe the optimization points, and provide your design solution.

---

## 4. Documentation Correction

If you find description errors in some operator documents in this project, feel free to create an Issue for feedback and fixes. For documentation standards, see the [Documentation contribution guide](docs/CONTRIBUTING_DOCS.md).

Follow the [Issue submission/Issue handling task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Documentation|文档反馈` type Issue and point out the problem in the corresponding document.

---

## 5. Help Others Resolve Issues

If you have a suitable solution to a problem that someone else in the community encounters, feel free to comment in the Issue to exchange ideas, help others solve problems and pain points, and jointly improve usability.

If the corresponding Issue requires code modification, enter `/assign` or `/assign @yourself` in the Issue comment box to assign the Issue to yourself, and then track and help resolve the problem.

---

## Appendix

### Project-Standard Operator Deliverables

```text
${op_class}                           # Operator classification
├── ${op_name}                        # Operator name
│   ├── op_host                       # Operator definition and Tiling implementation
│   │   ├── ${op_name}_def.cpp        # Operator definition file
│   │   ├── ${op_name}_tiling.cpp     # Operator Tiling implementation file
│   │   ├── ${op_name}_tiling_${sub_case}.cpp # Optional, Tiling implementation for a sub-scenario. ${sub_case} indicates the sub-scenario (for example, arch35)
│   │   ├── ${op_name}_tiling_${sub_case}.h # Optional, header file of the Tiling implementation for a sub-scenario
│   │   └── CMakeLists.txt
│   ├── op_kernel                     # Operator Kernel directory
│   │   ├── ${op_name}.cpp            # Kernel entry file, which contains the main function and scheduling logic
│   │   ├── ${op_name}.h              # Kernel implementation file, which defines the Kernel header file and contains function descriptions, structure definitions, and logic implementation
│   │   ├── ${op_name}_tiling_data.h  # TilingData file, which stores configuration information related to the Tiling strategy
│   │   └── ${op_name}_tiling_key.h   # TilingKey file, which defines the key of the Tiling strategy and identifies different partitioning methods
│   ├── CMakeLists.txt                # Operator compilation configuration file. Keep the original file.
│   ├── README.md                     # Operator description document
│   └── tests                         # Operator test files
│       └── ut                        # Operator UT test files
```

For the complete operator directory structure (including all optional deliverables), see [Project directory structure](docs/zh/install/dir_structure.md#项目目录).

> **NOTE**: For Tiling implementation files under the op_host directory that participate in compilation, the file name must contain the `_tiling` identifier (such as `${op_name}_tiling.cpp` and `${op_name}_tiling_${sub_case}.cpp`). Otherwise, the compilation system cannot recognize them. When you split Tiling implementations by sub-scenario (such as a specific architecture arch35), follow this naming rule.

### PR Submission Requirements

**Deliverable requirements:**

| Type | Requirement | Reference |
|------|------|---------|
| Code deliverables | Provide the operator Tiling implementation under op_host, the operator Kernel implementation under op_kernel, and operator UT test files | [Operator development guide](docs/zh/develop/aicore_develop_guide.md) |
| Documentation deliverables | The operator README document is mandatory; provide other documents as needed | [Documentation contribution guide](docs/CONTRIBUTING_DOCS.md) |

**Compliance check:**

- [ ] Does the code comply with the [C++ coding standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)?
- [ ] Does the code comply with the basic programming standards for standard operators?
- [ ] Does the code compile successfully?
- [ ] Does the Markdown document syntax comply with the standards?

**Submission standards:**

- **Contribution directory**: Submit to the specified directory `experimental/${op_class}` as advised by SIG members. You can refer to the placement rules of existing operator files.
- **PR submission**: Use the `git` command to submit a PR to the target branch. Check whether the PR title is clear, whether the PR description is standard (state the changed content and reasons, and link the related Issue), and whether the CLA is signed.
