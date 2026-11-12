# Operator List

<!-- md-trans-meta sourceCommit=1c655dcd79f7aa4a4a10d66c09810576779a0c65 translatedAt=2026-11-03T13:03:11Z -->

> Notes:
>
> - **Operator Directory**: The directory name is the lowercase underscore form of the operator name. Each directory contains all deliverables of the operator, including the code implementation, examples, and documentation. For details about the directories, see [Project Directories](../zh/install/dir_structure.md).
>
> - **Hardware Execution Unit**: Most operators run on AI Core, while a few run on AI CPU. By default, the operators mentioned in this project generally refer to AI Core operators. For details about AI Core and AI CPU, see "Concepts and Terminology > Hardware Architecture and Data Processing Principles" in [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC).
> - **Operator API List**: To make it easier to call operators, CANN provides a set of C APIs for executing operators, generally prefixed with aclnn. For the full list of APIs, see [aclnn List](op_api_list.md).
> - **V Version Evolution**: Note that some operators have multiple V versions. Simply select the highest V version. (Operators of a later version are compatible with all capabilities of earlier versions.)

The operator categories and operator lists provided by this project are as follows:

<table><thead>
  <tr>
    <th rowspan="2">Operator Category</th>
    <th rowspan="2">Operator Directory</th>
    <th colspan="2">Operator Implementation</th>
    <th>aclnn Invocation</th>
    <th>Graph Mode Invocation</th>
    <th rowspan="2">Hardware Execution Unit</th>
    <th rowspan="2">Description</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>op_api</th>
    <th>op_graph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>control</td>
    <td><a href="../../control/assert_async/README_en.md">assert_async</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Asserts that the single-element condition tensor is non-zero; prints the message and fails the operator when it is 0, semantically aligned with PyTorch's torch._assert_async.</td>
  </tr>
</tbody>
</table>
