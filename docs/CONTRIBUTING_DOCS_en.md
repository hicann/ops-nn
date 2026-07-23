# Documentation Contribution Guide

We welcome your contributions to the project documentation. High-quality documentation is crucial for project success. This guide will help you efficiently submit documentation that meets the standards.

## Contribution Scope

We welcome any contributions that can improve documentation quality, including but not limited to:

- Correction and Improvement: Fix typos, grammar errors, incorrect code examples, outdated information, or broken links.

- Clarification and Optimization: Make descriptions clearer and easier to understand, optimize sentence structure, and supplement background knowledge.

- Content Supplement: Add usage examples, API documentation, frequently asked questions (FAQ), best practices, or warning descriptions for existing features.

- New Content Creation: Write new chapters or tutorials for newly added features, such as operator README, API introduction documents, and so on. If you have questions, we recommend creating an Issue for discussion first.

- Localization Translation: Help us translate or proofread documents in other languages.

- Style and Navigation: Improve the layout, readability, and navigation structure of the documentation website.

## Contribution Process

1. **Preparation Work**

    - Determine the Task: If there are documentation issues, you can create new Issues. We recommend using the label category `[Documentation|文档反馈]` and providing a detailed description. Based on the existing Issues list, determine the documentation issues to be resolved.
    - Claim the Task: Comment `/assign @yourself` under the corresponding Issue to indicate that you will handle it and avoid duplicate work.

2. **Document Modification**

    - Select Branch: Please download the source code from the master or other Tag branches to the local machine.
    - Follow Format:
      - This project recommends using **Markdown format**.
      - Follow the existing writing style of the project.
      - Put static resources such as images in the corresponding directory. For example, images are generally in the `figures` folder under the docs directory. You can adjust them yourself in special cases.
    - Careful Addition and Deletion: When modifying content, please try to maintain the original line width and line break conventions.

3. **Submit Changes**

    - Atomic Commit: Each commit should focus on an independent modification. For example, "Fix spelling errors in xx guide" and "Update example code in API reference" should be submitted separately.

    - Write Clear Commit Messages:

      ```text
      Brief description (no more than 50 characters)

      If necessary, provide a more detailed description here. Explain the reason and content of the modification, rather than what specifically was changed (the code itself will show).
      Associated Issue: #123
      ```

4. **Initiate Pull Request**

    - Target Branch: Please merge the PR into the target branch of the project.
    - Title and Description:
      - PR Title: Should clearly summarize the modification, for example: `[Docs] Fix configuration example in quick start`.
      - PR Description: Detailed explanation of your changes, motivation, and associated Issues (use Closes #123 or Fixes #456).
    - Preview Check: Please check the document effect in local or online browsing in advance to ensure that the rendering meets expectations.
    - Wait for Review: Maintainers will review and may propose modification suggestions. Please follow up on the discussion in a timely manner.

## Writing Standards

Before developers write project documentation, please be sure to read the following standards first. If you have questions, you are welcome to make suggestions at any time!

- Prerequisites: Please first learn the unified writing standards provided by the CANN organization. For details, see [CANN Document Writing Standards](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md).

  - Document Content Requirements: Introduce the required and optional document deliverables in the project.
  - Directory Structure Standards: Introduce the principles of directory division, such as Chinese and English management.
  - Content Element Standards: Introduce rules for different writing elements, such as file naming, titles, fonts, images, code blocks, links, and so on.

- Precautions:

  In addition to the above writing rules, you also need to pay attention to the following:

  - Tone: Use a friendly, professional, and neutral tone. For beginners, avoid unnecessary jargon.
  - Terminology: Maintain terminology consistency (such as uniformly using "click" instead of "single click"). Please refer to the project terminology table (if available).
  - Code Examples:
    - Ensure that all code examples are runnable and tested.
    - Provide sufficient context and explanation.
    - Indicate the environment or prerequisites required for code running.
  - Punctuation and Format:
    - When mixing Chinese and English, use full-width punctuation. Punctuation marks must conform to the Chinese/English context.
    - Use appropriate hierarchy for titles (#, ##, ###).
    - Use lists and tables to organize complex information.
  - Links: Use descriptive link text, avoid "click here", and ensure that link resources are authentic and reliable.
  - Images:
    - Common Formats: We recommend the png format. Try to keep the style consistent with existing images.
    - Resolution and Clarity: Must be clear and of moderate size. Avoid blurring or excessive compression.
    - File Size: We do not recommend that a single image exceeds 10M.
  - Copyright: For all quoted images, literature, and other resources, please ensure compliance.

## Get Help

If you have any questions during the contribution process:

1. Check Existing Documentation: If there are problems with templates or standards, please first check the existing guides, API documentation, or README of the project.
2. Initiate Discussion: You can create a new Issue or leave a message directly in the relevant Issue or PR.

## Operator README Template

For `experimental` newly contributed operators, the operator README is a required document deliverable. You can refer to the **simple template** provided in this section. You are also supported to expand the content based on this template.

- Document Format: We recommend the Markdown file format. You can use native or Html syntax. Please ensure that all syntax conforms to official standards.
- Document Function: Clearly explain the operator function, implementation principle, parameter specifications, and operator invocation methods.
- Chapter Title: Prioritize using template chapter names (such as Function Description, Parameter Description, and so on). The title hierarchy is ##. If there are special cases, please increase the hierarchy in order. Support chapter customization and expansion. Optional chapters can be presented as needed.
- Content Requirements: For the writing goals and writing standards of each chapter, please refer to the detailed description below. For easy understanding, we will take the [AddExample](../examples/add_example/README.md) operator README as an example.

### Product Support Status

> **Writing Standard**: We recommend the table format. List the supported product models and mark them with √. For product form introduction, see [Ascend Product Form Description](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html).

| Product                                                         | Support Status |
| :----------------------------------------- | :------:|
| Atlas A3 Training Series Products/Atlas A3 Inference Series Products     |    √     |
| Atlas A2 Training Series Products/Atlas A2 Inference Series Products |    √     |

### Function Description

> [!NOTE]
>
> **Writing Goal**: Clarify the operator function, calculation principle, parameter specifications, invocation methods, usage scenarios, and so on.
>
> **Writing Standard**: We recommend the unordered list format, which generally includes the following dimensions
>
> - Operator Function (Required): Please explain the function concisely and clearly in one sentence.
> - Calculation Formula (Optional): For complex functions, you can use formulas to introduce the operator implementation principle or calculation process in different scenarios.
> - Other Dimensions (Optional): Support unordered list expansion. Please customize according to the actual situation, such as calculation examples, flowcharts, and so on.

- Operator Function: Complete tensor addition calculation.
- Calculation Formula:
  $$
  y = x1 + x2
  $$

### Parameter Description

> [!NOTE]
>
> **Writing Goal**: Clarify the meaning, function, specifications, and other information of the parameters defined by the operator.
>
> **Writing Standard**: Use the table format, which generally includes the following dimensions
>
> - Parameter Name: Explain the parameters in the operator definition file. Keep the order consistent, such as `op_host/add_example_def.cpp` or `op_graph/add_example_proto.h`.
> - Input/Output/Attribute: Clarify the parameter positioning. The default is required. If it is optional, it is generally an optional input/optional output/optional attribute.
> - Description: Provide the parameter meaning, function, usage scenario, and other introductions, including the mapping relationship with the above formula variables.
> - Data Type: The data type supported by the parameter. The tensor data type is generally in the `DT_XXX` form. For easy writing, you can omit the `DT_` prefix.
> - Data Format: The data layout mode supported by the parameter. The tensor format is generally in the `FORMAT_xxx` form. For easy writing, you can omit the `FORMAT_` prefix.
> - Other Dimensions (Optional): Support table field expansion. Please customize according to the actual situation, such as shape specifications.

|Parameter Name|Input/Output/Attribute|Description|Data Type|Data Format|
|-----|-----------|----|---------|------|
|x1|Input|Indicates the first tensor of the add_example calculation, that is, `x1` in the formula.|FLOAT, FLOAT16, INT32|ND|
|x2|Input|Indicates the second tensor of the add_example calculation, that is, `x2` in the formula.|The data type is consistent with x1|ND|
|y| Output           | Indicates the result tensor of the add_example calculation, that is, `y` in the formula. |FLOAT, FLOAT16, INT32|ND|

### Constraint Description (Optional)

> [!NOTE]
>
> **Writing Goal**: Clarify the precautions during operator use, such as parameter combination constraints, applicable scenarios, impact on business, operator performance or precision, and so on.
>
> **Writing Standard**: **This chapter is optional**. If there are no constraints, this chapter content does not need to be presented; if there are, please use the unordered list format.

None

### Invocation Description

> [!NOTE]
>
> **Writing Goal**: Provide the operator invocation method. Try to provide sample code that can be directly copied and run for quick verification.
>
> **Writing Standard**: We recommend the table format. If the content is complex, you can use other forms.
>
> - Invocation Method: Support aclnn, graph mode, and other invocation methods. You can also customize. Please provide at least one method.
> - Sample Code: Please provide invocation example code in the `examples` directory of the operator, such as `examples/test_aclnn_add_example.cpp`. The file naming rule is test_${invoke_mode}_${op_name}. ${invoke_mode} indicates the invocation method, and ${op_name} indicates the operator name.
> - Description: Supplementary descriptions for different invocation methods, such as invocation scenarios, invocation principles, compilation and running guidance, and so on. Please customize according to the actual situation.

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn Invocation</td>
    <td><a href="../examples/add_example/examples/test_aclnn_add_example.cpp">test_aclnn_add_example</a></td>
    <td rowspan="2">See <a href="./en/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
  <tr>
    <td>Graph Mode Invocation</td>
    <td><a href="../examples/add_example/examples/test_geir_add_example.cpp">test_geir_add_example</a></td>
  </tr>
</tbody>
</table>

### Reference Resources (Optional)

> [!NOTE]
>
> **Writing Goal**: Provide other supplementary introductions besides operator function, specifications, and invocation, such as operator design documents (Tiling/Kernel design), reference literature, and so on.
>
> **Writing Standard**: **This chapter is optional**. If there are no constraints, this chapter content does not need to be presented; if there are, please use the unordered list format.

None
