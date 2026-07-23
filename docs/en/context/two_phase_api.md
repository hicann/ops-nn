# Two-stage Interface

When calling an operator API based on the single-operator API execution method, it is usually divided into "two stages", with the following pattern:

```Cpp
aclnnStatus aclxxXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclxxXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

You must first call the first-stage interface aclxxXxxGetWorkspaceSize to calculate how much workspace memory is required during this API call. After obtaining the calculated workspaceSize, apply for NPU memory according to the workspaceSize, and then call the second-stage interface aclxxXxx to execute the calculation.

Here, "aclxx" represents the operator interface prefix, such as aclnn; and "Xxx" represents the corresponding operator type, such as the Add operator.

> Note:
>
> - workspace refers to the temporary memory required by the API to complete the calculation on the AI processor, in addition to input/output.
> - The second-stage interface aclxxXxx(...) cannot be called repeatedly. The following calling method will cause an exception:
>
>   ```Cpp
>   aclxxXxxGetWorkspaceSize(...)
>   aclxxXxx(...)
>   aclxxXxx(...)
>   ```
