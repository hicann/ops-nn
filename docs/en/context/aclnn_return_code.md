# aclnn Return Codes

When calling aclnn APIs, common interface return codes are shown in [Table 1](#table1).
For abnormal status code values, you can use the aclGetRecentErrMsg interface (refer to [ACL API (C)](https://www.hiascend.com/document/detail/en/canncommercial/latest/API/appdevgapi/aclcppdevg_03_0004.html)) to obtain exception information. You can troubleshoot the problem based on the error message or contact technical support.

**Table 1**  Return Status Codes

<a name="table1"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="30.543054305430545%" id="mcps1.2.4.1.1"><p>Status Code Name</p>
</th>
<th class="cellrowborder" valign="top" width="15.971597159715973%" id="mcps1.2.4.1.2"><p>Status Code Value</p>
</th>
<th class="cellrowborder" valign="top" width="53.48534853485349%" id="mcps1.2.4.1.3"><p>Status Code Description</p>
</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p>ACLNN_SUCCESS</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p>0</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p>Success.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_PARAM_NULLPTR</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p>161001</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p>Parameter validation error, illegal nullptr exists in parameters.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_PARAM_INVALID</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p>161002</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p>Parameter validation error, such as two input data types not satisfying the input type promotion relationship.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_RUNTIME_ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p>361001</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p>API internally calls npu runtime interface abnormally.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_XXX</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p>561xxx</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p>API internal exception occurred.</p>

</td>
</tr>
</tbody>
</table>

For more information about ACLNN_ERR_INNER_XXX status codes, see [Table 2](#table2).

**Table 2**  Exception Status Codes

<a name="table2"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="30.183018301830185%" id="mcps1.2.4.1.1"><p>Status Code Name</p>
</th>
<th class="cellrowborder" valign="top" width="16.521652165216523%" id="mcps1.2.4.1.2"><p>Status Code Value</p>
</th>
<th class="cellrowborder" valign="top" width="53.295329532953296%" id="mcps1.2.4.1.3"><p>Status Code Description</p>
</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561000</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: API internal exception occurred.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_INFERSHAPE_ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561001</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: API internal output shape deduction error occurred.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_TILING_ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561002</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: API internal tiling for npu kernel exception occurred.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_FIND_KERNEL_ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561003</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: API internal npu kernel lookup exception (possibly because operator binary package is not installed).</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_CREATE_EXECUTOR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561101</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: API internal aclOpExecutor creation failed (possibly due to operating system exception).</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561102</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: API internal uniqueExecutor ReleaseTo not called.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_NULLPTR</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561103</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: aclnn API internal exception occurred, nullptr exception appeared.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561104</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: aclnn API internal exception occurred, operator attribute count exception.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_KEY_CONFILICT</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561105</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: aclnn API internal exception occurred, operator kernel matching hash key conflict.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_INVALID_IMPL_MODE</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561106</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: aclnn API internal exception occurred, operator implementation mode parameter error.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561107</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: aclnn API internal exception occurred, environment variable ASCEND_OPP_PATH to be configured not detected.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_LOAD_JSON_FAILED</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561108</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: aclnn API internal exception occurred, failed to load operator information json file in operator kernel library.</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="30.183018301830185%" headers="mcps1.2.4.1.1 "><p>ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND</p>
</td>
<td class="cellrowborder" valign="top" width="16.521652165216523%" headers="mcps1.2.4.1.2 "><p>561109</p>
</td>
<td class="cellrowborder" valign="top" width="53.295329532953296%" headers="mcps1.2.4.1.3 "><p>Internal exception: aclnn API internal exception occurred, failed to load a field in operator information json file in operator kernel library.</p>
</td>
</tr>
</tbody>
</table>
