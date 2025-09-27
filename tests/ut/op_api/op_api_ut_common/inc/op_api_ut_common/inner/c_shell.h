#ifndef _OP_API_UT_COMMON_C_SHELL_H_
#define _OP_API_UT_COMMON_C_SHELL_H_

#include <string>
#include <sstream>

using namespace std;

int ExecuteShellCmd(const string& cmd, stringstream& ss);
void SetOpUtSrcPath(const char* p);
const string& GetOpUtSrcPath();
const string& GetExePath();
int GetEnv(const string& name, string& value);

#endif