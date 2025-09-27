#ifndef _OP_API_UT_COMMON_STRING_UTILS_H_
#define _OP_API_UT_COMMON_STRING_UTILS_H_

#include <string>
#include <vector>

using namespace std;

vector<string> Split(const string &s, const string& spliter);
string &Ltrim(string &s);
string &Rtrim(string &s);
string Trim(string &s);

#endif