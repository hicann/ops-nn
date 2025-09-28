#ifndef _OP_API_UT_COMMON_FILE_IO_H_
#define _OP_API_UT_COMMON_FILE_IO_H_

#include <string>
#include <vector>
#include <memory>

using namespace std;

void * ReadBinFile(const string& file_name, size_t& size);
int WriteBinFile(const void* host_mem, const string& file_name, size_t size);
bool FileExists(const string& file_name);
void DeleteUtTmpFiles(const string& file_prefix);
void DeleteCwdFilesEndsWith(const string& file_suffix);
string RealPath(const string& path);
void SetUtTmpFileSwitch();
bool GetUtTmpFileSwitch();
bool IsDir(const string& path);
void GetFilesWithSuffix(const string& path, const string& suffix, vector<string>& files);
unique_ptr<char[]> GetBinFromFile(const string& path, uint32_t &data_len);

#endif