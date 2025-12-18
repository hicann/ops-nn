#include <cstdlib>
#include <iostream>
#include <map>

#include "op_api_ut_common/inner/rts_interface.h"
#include "op_api_ut_common/inner/types.h"

using namespace std;

#define DEVICE_ID 0


void * MallocDeviceMemory(unsigned long size) {
    return nullptr;
}

void FreeDeviceMemory(void * device_mem_ptr) {
   return;
}

int MemcpyToDevice(const void* host_mem, void* dev_mem, unsigned long size) {
    return 0;
}

int MemcpyFromDevice(void* host_mem, const void* dev_mem, unsigned long size) {
    return 0;
}

bool IsFileExistsAccessable(string& name) {
    return (access(name.c_str(), F_OK ) != -1 );
}

int RtsInit() {
  return 0;
}

void RtsUnInit() {
  return;
}

int SynchronizeStream() {
  return 0;
}
