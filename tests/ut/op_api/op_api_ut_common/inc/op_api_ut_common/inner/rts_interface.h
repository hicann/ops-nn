#ifndef _OP_API_UT_COMMON_RTS_INTERFACE_H_
#define _OP_API_UT_COMMON_RTS_INTERFACE_H_

//#include "runtime/stream.h"

void * MallocDeviceMemory(unsigned long size);
void FreeDeviceMemory(void * device_mem_ptr);
int MemcpyToDevice(const void* host_mem, void* dev_mem, unsigned long size);
int MemcpyFromDevice(void* host_mem, const void* dev_mem, unsigned long size);
int RtsInit();
void RtsUnInit();
int SynchronizeStream();

#endif