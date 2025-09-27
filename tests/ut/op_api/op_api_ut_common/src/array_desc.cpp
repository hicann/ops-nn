#include "opdev/common_types.h"

#include "op_api_ut_common/array_desc.h"

using namespace std;

static void AclIntArrayReleaseWrapper(aclIntArray * p) {
  Release(p);
}

unique_ptr<aclIntArray, void (*)(aclIntArray*)> IntArrayDesc::ToAclType() const {
  unique_ptr<aclIntArray, decltype(&AclIntArrayReleaseWrapper)> p(this->ToAclTypeRawPtr(),
                                                                  AclIntArrayReleaseWrapper);
  return move(p);
}

aclIntArray * IntArrayDesc::ToAclTypeRawPtr() const {
  return new aclIntArray(arr_.data(), arr_.size());
}

////////////////////////////////////

static void AclFloatArrayReleaseWrapper(aclFloatArray * p) {
  Release(p);
}

unique_ptr<aclFloatArray, void (*)(aclFloatArray*)> FloatArrayDesc::ToAclType() const {
  unique_ptr<aclFloatArray, decltype(&AclFloatArrayReleaseWrapper)> p(this->ToAclTypeRawPtr(),
                                                                    AclFloatArrayReleaseWrapper);
  return move(p);
}

aclFloatArray * FloatArrayDesc::ToAclTypeRawPtr() const {
  return new aclFloatArray(arr_.data(), arr_.size());
}

////////////////////////////////////

static void AclBoolArrayReleaseWrapper(aclBoolArray * p) {
  Release(p);
}

unique_ptr<aclBoolArray, void (*)(aclBoolArray*)> BoolArrayDesc::ToAclType() const {
  unique_ptr<aclBoolArray, decltype(&AclBoolArrayReleaseWrapper)> p(this->ToAclTypeRawPtr(),
                                                                     AclBoolArrayReleaseWrapper);
  return move(p);
}

aclBoolArray * BoolArrayDesc::ToAclTypeRawPtr() const {
  auto p = new bool[arr_.size()];
  for (size_t i = 0; i < arr_.size(); ++i) {
    p[i] = arr_[i];
  }
  auto aclPtr = new aclBoolArray(p, arr_.size());
  delete[] p;
  return aclPtr;
}
