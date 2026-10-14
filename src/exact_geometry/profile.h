#pragma once

// Developer-only private build instrumentation. The regular extension compiles
// these scopes away. NVTX ranges associate launches with Nsight GPU timelines;
// no CUDA event, synchronization, geometry or admission behavior is added.
#ifdef CUMESH_PRIVATE_NVTX_PROFILE
#include <cstdlib>
#include <cstring>
#include <nvtx3/nvToolsExt.h>
#endif

namespace cumesh::private_profile {
class Range {
public:
  explicit Range(const char *name) {
#ifdef CUMESH_PRIVATE_NVTX_PROFILE
    const char *enabled = std::getenv("CUMESH_PRIVATE_NVTX_PROFILE");
    active_ = enabled && std::strcmp(enabled, "1") == 0;
    if (active_)
      nvtxRangePushA(name);
#else
    (void)name;
#endif
  }
  ~Range() { finish(); }
  Range(const Range &) = delete;
  Range &operator=(const Range &) = delete;
  void finish() noexcept {
#ifdef CUMESH_PRIVATE_NVTX_PROFILE
    if (active_) {
      nvtxRangePop();
      active_ = false;
    }
#endif
  }

private:
#ifdef CUMESH_PRIVATE_NVTX_PROFILE
  bool active_ = false;
#endif
};
} // namespace cumesh::private_profile
