// Test oracle for GPU traversal/scheduling, using the mapped host predicates.
#include "exact_geometry/predicates.h"
extern "C" void exact_pair_codes(const double *coordinates, const int *groups,
                                 int *out, int count) {
  using namespace cumesh::exact_geometry;
  for (int i = 0; i < count; ++i) {
    Point p[6];
    for (int j = 0; j < 6; ++j)
      p[j] = {coordinates[18 * i + 3 * j], coordinates[18 * i + 3 * j + 1],
              coordinates[18 * i + 3 * j + 2]};
    const int *g = groups + 6 * i;
    out[i] = pair(p[0], p[1], p[2], p[3], p[4], p[5], g[0], g[1], g[2], g[3],
                  g[4], g[5]);
  }
}
