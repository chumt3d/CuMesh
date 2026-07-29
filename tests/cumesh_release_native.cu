#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "cumesh.h"
#include "shared.h"

namespace cumesh {
std::uint64_t test_quarantine_retry_stream_synchronize_count() noexcept;
std::uint64_t test_quarantine_retry_event_wait_count() noexcept;
std::uint64_t test_quarantine_retry_legacy_free_attempt_count() noexcept;
void test_fail_quarantine_retry_completion(std::int64_t failures) noexcept;
void test_clear_injected_quarantine_legacy_free_attempts() noexcept;
int test_quarantine_state_for_pointer(const void* ptr) noexcept;
}

namespace {

__global__ void ordered_use_kernel(float3 *vertices, int *output) {
  vertices[0] = make_float3(37.0F, 0.0F, 0.0F);
  output[0] = static_cast<int>(vertices[0].x);
}

__global__ void delay_kernel(unsigned long long cycles) {
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

struct RetryCallbackState {
  std::mutex mutex;
  std::condition_variable condition;
  bool entered = false;
  bool release = false;
  bool registration_blocked = false;
  int device_index = -1;
  cudaStream_t stream = nullptr;
};

void CUDART_CB retry_reentrant_allocation_callback(void *opaque) {
  auto *state = static_cast<RetryCallbackState *>(opaque);
  bool registration_blocked = false;
  try {
    auto *owner = cumesh::register_cuda_allocation_owner(state->device_index,
                                                         state->stream);
    cumesh::release_cuda_allocation_owner(owner);
  } catch (const c10::Error &) {
    registration_blocked = true;
  }
  std::unique_lock<std::mutex> lock(state->mutex);
  state->registration_blocked = registration_blocked;
  state->entered = true;
  state->condition.notify_all();
  state->condition.wait(lock, [&]() { return state->release; });
}

void populate_all_buffers(cumesh::CuMesh &mesh, cudaStream_t stream,
                          int device_index) {
  if (mesh.allocated_buffer_count() == 0) {
    auto options = torch::TensorOptions().device(
        torch::Device(torch::kCUDA, device_index));
    mesh.init(torch::zeros({1, 3}, options.dtype(torch::kFloat32)),
              torch::zeros({1, 3}, options.dtype(torch::kInt32)));
  }
  auto initialize = [&](auto &buffer) {
    if (buffer.ptr != nullptr) {
      return;
    }
    buffer.init_async(1, stream, device_index);
    buffer.size = 1;
  };

  initialize(mesh.vertices);
  initialize(mesh.faces);
  initialize(mesh.face_areas);
  initialize(mesh.face_normals);
  initialize(mesh.vertex_normals);
  initialize(mesh.edges);
  initialize(mesh.boundaries);
  initialize(mesh.vert_is_boundary);
  initialize(mesh.vert_is_manifold);
  initialize(mesh.vert2edge);
  initialize(mesh.vert2edge_cnt);
  initialize(mesh.vert2edge_offset);
  initialize(mesh.vert2bound);
  initialize(mesh.vert2bound_cnt);
  initialize(mesh.vert2bound_offset);
  initialize(mesh.edge2face);
  initialize(mesh.edge2face_cnt);
  initialize(mesh.edge2face_offset);
  initialize(mesh.face2edge);
  initialize(mesh.vert2face);
  initialize(mesh.vert2face_cnt);
  initialize(mesh.vert2face_offset);
  initialize(mesh.manifold_face_adj);
  initialize(mesh.manifold_bound_adj);
  initialize(mesh.conn_comp_ids);
  initialize(mesh.bound_conn_comp_ids);
  initialize(mesh.loop_boundaries);
  initialize(mesh.loop_boundaries_offset);
  initialize(mesh.vertices_map);
  initialize(mesh.faces_map);
  initialize(mesh.edge_collapse_costs);
  initialize(mesh.propagated_costs);
  initialize(mesh.atlas_chart_ids);
  initialize(mesh.atlas_chart_vertex_map);
  initialize(mesh.atlas_chart_faces);
  initialize(mesh.atlas_chart_faces_offset);
  initialize(mesh.atlas_chart_vertex_offset);
  initialize(mesh.atlas_chart_uvs);
  initialize(mesh.atlas_chart_normal_cones);
  initialize(mesh.atlas_chart_adj);
  initialize(mesh.atlas_chart_adj_length);
  initialize(mesh.atlas_chart_perims);
  initialize(mesh.atlas_chart_areas);
  initialize(mesh.atlas_chart2edge);
  initialize(mesh.atlas_chart2edge_cnt);
  initialize(mesh.atlas_chart2edge_offset);
  initialize(mesh.temp_storage);
  initialize(mesh.cub_temp_storage);
}

py::tuple exercise_explicit_release(bool synchronize) {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();

  cumesh::CuMesh mesh;
  populate_all_buffers(mesh, stream, device_index);
  const std::size_t count_before = mesh.allocated_buffer_count();
  auto output =
      torch::zeros({1}, torch::TensorOptions()
                            .dtype(torch::kInt32)
                            .device(torch::Device(torch::kCUDA, device_index)));
  ordered_use_kernel<<<1, 1, 0, stream>>>(mesh.vertices.ptr,
                                          output.data_ptr<int>());
  CUDA_CHECK(cudaGetLastError());

  const cumesh::ReleaseResult result = mesh.release_async(synchronize);
  return py::make_tuple(count_before, result.buffers_released,
                        mesh.allocated_buffer_count(), result.device_index,
                        result.stream_synchronize_count, output);
}

py::tuple exercise_destructor_fallback() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();

  const std::uint64_t sync_count_before =
      cumesh::CuMesh::destructor_device_synchronize_count();
  const std::uint64_t error_count_before =
      cumesh::CuMesh::destructor_release_error_count();
  {
    cumesh::CuMesh mesh;
    populate_all_buffers(mesh, stream, device_index);
  }
  return py::make_tuple(
      sync_count_before, cumesh::CuMesh::destructor_device_synchronize_count(),
      error_count_before, cumesh::CuMesh::destructor_release_error_count());
}

py::tuple exercise_destructor_sync_failure_retains_persistent_quarantine() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();
  const std::size_t quarantine_before =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  {
    cumesh::CuMesh mesh;
    populate_all_buffers(mesh, stream, device_index);
    cumesh::CuMesh::test_fail_destructor_device_synchronize_once();
  }
  const std::size_t retained =
      cumesh::CuMesh::quarantined_cuda_allocation_count() - quarantine_before;
  bool new_admission_blocked = false;
  try {
    cumesh::ScopedCudaTemporary<int> blocked(1, stream, device_index);
  } catch (const c10::Error&) {
    new_admission_blocked = true;
  }
  const cudaError_t retry_error = cumesh::retry_cuda_allocation_quarantine();
  return py::make_tuple(retained, new_admission_blocked,
                        static_cast<int>(retry_error),
                        cumesh::CuMesh::quarantined_cuda_allocation_count());
}

py::tuple exercise_persistent_quarantine_retry_reentrancy() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();
  RetryCallbackState callback_state{};
  callback_state.device_index = device_index;
  callback_state.stream = stream;
  {
    cumesh::CuMesh mesh;
    populate_all_buffers(mesh, stream, device_index);
    CUDA_CHECK(cudaLaunchHostFunc(stream, retry_reentrant_allocation_callback,
                                  &callback_state));
    cumesh::CuMesh::test_fail_destructor_device_synchronize_once();
  }
  TORCH_CHECK(cumesh::CuMesh::quarantined_cuda_allocation_count() == 48,
              "Expected retained persistent quarantine before retry");
  cudaError_t owner_retry_error = cudaErrorUnknown;
  std::thread retry_owner([&]() {
    owner_retry_error = cumesh::retry_cuda_allocation_quarantine();
  });
  {
    std::unique_lock<std::mutex> lock(callback_state.mutex);
    TORCH_CHECK(callback_state.condition.wait_for(
                    lock, std::chrono::seconds(5),
                    [&]() { return callback_state.entered; }),
                "Persistent retry callback did not enter");
  }
  const cudaError_t concurrent_retry_error = cumesh::retry_cuda_allocation_quarantine();
  {
    std::lock_guard<std::mutex> lock(callback_state.mutex);
    callback_state.release = true;
  }
  callback_state.condition.notify_all();
  retry_owner.join();
  return py::make_tuple(static_cast<int>(owner_retry_error),
                        static_cast<int>(concurrent_retry_error),
                        static_cast<int>(cudaErrorNotReady),
                        callback_state.registration_blocked,
                        cumesh::CuMesh::quarantined_cuda_allocation_count());
}

py::tuple exercise_destructor_first_free_error_preserves_later_retryable_owners() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();
  const std::uint64_t free_attempts_before =
      cumesh::CuMesh::test_destructor_legacy_free_attempt_count();
  void* first_slot = nullptr;
  void* tail_slot = nullptr;
  {
    cumesh::CuMesh mesh;
    populate_all_buffers(mesh, stream, device_index);
    first_slot = mesh.vertices.ptr;
    tail_slot = mesh.faces.ptr;
    cumesh::CuMesh::test_fail_destructor_legacy_free_once();
  }
  const std::size_t retained_before_retry =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  const std::uint64_t initial_free_attempts =
      cumesh::CuMesh::test_destructor_legacy_free_attempt_count() -
      free_attempts_before;
  const int first_slot_state = cumesh::test_quarantine_state_for_pointer(first_slot);
  const int tail_slot_state = cumesh::test_quarantine_state_for_pointer(tail_slot);
  bool new_admission_blocked = false;
  try {
    cumesh::ScopedCudaTemporary<int> blocked(1, stream, device_index);
  } catch (const c10::Error&) {
    new_admission_blocked = true;
  }
  const cudaError_t first_retry = cumesh::retry_cuda_allocation_quarantine();
  const std::size_t retained_after_retry =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  cumesh::test_clear_injected_quarantine_legacy_free_attempts();
  const cudaError_t cleanup_retry = cumesh::retry_cuda_allocation_quarantine();
  return py::make_tuple(
      retained_before_retry, initial_free_attempts, new_admission_blocked,
      static_cast<int>(first_retry), static_cast<int>(cudaErrorNotReady),
      first_slot_state, tail_slot_state,
      retained_after_retry, static_cast<int>(cleanup_retry),
      cumesh::CuMesh::quarantined_cuda_allocation_count());
}

py::tuple exercise_explicit_release_identity_and_destructor() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  const std::uint64_t sync_count_before =
      cumesh::CuMesh::destructor_device_synchronize_count();

  std::uint64_t owner_id = 0;
  std::uint64_t generation = 0;
  std::uint64_t token_owner_id = 0;
  std::uint64_t token_generation = 0;
  std::uint64_t repeated_token_owner_id = 0;
  std::uint64_t repeated_token_generation = 0;
  std::size_t first_released = 0;
  std::size_t second_released = 0;
  std::string second_status;
  {
    cumesh::CuMesh mesh;
    auto options = torch::TensorOptions().device(
        torch::Device(torch::kCUDA, device_index));
    mesh.init(torch::zeros({3, 3}, options.dtype(torch::kFloat32)),
              torch::tensor({{0, 1, 2}}, options.dtype(torch::kInt32)));
    owner_id = mesh.owner_id();
    generation = mesh.generation();
    const cumesh::ReleaseResult first = mesh.release_async(false);
    const cumesh::ReleaseResult second = mesh.release_async(true);
    TORCH_CHECK(first.completion_token != nullptr,
                "Expected the first release to return a completion token");
    token_owner_id = first.completion_token->owner_id();
    token_generation = first.completion_token->generation();
    TORCH_CHECK(second.completion_token != nullptr,
                "Expected repeated release to preserve its completion token");
    repeated_token_owner_id = second.completion_token->owner_id();
    repeated_token_generation = second.completion_token->generation();
    first_released = first.buffers_released;
    second_released = second.buffers_released;
    second_status = second.status;
  }
  return py::make_tuple(
      owner_id, generation, token_owner_id, token_generation,
      repeated_token_owner_id, repeated_token_generation, first_released,
      second_released, second_status,
      cumesh::CuMesh::destructor_device_synchronize_count() -
          sync_count_before);
}

py::tuple exercise_init_failure_rolls_back_and_allows_next_generation() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  auto options = torch::TensorOptions().device(
      torch::Device(torch::kCUDA, device_index));
  const auto vertices = torch::tensor(
      {{1.0F, 2.0F, 3.0F}}, options.dtype(torch::kFloat32));
  const auto faces =
      torch::tensor({{0, 0, 0}}, options.dtype(torch::kInt32));

  cumesh::CuMesh mesh;
  cumesh::CuMesh::test_fail_buffer_attributes_once();
  bool rejected = false;
  try {
    mesh.init(vertices, faces);
  } catch (const c10::Error &) {
    rejected = true;
  }
  const std::size_t retained_after_failure = mesh.allocated_buffer_count();
  const std::size_t global_quarantine_after_failure =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  const std::uint64_t generation_after_failure = mesh.generation();

  mesh.init(vertices, faces);
  const std::uint64_t generation_after_success = mesh.generation();
  auto [read_vertices, read_faces] = mesh.read();
  const cumesh::ReleaseResult release = mesh.release_async(true);
  return py::make_tuple(
      rejected, retained_after_failure, global_quarantine_after_failure,
      generation_after_failure, generation_after_success, read_vertices,
      read_faces, release.buffers_released);
}

py::tuple exercise_clear_cache() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();

  cumesh::CuMesh mesh;
  populate_all_buffers(mesh, stream, device_index);
  mesh.num_conn_comps = 7;
  mesh.num_bound_conn_comps = 8;
  mesh.num_bound_loops = 9;
  mesh.atlas_num_charts = 10;
  const std::size_t count_before = mesh.allocated_buffer_count();
  mesh.clear_cache();
  const std::size_t count_after = mesh.allocated_buffer_count();
  const int metadata_sum = mesh.num_conn_comps + mesh.num_bound_conn_comps +
                           mesh.num_bound_loops + mesh.atlas_num_charts;
  const cumesh::ReleaseResult result = mesh.release_async(true);
  return py::make_tuple(count_before, count_after, metadata_sum,
                        result.buffers_released);
}

py::tuple exercise_full_swap() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();

  cumesh::CuMesh prior;
  populate_all_buffers(prior, stream, device_index);
  prior.num_conn_comps = 11;
  prior.num_bound_conn_comps = 12;
  prior.num_bound_loops = 13;
  prior.atlas_num_charts = 14;

  cumesh::CuMesh replacement;
  auto replacement_options =
      torch::TensorOptions().device(torch::Device(torch::kCUDA, device_index));
  replacement.init(
      torch::zeros({1, 3}, replacement_options.dtype(torch::kFloat32)),
      torch::zeros({1, 3}, replacement_options.dtype(torch::kInt32)));
  replacement.num_conn_comps = 21;
  replacement.num_bound_conn_comps = 22;
  replacement.num_bound_loops = 23;
  replacement.atlas_num_charts = 24;

  float3 *prior_vertices_before = prior.vertices.ptr;
  int3 *prior_faces_before = prior.faces.ptr;
  float3 *replacement_vertices_before = replacement.vertices.ptr;
  int3 *replacement_faces_before = replacement.faces.ptr;
  prior.swap(replacement);
  const bool pointer_owners_exchanged =
      prior.vertices.ptr == replacement_vertices_before &&
      prior.faces.ptr == replacement_faces_before &&
      replacement.vertices.ptr == prior_vertices_before &&
      replacement.faces.ptr == prior_faces_before;
  const std::size_t selected_count = prior.allocated_buffer_count();
  const std::size_t retained_count = replacement.allocated_buffer_count();
  const int selected_metadata_sum =
      prior.num_conn_comps + prior.num_bound_conn_comps +
      prior.num_bound_loops + prior.atlas_num_charts;
  const int retained_metadata_sum =
      replacement.num_conn_comps + replacement.num_bound_conn_comps +
      replacement.num_bound_loops + replacement.atlas_num_charts;
  const cumesh::ReleaseResult retained_release =
      replacement.release_async(true);
  const cumesh::ReleaseResult selected_release = prior.release_async(true);
  return py::make_tuple(
      pointer_owners_exchanged, selected_count, retained_count,
      selected_metadata_sum, retained_metadata_sum,
      retained_release.buffers_released, selected_release.buffers_released);
}

py::tuple exercise_partial_release_quarantine() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();

  cumesh::CuMesh mesh;
  populate_all_buffers(mesh, stream, device_index);
  // all_buffers() starts vertices, faces, face_areas, face_normals,
  // vertex_normals.  The pre-call hook below stops at face_normals, making
  // these two raw identities an exact boundary/tail ownership assertion.
  void* boundary_slot = mesh.face_normals.ptr;
  void* tail_slot = mesh.vertex_normals.ptr;
  cumesh::CuMesh::test_fail_release_after(3);
  const cumesh::ReleaseResult partial = mesh.release_async(false);
  if (partial.completion_token != nullptr) {
    partial.completion_token->synchronize();
  }
  const std::size_t retained_after_partial = mesh.allocated_buffer_count();
  const int boundary_state = cumesh::test_quarantine_state_for_pointer(boundary_slot);
  const int tail_state = cumesh::test_quarantine_state_for_pointer(tail_slot);
  // With no local buffers left, a pending-token observation must retain the
  // same release's durable global-Q failure rather than report completion.
  const cumesh::ReleaseResult pending_observation = mesh.release_async(true);

  // The fourth release API boundary is ambiguous and is globally retained;
  // the 44 later members were never attempted and are also transferred out
  // of the mesh.  A mesh-local retry must not issue another cudaFreeAsync.
  const cudaError_t first_global_retry = cumesh::retry_cuda_allocation_quarantine();
  const std::size_t retained_after_first_global_retry =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  cumesh::test_clear_injected_quarantine_legacy_free_attempts();
  const cudaError_t cleanup_global_retry = cumesh::retry_cuda_allocation_quarantine();
  const cumesh::ReleaseResult clean_observation = mesh.release_async(true);
  return py::make_tuple(partial.status, partial.buffers_released,
                        partial.quarantined_buffer_count, partial.error_code,
                        retained_after_partial,
                        boundary_state, tail_state,
                        pending_observation.status,
                        cumesh::CuMesh::quarantined_cuda_allocation_count(),
                        static_cast<int>(first_global_retry),
                        static_cast<int>(cudaErrorNotReady),
                        retained_after_first_global_retry,
                        static_cast<int>(cleanup_global_retry),
                        clean_observation.status,
                        mesh.allocated_buffer_count());
}

void exercise_stream_device_mismatch() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  const c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(device_index);
  cumesh::CuMesh::test_validate_stream_device(stream, device_index + 1);
}

py::tuple exercise_pointer_device_preflight_mismatch() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();

  cumesh::CuMesh mesh;
  populate_all_buffers(mesh, stream, device_index);
  mesh.vertices.device_index = device_index + 1;
  bool rejected = false;
  try {
    (void)mesh.release_async(false);
  } catch (const c10::Error &) {
    rejected = true;
  }
  mesh.vertices.device_index = device_index;
  const std::size_t retained = mesh.allocated_buffer_count();
  const cumesh::ReleaseResult cleanup = mesh.release_async(true);
  return py::make_tuple(rejected, retained, cleanup.buffers_released);
}

py::tuple exercise_release_completion_failure(const std::string &failure) {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  const c10::cuda::CUDAStream owner_stream =
      failure == "wait" ? c10::cuda::getStreamFromPool(false, device_index)
                        : c10::cuda::getCurrentCUDAStream(device_index);
  cumesh::CuMesh mesh;
  {
    c10::cuda::CUDAStreamGuard owner_guard(owner_stream);
    populate_all_buffers(mesh, owner_stream.stream(), device_index);
  }

  if (failure == "record") {
    cumesh::CuMesh::test_fail_release_event_record_once();
  } else if (failure == "token") {
    cumesh::CuMesh::test_fail_release_token_allocation_once();
  } else if (failure == "wait") {
    cumesh::CuMesh::test_fail_release_caller_wait_once();
  } else if (failure == "sync") {
    cumesh::CuMesh::test_fail_release_synchronize_once();
  } else {
    TORCH_CHECK(false, "Unknown release completion fault: ", failure);
  }

  const bool request_sync = failure == "sync";
  const cumesh::ReleaseResult failed = mesh.release_async(request_sync);
  const std::size_t retained = mesh.allocated_buffer_count();
  const cumesh::ReleaseResult retry = mesh.release_async(true);
  return py::make_tuple(failed.status, failed.buffers_released,
                        failed.quarantined_buffer_count, retained, retry.status,
                        retry.error_code, mesh.allocated_buffer_count());
}

py::tuple exercise_combined_primary_and_completion_failure() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard device_guard(device_index);
  const c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(device_index);
  cumesh::CuMesh mesh;
  populate_all_buffers(mesh, stream.stream(), device_index);

  // Three prefix buffers cross cudaFreeAsync.  The fourth and all later
  // buffers transfer to global Q while the completion event for that prefix
  // independently fails to record.
  cumesh::CuMesh::test_fail_release_after(3);
  cumesh::CuMesh::test_fail_release_event_record_once();
  const cumesh::ReleaseResult combined = mesh.release_async(false);
  const std::uint8_t combined_facts = mesh.test_release_lifecycle_fact_mask();
  const std::size_t combined_global_q =
      cumesh::CuMesh::quarantined_cuda_allocation_count();

  // Prove the prefix completion first.  Global Q must remain a separate
  // durable fact and continue to block a clean release result.
  const cumesh::ReleaseResult completion_proved = mesh.release_async(true);
  const std::uint8_t after_completion_facts =
      mesh.test_release_lifecycle_fact_mask();
  const std::size_t after_completion_global_q =
      cumesh::CuMesh::quarantined_cuda_allocation_count();

  const cudaError_t global_retry = cumesh::retry_cuda_allocation_quarantine();
  const cumesh::ReleaseResult recovered = mesh.release_async(true);
  return py::make_tuple(
      combined.status, combined.buffers_released, combined.primary_error_code,
      combined.cleanup_error_code, combined_facts, combined_global_q,
      completion_proved.status, after_completion_facts,
      after_completion_global_q, static_cast<int>(global_retry),
      recovered.status, mesh.test_release_lifecycle_fact_mask(),
      cumesh::CuMesh::quarantined_cuda_allocation_count(),
      mesh.allocated_buffer_count());
}

py::tuple exercise_remove_faces_validation() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  auto options =
      torch::TensorOptions().device(torch::Device(torch::kCUDA, device_index));
  cumesh::CuMesh mesh;
  mesh.init(
      torch::zeros({3, 3}, options.dtype(torch::kFloat32)),
      torch::tensor({{0, 1, 2}, {0, 2, 1}}, options.dtype(torch::kInt32)));
  const int faces_before = mesh.num_faces();

  auto rejected = [&](torch::Tensor mask) {
    try {
      mesh.remove_faces(mask);
      return false;
    } catch (const c10::Error &) {
      return true;
    }
  };
  auto wrong_shape = torch::ones({2, 1}, options.dtype(torch::kBool));
  auto wrong_dtype = torch::ones({2}, options.dtype(torch::kInt32));
  auto noncontiguous_base = torch::ones({2, 2}, options.dtype(torch::kBool));
  auto noncontiguous = noncontiguous_base.select(1, 0);
  const bool shape_rejected = rejected(wrong_shape);
  const bool dtype_rejected = rejected(wrong_dtype);
  const bool contiguity_rejected = rejected(noncontiguous);

  bool cross_device_rejected = true;
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count > 1) {
    auto wrong_device = torch::ones(
        {2},
        torch::TensorOptions()
            .dtype(torch::kBool)
            .device(torch::Device(torch::kCUDA, device_index == 0 ? 1 : 0)));
    cross_device_rejected = rejected(wrong_device);
  }
  const int faces_after = mesh.num_faces();
  (void)mesh.release_async(true);
  return py::make_tuple(shape_rejected, dtype_rejected, contiguity_rejected,
                        cross_device_rejected, faces_before, faces_after,
                        device_count);
}

py::tuple exercise_temporary_allocation_failure() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  auto options =
      torch::TensorOptions().device(torch::Device(torch::kCUDA, device_index));
  cumesh::CuMesh mesh;
  mesh.init(torch::tensor({{0.0F, 0.0F, 0.0F},
                           {1.0F, 0.0F, 0.0F},
                           {0.0F, 1.0F, 0.0F},
                           {0.0F, 0.0F, 1.0F}},
                          options.dtype(torch::kFloat32)),
            torch::tensor({{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}},
                          options.dtype(torch::kInt32)));
  mesh.get_edge_face_adjacency();
  const std::uint64_t before =
      cumesh::CuMesh::active_temporary_allocation_count();
  cumesh::CuMesh::test_fail_temporary_allocation_after(1);
  bool rejected = false;
  try {
    mesh.get_manifold_face_adjacency();
  } catch (const c10::Error &) {
    rejected = true;
  }
  CUDA_CHECK(cudaStreamSynchronize(
      c10::cuda::getCurrentCUDAStream(device_index).stream()));
  const std::uint64_t after =
      cumesh::CuMesh::active_temporary_allocation_count();
  (void)mesh.release_async(true);
  return py::make_tuple(rejected, before, after);
}

py::tuple
exercise_output_exception_allocator_reuse(const std::string &failure) {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  const c10::cuda::CUDAStream owner_stream =
      c10::cuda::getStreamFromPool(false, device_index);
  const c10::cuda::CUDAStream caller_stream =
      c10::cuda::getStreamFromPool(false, device_index);
  auto options =
      torch::TensorOptions().device(torch::Device(torch::kCUDA, device_index));
  cumesh::CuMesh mesh;
  {
    c10::cuda::CUDAStreamGuard owner_guard(owner_stream);
    mesh.init(
        torch::tensor({{2.0F, 3.0F, 4.0F}}, options.dtype(torch::kFloat32)),
        torch::tensor({{0, 0, 0}}, options.dtype(torch::kInt32)));
    if (failure == "map") {
      mesh.compute_face_normals();
    }
    delay_kernel<<<1, 1, 0, owner_stream.stream()>>>(200000000ULL);
    CUDA_CHECK(cudaGetLastError());
  }
  if (failure == "allocation") {
    cumesh::CuMesh::test_fail_output_allocation_after(1);
  } else if (failure == "copy") {
    cumesh::CuMesh::test_fail_output_copy_after(0);
  } else if (failure == "map") {
    cumesh::CuMesh::test_fail_output_map_after(1);
  } else {
    TORCH_CHECK(false, "Unknown output fault: ", failure);
  }

  bool rejected = false;
  torch::Tensor reuse;
  {
    c10::cuda::CUDAStreamGuard caller_guard(caller_stream);
    try {
      if (failure == "map") {
        (void)mesh.read_all_cache();
      } else {
        (void)mesh.read();
      }
    } catch (const c10::Error &) {
      rejected = true;
    }
    reuse = torch::full({1, 3}, 91.0F, options.dtype(torch::kFloat32));
  }
  owner_stream.synchronize();
  caller_stream.synchronize();
  auto reuse_cpu = reuse.cpu();
  const bool reuse_intact = reuse_cpu[0][0].item<float>() == 91.0F &&
                            reuse_cpu[0][1].item<float>() == 91.0F &&
                            reuse_cpu[0][2].item<float>() == 91.0F;
  const cumesh::ReleaseResult cleanup = mesh.release_async(true);
  return py::make_tuple(rejected, reuse_intact, cleanup.error_code);
}

py::tuple exercise_event_construction_failure() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  const c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(device_index);
  cumesh::CuMesh mesh;
  populate_all_buffers(mesh, stream.stream(), device_index);
  cumesh::CuMesh::test_fail_release_event_create_once();

  const cumesh::ReleaseResult failed = mesh.release_async(false);
  const std::string failed_status = failed.status;
  const bool rejected =
      failed.error_code == static_cast<int>(cudaErrorMemoryAllocation);

  bool operation_blocked = false;
  try {
    (void)mesh.read();
  } catch (const c10::Error &) {
    operation_blocked = true;
  }
  const std::size_t retained = mesh.allocated_buffer_count();
  const cumesh::ReleaseResult retry = mesh.release_async(true);
  return py::make_tuple(failed_status, rejected, operation_blocked, retained,
                        retry.status, retry.buffers_released,
                        mesh.allocated_buffer_count());
}

py::tuple exercise_temporary_cleanup_quarantine() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(device_index).stream();
  const std::size_t before =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  {
    cumesh::ScopedCudaTemporary<int> first(1, stream, device_index);
    cumesh::ScopedCudaTemporary<int> second(1, stream, device_index);
    cumesh::CuMesh::test_fail_temporary_cleanup(2, 2, 2);
  }
  const std::size_t quarantined =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  const std::uint64_t allocations_before_blocked_attempts =
      cumesh::CuMesh::total_temporary_allocation_count();
  int blocked_attempts = 0;
  for (int attempt = 0; attempt < 3; ++attempt) {
    try {
      cumesh::ScopedCudaTemporary<int> blocked(1, stream, device_index);
    } catch (const c10::Error &) {
      ++blocked_attempts;
    }
  }
  const std::size_t after_blocked_attempts =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  const std::uint64_t blocked_attempt_allocations =
      cumesh::CuMesh::total_temporary_allocation_count() -
      allocations_before_blocked_attempts;
  const std::uint64_t retry_synchronizations_before =
      cumesh::test_quarantine_retry_stream_synchronize_count();
  cumesh::CuMesh::retry_quarantined_cuda_allocations();
  const std::uint64_t retry_synchronizations =
      cumesh::test_quarantine_retry_stream_synchronize_count() -
      retry_synchronizations_before;
  const std::size_t after_retry =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  return py::make_tuple(before, quarantined, blocked_attempts,
                        after_blocked_attempts, after_retry,
                        cumesh::CuMesh::active_temporary_allocation_count(),
                        retry_synchronizations, blocked_attempt_allocations);
}

py::tuple exercise_buffer_attribute_cleanup_quarantine() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  const std::size_t before =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  cumesh::Buffer<int> failed_buffer;
  cumesh::CuMesh::test_fail_buffer_attributes_and_cleanup_once();
  bool rejected = false;
  try {
    failed_buffer.init(1);
  } catch (const c10::Error &) {
    rejected = true;
  }
  const std::size_t quarantined =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  cumesh::Buffer<int> blocked_buffer;
  bool allocation_blocked = false;
  try {
    blocked_buffer.init(1);
  } catch (const c10::Error &) {
    allocation_blocked = true;
  }
  const std::size_t after_blocked =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  cumesh::CuMesh::retry_quarantined_cuda_allocations();
  return py::make_tuple(before, rejected, quarantined, allocation_blocked,
                        after_blocked,
                        cumesh::CuMesh::quarantined_cuda_allocation_count());
}

py::tuple exercise_quarantine_retry_groups_by_device_and_stream() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  const c10::cuda::CUDAStream first_stream =
      c10::cuda::getCurrentCUDAStream(device_index);
  const c10::cuda::CUDAStream second_stream =
      c10::cuda::getStreamFromPool(false, device_index);
  TORCH_CHECK(first_stream != second_stream,
              "Expected two distinct same-device streams for retry grouping");
  {
    cumesh::ScopedCudaTemporary<int> first_a(
        1, first_stream.stream(), device_index);
    cumesh::ScopedCudaTemporary<int> first_b(
        1, first_stream.stream(), device_index);
    cumesh::ScopedCudaTemporary<int> second_a(
        1, second_stream.stream(), device_index);
    cumesh::ScopedCudaTemporary<int> second_b(
        1, second_stream.stream(), device_index);
    cumesh::CuMesh::test_fail_temporary_cleanup(4, 4, 4);
  }
  TORCH_CHECK(cumesh::CuMesh::quarantined_cuda_allocation_count() == 4,
              "Expected four quarantined allocations across two streams");

  const std::uint64_t synchronize_before =
      cumesh::test_quarantine_retry_stream_synchronize_count();
  const std::uint64_t waits_before =
      cumesh::test_quarantine_retry_event_wait_count();
  cumesh::test_fail_quarantine_retry_completion(1);
  const cudaError_t first_retry = cumesh::retry_cuda_allocation_quarantine();
  const std::size_t after_first_retry =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  const std::uint64_t first_retry_synchronizations =
      cumesh::test_quarantine_retry_stream_synchronize_count() -
      synchronize_before;
  const std::uint64_t first_retry_waits =
      cumesh::test_quarantine_retry_event_wait_count() - waits_before;

  const std::uint64_t retry_synchronize_before =
      cumesh::test_quarantine_retry_stream_synchronize_count();
  const std::uint64_t retry_waits_before =
      cumesh::test_quarantine_retry_event_wait_count();
  const cudaError_t second_retry = cumesh::retry_cuda_allocation_quarantine();
  const std::size_t after_second_retry =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  const std::uint64_t second_retry_synchronizations =
      cumesh::test_quarantine_retry_stream_synchronize_count() -
      retry_synchronize_before;
  const std::uint64_t second_retry_waits =
      cumesh::test_quarantine_retry_event_wait_count() - retry_waits_before;

  return py::make_tuple(
      static_cast<int>(first_retry), static_cast<int>(cudaErrorUnknown),
      after_first_retry, first_retry_synchronizations, first_retry_waits,
      static_cast<int>(second_retry), after_second_retry,
      second_retry_synchronizations, second_retry_waits);
}

py::tuple exercise_sync_failure_never_falls_back(bool destructor_path) {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(device_index).stream();
  const std::size_t quarantine_before =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  bool reset_rejected = false;
  if (destructor_path) {
    {
      cumesh::ScopedCudaTemporary<int> temporary(1, stream, device_index);
      cumesh::CuMesh::test_fail_temporary_cleanup(1, 1, 0);
    }
  } else {
    cumesh::ScopedCudaTemporary<int> temporary(1, stream, device_index);
    cumesh::CuMesh::test_fail_temporary_cleanup(1, 1, 0);
    try {
      temporary.reset();
    } catch (const c10::Error &) {
      reset_rejected = true;
    }
  }
  const std::size_t quarantined =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  const std::uint64_t fallback_attempts = 0;
  const cudaError_t retry_error = cumesh::retry_cuda_allocation_quarantine();
  return py::make_tuple(
      reset_rejected, quarantined - quarantine_before, fallback_attempts,
      static_cast<int>(retry_error),
      cumesh::CuMesh::quarantined_cuda_allocation_count());
}

py::tuple exercise_fallback_error_is_never_legacy_freed_twice(bool destructor_path) {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(device_index).stream();
  const std::size_t quarantine_before =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  bool reset_rejected = false;
  if (destructor_path) {
    {
      cumesh::ScopedCudaTemporary<int> temporary(1, stream, device_index);
      // The fallback hook returns cudaErrorUnknown before cudaFree.  Production
      // still has to treat this as an ambiguous one-shot legacy-free attempt.
      cumesh::CuMesh::test_fail_temporary_cleanup(1, 0, 1);
    }
  } else {
    cumesh::ScopedCudaTemporary<int> temporary(1, stream, device_index);
    cumesh::CuMesh::test_fail_temporary_cleanup(1, 0, 1);
    try {
      temporary.reset();
    } catch (const c10::Error &) {
      reset_rejected = true;
    }
  }
  const std::size_t quarantined =
      cumesh::CuMesh::quarantined_cuda_allocation_count() - quarantine_before;
  const std::uint64_t fallback_attempts = 0;
  const std::uint64_t retry_free_before =
      cumesh::test_quarantine_retry_legacy_free_attempt_count();
  const cudaError_t retry_error = cumesh::retry_cuda_allocation_quarantine();
  const std::uint64_t retry_free_attempts =
      cumesh::test_quarantine_retry_legacy_free_attempt_count() - retry_free_before;
  const std::size_t retained_after_retry =
      cumesh::CuMesh::quarantined_cuda_allocation_count();

  // Pre-call injection never crosses cudaFreeAsync; the exact-device retry
  // owns the one permitted legacy free after synchronization.
  cumesh::test_clear_injected_quarantine_legacy_free_attempts();
  const cudaError_t cleanup_error = cumesh::retry_cuda_allocation_quarantine();
  return py::make_tuple(
      reset_rejected, quarantined, fallback_attempts, static_cast<int>(retry_error),
      static_cast<int>(cudaSuccess), retry_free_attempts, retained_after_retry,
      static_cast<int>(cleanup_error),
      cumesh::CuMesh::quarantined_cuda_allocation_count());
}

py::tuple exercise_pending_admission_epoch_revoke(bool revoke_fails) {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(device_index).stream();
  auto* owner = cumesh::register_cuda_allocation_owner(device_index, stream);
  CUDA_CHECK(cumesh::prepare_cuda_allocation_owner_admission(owner, stream));
  int* allocation = nullptr;
  CUDA_CHECK(cudaMallocAsync(&allocation, sizeof(int), stream));
  cumesh::update_cuda_allocation_owner(owner, allocation);
  if (revoke_fails) {
    cumesh::buffer_cleanup_failures.store(1, std::memory_order_relaxed);
  }
  // A separate quarantine winner would take this same epoch transition; using
  // the pending record keeps the fixture bounded while exercising the exact
  // no-publication/revoke edge.
  cumesh::quarantine_cuda_allocation_owner(owner);
  const bool committed = cumesh::commit_cuda_allocation_owner_admission(owner);
  const cudaError_t revoke_error =
      cumesh::revoke_cuda_allocation_owner_admission(owner, allocation, stream);
  const std::size_t after_revoke =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  const cudaError_t retry_error = cumesh::retry_cuda_allocation_quarantine();
  return py::make_tuple(committed, static_cast<int>(revoke_error), after_revoke,
                        static_cast<int>(retry_error),
                        cumesh::CuMesh::quarantined_cuda_allocation_count());
}

py::tuple exercise_async_free_event_record_failure_recovers_by_device_sync() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(device_index).stream();
  auto* owner = cumesh::register_cuda_allocation_owner(device_index, stream);
  CUDA_CHECK(cumesh::prepare_cuda_allocation_owner_admission(owner, stream));
  int* allocation = nullptr;
  CUDA_CHECK(cudaMallocAsync(&allocation, sizeof(int), stream));
  cumesh::update_cuda_allocation_owner(owner, allocation);
  // Force the exact successful-async-free/event-record-failure branch.  The
  // later completion event proves that free before retry retires the record.
  cumesh::admission_revoke_event_record_failures.store(1,
                                                        std::memory_order_relaxed);
  cumesh::quarantine_cuda_allocation_owner(owner);
  const cudaError_t revoke_error =
      cumesh::revoke_cuda_allocation_owner_admission(owner, allocation, stream);
  const std::uint64_t legacy_attempts_before =
      cumesh::test_quarantine_retry_legacy_free_attempt_count();
  const std::uint64_t syncs_before =
      cumesh::test_quarantine_retry_stream_synchronize_count();
  const cudaError_t recovery_retry = cumesh::retry_cuda_allocation_quarantine();
  const std::uint64_t legacy_attempts_during_blocked_retry =
      cumesh::test_quarantine_retry_legacy_free_attempt_count() -
      legacy_attempts_before;
  const std::uint64_t recovery_syncs =
      cumesh::test_quarantine_retry_stream_synchronize_count() - syncs_before;
  const std::size_t retained_after_recovery_retry =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  return py::make_tuple(
      static_cast<int>(revoke_error), static_cast<int>(cudaErrorUnknown),
      static_cast<int>(recovery_retry), static_cast<int>(cudaSuccess),
      retained_after_recovery_retry, legacy_attempts_during_blocked_retry,
      recovery_syncs,
      cumesh::CuMesh::quarantined_cuda_allocation_count());
}

py::tuple exercise_committed_transfer_survives_later_quarantine() {
  struct Publication {
    int* pointer = nullptr;
    std::size_t capacity = 0;
    int device_index = -1;
    bool published = false;
  } first_publication, transfer_publication, blocked_publication;
  auto publish = +[](void* opaque) noexcept {
    auto* publication = static_cast<Publication*>(opaque);
    publication->published = true;
  };

  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(device_index).stream();

  auto* committed_owner = cumesh::register_cuda_allocation_owner(device_index, stream);
  CUDA_CHECK(cumesh::prepare_cuda_allocation_owner_admission(committed_owner, stream));
  int* committed_allocation = nullptr;
  CUDA_CHECK(cudaMallocAsync(&committed_allocation, sizeof(int), stream));
  cumesh::update_cuda_allocation_owner(committed_owner, committed_allocation);
  const bool first_commit = cumesh::commit_cuda_allocation_owner_admission(
      committed_owner, publish, &first_publication);

  auto* quarantine_winner = cumesh::register_cuda_allocation_owner(device_index, stream);
  cumesh::quarantine_cuda_allocation_owner(quarantine_winner);
  transfer_publication.pointer = committed_allocation;
  transfer_publication.capacity = 1;
  transfer_publication.device_index = device_index;
  const bool transferred_after_q = cumesh::transfer_cuda_allocation_owner_publication(
      committed_owner, publish, &transfer_publication);
  const bool transfer_metadata_visible = transfer_publication.published &&
      transfer_publication.pointer == committed_allocation &&
      transfer_publication.capacity == 1 &&
      transfer_publication.device_index == device_index;
  CUDA_CHECK(cudaFreeAsync(committed_allocation, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cumesh::release_cuda_allocation_owner(committed_owner);
  CUDA_CHECK(cumesh::revoke_cuda_allocation_owner_admission(
      quarantine_winner, nullptr, stream));

  auto* pending_owner = cumesh::register_cuda_allocation_owner(device_index, stream);
  CUDA_CHECK(cumesh::prepare_cuda_allocation_owner_admission(pending_owner, stream));
  int* pending_allocation = nullptr;
  CUDA_CHECK(cudaMallocAsync(&pending_allocation, sizeof(int), stream));
  cumesh::update_cuda_allocation_owner(pending_owner, pending_allocation);
  auto* second_quarantine_winner =
      cumesh::register_cuda_allocation_owner(device_index, stream);
  cumesh::quarantine_cuda_allocation_owner(second_quarantine_winner);
  const bool blocked_commit = cumesh::commit_cuda_allocation_owner_admission(
      pending_owner, publish, &blocked_publication);
  const cudaError_t pending_revoke = cumesh::revoke_cuda_allocation_owner_admission(
      pending_owner, pending_allocation, stream);
  CUDA_CHECK(cumesh::revoke_cuda_allocation_owner_admission(
      second_quarantine_winner, nullptr, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return py::make_tuple(
      first_commit, first_publication.published, transferred_after_q,
      transfer_metadata_visible, blocked_commit, blocked_publication.published,
      static_cast<int>(pending_revoke),
      cumesh::CuMesh::quarantined_cuda_allocation_count());
}

py::tuple exercise_completion_device_guards() {
  int original_device = -1;
  int device_count = 0;
  CUDA_CHECK(cudaGetDevice(&original_device));
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  bool injected_identity_rejected = false;
  try {
    cumesh::CuMesh::test_validate_completion_device_identity(
        original_device + 1, original_device);
  } catch (const c10::Error &) {
    injected_identity_rejected = true;
  }

  bool real_multi_gpu_executed = false;
  bool current_device_restored = true;
  if (device_count > 1) {
    const int mesh_device = original_device == 0 ? 1 : 0;
    cumesh::CuMesh mesh;
    {
      c10::cuda::CUDAGuard mesh_guard(mesh_device);
      const c10::cuda::CUDAStream stream =
          c10::cuda::getCurrentCUDAStream(mesh_device);
      populate_all_buffers(mesh, stream.stream(), mesh_device);
    }
    CUDA_CHECK(cudaSetDevice(original_device));
    const cumesh::ReleaseResult result = mesh.release_async(true);
    int current_device = -1;
    CUDA_CHECK(cudaGetDevice(&current_device));
    current_device_restored = current_device == original_device &&
                              result.status == "release_completed";
    real_multi_gpu_executed = true;
  }
  return py::make_tuple(injected_identity_rejected, real_multi_gpu_executed,
                        current_device_restored, device_count);
}

py::tuple exercise_compress_ids_allocation_count() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(device_index).stream();
  cumesh::Buffer<int> ids;
  cumesh::Buffer<int> inverse;
  cumesh::Buffer<char> cub_storage;
  ids.init(4);
  ids.size = 4;
  inverse.init(4);
  inverse.size = 4;
  const int host_ids[4] = {4, 2, 4, 7};
  CUDA_CHECK(cudaMemcpyAsync(ids.ptr, host_ids, sizeof(host_ids),
                             cudaMemcpyHostToDevice, stream));
  const std::uint64_t before_without_inverse =
      cumesh::CuMesh::total_temporary_allocation_count();
  (void)cumesh::compress_ids(ids.ptr, ids.size, cub_storage,
                             static_cast<int *>(nullptr));
  const std::uint64_t without_inverse =
      cumesh::CuMesh::total_temporary_allocation_count() -
      before_without_inverse;
  CUDA_CHECK(cudaMemcpyAsync(ids.ptr, host_ids, sizeof(host_ids),
                             cudaMemcpyHostToDevice, stream));
  const std::uint64_t before_with_inverse =
      cumesh::CuMesh::total_temporary_allocation_count();
  (void)cumesh::compress_ids(ids.ptr, ids.size, cub_storage, inverse.ptr);
  const std::uint64_t with_inverse =
      cumesh::CuMesh::total_temporary_allocation_count() - before_with_inverse;
  CUDA_CHECK(ids.release_async(stream, device_index));
  CUDA_CHECK(inverse.release_async(stream, device_index));
  CUDA_CHECK(cub_storage.release_async(stream, device_index));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return py::make_tuple(without_inverse, with_inverse);
}

py::tuple exercise_reentrant_concurrent_quarantine_retry() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(device_index).stream();

  RetryCallbackState callback_state;
  callback_state.device_index = device_index;
  callback_state.stream = stream;
  {
    cumesh::ScopedCudaTemporary<int> temporary(1, stream, device_index);
    cumesh::CuMesh::test_fail_temporary_cleanup(1, 1, 1);
    // This callback is deliberately ahead of the owner completion event that
    // the destructor records.  The recovery barrier must therefore remain
    // claimed while the callback verifies that new allocation is blocked.
    CUDA_CHECK(cudaLaunchHostFunc(stream, retry_reentrant_allocation_callback,
                                  &callback_state));
  }
  TORCH_CHECK(cumesh::CuMesh::quarantined_cuda_allocation_count() == 1,
              "Expected one quarantined allocation before concurrent retry");

  cudaError_t owner_retry_error = cudaErrorUnknown;
  std::thread owner_retry([&]() {
    owner_retry_error = cumesh::retry_cuda_allocation_quarantine();
  });
  {
    std::unique_lock<std::mutex> lock(callback_state.mutex);
    callback_state.condition.wait(lock,
                                  [&]() { return callback_state.entered; });
  }
  const cudaError_t concurrent_retry_error =
      cumesh::retry_cuda_allocation_quarantine();
  {
    std::lock_guard<std::mutex> lock(callback_state.mutex);
    callback_state.release = true;
  }
  callback_state.condition.notify_all();
  owner_retry.join();
  return py::make_tuple(callback_state.registration_blocked,
                        static_cast<int>(owner_retry_error),
                        static_cast<int>(concurrent_retry_error),
                        static_cast<int>(cudaErrorNotReady),
                        cumesh::CuMesh::quarantined_cuda_allocation_count());
}

py::tuple exercise_external_destroyed_stream_recovery() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  cudaStream_t origin = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&origin, cudaStreamNonBlocking));
  const c10::cuda::CUDAStream external =
      c10::cuda::getStreamFromExternal(origin, device_index);
  TORCH_CHECK(external.stream() == origin,
              "Expected the external CUDA stream wrapper to preserve its raw handle");
  unsigned long long origin_id = 0;
  CUDA_CHECK(cudaStreamGetId(origin, &origin_id));
  const std::size_t quarantine_before =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  {
    cumesh::ScopedCudaTemporary<int> temporary(1, external.stream(), device_index);
    cumesh::CuMesh::test_fail_temporary_cleanup(1, 1, 0);
  }
  const std::size_t quarantined =
      cumesh::CuMesh::quarantined_cuda_allocation_count() - quarantine_before;
  const cudaError_t destroy_error = cudaStreamDestroy(origin);
  cudaStream_t replacement = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&replacement, cudaStreamNonBlocking));
  unsigned long long replacement_id = 0;
  CUDA_CHECK(cudaStreamGetId(replacement, &replacement_id));
  const std::uint64_t waits_before = cumesh::test_quarantine_retry_event_wait_count();
  const std::uint64_t syncs_before =
      cumesh::test_quarantine_retry_stream_synchronize_count();
  const cudaError_t retry_error = cumesh::retry_cuda_allocation_quarantine();
  const std::uint64_t waits = cumesh::test_quarantine_retry_event_wait_count() - waits_before;
  const std::uint64_t syncs =
      cumesh::test_quarantine_retry_stream_synchronize_count() - syncs_before;
  const cudaError_t replacement_destroy_error = cudaStreamDestroy(replacement);
  return py::make_tuple(
      static_cast<int>(destroy_error), static_cast<int>(retry_error),
      static_cast<int>(replacement_destroy_error), quarantined,
      cumesh::CuMesh::quarantined_cuda_allocation_count(), origin_id,
      replacement_id, waits, syncs);
}

py::tuple exercise_per_thread_stream_recovery() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  unsigned long long per_thread_id = 0;
  CUDA_CHECK(cudaStreamGetId(cudaStreamPerThread, &per_thread_id));
  const std::size_t quarantine_before =
      cumesh::CuMesh::quarantined_cuda_allocation_count();
  {
    cumesh::ScopedCudaTemporary<int> temporary(
        1, cudaStreamPerThread, device_index);
    cumesh::CuMesh::test_fail_temporary_cleanup(1, 1, 0);
  }
  const std::size_t quarantined =
      cumesh::CuMesh::quarantined_cuda_allocation_count() - quarantine_before;
  const std::uint64_t waits_before = cumesh::test_quarantine_retry_event_wait_count();
  const cudaError_t retry_error = cumesh::retry_cuda_allocation_quarantine();
  const std::uint64_t waits = cumesh::test_quarantine_retry_event_wait_count() - waits_before;
  return py::make_tuple(static_cast<int>(retry_error), quarantined,
                        cumesh::CuMesh::quarantined_cuda_allocation_count(),
                        per_thread_id, waits);
}

py::tuple exercise_completion_origin_stream_identity() {
  int device_index = -1;
  CUDA_CHECK(cudaGetDevice(&device_index));
  c10::cuda::CUDAGuard guard(device_index);
  const c10::cuda::CUDAStream origin =
      c10::cuda::getCurrentCUDAStream(device_index);
  const c10::cuda::CUDAStream different =
      c10::cuda::getStreamFromPool(false, device_index);
  TORCH_CHECK(origin.id() != different.id(),
              "Expected a distinct same-device CUDA stream for origin test");
  cumesh::ReleaseCompletionToken token(
      device_index, static_cast<std::int64_t>(origin.id()), 17, 23);
  const cudaError_t different_stream_error = token.record(different);
  const cudaError_t origin_error = token.record(origin);
  const cudaError_t synchronize_error = token.synchronize_noexcept();
  return py::make_tuple(static_cast<int>(different_stream_error),
                        static_cast<int>(cudaErrorInvalidResourceHandle),
                        static_cast<int>(origin_error),
                        static_cast<int>(synchronize_error), device_index,
                        static_cast<std::int64_t>(origin.id()),
                        static_cast<std::int64_t>(different.id()));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("exercise_explicit_release", &exercise_explicit_release);
  module.def("exercise_destructor_fallback", &exercise_destructor_fallback);
  module.def("exercise_destructor_sync_failure_retains_persistent_quarantine",
             &exercise_destructor_sync_failure_retains_persistent_quarantine);
  module.def("exercise_persistent_quarantine_retry_reentrancy",
             &exercise_persistent_quarantine_retry_reentrancy);
  module.def("exercise_destructor_first_free_error_preserves_later_retryable_owners",
             &exercise_destructor_first_free_error_preserves_later_retryable_owners);
  module.def("exercise_explicit_release_identity_and_destructor",
             &exercise_explicit_release_identity_and_destructor);
  module.def("exercise_init_failure_rolls_back_and_allows_next_generation",
             &exercise_init_failure_rolls_back_and_allows_next_generation);
  module.def("exercise_clear_cache", &exercise_clear_cache);
  module.def("exercise_full_swap", &exercise_full_swap);
  module.def("exercise_partial_release_quarantine",
             &exercise_partial_release_quarantine);
  module.def("exercise_stream_device_mismatch",
             &exercise_stream_device_mismatch);
  module.def("exercise_pointer_device_preflight_mismatch",
             &exercise_pointer_device_preflight_mismatch);
  module.def("exercise_release_completion_failure",
             &exercise_release_completion_failure);
  module.def("exercise_combined_primary_and_completion_failure",
             &exercise_combined_primary_and_completion_failure);
  module.def("exercise_remove_faces_validation",
             &exercise_remove_faces_validation);
  module.def("exercise_temporary_allocation_failure",
             &exercise_temporary_allocation_failure);
  module.def("exercise_output_exception_allocator_reuse",
             &exercise_output_exception_allocator_reuse);
  module.def("exercise_event_construction_failure",
             &exercise_event_construction_failure);
  module.def("exercise_temporary_cleanup_quarantine",
             &exercise_temporary_cleanup_quarantine);
  module.def("exercise_buffer_attribute_cleanup_quarantine",
             &exercise_buffer_attribute_cleanup_quarantine);
  module.def("exercise_quarantine_retry_groups_by_device_and_stream",
             &exercise_quarantine_retry_groups_by_device_and_stream);
  module.def("exercise_sync_failure_never_falls_back",
             &exercise_sync_failure_never_falls_back);
  module.def("exercise_fallback_error_is_never_legacy_freed_twice",
             &exercise_fallback_error_is_never_legacy_freed_twice);
  module.def("exercise_pending_admission_epoch_revoke",
             &exercise_pending_admission_epoch_revoke);
  module.def("exercise_async_free_event_record_failure_recovers_by_device_sync",
             &exercise_async_free_event_record_failure_recovers_by_device_sync);
  module.def("exercise_committed_transfer_survives_later_quarantine",
             &exercise_committed_transfer_survives_later_quarantine);
  module.def("exercise_completion_device_guards",
             &exercise_completion_device_guards);
  module.def("exercise_compress_ids_allocation_count",
             &exercise_compress_ids_allocation_count);
  module.def("exercise_reentrant_concurrent_quarantine_retry",
             &exercise_reentrant_concurrent_quarantine_retry);
  module.def("exercise_external_destroyed_stream_recovery",
             &exercise_external_destroyed_stream_recovery);
  module.def("exercise_per_thread_stream_recovery",
             &exercise_per_thread_stream_recovery);
  module.def("exercise_completion_origin_stream_identity",
             &exercise_completion_origin_stream_identity);
}
