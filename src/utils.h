#pragma once
#include <optional>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

#define CUDA_CHECK(call)                                \
do {                                                    \
    const cudaError_t error_code = call;                \
    if (error_code != cudaSuccess) {                    \
        TORCH_CHECK(false,                              \
            "[CuMesh] CUDA error:\n",                   \
            "    File:       ", __FILE__, "\n",         \
            "    Line:       ", __LINE__, "\n",         \
            "    Error code: ", error_code, "\n",       \
            "    Error text: ",                         \
            cudaGetErrorString(error_code), "\n");      \
    }                                                   \
} while (0)

namespace cumesh {

#ifdef CUMESH_ENABLE_TEST_HOOKS
inline std::atomic<std::int64_t> temporary_allocation_successes_before_failure{-1};
inline std::atomic<std::int64_t> output_allocation_successes_before_failure{-1};
inline std::atomic<std::int64_t> output_copy_successes_before_failure{-1};
inline std::atomic<std::int64_t> output_map_successes_before_failure{-1};
inline std::atomic<std::int64_t> temporary_async_free_failures{0};
inline std::atomic<std::int64_t> buffer_attribute_failures{0};
inline std::atomic<std::int64_t> buffer_cleanup_failures{0};
inline std::atomic<std::int64_t> admission_revoke_event_record_failures{0};
#endif

inline std::atomic<std::uint64_t> active_temporary_allocations{0};
inline std::atomic<std::uint64_t> total_temporary_allocations{0};

struct CudaAllocationOwnerRecord;
using CudaAllocationOwnerPublication = void (*)(void*) noexcept;

// A record is fully constructed before a batch is published.  In particular,
// a destructor failure never exposes a prefix of its surviving Buffer owners
// to a retry while the rest are still object-local.
struct PersistentCudaAllocationQuarantineSpec {
    void* ptr;
    int device_index;
    cudaError_t primary_error;
    bool legacy_free_attempted;
    bool async_free_attempted;
    bool needs_device_synchronize;
    bool test_synthetic_free_attempt;
};

CudaAllocationOwnerRecord* register_cuda_allocation_owner(
    int device_index,
    cudaStream_t stream
);
cudaError_t prepare_cuda_allocation_owner_admission(
    CudaAllocationOwnerRecord* owner,
    cudaStream_t stream
) noexcept;
bool commit_cuda_allocation_owner_admission(
    CudaAllocationOwnerRecord* owner,
    CudaAllocationOwnerPublication publish,
    void* publish_context
) noexcept;
bool commit_cuda_allocation_owner_admission(CudaAllocationOwnerRecord* owner) noexcept;
bool transfer_cuda_allocation_owner_publication(
    CudaAllocationOwnerRecord* owner,
    CudaAllocationOwnerPublication publish,
    void* publish_context
) noexcept;
cudaError_t revoke_cuda_allocation_owner_admission(
    CudaAllocationOwnerRecord* owner,
    void* ptr,
    cudaStream_t stream
) noexcept;
void update_cuda_allocation_owner(CudaAllocationOwnerRecord* owner, void* ptr) noexcept;
void begin_cuda_allocation_owner_release(CudaAllocationOwnerRecord* owner) noexcept;
void release_cuda_allocation_owner(CudaAllocationOwnerRecord* owner) noexcept;
void quarantine_cuda_allocation_owner(CudaAllocationOwnerRecord* owner) noexcept;
void mark_cuda_allocation_owner_never_attempted(
    CudaAllocationOwnerRecord* owner,
    bool known_device
) noexcept;
void mark_cuda_allocation_owner_async_completion(
    CudaAllocationOwnerRecord* owner,
    bool completion_recorded
) noexcept;
void mark_cuda_allocation_owner_permanent_ambiguity(
    CudaAllocationOwnerRecord* owner
) noexcept;
void retain_persistent_cuda_allocation_quarantine(
    void* ptr,
    int device_index,
    cudaError_t primary_error,
    bool free_attempted,
    bool needs_device_synchronize,
    bool test_synthetic_free_attempt = false
) noexcept;
void retain_persistent_cuda_allocation_quarantine_batch(
    const PersistentCudaAllocationQuarantineSpec* specs,
    std::size_t count
) noexcept;
void quarantine_buffer_async_free_failure(
    void* ptr,
    int device_index,
    cudaError_t error
) noexcept;
void require_cuda_allocation_quarantine_clear();
std::size_t cuda_allocation_quarantine_count() noexcept;
std::optional<std::uint64_t> cuda_allocation_quarantine_epoch_if_clear() noexcept;
cudaError_t retry_cuda_allocation_quarantine() noexcept;
cudaError_t record_cuda_allocation_owner_completion(
    CudaAllocationOwnerRecord* owner,
    cudaStream_t stream
) noexcept;
cudaError_t validate_cuda_allocation_owner_stream(
    CudaAllocationOwnerRecord* owner,
    cudaStream_t stream
) noexcept;
void record_cuda_allocation_owner_primary_failure(
    CudaAllocationOwnerRecord* owner,
    cudaError_t error
) noexcept;
void record_cuda_allocation_owner_cleanup_failure(
    CudaAllocationOwnerRecord* owner,
    cudaError_t error
) noexcept;
void record_cuda_allocation_owner_completion_sync(
    CudaAllocationOwnerRecord* owner,
    cudaError_t error
) noexcept;
void mark_cuda_allocation_owner_legacy_free_attempt(CudaAllocationOwnerRecord* owner) noexcept;
void record_cuda_allocation_owner_fallback_result(
    CudaAllocationOwnerRecord* owner,
    cudaError_t error
) noexcept;

#ifdef CUMESH_ENABLE_TEST_HOOKS
inline void test_maybe_fail(std::atomic<std::int64_t>& remaining, const char* message) {
    const std::int64_t value = remaining.load(std::memory_order_relaxed);
    if (value == 0) {
        remaining.store(-1, std::memory_order_relaxed);
        TORCH_CHECK(false, "[CuMesh] Injected failure: ", message);
    }
    if (value > 0) {
        remaining.fetch_sub(1, std::memory_order_relaxed);
    }
}
#endif

#ifdef CUMESH_ENABLE_TEST_HOOKS
inline bool consume_test_failure_count(std::atomic<std::int64_t>& remaining) noexcept {
    std::int64_t value = remaining.load(std::memory_order_relaxed);
    while (value > 0) {
        if (remaining.compare_exchange_weak(
                value,
                value - 1,
                std::memory_order_relaxed
            )) {
            return true;
        }
    }
    return false;
}
#endif

inline void record_tensor_on_stream(
    const torch::Tensor& tensor,
    const c10::cuda::CUDAStream& stream
) {
    c10::cuda::CUDACachingAllocator::recordStream(
        tensor.storage().data_ptr(),
        stream
    );
}

inline int pointer_device_index(const void* ptr) {
    TORCH_CHECK(ptr != nullptr, "[CuMesh] Cannot derive a CUDA device from a null pointer");
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, ptr));
    TORCH_CHECK(
        attributes.type == cudaMemoryTypeDevice && attributes.device >= 0,
        "[CuMesh] Pointer does not identify CUDA device storage"
    );
    return attributes.device;
}

template<typename T>
struct Buffer;

template<typename T>
class ScopedCudaTemporary {
public:
    ScopedCudaTemporary(
        std::size_t count,
        cudaStream_t stream,
        int device_index,
        bool account_as_temporary = true
    ) : stream_(stream),
        device_index_(device_index),
        account_as_temporary_(account_as_temporary) {
        TORCH_CHECK(device_index >= 0, "[CuMesh] Temporary allocation requires a CUDA device");
        if (count == 0) {
            return;
        }
        require_cuda_allocation_quarantine_clear();
#ifdef CUMESH_ENABLE_TEST_HOOKS
        test_maybe_fail(
            temporary_allocation_successes_before_failure,
            "temporary CUDA allocation"
        );
#endif
        c10::cuda::CUDAGuard device_guard(device_index_);
        owner_ = register_cuda_allocation_owner(device_index_, stream_);
        const cudaError_t preparation_error =
            prepare_cuda_allocation_owner_admission(owner_, stream_);
        if (preparation_error != cudaSuccess) {
            release_cuda_allocation_owner(owner_);
            owner_ = nullptr;
            CUDA_CHECK(preparation_error);
        }
        T* allocation = nullptr;
        const cudaError_t allocation_error = cudaMallocAsync(&allocation, count * sizeof(T), stream_);
        if (allocation_error != cudaSuccess) {
            (void)revoke_cuda_allocation_owner_admission(owner_, allocation, stream_);
            owner_ = nullptr;
            CUDA_CHECK(allocation_error);
        }
        update_cuda_allocation_owner(owner_, allocation);
        struct ScopedAdmissionPublication {
            ScopedCudaTemporary* temporary;
            T* allocation;
        } publication{this, allocation};
        if (!commit_cuda_allocation_owner_admission(
                owner_,
                +[](void* opaque) noexcept {
                    auto* publication = static_cast<ScopedAdmissionPublication*>(opaque);
                    publication->temporary->ptr_ = publication->allocation;
                },
                &publication
            )) {
            const cudaError_t revoke_error =
                revoke_cuda_allocation_owner_admission(owner_, allocation, stream_);
            owner_ = nullptr;
            TORCH_CHECK(
                false,
                "[CuMesh] CUDA allocation admission was revoked by a quarantine epoch change: ",
                cudaGetErrorString(revoke_error)
            );
        }
        if (account_as_temporary_) {
            active_temporary_allocations.fetch_add(1, std::memory_order_relaxed);
            total_temporary_allocations.fetch_add(1, std::memory_order_relaxed);
        }
    }

    ~ScopedCudaTemporary() noexcept {
        release_noexcept();
    }

    ScopedCudaTemporary(const ScopedCudaTemporary&) = delete;
    ScopedCudaTemporary& operator=(const ScopedCudaTemporary&) = delete;
    ScopedCudaTemporary(ScopedCudaTemporary&&) = delete;
    ScopedCudaTemporary& operator=(ScopedCudaTemporary&&) = delete;

    T* get() const noexcept {
        return ptr_;
    }

    operator T*() const noexcept {
        return ptr_;
    }

    bool release_ownership_to(
        Buffer<T>& destination,
        std::size_t destination_size,
        std::size_t destination_capacity,
        int destination_device
    ) noexcept;

    void reset() {
        if (ptr_ == nullptr) {
            return;
        }
        c10::cuda::CUDAGuard device_guard(device_index_);
        begin_cuda_allocation_owner_release(owner_);
        cudaError_t error = validate_cuda_allocation_owner_stream(owner_, stream_);
        bool cuda_api_called = false;
#ifdef CUMESH_ENABLE_TEST_HOOKS
        if (error == cudaSuccess) {
            if (consume_test_failure_count(temporary_async_free_failures)) {
                error = cudaErrorUnknown;
            } else {
                cuda_api_called = true;
                error = cudaFreeAsync(ptr_, stream_);
            }
        }
#else
        if (error == cudaSuccess) {
            cuda_api_called = true;
            error = cudaFreeAsync(ptr_, stream_);
        }
#endif
        if (error == cudaSuccess) {
            mark_cuda_allocation_owner_async_completion(owner_, false);
            const cudaError_t completion_error = record_cuda_allocation_owner_completion(owner_, stream_);
            if (completion_error == cudaSuccess) {
                complete_release();
                return;
            }
            record_cuda_allocation_owner_cleanup_failure(owner_, completion_error);
            quarantine_release();
            TORCH_CHECK(
                false,
                "[CuMesh] Temporary CUDA async free completion could not be recorded: ",
                cudaGetErrorString(completion_error)
            );
        }
        record_cuda_allocation_owner_primary_failure(owner_, error);
        if (cuda_api_called) {
            mark_cuda_allocation_owner_permanent_ambiguity(owner_);
        } else {
            mark_cuda_allocation_owner_never_attempted(owner_, device_index_ >= 0);
        }
        quarantine_release();
        TORCH_CHECK(
            false,
            "[CuMesh] Temporary CUDA cleanup quarantined at the cudaFreeAsync boundary: ",
            cudaGetErrorString(error)
        );
    }

private:
    void complete_release() noexcept {
        ptr_ = nullptr;
        if (account_as_temporary_) {
            active_temporary_allocations.fetch_sub(1, std::memory_order_relaxed);
        }
        release_cuda_allocation_owner(owner_);
        owner_ = nullptr;
    }

    void quarantine_release() noexcept {
        quarantine_cuda_allocation_owner(owner_);
        owner_ = nullptr;
        ptr_ = nullptr;
        if (account_as_temporary_) {
            active_temporary_allocations.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    void release_noexcept() noexcept {
        if (ptr_ == nullptr) {
            return;
        }
        const cudaError_t original_device_error = cudaGetDevice(&original_device_);
        const cudaError_t set_device_error = cudaSetDevice(device_index_);
        cudaError_t release_error = set_device_error;
        bool cuda_api_called = false;
        if (set_device_error == cudaSuccess) {
            begin_cuda_allocation_owner_release(owner_);
            release_error = validate_cuda_allocation_owner_stream(owner_, stream_);
            if (release_error == cudaSuccess) {
                #ifdef CUMESH_ENABLE_TEST_HOOKS
                if (consume_test_failure_count(temporary_async_free_failures)) {
                    release_error = cudaErrorUnknown;
                } else {
                    cuda_api_called = true;
                    release_error = cudaFreeAsync(ptr_, stream_);
                }
                #else
                cuda_api_called = true;
                release_error = cudaFreeAsync(ptr_, stream_);
                #endif
            }
            if (release_error != cudaSuccess) {
                record_cuda_allocation_owner_primary_failure(owner_, release_error);
                if (cuda_api_called) {
                    mark_cuda_allocation_owner_permanent_ambiguity(owner_);
                } else {
                    mark_cuda_allocation_owner_never_attempted(owner_, device_index_ >= 0);
                }
            }
        }
        if (release_error == cudaSuccess) {
            mark_cuda_allocation_owner_async_completion(owner_, false);
            const cudaError_t completion_error = record_cuda_allocation_owner_completion(owner_, stream_);
            if (completion_error == cudaSuccess) {
                complete_release();
            } else {
                record_cuda_allocation_owner_cleanup_failure(owner_, completion_error);
                quarantine_release();
            }
        } else {
            if (!cuda_api_called && set_device_error != cudaSuccess) {
                mark_cuda_allocation_owner_never_attempted(owner_, device_index_ >= 0);
            }
            quarantine_release();
        }
        if (original_device_error == cudaSuccess) {
            (void)cudaSetDevice(original_device_);
        }
    }

    T* ptr_ = nullptr;
    cudaStream_t stream_ = nullptr;
    int device_index_ = -1;
    int original_device_ = -1;
    CudaAllocationOwnerRecord* owner_ = nullptr;
    bool account_as_temporary_ = true;
};


/**
 * A GPU buffer class that manages device memory.
 */
template<typename T>
struct Buffer {
    T* ptr;
    size_t size;
    size_t capacity;
    int device_index;

    Buffer() : ptr(nullptr), size(0), capacity(0), device_index(-1) {}

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    bool is_empty() const {
        return size == 0;
    }

    void init(size_t capacity) {
        int allocation_device = -1;
        CUDA_CHECK(cudaGetDevice(&allocation_device));
        cudaStream_t allocation_stream = c10::cuda::getCurrentCUDAStream(allocation_device).stream();
        init_async(capacity, allocation_stream, allocation_device);
    }

    void init_async(size_t capacity, cudaStream_t stream, int stream_device) {
        TORCH_CHECK(ptr == nullptr, "[CuMesh] Cannot initialize an allocated buffer");
        TORCH_CHECK(stream_device >= 0, "[CuMesh] Async allocation requires an explicit stream device");
        if (capacity == 0) {
            reset();
            return;
        }

        require_cuda_allocation_quarantine_clear();
        c10::cuda::CUDAGuard device_guard(stream_device);
        auto* allocation_owner = register_cuda_allocation_owner(stream_device, stream);
        const cudaError_t preparation_error =
            prepare_cuda_allocation_owner_admission(allocation_owner, stream);
        if (preparation_error != cudaSuccess) {
            release_cuda_allocation_owner(allocation_owner);
            CUDA_CHECK(preparation_error);
        }
        T* allocation = nullptr;
        const cudaError_t allocation_error = cudaMallocAsync(&allocation, capacity * sizeof(T), stream);
        if (allocation_error != cudaSuccess) {
            (void)revoke_cuda_allocation_owner_admission(allocation_owner, allocation, stream);
            CUDA_CHECK(allocation_error);
        }
        update_cuda_allocation_owner(allocation_owner, allocation);
        cudaPointerAttributes attributes{};
        cudaError_t attributes_error = cudaSuccess;
#ifdef CUMESH_ENABLE_TEST_HOOKS
        attributes_error = consume_test_failure_count(buffer_attribute_failures)
            ? cudaErrorUnknown
            : cudaPointerGetAttributes(&attributes, allocation);
#else
        attributes_error = cudaPointerGetAttributes(&attributes, allocation);
#endif
        if (attributes_error != cudaSuccess) {
            (void)revoke_cuda_allocation_owner_admission(allocation_owner, allocation, stream);
            CUDA_CHECK(attributes_error);
        }
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != stream_device) {
            (void)revoke_cuda_allocation_owner_admission(allocation_owner, allocation, stream);
            TORCH_CHECK(false, "[CuMesh] Async allocation device does not match its declared stream device");
        }
        struct BufferAdmissionPublication {
            Buffer* buffer;
            T* allocation;
            size_t capacity;
            int device_index;
        } publication{this, allocation, capacity, attributes.device};
        if (!commit_cuda_allocation_owner_admission(
                allocation_owner,
                +[](void* opaque) noexcept {
                    auto* publication = static_cast<BufferAdmissionPublication*>(opaque);
                    publication->buffer->ptr = publication->allocation;
                    publication->buffer->size = 0;
                    publication->buffer->capacity = publication->capacity;
                    publication->buffer->device_index = publication->device_index;
                },
                &publication
            )) {
            const cudaError_t revoke_error =
                revoke_cuda_allocation_owner_admission(allocation_owner, allocation, stream);
            TORCH_CHECK(
                false,
                "[CuMesh] Buffer allocation admission was revoked by a quarantine epoch change: ",
                cudaGetErrorString(revoke_error)
            );
        }
        release_cuda_allocation_owner(allocation_owner);
    }

    void free() {
        if (ptr == nullptr) {
            reset();
            return;
        }

        TORCH_CHECK(device_index >= 0, "[CuMesh] Allocated buffer has no CUDA device owner");
        c10::cuda::CUDAGuard device_guard(device_index);
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();
        CUDA_CHECK(release_async(stream, device_index));
    }

    cudaError_t release_async(
        cudaStream_t stream,
        int stream_device,
        bool transfer_failure_to_global_quarantine = true
    ) noexcept {
        if (ptr == nullptr) {
            reset();
            return cudaSuccess;
        }
        if (device_index != stream_device) {
            return cudaErrorInvalidDevice;
        }

        const cudaError_t error = cudaFreeAsync(ptr, stream);
        if (error == cudaSuccess) {
            reset();
        } else if (transfer_failure_to_global_quarantine) {
            // cudaFreeAsync returning an error after the API boundary leaves
            // the pointer's ownership ambiguous.  Do not leave it on this
            // Buffer where another release can issue a second async free.
            quarantine_buffer_async_free_failure(ptr, device_index, error);
            reset();
        }
        return error;
    }

    cudaError_t free_after_device_sync() noexcept {
        if (ptr == nullptr) {
            reset();
            return cudaSuccess;
        }

        const cudaError_t error = cudaFree(ptr);
        if (error == cudaSuccess) {
            reset();
        }
        return error;
    }

    void reset() noexcept {
        ptr = nullptr;
        size = 0;
        capacity = 0;
        device_index = -1;
    }

    void swap(Buffer& other) noexcept {
        using std::swap;
        swap(ptr, other.ptr);
        swap(size, other.size);
        swap(capacity, other.capacity);
        swap(device_index, other.device_index);
    }

    void resize(size_t size) {
        if (size > capacity) {
            TORCH_CHECK(
                device_index >= 0 || ptr == nullptr,
                "[CuMesh] Allocated buffer has no CUDA device owner"
            );
            int stream_device = device_index;
            if (stream_device < 0) {
                CUDA_CHECK(cudaGetDevice(&stream_device));
            }
            c10::cuda::CUDAGuard device_guard(stream_device);
            cudaStream_t stream = c10::cuda::getCurrentCUDAStream(stream_device).stream();
            ScopedCudaTemporary<T> replacement(
                size,
                stream,
                stream_device,
                false
            );
            if (ptr != nullptr) {
                CUDA_CHECK(release_async(stream, stream_device));
            }
            TORCH_CHECK(
                replacement.release_ownership_to(*this, size, size, stream_device),
                "[CuMesh] Buffer replacement metadata transfer failed"
            );
            return;
        }
        this->size = size;
    }

    void extend(size_t size) {
        size_t new_size = size + this->size;
        if (new_size > capacity) {
            TORCH_CHECK(device_index >= 0 || ptr == nullptr, "[CuMesh] Allocated buffer has no CUDA device owner");
            const int allocation_device = device_index;
            c10::cuda::OptionalCUDAGuard device_guard;
            if (allocation_device >= 0) {
                device_guard.set_index(allocation_device);
            }
            int stream_device = allocation_device;
            if (stream_device < 0) {
                CUDA_CHECK(cudaGetDevice(&stream_device));
            }
            cudaStream_t stream = c10::cuda::getCurrentCUDAStream(stream_device).stream();
            ScopedCudaTemporary<T> new_storage(
                new_size,
                stream,
                stream_device,
                false
            );
            T* new_ptr = new_storage.get();
            if (ptr != nullptr && this->size != 0) {
                CUDA_CHECK(cudaMemcpyAsync(new_ptr, ptr, this->size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
            }
            if (ptr != nullptr) {
                CUDA_CHECK(release_async(stream, stream_device));
            }
            TORCH_CHECK(
                new_storage.release_ownership_to(*this, new_size, new_size, stream_device),
                "[CuMesh] Buffer growth metadata transfer failed"
            );
            return;
        }
        this->size = new_size;
    }

    void zero() {
        TORCH_CHECK(device_index >= 0, "[CuMesh] Cannot zero a buffer without a CUDA device owner");
        c10::cuda::CUDAGuard device_guard(device_index);
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();
        CUDA_CHECK(cudaMemsetAsync(ptr, 0, size * sizeof(T), stream));
    }

};

template<typename T>
bool ScopedCudaTemporary<T>::release_ownership_to(
    Buffer<T>& destination,
    std::size_t destination_size,
    std::size_t destination_capacity,
    int destination_device
) noexcept {
    if (ptr_ == nullptr || owner_ == nullptr) {
        return false;
    }
    struct BufferTransferPublication {
        Buffer<T>* destination;
        T* allocation;
        std::size_t size;
        std::size_t capacity;
        int device_index;
    } publication{&destination, ptr_, destination_size, destination_capacity, destination_device};
    const bool transferred = transfer_cuda_allocation_owner_publication(
        owner_,
        +[](void* opaque) noexcept {
            auto* publication = static_cast<BufferTransferPublication*>(opaque);
            publication->destination->ptr = publication->allocation;
            publication->destination->size = publication->size;
            publication->destination->capacity = publication->capacity;
            publication->destination->device_index = publication->device_index;
        },
        &publication
    );
    if (!transferred) {
        return false;
    }
    if (account_as_temporary_) {
        active_temporary_allocations.fetch_sub(1, std::memory_order_relaxed);
    }
    release_cuda_allocation_owner(owner_);
    owner_ = nullptr;
    ptr_ = nullptr;
    return true;
}


/**
 * Swap the contents of two buffers.
 */
template<typename T1, typename T2>
void swap_buffers(Buffer<T1>& b1, Buffer<T2>& b2) {
    void* b1_ptr = reinterpret_cast<void*>(b1.ptr);
    void* b2_ptr = reinterpret_cast<void*>(b2.ptr);
    size_t b1_capacity_bytes = b1.capacity * sizeof(T1);
    size_t b2_capacity_bytes = b2.capacity * sizeof(T2);
    size_t b1_size_bytes = b1.size * sizeof(T1);
    size_t b2_size_bytes = b2.size * sizeof(T2);
    
    b1.ptr = reinterpret_cast<T1*>(b2_ptr);
    b2.ptr = reinterpret_cast<T2*>(b1_ptr);
    b1.capacity = b2_capacity_bytes / sizeof(T1);
    b2.capacity = b1_capacity_bytes / sizeof(T2);
    b1.size = b2_size_bytes / sizeof(T1);
    b2.size = b1_size_bytes / sizeof(T2);
    std::swap(b1.device_index, b2.device_index);
}


} // namespace cumesh
