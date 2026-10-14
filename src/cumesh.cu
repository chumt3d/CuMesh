#include "cumesh.h"

#include <c10/cuda/CUDAGuard.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <exception>
#include <memory>
#include <new>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace cumesh {

enum class CudaAllocationOwnerState : std::uint8_t {
    pending_admission,
    committed_active,
    releasing,
    revoked_cleanup,
    quarantined_never_attempted,
    quarantined_async_completion_unproved,
    quarantined_async_completion_recorded,
    quarantined_claimed,
    permanent_ambiguous,
    retired,
};

struct CudaAllocationOwnerRecord {
    void* ptr = nullptr;
    cudaStream_t stream = nullptr;
    int device_index = -1;
    unsigned long long stream_id = 0;
    cudaEvent_t completion_event = nullptr;
    cudaError_t primary_error = cudaSuccess;
    cudaError_t cleanup_error = cudaSuccess;
    cudaError_t stream_validation_error = cudaSuccess;
    cudaError_t event_record_error = cudaSuccess;
    cudaError_t completion_sync_error = cudaSuccess;
    cudaError_t recovery_wait_error = cudaSuccess;
    cudaError_t barrier_record_error = cudaSuccess;
    cudaError_t barrier_synchronize_error = cudaSuccess;
    cudaError_t set_device_error = cudaSuccess;
    cudaError_t restore_error = cudaSuccess;
    cudaError_t fallback_error = cudaSuccess;
    cudaError_t retry_error = cudaSuccess;
    bool completion_recorded = false;
    bool completion_proved = false;
    bool quarantined = false;
    bool retry_claimed = false;
    bool revoke_claimed = false;
    bool free_enqueued = false;
    bool async_free_enqueued = false;
    bool persistent_fallback = false;
    bool needs_device_synchronize = false;
    bool test_synthetic_free_attempt = false;
    CudaAllocationOwnerState state = CudaAllocationOwnerState::pending_admission;
    CudaAllocationOwnerState state_before_retry_claim = CudaAllocationOwnerState::retired;
    std::uint64_t admission_epoch = 0;
    CudaAllocationOwnerRecord* previous = nullptr;
    CudaAllocationOwnerRecord* next = nullptr;
    CudaAllocationOwnerRecord* retry_next = nullptr;
    CudaAllocationOwnerRecord* retry_group_next = nullptr;
    CudaAllocationOwnerRecord* quarantine_previous = nullptr;
    CudaAllocationOwnerRecord* quarantine_next = nullptr;
};

namespace {

struct CudaAllocationQuarantineOwner {
    std::mutex mutex;
    CudaAllocationOwnerRecord* active_head = nullptr;
    CudaAllocationOwnerRecord* quarantine_head = nullptr;
    std::size_t quarantined_count = 0;
    std::uint64_t quarantine_epoch = 0;
    // A 64-bit admission epoch must never wrap.  Once exhausted, leave the
    // allocator closed rather than letting an old pending token look current.
    bool admission_closed = false;
};

// Retry never touches an origin stream handle.  Each CUDA device instead owns
// one durable recovery stream and one barrier event.  A retry enqueues one
// wait for every owner-managed completion event on that stream, records the
// barrier, then performs one host wait before any legacy cudaFree call.
struct CudaAllocationRecoveryDevice {
    cudaStream_t stream = nullptr;
    cudaEvent_t barrier_event = nullptr;
};

#ifdef CUMESH_ENABLE_TEST_HOOKS
std::atomic<std::uint64_t> quarantine_retry_stream_synchronizations{0};
std::atomic<std::uint64_t> quarantine_retry_event_waits{0};
std::atomic<std::uint64_t> quarantine_retry_legacy_free_attempts{0};
std::atomic<std::int64_t> quarantine_retry_completion_failures{0};
#endif

CudaAllocationQuarantineOwner& cuda_allocation_owner() {
    static auto* owner = new CudaAllocationQuarantineOwner();
    return *owner;
}

cudaError_t recovery_device_for(
    CudaAllocationRecoveryDevice** result
) noexcept {
    *result = nullptr;
    try {
        auto* recovery = new (std::nothrow) CudaAllocationRecoveryDevice();
        if (recovery == nullptr) {
            return cudaErrorMemoryAllocation;
        }
        cudaError_t error = cudaStreamCreateWithFlags(&recovery->stream, cudaStreamNonBlocking);
        if (error == cudaSuccess) {
            error = cudaEventCreateWithFlags(&recovery->barrier_event, cudaEventDisableTiming);
        }
        if (error != cudaSuccess) {
            if (recovery->barrier_event != nullptr) {
                (void)cudaEventDestroy(recovery->barrier_event);
            }
            if (recovery->stream != nullptr) {
                (void)cudaStreamDestroy(recovery->stream);
            }
            delete recovery;
            return error;
        }
        // Recovery resources belong to this retry invocation.  A CUDA wait or
        // synchronize can therefore re-enter retry without a host mutex
        // spanning a CUDA call or sharing this stream/event with a later Q.
        *result = recovery;
        return cudaSuccess;
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    } catch (...) {
        return cudaErrorUnknown;
    }
}

void destroy_recovery_device(CudaAllocationRecoveryDevice* recovery) noexcept {
    if (recovery == nullptr) {
        return;
    }
    if (recovery->barrier_event != nullptr) {
        (void)cudaEventDestroy(recovery->barrier_event);
    }
    if (recovery->stream != nullptr) {
        (void)cudaStreamDestroy(recovery->stream);
    }
    delete recovery;
}

void unlink_allocation_owner_locked(
    CudaAllocationQuarantineOwner& owner,
    CudaAllocationOwnerRecord* record
) noexcept {
    if (record->previous != nullptr) {
        record->previous->next = record->next;
    } else {
        owner.active_head = record->next;
    }
    if (record->next != nullptr) {
        record->next->previous = record->previous;
    }
}

void unlink_quarantine_owner_locked(
    CudaAllocationQuarantineOwner& owner,
    CudaAllocationOwnerRecord* record
) noexcept {
    if (record->quarantine_previous != nullptr) {
        record->quarantine_previous->quarantine_next = record->quarantine_next;
    } else {
        owner.quarantine_head = record->quarantine_next;
    }
    if (record->quarantine_next != nullptr) {
        record->quarantine_next->quarantine_previous = record->quarantine_previous;
    }
    record->quarantine_previous = nullptr;
    record->quarantine_next = nullptr;
}

void destroy_allocation_owner_record(CudaAllocationOwnerRecord* record) noexcept {
    if (record == nullptr) {
        return;
    }
    if (record->completion_event != nullptr) {
        (void)cudaEventDestroy(record->completion_event);
    }
    record->state = CudaAllocationOwnerState::retired;
    delete record;
}

} // namespace

CudaAllocationOwnerRecord* register_cuda_allocation_owner(
    int device_index,
    cudaStream_t stream
) {
    auto* record = new CudaAllocationOwnerRecord();
    record->stream = stream;
    record->device_index = device_index;
    auto& owner = cuda_allocation_owner();
    bool blocked = false;
    {
        std::lock_guard<std::mutex> lock(owner.mutex);
        blocked = owner.quarantined_count != 0 || owner.admission_closed;
        if (!blocked) {
            record->admission_epoch = owner.quarantine_epoch;
            record->next = owner.active_head;
            if (owner.active_head != nullptr) {
                owner.active_head->previous = record;
            }
            owner.active_head = record;
        }
    }
    if (blocked) {
        destroy_allocation_owner_record(record);
        TORCH_CHECK(
            false,
            "[CuMesh] CUDA allocation is blocked until quarantined allocations are retried"
        );
    }
    return record;
}

cudaError_t prepare_cuda_allocation_owner_admission(
    CudaAllocationOwnerRecord* record,
    cudaStream_t stream
) noexcept {
    if (record == nullptr || record->state != CudaAllocationOwnerState::pending_admission) {
        return cudaErrorInvalidResourceHandle;
    }
    const cudaError_t stream_id_error = cudaStreamGetId(stream, &record->stream_id);
    if (stream_id_error != cudaSuccess) {
        return stream_id_error;
    }
    return cudaEventCreateWithFlags(&record->completion_event, cudaEventDisableTiming);
}

bool commit_cuda_allocation_owner_admission(
    CudaAllocationOwnerRecord* record,
    CudaAllocationOwnerPublication publish,
    void* publish_context
) noexcept {
    if (record == nullptr || record->ptr == nullptr) {
        return false;
    }
    auto& owner = cuda_allocation_owner();
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.quarantined_count != 0 || owner.admission_closed
        || owner.quarantine_epoch != record->admission_epoch
        || record->state != CudaAllocationOwnerState::pending_admission) {
        return false;
    }
    // This is the sole pointer-publication linearization point.  The callback
    // is deliberately host-only and noexcept: no CUDA/runtime call can occur
    // while the allocation-owner mutex is held.
    if (publish != nullptr) {
        publish(publish_context);
    }
    record->state = CudaAllocationOwnerState::committed_active;
    return true;
}

bool commit_cuda_allocation_owner_admission(CudaAllocationOwnerRecord* record) noexcept {
    return commit_cuda_allocation_owner_admission(record, nullptr, nullptr);
}

bool transfer_cuda_allocation_owner_publication(
    CudaAllocationOwnerRecord* record,
    CudaAllocationOwnerPublication publish,
    void* publish_context
) noexcept {
    if (record == nullptr || publish == nullptr) {
        return false;
    }
    auto& owner = cuda_allocation_owner();
    std::lock_guard<std::mutex> lock(owner.mutex);
    // This transfer is for an allocation already admitted before any consumer
    // enqueue.  A later Q generation blocks only new admissions, never this
    // committed allocation's host-only destination-metadata handoff.
    if (record->quarantined || record->retry_claimed
        || record->state != CudaAllocationOwnerState::committed_active) {
        return false;
    }
    publish(publish_context);
    return true;
}

void update_cuda_allocation_owner(CudaAllocationOwnerRecord* owner, void* ptr) noexcept {
    owner->ptr = ptr;
}

void begin_cuda_allocation_owner_release(CudaAllocationOwnerRecord* record) noexcept {
    if (record == nullptr) {
        return;
    }
    auto& owner = cuda_allocation_owner();
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (!record->quarantined && record->state == CudaAllocationOwnerState::committed_active) {
        record->state = CudaAllocationOwnerState::releasing;
    }
}

void release_cuda_allocation_owner(CudaAllocationOwnerRecord* record) noexcept {
    if (record == nullptr) {
        return;
    }
    auto& owner = cuda_allocation_owner();
    {
        std::lock_guard<std::mutex> lock(owner.mutex);
        if (record->quarantined) {
            if (record->retry_claimed
                || (!record->revoke_claimed
                    && record->state != CudaAllocationOwnerState::revoked_cleanup
                    && record->state != CudaAllocationOwnerState::quarantined_async_completion_recorded)) {
                return;
            }
            unlink_quarantine_owner_locked(owner, record);
            --owner.quarantined_count;
            record->quarantined = false;
            record->revoke_claimed = false;
        } else {
            unlink_allocation_owner_locked(owner, record);
        }
        record->state = CudaAllocationOwnerState::retired;
    }
    destroy_allocation_owner_record(record);
}

cudaError_t validate_cuda_allocation_owner_stream(
    CudaAllocationOwnerRecord* record,
    cudaStream_t stream
) noexcept;

cudaError_t record_cuda_allocation_owner_completion(
    CudaAllocationOwnerRecord* record,
    cudaStream_t stream
) noexcept {
    if (record == nullptr || record->completion_event == nullptr) {
        return cudaErrorInvalidResourceHandle;
    }
    const cudaError_t identity_error = validate_cuda_allocation_owner_stream(record, stream);
    if (identity_error != cudaSuccess) {
        record->event_record_error = identity_error;
        return record->event_record_error;
    }
#ifdef CUMESH_ENABLE_TEST_HOOKS
    record->event_record_error = consume_test_failure_count(
        admission_revoke_event_record_failures
    ) ? cudaErrorUnknown : cudaEventRecord(record->completion_event, stream);
#else
    record->event_record_error = cudaEventRecord(record->completion_event, stream);
#endif
    if (record->event_record_error == cudaSuccess) {
        record->completion_recorded = true;
        if (record->async_free_enqueued) {
            auto& owner = cuda_allocation_owner();
            std::lock_guard<std::mutex> lock(owner.mutex);
            record->state = CudaAllocationOwnerState::quarantined_async_completion_recorded;
        }
    }
    return record->event_record_error;
}

cudaError_t validate_cuda_allocation_owner_stream(
    CudaAllocationOwnerRecord* record,
    cudaStream_t stream
) noexcept {
    if (record == nullptr) {
        return cudaErrorInvalidResourceHandle;
    }
    unsigned long long stream_id = 0;
    const cudaError_t error = cudaStreamGetId(stream, &stream_id);
    record->stream_validation_error = error == cudaSuccess && stream_id == record->stream_id
        ? cudaSuccess
        : (error == cudaSuccess ? cudaErrorInvalidResourceHandle : error);
    return record->stream_validation_error;
}

void record_cuda_allocation_owner_completion_sync(
    CudaAllocationOwnerRecord* record,
    cudaError_t error
) noexcept {
    if (record != nullptr && error != cudaSuccess && record->completion_sync_error == cudaSuccess) {
        record->completion_sync_error = error;
    }
}

void mark_cuda_allocation_owner_legacy_free_attempt(
    CudaAllocationOwnerRecord* record
) noexcept {
    if (record != nullptr) {
        // Mark before cudaFree: an error does not prove that the driver left
        // the allocation untouched, so retry must never issue a second free.
        record->free_enqueued = true;
    }
}

void record_cuda_allocation_owner_fallback_result(
    CudaAllocationOwnerRecord* record,
    cudaError_t error
) noexcept {
    if (record != nullptr) {
        record->fallback_error = error;
    }
}

void record_cuda_allocation_owner_primary_failure(
    CudaAllocationOwnerRecord* record,
    cudaError_t error
) noexcept {
    if (record != nullptr && record->primary_error == cudaSuccess) {
        record->primary_error = error;
    }
}

void record_cuda_allocation_owner_cleanup_failure(
    CudaAllocationOwnerRecord* record,
    cudaError_t error
) noexcept {
    if (record != nullptr && error != cudaSuccess && record->cleanup_error == cudaSuccess) {
        record->cleanup_error = error;
    }
}

void quarantine_cuda_allocation_owner(CudaAllocationOwnerRecord* record) noexcept {
    if (record != nullptr) {
        auto& owner = cuda_allocation_owner();
        std::lock_guard<std::mutex> lock(owner.mutex);
        if (!record->quarantined) {
            unlink_allocation_owner_locked(owner, record);
            record->quarantined = true;
            // The caller classifies the CUDA API boundary before Q.  Never
            // derive retry eligibility from stale booleans after that point.
            if (record->state != CudaAllocationOwnerState::quarantined_never_attempted
                && record->state != CudaAllocationOwnerState::quarantined_async_completion_unproved
                && record->state != CudaAllocationOwnerState::quarantined_async_completion_recorded
                && record->state != CudaAllocationOwnerState::permanent_ambiguous) {
                record->state = CudaAllocationOwnerState::quarantined_never_attempted;
            }
            record->quarantine_next = owner.quarantine_head;
            if (owner.quarantine_head != nullptr) {
                owner.quarantine_head->quarantine_previous = record;
            }
            owner.quarantine_head = record;
            ++owner.quarantined_count;
            if (owner.quarantine_epoch == std::numeric_limits<std::uint64_t>::max()) {
                owner.admission_closed = true;
            } else {
                ++owner.quarantine_epoch;
            }
        }
    }
}

void mark_cuda_allocation_owner_never_attempted(
    CudaAllocationOwnerRecord* record,
    bool known_device
) noexcept {
    if (record == nullptr) {
        return;
    }
    auto& owner = cuda_allocation_owner();
    std::lock_guard<std::mutex> lock(owner.mutex);
    record->needs_device_synchronize = known_device;
    record->state = known_device
        ? CudaAllocationOwnerState::quarantined_never_attempted
        : CudaAllocationOwnerState::permanent_ambiguous;
}

void mark_cuda_allocation_owner_async_completion(
    CudaAllocationOwnerRecord* record,
    bool completion_recorded
) noexcept {
    if (record == nullptr) {
        return;
    }
    auto& owner = cuda_allocation_owner();
    std::lock_guard<std::mutex> lock(owner.mutex);
    record->async_free_enqueued = true;
    record->needs_device_synchronize = !completion_recorded;
    record->state = completion_recorded
        ? CudaAllocationOwnerState::quarantined_async_completion_recorded
        : CudaAllocationOwnerState::quarantined_async_completion_unproved;
}

void mark_cuda_allocation_owner_permanent_ambiguity(
    CudaAllocationOwnerRecord* record
) noexcept {
    if (record != nullptr) {
        auto& owner = cuda_allocation_owner();
        std::lock_guard<std::mutex> lock(owner.mutex);
        record->state = CudaAllocationOwnerState::permanent_ambiguous;
    }
}

void retain_persistent_cuda_allocation_quarantine(
    void* ptr,
    int device_index,
    cudaError_t primary_error,
    bool free_attempted,
    bool needs_device_synchronize,
    bool test_synthetic_free_attempt
) noexcept {
    const PersistentCudaAllocationQuarantineSpec spec{
        ptr, device_index, primary_error, free_attempted, false,
        needs_device_synchronize, test_synthetic_free_attempt
    };
    retain_persistent_cuda_allocation_quarantine_batch(&spec, 1);
}

void retain_persistent_cuda_allocation_quarantine_batch(
    const PersistentCudaAllocationQuarantineSpec* specs,
    std::size_t count
) noexcept {
    if (specs == nullptr || count == 0) {
        return;
    }
    CudaAllocationOwnerRecord* batch_head = nullptr;
    CudaAllocationOwnerRecord* batch_tail = nullptr;
    std::size_t retained_count = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto& spec = specs[index];
        if (spec.ptr == nullptr) {
            continue;
        }
        auto* record = new (std::nothrow) CudaAllocationOwnerRecord();
        if (record == nullptr) {
            while (batch_head != nullptr) {
                auto* next = batch_head->quarantine_next;
                delete batch_head;
                batch_head = next;
            }
            // A destructor must never silently drop a live device pointer.
            std::terminate();
        }
        record->ptr = spec.ptr;
        record->device_index = spec.device_index;
        record->primary_error = spec.primary_error;
        record->free_enqueued = spec.legacy_free_attempted;
        record->async_free_enqueued = spec.async_free_attempted;
        record->persistent_fallback = true;
        record->needs_device_synchronize = spec.needs_device_synchronize;
        record->test_synthetic_free_attempt = spec.test_synthetic_free_attempt;
        record->quarantined = true;
        record->state = spec.legacy_free_attempted || spec.async_free_attempted || spec.device_index < 0
            ? CudaAllocationOwnerState::permanent_ambiguous
            : CudaAllocationOwnerState::quarantined_never_attempted;
        record->quarantine_previous = batch_tail;
        if (batch_tail != nullptr) {
            batch_tail->quarantine_next = record;
        } else {
            batch_head = record;
        }
        batch_tail = record;
        ++retained_count;
    }
    if (retained_count == 0) {
        return;
    }
    auto& owner = cuda_allocation_owner();
    std::lock_guard<std::mutex> lock(owner.mutex);
    batch_tail->quarantine_next = owner.quarantine_head;
    if (owner.quarantine_head != nullptr) {
        owner.quarantine_head->quarantine_previous = batch_tail;
    }
    batch_head->quarantine_previous = nullptr;
    owner.quarantine_head = batch_head;
    owner.quarantined_count += retained_count;
    if (owner.quarantine_epoch == std::numeric_limits<std::uint64_t>::max()) {
        owner.admission_closed = true;
    } else {
        ++owner.quarantine_epoch;
    }
}

void quarantine_buffer_async_free_failure(
    void* ptr,
    int device_index,
    cudaError_t error
) noexcept {
    const PersistentCudaAllocationQuarantineSpec spec{
        ptr, device_index, error, false, true, false, false
    };
    retain_persistent_cuda_allocation_quarantine_batch(&spec, 1);
}

cudaError_t revoke_cuda_allocation_owner_admission(
    CudaAllocationOwnerRecord* record,
    void* ptr,
    cudaStream_t stream
) noexcept {
    if (record == nullptr) {
        return cudaErrorInvalidResourceHandle;
    }
    {
        auto& owner = cuda_allocation_owner();
        std::lock_guard<std::mutex> lock(owner.mutex);
        if (record->retry_claimed || record->revoke_claimed) {
            return cudaErrorNotReady;
        }
        // This excludes global retry before any CUDA call.  CUDA work remains
        // outside the owner mutex; every terminal state is published before
        // this claim is released.
        record->revoke_claimed = true;
        record->ptr = ptr;
        record->state = CudaAllocationOwnerState::revoked_cleanup;
    }
    if (ptr == nullptr) {
        release_cuda_allocation_owner(record);
        return cudaSuccess;
    }
    bool cuda_api_called = true;
    const cudaError_t free_error =
#ifdef CUMESH_ENABLE_TEST_HOOKS
        ([&]() noexcept {
            if (consume_test_failure_count(buffer_cleanup_failures)) {
                cuda_api_called = false;
                return cudaErrorUnknown;
            }
            return cudaFreeAsync(ptr, stream);
        })();
#else
        cudaFreeAsync(ptr, stream);
#endif
    if (free_error == cudaSuccess) {
        mark_cuda_allocation_owner_async_completion(record, false);
        const cudaError_t event_error = record_cuda_allocation_owner_completion(record, stream);
        if (event_error == cudaSuccess) {
            // The uncommitted pointer has never become visible to a consumer;
            // its exact-stream free is ordered before this completion event.
            release_cuda_allocation_owner(record);
            return cudaSuccess;
        }
        record_cuda_allocation_owner_cleanup_failure(record, event_error);
    } else {
        record_cuda_allocation_owner_primary_failure(record, free_error);
        if (cuda_api_called) {
            // CUDA 12.4: after cudaFreeAsync returns, pointer access and
            // attribute queries are undefined.  An error therefore remains
            // permanently ambiguous; do not enqueue an event or legacy free.
            mark_cuda_allocation_owner_permanent_ambiguity(record);
        } else {
            mark_cuda_allocation_owner_never_attempted(record, record->device_index >= 0);
        }
    }
    quarantine_cuda_allocation_owner(record);
    {
        auto& owner = cuda_allocation_owner();
        std::lock_guard<std::mutex> lock(owner.mutex);
        record->revoke_claimed = false;
    }
    return free_error == cudaSuccess ? cudaErrorUnknown : free_error;
}

void require_cuda_allocation_quarantine_clear() {
    TORCH_CHECK(
        cuda_allocation_quarantine_count() == 0,
        "[CuMesh] CUDA allocation is blocked until quarantined allocations are retried"
    );
}

std::size_t cuda_allocation_quarantine_count() noexcept {
    auto& owner = cuda_allocation_owner();
    std::lock_guard<std::mutex> lock(owner.mutex);
    return owner.quarantined_count;
}

std::optional<std::uint64_t> cuda_allocation_quarantine_epoch_if_clear() noexcept {
    auto& owner = cuda_allocation_owner();
    std::lock_guard<std::mutex> lock(owner.mutex);
    if (owner.quarantined_count != 0 || owner.admission_closed)
        return std::nullopt;
    return owner.quarantine_epoch;
}

cudaError_t retry_cuda_allocation_quarantine() noexcept {
    auto& owner = cuda_allocation_owner();
    CudaAllocationOwnerRecord* claimed_head = nullptr;
    CudaAllocationOwnerRecord* claimed_tail = nullptr;
    std::uint64_t claimed_epoch = 0;
    {
        std::lock_guard<std::mutex> lock(owner.mutex);
        claimed_epoch = owner.quarantine_epoch;
        for (auto* record = owner.quarantine_head; record != nullptr;
             record = record->quarantine_next) {
            if (!record->quarantined || record->retry_claimed || record->revoke_claimed) {
                continue;
            }
            if (record->state == CudaAllocationOwnerState::permanent_ambiguous) {
                continue;
            }
            if (record->device_index < 0) {
                record->state = CudaAllocationOwnerState::permanent_ambiguous;
                continue;
            }
            if (record->state != CudaAllocationOwnerState::quarantined_never_attempted
                && record->state != CudaAllocationOwnerState::quarantined_async_completion_unproved
                && record->state != CudaAllocationOwnerState::quarantined_async_completion_recorded) {
                record->state = CudaAllocationOwnerState::permanent_ambiguous;
                continue;
            }
            record->retry_claimed = true;
            record->state_before_retry_claim = record->state;
            record->state = CudaAllocationOwnerState::quarantined_claimed;
            record->retry_next = nullptr;
            if (claimed_tail == nullptr) {
                claimed_head = record;
            } else {
                claimed_tail->retry_next = record;
            }
            claimed_tail = record;
        }
        if (claimed_head == nullptr) {
            return owner.quarantined_count == 0 ? cudaSuccess : cudaErrorNotReady;
        }
    }

    auto reset_claims = [&]() noexcept {
        std::lock_guard<std::mutex> lock(owner.mutex);
        auto* record = claimed_head;
        while (record != nullptr) {
            auto* retry_next = record->retry_next;
            record->retry_claimed = false;
            record->state = record->state_before_retry_claim;
            record->retry_next = nullptr;
            record->retry_group_next = nullptr;
            record = retry_next;
        }
    };

    // Group only by device.  The recovery stream waits on every individual
    // completion event, so no host-side sequence number is used to infer the
    // enqueue order of concurrent event records on an origin stream.
    std::unordered_map<int, CudaAllocationOwnerRecord*> device_groups;
    try {
        auto* record = claimed_head;
        while (record != nullptr) {
            auto [group, inserted] = device_groups.emplace(record->device_index, nullptr);
            (void)inserted;
            record->retry_group_next = group->second;
            group->second = record;
            record = record->retry_next;
        }
    } catch (const std::bad_alloc&) {
        reset_claims();
        return cudaErrorMemoryAllocation;
    } catch (...) {
        reset_claims();
        return cudaErrorUnknown;
    }

    int original_device = -1;
    const cudaError_t current_device_error = cudaGetDevice(&original_device);
    if (current_device_error != cudaSuccess) {
        reset_claims();
        return current_device_error;
    }

    cudaError_t first_error = cudaSuccess;
    for (auto& [device_index, group_head] : device_groups) {
        cudaError_t group_error = cudaSetDevice(device_index);
        for (auto* record = group_head; record != nullptr; record = record->retry_group_next) {
            record->set_device_error = group_error;
        }

        bool needs_device_synchronize = false;
        for (auto* record = group_head; record != nullptr; record = record->retry_group_next) {
            needs_device_synchronize = needs_device_synchronize || record->needs_device_synchronize;
        }
        CudaAllocationRecoveryDevice* recovery = nullptr;
        if (group_error == cudaSuccess && needs_device_synchronize) {
#ifdef CUMESH_ENABLE_TEST_HOOKS
            quarantine_retry_stream_synchronizations.fetch_add(1, std::memory_order_relaxed);
            group_error = consume_test_failure_count(quarantine_retry_completion_failures)
                ? cudaErrorUnknown
                : cudaDeviceSynchronize();
#else
            group_error = cudaDeviceSynchronize();
#endif
            if (group_error != cudaSuccess) {
                for (auto* record = group_head; record != nullptr;
                     record = record->retry_group_next) {
                    record->barrier_synchronize_error = group_error;
                }
            }
        } else if (group_error == cudaSuccess) {
            group_error = recovery_device_for(&recovery);
        }
        if (group_error == cudaSuccess && !needs_device_synchronize) {
            for (auto* record = group_head; record != nullptr; record = record->retry_group_next) {
                if (!record->completion_recorded || record->completion_event == nullptr) {
                    record->cleanup_error = cudaErrorInvalidResourceHandle;
                    group_error = cudaErrorInvalidResourceHandle;
                    break;
                }
#ifdef CUMESH_ENABLE_TEST_HOOKS
                quarantine_retry_event_waits.fetch_add(1, std::memory_order_relaxed);
#endif
                const cudaError_t wait_error = cudaStreamWaitEvent(
                    recovery->stream,
                    record->completion_event,
                    0
                );
                if (wait_error != cudaSuccess) {
                    record->recovery_wait_error = wait_error;
                    record->cleanup_error = wait_error;
                    group_error = wait_error;
                    break;
                }
            }
            if (group_error == cudaSuccess) {
                group_error = cudaEventRecord(recovery->barrier_event, recovery->stream);
                if (group_error != cudaSuccess) {
                    for (auto* record = group_head; record != nullptr;
                         record = record->retry_group_next) {
                        record->barrier_record_error = group_error;
                    }
                }
            }
            if (group_error == cudaSuccess) {
#ifdef CUMESH_ENABLE_TEST_HOOKS
                quarantine_retry_stream_synchronizations.fetch_add(1, std::memory_order_relaxed);
                group_error = consume_test_failure_count(quarantine_retry_completion_failures)
                    ? cudaErrorUnknown
                    : cudaEventSynchronize(recovery->barrier_event);
#else
                group_error = cudaEventSynchronize(recovery->barrier_event);
#endif
                if (group_error != cudaSuccess) {
                    for (auto* record = group_head; record != nullptr;
                         record = record->retry_group_next) {
                        record->barrier_synchronize_error = group_error;
                    }
                }
            }
        }
        if (group_error != cudaSuccess && first_error == cudaSuccess) {
            first_error = group_error;
        }

        if (group_error == cudaSuccess) {
            bool legacy_free_ambiguous_for_device = false;
            for (auto* record = group_head; record != nullptr; record = record->retry_group_next) {
                record->completion_proved = true;
                // A cudaFree error does not prove that the driver left the
                // allocation untouched.  Never issue a second legacy free
                // for that owner; retain it as quarantined telemetry instead.
                cudaError_t legacy_free_error = cudaSuccess;
                if (record->state_before_retry_claim
                    == CudaAllocationOwnerState::quarantined_async_completion_recorded
                    || record->state_before_retry_claim
                        == CudaAllocationOwnerState::quarantined_async_completion_unproved) {
                    // The exact async free was already enqueued and has a
                    // durable completion event.  The barrier proves it; a
                    // second free would violate the CUDA API boundary.
                    record->retry_error = cudaSuccess;
                    legacy_free_error = cudaSuccess;
                } else if (record->free_enqueued) {
                    record->retry_error = cudaErrorNotReady;
                    legacy_free_error = record->retry_error;
                } else if (legacy_free_ambiguous_for_device) {
                    // A preceding direct cudaFree failed ambiguously on this
                    // device.  Do not turn later never-attempted owners into
                    // permanent ambiguity; leave them retryable after a new
                    // device-completion proof.
                    record->retry_error = cudaErrorNotReady;
                    record->needs_device_synchronize = true;
                    legacy_free_error = record->retry_error;
                } else {
                    // Persistent destructor records are retried only after a
                    // successful device-wide completion proof on their exact
                    // device.  Normal records use the durable event barrier.
                    record->free_enqueued = true;
#ifdef CUMESH_ENABLE_TEST_HOOKS
                    quarantine_retry_legacy_free_attempts.fetch_add(1, std::memory_order_relaxed);
#endif
                    record->fallback_error = cudaFree(record->ptr);
                    legacy_free_error = record->fallback_error;
                    record->retry_error = legacy_free_error;
                    if (legacy_free_error != cudaSuccess) {
                        record->state_before_retry_claim = CudaAllocationOwnerState::permanent_ambiguous;
                        legacy_free_ambiguous_for_device = true;
                    }
                }
                if (legacy_free_error != cudaSuccess && first_error == cudaSuccess) {
                    first_error = legacy_free_error;
                }
            }
        }

        CudaAllocationOwnerRecord* retired_head = nullptr;
        {
            std::lock_guard<std::mutex> lock(owner.mutex);
            auto* record = group_head;
            while (record != nullptr) {
                auto* group_next = record->retry_group_next;
                // `fallback_error` may still carry its constructor default on
                // a record retained after an ambiguous direct cudaFree.  The
                // authoritative outcome is retry_error: only this retry's
                // exact successful free, or a proven prior async free, may
                // retire an owner.
                if (group_error == cudaSuccess && record->retry_error == cudaSuccess) {
                    --owner.quarantined_count;
                    unlink_quarantine_owner_locked(owner, record);
                    record->state = CudaAllocationOwnerState::retired;
                    record->retry_next = retired_head;
                    record->retry_group_next = nullptr;
                    retired_head = record;
                } else {
                    if (group_error != cudaSuccess) {
                        record->cleanup_error = group_error;
                    }
                    record->retry_claimed = false;
                    record->state = record->state_before_retry_claim;
                    record->retry_next = nullptr;
                    record->retry_group_next = nullptr;
                }
                record = group_next;
            }
        }
        while (retired_head != nullptr) {
            auto* retired_next = retired_head->retry_next;
            destroy_allocation_owner_record(retired_head);
            retired_head = retired_next;
        }
        destroy_recovery_device(recovery);
    }

    const cudaError_t restore_error = cudaSetDevice(original_device);
    if (restore_error != cudaSuccess) {
        std::lock_guard<std::mutex> lock(owner.mutex);
        for (auto* record = owner.quarantine_head; record != nullptr;
             record = record->quarantine_next) {
            if (record->quarantined) {
                record->restore_error = restore_error;
            }
        }
    }
    if (restore_error != cudaSuccess && first_error == cudaSuccess) {
        first_error = restore_error;
    }
    {
        std::lock_guard<std::mutex> lock(owner.mutex);
        if (owner.quarantined_count != 0 || owner.quarantine_epoch != claimed_epoch) {
            if (first_error == cudaSuccess) {
                first_error = cudaErrorNotReady;
            }
        }
    }
    return first_error;
}

namespace {

std::atomic<std::uint64_t> destructor_release_errors{0};
std::atomic<std::uint64_t> destructor_device_synchronizations{0};
std::atomic<std::uint64_t> next_mesh_owner_id{1};

class CudaEventConstructionFailure final : public std::runtime_error {
public:
    explicit CudaEventConstructionFailure(cudaError_t error)
        : std::runtime_error(cudaGetErrorString(error)), error_(error) {}

    cudaError_t error() const noexcept {
        return error_;
    }

private:
    cudaError_t error_;
};

#ifdef CUMESH_ENABLE_TEST_HOOKS
std::atomic<std::int64_t> release_successes_before_failure{-1};
std::atomic<bool> fail_release_event_record{false};
std::atomic<bool> fail_release_event_create{false};
std::atomic<bool> fail_release_token_allocation{false};
std::atomic<bool> fail_release_caller_wait{false};
std::atomic<bool> fail_release_synchronize{false};
std::atomic<bool> fail_destructor_device_synchronize{false};
std::atomic<bool> fail_destructor_legacy_free{false};
std::atomic<std::uint64_t> destructor_legacy_free_attempts{0};

bool consume_failure(std::atomic<bool>& failure) noexcept {
    return failure.exchange(false, std::memory_order_relaxed);
}
#endif

template<typename Operation>
cudaError_t guarded_cuda_call(int device_index, Operation&& operation) noexcept {
    int original_device = -1;
    const cudaError_t current_device_error = cudaGetDevice(&original_device);
    if (current_device_error != cudaSuccess) {
        return current_device_error;
    }
    const cudaError_t set_device_error = cudaSetDevice(device_index);
    if (set_device_error != cudaSuccess) {
        return set_device_error;
    }
    const cudaError_t operation_error = operation();
    const cudaError_t restore_error = cudaSetDevice(original_device);
    return operation_error == cudaSuccess ? restore_error : operation_error;
}

cudaError_t enqueue_stream_dependency(
    const c10::cuda::CUDAStream& producer,
    const c10::cuda::CUDAStream& consumer
) noexcept {
    if (producer == consumer) {
        return cudaSuccess;
    }
    if (producer.device_index() != consumer.device_index()) {
        return cudaErrorInvalidDevice;
    }
    return guarded_cuda_call(
        producer.device_index(),
        [&]() {
            cudaEvent_t event = nullptr;
            cudaError_t first_error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            if (first_error == cudaSuccess) {
                first_error = cudaEventRecord(event, producer.stream());
            }
            if (first_error == cudaSuccess) {
                first_error = cudaStreamWaitEvent(consumer.stream(), event, 0);
            }
            if (event != nullptr) {
                const cudaError_t destroy_error = cudaEventDestroy(event);
                if (first_error == cudaSuccess) {
                    first_error = destroy_error;
                }
            }
            return first_error;
        }
    );
}

template<typename Tuple>
std::size_t count_allocated(const Tuple& buffers) noexcept {
    std::size_t count = 0;
    std::apply(
        [&](const auto&... buffer) {
            ((count += buffer.ptr != nullptr ? 1 : 0), ...);
        },
        buffers
    );
    return count;
}

template<typename Tuple>
int require_single_device(const Tuple& buffers, const char* operation) {
    int device_index = -1;
    std::apply(
        [&](const auto&... buffer) {
            auto inspect = [&](const auto& item) {
                if (item.ptr == nullptr) {
                    return;
                }
                TORCH_CHECK(
                    item.device_index >= 0,
                    "[CuMesh] ", operation, ": allocated buffer has no CUDA device owner"
                );
                TORCH_CHECK(
                    device_index < 0 || device_index == item.device_index,
                    "[CuMesh] ", operation, ": buffers span multiple CUDA devices"
                );
                device_index = item.device_index;
            };
            (inspect(buffer), ...);
        },
        buffers
    );
    return device_index;
}

template<typename Tuple>
void preflight_pointer_ownership(const Tuple& buffers, int expected_device, const char* operation) {
    std::apply(
        [&](const auto&... buffer) {
            auto inspect = [&](const auto& item) {
                if (item.ptr == nullptr) {
                    return;
                }
                TORCH_CHECK(
                    item.device_index == expected_device,
                    "[CuMesh] ", operation, ": buffer metadata device does not match canonical device"
                );
                cudaPointerAttributes attributes{};
                CUDA_CHECK(cudaPointerGetAttributes(&attributes, item.ptr));
                TORCH_CHECK(
                    attributes.type == cudaMemoryTypeDevice && attributes.device == expected_device,
                    "[CuMesh] ", operation, ": pointer ownership does not match canonical CUDA device"
                );
            };
            (inspect(buffer), ...);
        },
        buffers
    );
}

void require_async_release_support(cudaStream_t stream, int device_index, const char* operation) {
    int memory_pools_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memory_pools_supported, cudaDevAttrMemoryPoolsSupported, device_index));
    TORCH_CHECK(
        memory_pools_supported != 0,
        "[CuMesh] ", operation, ": CUDA device does not support stream-ordered memory release"
    );

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    TORCH_CHECK(
        capture_status == cudaStreamCaptureStatusNone,
        "[CuMesh] ", operation, ": release of CuMesh-owned allocations is not allowed during CUDA graph capture"
    );
}

struct ReleaseEnqueueOutcome {
    std::size_t released;
    cudaError_t error;
};

template <typename BufferType>
cudaError_t enqueue_buffer_release(BufferType &buffer, cudaStream_t stream,
                                   int device_index,
                                   bool *cuda_api_called) noexcept {
  *cuda_api_called = false;
#ifdef CUMESH_ENABLE_TEST_HOOKS
  std::int64_t remaining =
      release_successes_before_failure.load(std::memory_order_relaxed);
  if (remaining == 0) {
    // The injected boundary is one-shot.  Leaving this at zero poisons
    // every later real release in the same compiled fixture and turns a
    // source-derived fault case into order-dependent synthetic behavior.
    release_successes_before_failure.store(-1, std::memory_order_relaxed);
    return cudaErrorUnknown;
  }
  if (remaining > 0) {
    release_successes_before_failure.fetch_sub(1, std::memory_order_relaxed);
  }
#endif
  *cuda_api_called = true;
  return buffer.release_async(stream, device_index, false);
}

template<typename Tuple>
ReleaseEnqueueOutcome release_tuple_async(Tuple&& buffers, cudaStream_t stream, int device_index) {
    ReleaseEnqueueOutcome outcome{0, cudaSuccess};
    constexpr std::size_t buffer_count = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    std::array<PersistentCudaAllocationQuarantineSpec, buffer_count> retained{};
    std::size_t retained_count = 0;
    std::apply(
        [&](auto&... buffer) {
            auto release_one = [&](auto& item) {
                if (item.ptr == nullptr) {
                    return;
                }
                if (outcome.error == cudaSuccess) {
                    bool cuda_api_called = false;
                    outcome.error = enqueue_buffer_release(
                        item, stream, device_index, &cuda_api_called
                    );
                    if (outcome.error != cudaSuccess) {
                        // A test-hook/pre-call failure is never attempted;
                        // a returned production API error is permanently
                        // ambiguous at this exact tuple slot.
                        const bool failed_async_boundary = cuda_api_called;
#ifdef CUMESH_ENABLE_TEST_HOOKS
                        const bool synthetic_failed_async_boundary = !cuda_api_called;
#endif
                        retained[retained_count] = PersistentCudaAllocationQuarantineSpec{
                            item.ptr, item.device_index, outcome.error, false,
                            failed_async_boundary, !cuda_api_called,
#ifdef CUMESH_ENABLE_TEST_HOOKS
                            synthetic_failed_async_boundary
#else
                            false
#endif
                        };
                        ++retained_count;
                        return;
                    }
                }
                if (outcome.error == cudaSuccess) {
                    ++outcome.released;
                    return;
                }
                // The failing cudaFreeAsync is permanent ambiguity.  Every
                // later tuple member was never attempted; retain those too so
                // the object cannot issue a second release after a partial
                // failure.  Publish the full disposition batch atomically.
                // The first retained item was already classified by the API
                // call above.  Every later item was never attempted.
                retained[retained_count] = PersistentCudaAllocationQuarantineSpec{
                    item.ptr,
                    item.device_index,
                    outcome.error,
                    false,
                    false,
                    true,
                    false
                };
                ++retained_count;
            };
            (release_one(buffer), ...);
        },
        buffers
    );
    if (retained_count != 0) {
        retain_persistent_cuda_allocation_quarantine_batch(retained.data(), retained_count);
        std::apply(
            [&](auto&... buffer) {
                auto reset_retained = [&](auto& item) {
                    for (std::size_t index = 0; index < retained_count; ++index) {
                        if (item.ptr == retained[index].ptr) {
                            item.reset();
                            return;
                        }
                    }
                };
                (reset_retained(buffer), ...);
            },
            buffers
        );
    }
    return outcome;
}

template<typename LeftTuple, typename RightTuple, std::size_t... Index>
void swap_buffer_tuples(
    LeftTuple&& left,
    RightTuple&& right,
    std::index_sequence<Index...>
) noexcept {
    (std::get<Index>(left).swap(std::get<Index>(right)), ...);
}

} // namespace

ReleaseCompletionToken::ReleaseCompletionToken(
    int device_index,
    std::int64_t stream_id,
    std::uint64_t owner_id,
    std::uint64_t generation
) : event_(nullptr),
    device_index_(device_index),
    stream_id_(stream_id),
    owner_id_(owner_id),
    generation_(generation) {
    c10::cuda::CUDAGuard device_guard(device_index_);
    cudaEvent_t event = nullptr;
#ifdef CUMESH_ENABLE_TEST_HOOKS
    const cudaError_t error = consume_failure(fail_release_event_create)
        ? cudaErrorMemoryAllocation
        : cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
#else
    const cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
#endif
    if (error != cudaSuccess) {
        if (event != nullptr) {
            (void)cudaEventDestroy(event);
        }
        throw CudaEventConstructionFailure(error);
    }
    event_ = event;
}

ReleaseCompletionToken::~ReleaseCompletionToken() noexcept {
    if (event_ == nullptr) {
        return;
    }
    try {
        c10::cuda::CUDAGuard device_guard(device_index_);
        const cudaError_t error = cudaEventDestroy(event_);
        if (error != cudaSuccess) {
            std::fprintf(
                stderr,
                "[CuMesh] Release completion event destruction failed with CUDA error %d: %s\n",
                static_cast<int>(error),
                cudaGetErrorString(error)
            );
        }
    } catch (...) {
        std::fprintf(stderr, "[CuMesh] Release completion event destruction raised unexpectedly\n");
    }
}

int ReleaseCompletionToken::device_index() const noexcept {
    return device_index_;
}

std::int64_t ReleaseCompletionToken::stream_id() const noexcept {
    return stream_id_;
}

std::uint64_t ReleaseCompletionToken::owner_id() const noexcept {
    return owner_id_;
}

std::uint64_t ReleaseCompletionToken::generation() const noexcept {
    return generation_;
}

bool ReleaseCompletionToken::is_complete() const {
    TORCH_CHECK(recorded_.load(std::memory_order_acquire), "[CuMesh] Release completion event has not been recorded");
    c10::cuda::CUDAGuard device_guard(device_index_);
    const cudaError_t error = cudaEventQuery(event_);
    if (error == cudaSuccess) {
        return true;
    }
    if (error == cudaErrorNotReady) {
        (void)cudaGetLastError();
        return false;
    }
    CUDA_CHECK(error);
    return false;
}

std::string ReleaseCompletionToken::status() const {
    return is_complete() ? "release_completed" : "release_enqueued";
}

std::int64_t ReleaseCompletionToken::wait_on_current_stream() const {
    c10::cuda::CUDAGuard device_guard(device_index_);
    const c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(device_index_);
    TORCH_CHECK(
        stream.device_index() == device_index_,
        "[CuMesh] Release completion wait stream device does not match release device"
    );
    CUDA_CHECK(wait(stream));
    return static_cast<std::int64_t>(stream.id());
}

void ReleaseCompletionToken::synchronize() const {
    c10::cuda::CUDAGuard device_guard(device_index_);
    CUDA_CHECK(synchronize_noexcept());
}

cudaError_t ReleaseCompletionToken::record(const c10::cuda::CUDAStream& stream) noexcept {
    if (stream.device_index() != device_index_) {
        return cudaErrorInvalidDevice;
    }
    if (static_cast<std::int64_t>(stream.id()) != stream_id_) {
        return cudaErrorInvalidResourceHandle;
    }
    const cudaError_t error = guarded_cuda_call(
        device_index_,
        [&]() { return cudaEventRecord(event_, stream.stream()); }
    );
    if (error == cudaSuccess) {
        recorded_.store(true, std::memory_order_release);
    }
    return error;
}

cudaError_t ReleaseCompletionToken::wait(const c10::cuda::CUDAStream& stream) const noexcept {
    if (!recorded_.load(std::memory_order_acquire)) {
        return cudaErrorInvalidResourceHandle;
    }
    if (stream.device_index() != device_index_) {
        return cudaErrorInvalidDevice;
    }
    return guarded_cuda_call(
        stream.device_index(),
        [&]() { return cudaStreamWaitEvent(stream.stream(), event_, 0); }
    );
}

cudaError_t ReleaseCompletionToken::synchronize_noexcept() const noexcept {
    if (!recorded_.load(std::memory_order_acquire)) {
        return cudaErrorInvalidResourceHandle;
    }
    return guarded_cuda_call(
        device_index_,
        [&]() { return cudaEventSynchronize(event_); }
    );
}

bool ReleaseCompletionToken::recorded() const noexcept {
    return recorded_.load(std::memory_order_acquire);
}

CuMeshOperationScope::CuMeshOperationScope(CuMesh& owner)
    : owner_(owner),
      lock_(owner.operation_mutex_),
      canonical_stream_(owner.canonical_stream_locked()),
      caller_stream_(c10::cuda::getCurrentCUDAStream(canonical_stream_.device_index())) {
    try {
    require_cuda_allocation_quarantine_clear();
    owner_.require_operable_locked();
    TORCH_CHECK(
        canonical_stream_.device_index() == owner_.canonical_device_index_,
        "[CuMesh] Canonical stream device does not match canonical mesh device"
    );
    TORCH_CHECK(
        caller_stream_.device_index() == owner_.canonical_device_index_,
        "[CuMesh] Caller stream device does not match canonical mesh device"
    );
    if (caller_stream_ != canonical_stream_) {
        ingress_event_.record(caller_stream_);
        ingress_event_.block(canonical_stream_);
    }
    stream_guard_.emplace(canonical_stream_);
    } catch (...) {
        owner_.invalidate_spatial_audit_locked();
        throw;
    }
}

CuMeshOperationScope::~CuMeshOperationScope() noexcept(false) {
    if (std::uncaught_exceptions() != 0) {
        owner_.invalidate_spatial_audit_locked();
    }
    if (bridged_to_caller_) {
        return;
    }
    if (std::uncaught_exceptions() == 0) {
        try {
            bridge_to_caller();
        } catch (...) {
            owner_.invalidate_spatial_audit_locked();
            throw;
        }
        return;
    }
    try {
        bridge_to_caller();
    } catch (...) {
        owner_.quarantine_operation_scope_failure_noexcept(
            "canonical_to_caller_bridge_failed_during_exception_unwind"
        );
    }
}

cudaStream_t CuMeshOperationScope::stream() const noexcept {
    return canonical_stream_.stream();
}

int CuMeshOperationScope::device_index() const noexcept {
    return canonical_stream_.device_index();
}

std::int64_t CuMeshOperationScope::stream_id() const noexcept {
    return static_cast<std::int64_t>(canonical_stream_.id());
}

c10::cuda::CUDAStream CuMeshOperationScope::caller_stream() const noexcept {
    return caller_stream_;
}

c10::cuda::CUDAStream CuMeshOperationScope::execution_stream() const noexcept {
    return canonical_stream_;
}

void CuMeshOperationScope::bridge_to_caller() {
    if (bridged_to_caller_) {
        return;
    }
    if (caller_stream_ == canonical_stream_) {
        bridged_to_caller_ = true;
        return;
    }
    at::cuda::CUDAEvent completion_event;
    completion_event.record(canonical_stream_);
    completion_event.block(caller_stream_);
    bridged_to_caller_ = true;
}

CuMesh::CuMesh()
    : owner_id_(next_mesh_owner_id.fetch_add(1, std::memory_order_relaxed)) {}

CuMesh::~CuMesh() noexcept {
    release_fallback_noexcept();
}

void CuMesh::swap(CuMesh& other) {
    if (this == &other) {
        return;
    }

    std::scoped_lock lock(operation_mutex_, other.operation_mutex_);
    invalidate_spatial_audit_locked();
    other.invalidate_spatial_audit_locked();
    require_cuda_allocation_quarantine_clear();
    require_operable_locked();
    other.require_operable_locked();

    auto left = all_buffers();
    auto right = other.all_buffers();
    static_assert(std::tuple_size_v<decltype(left)> == buffer_slot_count);
    static_assert(std::tuple_size_v<decltype(left)> == std::tuple_size_v<decltype(right)>);
    const int left_device = require_single_device(left, "swap left owner");
    const int right_device = require_single_device(right, "swap right owner");
    TORCH_CHECK(
        left_device < 0 || (
            canonical_stream_.has_value()
            && canonical_device_index_ == left_device
            && canonical_stream_->device_index() == left_device
        ),
        "[CuMesh] swap left owner canonical stream does not match its buffers"
    );
    TORCH_CHECK(
        right_device < 0 || (
            other.canonical_stream_.has_value()
            && other.canonical_device_index_ == right_device
            && other.canonical_stream_->device_index() == right_device
        ),
        "[CuMesh] swap right owner canonical stream does not match its buffers"
    );
    if (left_device >= 0) {
        preflight_pointer_ownership(left, left_device, "swap left owner");
    }
    if (right_device >= 0) {
        preflight_pointer_ownership(right, right_device, "swap right owner");
    }

    const int swap_device = left_device >= 0 ? left_device : right_device;
    std::optional<c10::cuda::CUDAStream> continuation_stream;
    const bool owners_share_device =
        left_device < 0 || right_device < 0 || left_device == right_device;
    if (swap_device >= 0 && owners_share_device) {
        c10::cuda::CUDAGuard device_guard(swap_device);
        continuation_stream = c10::cuda::getCurrentCUDAStream(swap_device);
        auto join_owner = [&](
            int owner_device,
            const std::optional<c10::cuda::CUDAStream>& owner_stream
        ) {
            if (owner_device < 0) {
                return;
            }
            TORCH_CHECK(
                owner_device == swap_device && owner_stream.has_value(),
                "[CuMesh] swap owner is missing a same-device canonical stream"
            );
            if (*owner_stream != *continuation_stream) {
                at::cuda::CUDAEvent ready;
                ready.record(*owner_stream);
                ready.block(*continuation_stream);
            }
        };
        join_owner(left_device, canonical_stream_);
        join_owner(right_device, other.canonical_stream_);
    }
    swap_buffer_tuples(left, right, std::make_index_sequence<buffer_slot_count>{});

    using std::swap;
    swap(num_conn_comps, other.num_conn_comps);
    swap(num_bound_conn_comps, other.num_bound_conn_comps);
    swap(num_bound_loops, other.num_bound_loops);
    swap(atlas_num_charts, other.atlas_num_charts);
    ++generation_;
    ++other.generation_;
    if (continuation_stream.has_value()) {
        canonical_device_index_ = swap_device;
        canonical_stream_ = continuation_stream;
        other.canonical_device_index_ = swap_device;
        other.canonical_stream_ = continuation_stream;
    } else {
        swap(canonical_device_index_, other.canonical_device_index_);
        swap(canonical_stream_, other.canonical_stream_);
    }
}

ReleaseResult CuMesh::release_async(bool synchronize) {
    std::unique_lock<std::recursive_mutex> lock(operation_mutex_);
    invalidate_spatial_audit_locked();
    auto buffers = all_buffers();
    static_assert(std::tuple_size_v<decltype(buffers)> == buffer_slot_count);
    int device_index = require_single_device(buffers, "release_async");

    if (pending_release_token_ != nullptr) {
        TORCH_CHECK(
            canonical_stream_.has_value() && canonical_device_index_ >= 0,
            "[CuMesh] release_async: pending release has no canonical stream"
        );
        const c10::cuda::CUDAStream stream = *canonical_stream_;
        TORCH_CHECK(
            pending_release_token_->device_index() == canonical_device_index_
                && pending_release_token_->stream_id() == static_cast<std::int64_t>(stream.id())
                && pending_release_token_->owner_id() == owner_id_
                && pending_release_token_->generation() == pending_release_generation_,
            "[CuMesh] release_async: pending completion token origin does not match canonical device/stream"
        );
        c10::cuda::CUDAGuard pending_release_device_guard(canonical_device_index_);
        const c10::cuda::CUDAStream caller_stream = c10::cuda::getCurrentCUDAStream(canonical_device_index_);
        cudaError_t completion_error = cudaSuccess;
        if (!pending_release_token_->recorded()) {
#ifdef CUMESH_ENABLE_TEST_HOOKS
            completion_error = consume_failure(fail_release_event_record)
                ? cudaErrorUnknown
                : pending_release_token_->record(stream);
#else
            completion_error = pending_release_token_->record(stream);
#endif
        }
        if (completion_error == cudaSuccess && caller_stream != stream) {
#ifdef CUMESH_ENABLE_TEST_HOOKS
            completion_error = consume_failure(fail_release_caller_wait)
                ? cudaErrorUnknown
                : pending_release_token_->wait(caller_stream);
#else
            completion_error = pending_release_token_->wait(caller_stream);
#endif
        }
        int synchronize_count = 0;
        if (completion_error == cudaSuccess && synchronize) {
#ifdef CUMESH_ENABLE_TEST_HOOKS
            completion_error = consume_failure(fail_release_synchronize)
                ? cudaErrorUnknown
                : pending_release_token_->synchronize_noexcept();
#else
            completion_error = pending_release_token_->synchronize_noexcept();
#endif
            if (completion_error == cudaSuccess) {
                synchronize_count = 1;
            }
        }
        if (completion_error != cudaSuccess) {
            completion_proof_pending_ = true;
            quarantine_reason_ = std::string("release_completion_unproved: ") + cudaGetErrorString(completion_error);
            return ReleaseResult{
                "release_completion_failed_quarantined",
                pending_release_count_,
                canonical_device_index_,
                static_cast<std::int64_t>(stream.id()),
                synchronize_count,
                pending_release_token_->recorded() ? pending_release_token_ : nullptr,
                count_allocated(buffers),
                static_cast<int>(completion_error),
                quarantine_reason_,
                owner_id_,
                pending_release_generation_,
                static_cast<int>(completion_error),
                cudaGetErrorString(completion_error),
                0,
                ""
            };
        }
        completion_proof_pending_ = false;
        if (device_index < 0) {
            const auto token = pending_release_token_;
            const std::uint64_t released_generation = pending_release_generation_;
            if (synchronize) {
                pending_release_token_.reset();
                pending_release_count_ = 0;
                pending_release_generation_ = 0;
            }
            const std::size_t global_quarantine_count = cuda_allocation_quarantine_count();
            // A same-release primary/global-Q failure stays durable even
            // after this call proved the independent completion token.
            if (durable_global_quarantine_ && global_quarantine_count == 0) {
                durable_global_quarantine_ = false;
                quarantine_reason_.clear();
            }
            if (completion_proof_pending_ || durable_global_quarantine_
                || global_quarantine_count != 0) {
                return ReleaseResult{
                    "release_primary_failure_quarantined",
                    0,
                    canonical_device_index_,
                    static_cast<std::int64_t>(stream.id()),
                    synchronize_count,
                    synchronize ? nullptr : token,
                    0,
                    static_cast<int>(cudaErrorUnknown),
                    quarantine_reason_,
                    owner_id_,
                    released_generation,
                    0,
                    "",
                    static_cast<int>(cudaErrorUnknown),
                    quarantine_reason_
                };
            }
            return ReleaseResult{
                synchronize ? "release_completed" : "release_enqueued",
                0,
                canonical_device_index_,
                static_cast<std::int64_t>(stream.id()),
                synchronize_count,
                token,
                0,
                0,
                "",
                owner_id_,
                released_generation,
                0,
                "",
                0,
                ""
            };
        }
        pending_release_token_.reset();
        pending_release_count_ = 0;
        pending_release_generation_ = 0;
    }

    if (device_index < 0) {
        const std::size_t global_quarantine_count = cuda_allocation_quarantine_count();
        const bool recovered_durable_global_quarantine =
            durable_global_quarantine_ && global_quarantine_count == 0;
        if (recovered_durable_global_quarantine) {
            durable_global_quarantine_ = false;
            quarantine_reason_.clear();
        }
        if (completion_proof_pending_ || durable_global_quarantine_
            || global_quarantine_count != 0) {
            return ReleaseResult{
                "release_completion_failed_quarantined",
                0,
                canonical_device_index_,
                canonical_stream_.has_value() ? static_cast<std::int64_t>(canonical_stream_->id()) : -1,
                0,
                nullptr,
                0,
                static_cast<int>(cudaErrorUnknown),
                quarantine_reason_,
                owner_id_,
                generation_,
                static_cast<int>(cudaErrorUnknown),
                quarantine_reason_,
                0,
                ""
            };
        }
        reset_metadata();
        return ReleaseResult{
            recovered_durable_global_quarantine ? "release_completed" : "already_released",
            0,
            canonical_device_index_,
            canonical_stream_.has_value() ? static_cast<std::int64_t>(canonical_stream_->id()) : -1,
            0,
            nullptr,
            0,
            0,
            "",
            owner_id_,
            generation_,
            0,
            "",
            0,
            ""
        };
    }

    TORCH_CHECK(
        canonical_device_index_ == device_index && canonical_stream_.has_value(),
        "[CuMesh] release_async: canonical stream ownership is unavailable or mismatched"
    );
    const c10::cuda::CUDAStream stream = *canonical_stream_;
    TORCH_CHECK(
        stream.device_index() == device_index,
        "[CuMesh] release_async: canonical stream device does not match buffer device"
    );
    const c10::cuda::CUDAStream caller_stream = c10::cuda::getCurrentCUDAStream(device_index);
    TORCH_CHECK(
        caller_stream.device_index() == device_index,
        "[CuMesh] release_async: caller stream device does not match buffer device"
    );

    c10::cuda::CUDAStreamGuard stream_guard(stream);
    require_async_release_support(stream.stream(), device_index, "release_async");
    if (caller_stream != stream) {
        require_async_release_support(caller_stream.stream(), device_index, "release_async caller stream");
    }
    preflight_pointer_ownership(buffers, device_index, "release_async");

    const cudaError_t ingress_error = enqueue_stream_dependency(caller_stream, stream);
    if (ingress_error != cudaSuccess) {
        durable_global_quarantine_ = true;
        quarantine_reason_ = std::string("release_ingress_failed_quarantined: ")
            + cudaGetErrorString(ingress_error);
        return ReleaseResult{
            "release_ingress_failed_quarantined",
            0,
            device_index,
            static_cast<std::int64_t>(stream.id()),
            0,
            nullptr,
            count_allocated(buffers),
            static_cast<int>(ingress_error),
            quarantine_reason_,
            owner_id_,
            generation_,
            0,
            "",
            static_cast<int>(ingress_error),
            quarantine_reason_
        };
    }

    std::shared_ptr<ReleaseCompletionToken> completion_token;
    try {
#ifdef CUMESH_ENABLE_TEST_HOOKS
        if (consume_failure(fail_release_token_allocation)) {
            throw std::bad_alloc();
        }
#endif
        completion_token = std::make_shared<ReleaseCompletionToken>(
            device_index,
            static_cast<std::int64_t>(stream.id()),
            owner_id_,
            generation_
        );
    } catch (const std::bad_alloc&) {
        durable_global_quarantine_ = true;
        quarantine_reason_ = "release_token_allocation_failed";
        return ReleaseResult{
            "release_enqueue_failed_quarantined",
            0,
            device_index,
            static_cast<std::int64_t>(stream.id()),
            0,
            nullptr,
            count_allocated(buffers),
            static_cast<int>(cudaErrorMemoryAllocation),
            quarantine_reason_,
            owner_id_,
            generation_,
            0,
            "",
            static_cast<int>(cudaErrorMemoryAllocation),
            quarantine_reason_
        };
    } catch (const CudaEventConstructionFailure& error) {
        durable_global_quarantine_ = true;
        quarantine_reason_ = std::string("release_event_construction_failed_quarantined: ")
            + error.what();
        return ReleaseResult{
            "release_event_construction_failed_quarantined",
            0,
            device_index,
            static_cast<std::int64_t>(stream.id()),
            0,
            nullptr,
            count_allocated(buffers),
            static_cast<int>(error.error()),
            quarantine_reason_,
            owner_id_,
            generation_,
            0,
            "",
            static_cast<int>(error.error()),
            quarantine_reason_
        };
    }
    pending_release_token_ = completion_token;
    pending_release_count_ = 0;
    pending_release_generation_ = generation_;
    const ReleaseEnqueueOutcome outcome = release_tuple_async(buffers, stream.stream(), device_index);
    pending_release_count_ = outcome.released;

    cudaError_t event_record_error = cudaSuccess;
    if (outcome.released != 0) {
#ifdef CUMESH_ENABLE_TEST_HOOKS
        event_record_error = consume_failure(fail_release_event_record)
            ? cudaErrorUnknown
            : completion_token->record(stream);
#else
        event_record_error = completion_token->record(stream);
#endif
    }

    cudaError_t caller_wait_error = cudaSuccess;
    if (outcome.released != 0 && event_record_error == cudaSuccess && caller_stream != stream) {
#ifdef CUMESH_ENABLE_TEST_HOOKS
        caller_wait_error = consume_failure(fail_release_caller_wait)
            ? cudaErrorUnknown
            : completion_token->wait(caller_stream);
#else
        caller_wait_error = completion_token->wait(caller_stream);
#endif
    }

    cudaError_t completion_error = event_record_error != cudaSuccess
        ? event_record_error
        : caller_wait_error;

    int stream_synchronize_count = 0;
    if (synchronize && completion_token != nullptr && completion_error == cudaSuccess) {
#ifdef CUMESH_ENABLE_TEST_HOOKS
        completion_error = consume_failure(fail_release_synchronize)
            ? cudaErrorUnknown
            : completion_token->synchronize_noexcept();
#else
        completion_error = completion_token->synchronize_noexcept();
#endif
        if (completion_error == cudaSuccess) {
            stream_synchronize_count = 1;
        }
    }

    const std::size_t quarantined_buffer_count = count_allocated(buffers);
    const cudaError_t release_error =
        outcome.error != cudaSuccess ? outcome.error : completion_error;
    if (release_error != cudaSuccess) {
      // These facts are orthogonal.  A partial primary release can move the
      // untouched/ambiguous tail into global Q while the completion event
      // for the successfully enqueued prefix also fails.  Keep both facts
      // until each is independently proved or reclaimed.
      durable_global_quarantine_ = outcome.error != cudaSuccess;
      completion_proof_pending_ =
          outcome.released != 0 && completion_error != cudaSuccess;
      if (outcome.error != cudaSuccess && completion_error != cudaSuccess) {
        quarantine_reason_ = std::string("release_primary_failure: ") +
                             cudaGetErrorString(outcome.error) +
                             "; release_cleanup_failure: " +
                             cudaGetErrorString(completion_error);
      } else if (outcome.error != cudaSuccess) {
        quarantine_reason_ = std::string("release_primary_failure: ") +
                             cudaGetErrorString(outcome.error);
      } else {
        quarantine_reason_ = std::string("release_completion_unproved: ") +
                             cudaGetErrorString(completion_error);
      }
    } else if (quarantined_buffer_count == 0) {
      completion_proof_pending_ = false;
      durable_global_quarantine_ = false;
      quarantine_reason_.clear();
      reset_metadata();
    }

    const bool completion_proved = event_record_error == cudaSuccess
        && caller_wait_error == cudaSuccess
        && completion_error == cudaSuccess;
    if (outcome.released == 0 || (completion_proved && (quarantined_buffer_count != 0 || synchronize))) {
        pending_release_token_.reset();
        pending_release_count_ = 0;
        pending_release_generation_ = 0;
    }

    std::string status;
    if (release_error != cudaSuccess) {
        if (completion_error != cudaSuccess && outcome.released != 0) {
            status = "release_completion_failed_quarantined";
        } else {
            status = outcome.released == 0
                ? "release_enqueue_failed_quarantined"
                : "release_partially_enqueued_quarantined";
        }
    } else if (synchronize) {
        status = "release_completed";
    } else {
        status = "release_enqueued";
    }
    return ReleaseResult{
        status,
        outcome.released,
        device_index,
        static_cast<std::int64_t>(stream.id()),
        stream_synchronize_count,
        completion_token->recorded() ? completion_token : nullptr,
        quarantined_buffer_count,
        static_cast<int>(release_error),
        release_error == cudaSuccess ? "" : cudaGetErrorString(release_error),
        owner_id_,
        generation_,
        static_cast<int>(completion_error),
        completion_error == cudaSuccess ? "" : cudaGetErrorString(completion_error),
        static_cast<int>(outcome.error),
        outcome.error == cudaSuccess ? "" : cudaGetErrorString(outcome.error)
    };
}

std::size_t CuMesh::allocated_buffer_count() const noexcept {
    std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
    auto buffers = all_buffers();
    static_assert(std::tuple_size_v<decltype(buffers)> == buffer_slot_count);
    return count_allocated(buffers);
}

std::uint64_t CuMesh::owner_id() const noexcept {
    return owner_id_;
}

std::uint64_t CuMesh::generation() const noexcept {
    std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
    return generation_;
}

int CuMesh::device_index() const noexcept {
    std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
    return canonical_device_index_;
}

#ifdef CUMESH_ENABLE_TEST_HOOKS
void CuMesh::test_fail_release_after(std::int64_t successful_enqueues) noexcept {
    release_successes_before_failure.store(successful_enqueues, std::memory_order_relaxed);
}

void CuMesh::test_fail_release_event_record_once() noexcept {
    fail_release_event_record.store(true, std::memory_order_relaxed);
}

void CuMesh::test_fail_release_event_create_once() noexcept {
    fail_release_event_create.store(true, std::memory_order_relaxed);
}

void CuMesh::test_fail_release_token_allocation_once() noexcept {
    fail_release_token_allocation.store(true, std::memory_order_relaxed);
}

void CuMesh::test_fail_release_caller_wait_once() noexcept {
    fail_release_caller_wait.store(true, std::memory_order_relaxed);
}

void CuMesh::test_fail_release_synchronize_once() noexcept {
  fail_release_synchronize.store(true, std::memory_order_relaxed);
}

std::uint8_t CuMesh::test_release_lifecycle_fact_mask() const noexcept {
  std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
  return static_cast<std::uint8_t>(completion_proof_pending_ ? 1U : 0U) |
         static_cast<std::uint8_t>(durable_global_quarantine_ ? 2U : 0U);
}

void CuMesh::test_fail_temporary_allocation_after(
    std::int64_t successful_allocations) noexcept {
  temporary_allocation_successes_before_failure.store(
      successful_allocations, std::memory_order_relaxed);
}

void CuMesh::test_fail_output_allocation_after(std::int64_t successful_allocations) noexcept {
    output_allocation_successes_before_failure.store(successful_allocations, std::memory_order_relaxed);
}

void CuMesh::test_fail_output_copy_after(std::int64_t successful_copies) noexcept {
    output_copy_successes_before_failure.store(successful_copies, std::memory_order_relaxed);
}

void CuMesh::test_fail_output_map_after(std::int64_t successful_maps) noexcept {
    output_map_successes_before_failure.store(successful_maps, std::memory_order_relaxed);
}

void CuMesh::test_fail_temporary_cleanup(
    std::int64_t async_free_failures,
    std::int64_t synchronize_failures,
    std::int64_t fallback_free_failures
) noexcept {
    temporary_async_free_failures.store(async_free_failures, std::memory_order_relaxed);
    // Scoped cleanup no longer performs a production synchronize/fallback
    // after cudaFreeAsync.  Retain the ABI-shaped parameters for existing
    // callers but make their former behavior explicitly inert.
    (void)synchronize_failures;
    (void)fallback_free_failures;
}

void CuMesh::test_fail_destructor_device_synchronize_once() noexcept {
    fail_destructor_device_synchronize.store(true, std::memory_order_relaxed);
}

void CuMesh::test_fail_destructor_legacy_free_once() noexcept {
    fail_destructor_legacy_free.store(true, std::memory_order_relaxed);
}

std::uint64_t CuMesh::test_destructor_legacy_free_attempt_count() noexcept {
    return destructor_legacy_free_attempts.load(std::memory_order_relaxed);
}

void CuMesh::test_fail_buffer_attributes_and_cleanup_once() noexcept {
    buffer_attribute_failures.store(1, std::memory_order_relaxed);
    buffer_cleanup_failures.store(1, std::memory_order_relaxed);
}

void CuMesh::test_fail_buffer_attributes_once() noexcept {
    buffer_attribute_failures.store(1, std::memory_order_relaxed);
    buffer_cleanup_failures.store(0, std::memory_order_relaxed);
}

void CuMesh::test_validate_stream_device(const c10::cuda::CUDAStream& stream, int expected_device) {
    TORCH_CHECK(
        stream.device_index() == expected_device,
        "[CuMesh] Test stream device does not match expected canonical device"
    );
}

void CuMesh::test_validate_completion_device_identity(int token_device, int expected_device) {
    TORCH_CHECK(
        token_device == expected_device,
        "[CuMesh] Injected completion token device does not match expected release device"
    );
}

std::uint64_t test_quarantine_retry_stream_synchronize_count() noexcept {
    return quarantine_retry_stream_synchronizations.load(std::memory_order_relaxed);
}

std::uint64_t test_quarantine_retry_event_wait_count() noexcept {
    return quarantine_retry_event_waits.load(std::memory_order_relaxed);
}

std::uint64_t test_quarantine_retry_legacy_free_attempt_count() noexcept {
    return quarantine_retry_legacy_free_attempts.load(std::memory_order_relaxed);
}

void test_fail_quarantine_retry_completion(std::int64_t failures) noexcept {
    quarantine_retry_completion_failures.store(failures, std::memory_order_relaxed);
}

void test_clear_injected_quarantine_legacy_free_attempts() noexcept {
    auto& owner = cuda_allocation_owner();
    std::lock_guard<std::mutex> lock(owner.mutex);
    for (auto* record = owner.quarantine_head; record != nullptr;
         record = record->quarantine_next) {
        if (record->quarantined && record->persistent_fallback
            && record->test_synthetic_free_attempt) {
            // The destructor fixture injected an error before cudaFree.  This
            // is test-only cleanup after asserting the one-shot quarantine;
            // production never clears an ambiguous direct-free marker.
            record->free_enqueued = false;
            record->async_free_enqueued = false;
            record->test_synthetic_free_attempt = false;
            record->retry_error = cudaSuccess;
            record->needs_device_synchronize = true;
            record->state = CudaAllocationOwnerState::quarantined_never_attempted;
        }
    }
}

int test_quarantine_state_for_pointer(const void* ptr) noexcept {
    auto& owner = cuda_allocation_owner();
    std::lock_guard<std::mutex> lock(owner.mutex);
    for (auto* record = owner.quarantine_head; record != nullptr;
         record = record->quarantine_next) {
        if (record->ptr == ptr) {
            return static_cast<int>(record->state);
        }
    }
    return -1;
}

#endif

std::uint64_t CuMesh::active_temporary_allocation_count() noexcept {
    return active_temporary_allocations.load(std::memory_order_relaxed);
}

std::uint64_t CuMesh::total_temporary_allocation_count() noexcept {
    return total_temporary_allocations.load(std::memory_order_relaxed);
}

std::size_t CuMesh::quarantined_cuda_allocation_count() noexcept {
    return cuda_allocation_quarantine_count();
}

void CuMesh::retry_quarantined_cuda_allocations() {
    CUDA_CHECK(retry_cuda_allocation_quarantine());
    TORCH_CHECK(
        cuda_allocation_quarantine_count() == 0,
        "[CuMesh] CUDA allocation quarantine retry did not release every retained pointer"
    );
}

std::uint64_t CuMesh::destructor_release_error_count() noexcept {
    return destructor_release_errors.load(std::memory_order_relaxed);
}

std::uint64_t CuMesh::destructor_device_synchronize_count() noexcept {
    return destructor_device_synchronizations.load(std::memory_order_relaxed);
}

void CuMesh::reset_metadata() noexcept {
    num_conn_comps = 0;
    num_bound_conn_comps = 0;
    num_bound_loops = 0;
    atlas_num_charts = 0;
}

void CuMesh::invalidate_spatial_audit_locked() noexcept {
    spatial_audit_state_ = SpatialAuditState::unknown;
    ++geometry_revision_;
}

bool CuMesh::spatial_input_clean_after_selection_sync_locked() noexcept {
    const auto quarantine_epoch = cuda_allocation_quarantine_epoch_if_clear();
    // Recovery can empty global Q before this owner is used again. The
    // allocator's existing epoch remembers that intervening failure.
    if (!quarantine_epoch || spatial_audit_quarantine_epoch_ != *quarantine_epoch)
        invalidate_spatial_audit_locked();
    if (spatial_audit_cancelled_.exchange(false,std::memory_order_acq_rel))
        invalidate_spatial_audit_locked();
    if (spatial_audit_state_ == SpatialAuditState::pending_commit
        && spatial_audit_revision_ == geometry_revision_)
        spatial_audit_state_ = SpatialAuditState::verified_clean;
    return spatial_audit_state_ == SpatialAuditState::verified_clean
        && spatial_audit_revision_ == geometry_revision_;
}

void CuMesh::record_verified_spatial_input_locked(bool input_reused) noexcept {
    const auto quarantine_epoch = cuda_allocation_quarantine_epoch_if_clear();
    if (!quarantine_epoch || spatial_audit_cancelled_.load(std::memory_order_acquire)
        || simplification_cancelled()
        || (input_reused && spatial_audit_quarantine_epoch_ != *quarantine_epoch)) {
        invalidate_spatial_audit_locked();
        return;
    }
    spatial_audit_revision_ = geometry_revision_;
    // Only an actually completed full audit may establish a fresh binding.
    // Reaffirming a skipped audit must keep the epoch checked before reuse.
    if (!input_reused) spatial_audit_quarantine_epoch_ = *quarantine_epoch;
    spatial_audit_state_ = SpatialAuditState::verified_clean;
}

void CuMesh::record_pending_spatial_commit_locked() noexcept {
    const auto quarantine_epoch = cuda_allocation_quarantine_epoch_if_clear();
    if (!quarantine_epoch || spatial_audit_quarantine_epoch_ != *quarantine_epoch
        || spatial_audit_cancelled_.load(std::memory_order_acquire)
        || simplification_cancelled()) {
        invalidate_spatial_audit_locked();
        return;
    }
    spatial_audit_revision_ = geometry_revision_;
    spatial_audit_state_ = SpatialAuditState::pending_commit;
}

void CuMesh::reset_cache_metadata() noexcept {
    num_conn_comps = 0;
    num_bound_conn_comps = 0;
    num_bound_loops = 0;
    atlas_num_charts = 0;
}

void CuMesh::require_operable_locked() const {
    TORCH_CHECK(
        !completion_proof_pending_ && !durable_global_quarantine_,
        "[CuMesh] Mesh ownership is quarantined after a partial release: ", quarantine_reason_
    );
    TORCH_CHECK(
        pending_release_token_ == nullptr,
        "[CuMesh] Mesh has a pending release completion; observe or synchronize release_async before reuse"
    );
}

void CuMesh::quarantine_operation_scope_failure_noexcept(const char* reason) noexcept {
    invalidate_spatial_audit_locked();
    durable_global_quarantine_ = true;
    try {
        quarantine_reason_ = reason;
    } catch (...) {
        quarantine_reason_.clear();
    }
}

void CuMesh::establish_canonical_stream_locked(int device_index) {
    TORCH_CHECK(device_index >= 0, "[CuMesh] Cannot establish a canonical stream without a CUDA device");
    const c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(device_index);
    TORCH_CHECK(
        stream.device_index() == device_index,
        "[CuMesh] Canonical Torch stream device does not match input tensor device"
    );
    canonical_device_index_ = device_index;
    canonical_stream_ = stream;
}

c10::cuda::CUDAStream CuMesh::canonical_stream_locked() const {
    TORCH_CHECK(
        canonical_stream_.has_value() && canonical_device_index_ >= 0,
        "[CuMesh] Canonical CUDA stream is unavailable before mesh initialization"
    );
    return *canonical_stream_;
}

void CuMesh::release_fallback_noexcept() noexcept {
    std::unique_lock<std::recursive_mutex> lock(operation_mutex_);
    invalidate_spatial_audit_locked();
    auto buffers = all_buffers();
    static_assert(std::tuple_size_v<decltype(buffers)> == buffer_slot_count);
    if (count_allocated(buffers) == 0) {
        return;
    }

    auto retain_matching = [&](auto&& matches,
                               cudaError_t error,
                               bool legacy_free_attempted,
                               bool needs_device_synchronize,
                               bool test_synthetic_free_attempt = false) noexcept {
        std::array<PersistentCudaAllocationQuarantineSpec, buffer_slot_count> specs{};
        std::size_t count = 0;
        std::apply(
            [&](auto&... buffer) {
                auto collect = [&](auto& item) {
                    if (item.ptr != nullptr && matches(item)) {
                        specs[count++] = PersistentCudaAllocationQuarantineSpec{
                            item.ptr, item.device_index, error, legacy_free_attempted,
                            false, needs_device_synchronize, test_synthetic_free_attempt
                        };
                    }
                };
                (collect(buffer), ...);
            },
            buffers
        );
        // Construct every owner record before the single Q publication.  A
        // retry cannot observe a partial destructor failure batch.
        retain_persistent_cuda_allocation_quarantine_batch(specs.data(), count);
        std::apply(
            [&](auto&... buffer) {
                auto clear = [&](auto& item) {
                    if (item.ptr != nullptr && matches(item)) {
                        item.reset();
                    }
                };
                (clear(buffer), ...);
            },
            buffers
        );
    };

    std::array<int, buffer_slot_count> devices{};
    std::size_t device_count = 0;
    std::size_t invalid_owner_count = 0;
    std::apply(
        [&](const auto&... buffer) {
            auto collect_device = [&](const auto& item) {
                if (item.ptr == nullptr) {
                    return;
                }
                if (item.device_index < 0) {
                    ++invalid_owner_count;
                    return;
                }
                for (std::size_t index = 0; index < device_count; ++index) {
                    if (devices[index] == item.device_index) {
                        return;
                    }
                }
                devices[device_count++] = item.device_index;
            };
            (collect_device(buffer), ...);
        },
        buffers
    );

    std::size_t error_count = invalid_owner_count;
    cudaError_t first_error = invalid_owner_count == 0 ? cudaSuccess : cudaErrorInvalidDevice;
    int original_device = -1;
    const cudaError_t get_device_error = cudaGetDevice(&original_device);
    if (get_device_error != cudaSuccess) {
        ++error_count;
        first_error = first_error == cudaSuccess ? get_device_error : first_error;
        retain_matching([](const auto&) noexcept { return true; }, get_device_error, false, true);
        reset_metadata();
        destructor_release_errors.fetch_add(error_count, std::memory_order_relaxed);
        std::fprintf(
            stderr,
            "[CuMesh] Destructor release could not read the CUDA device; retained persistent pointers in quarantine\n"
        );
        return;
    }

    if (invalid_owner_count != 0) {
        retain_matching(
            [](const auto& item) noexcept { return item.device_index < 0; },
            cudaErrorInvalidDevice, false, false
        );
    }

    for (std::size_t device_position = 0; device_position < device_count; ++device_position) {
        const int device_index = devices[device_position];
        const cudaError_t set_device_error = cudaSetDevice(device_index);
        if (set_device_error != cudaSuccess) {
            ++error_count;
            first_error = first_error == cudaSuccess ? set_device_error : first_error;
            retain_matching(
                [device_index](const auto& item) noexcept { return item.device_index == device_index; },
                set_device_error, false, true
            );
            continue;
        }

        destructor_device_synchronizations.fetch_add(1, std::memory_order_relaxed);
        const cudaError_t synchronize_error =
#ifdef CUMESH_ENABLE_TEST_HOOKS
            consume_failure(fail_destructor_device_synchronize)
                ? cudaErrorUnknown
                : cudaDeviceSynchronize();
#else
            cudaDeviceSynchronize();
#endif
        if (synchronize_error != cudaSuccess) {
            ++error_count;
            first_error = first_error == cudaSuccess ? synchronize_error : first_error;
            retain_matching(
                [device_index](const auto& item) noexcept { return item.device_index == device_index; },
                synchronize_error, false, true
            );
            continue;
        }

        bool legacy_free_ambiguous_for_device = false;
        void* ambiguous_pointer = nullptr;
        cudaError_t ambiguous_error = cudaSuccess;
        bool ambiguous_test_synthetic_attempt = false;
        std::apply(
            [&](auto&... buffer) {
                auto release_one = [&](auto& item) {
                    if (item.ptr == nullptr || item.device_index != device_index) {
                        return;
                    }
                    if (legacy_free_ambiguous_for_device) {
                        return;
                    }
                    const bool injected_free_error =
#ifdef CUMESH_ENABLE_TEST_HOOKS
                        consume_failure(fail_destructor_legacy_free);
#else
                        false;
#endif
#ifdef CUMESH_ENABLE_TEST_HOOKS
                    destructor_legacy_free_attempts.fetch_add(1, std::memory_order_relaxed);
#endif
                    const cudaError_t free_error = injected_free_error
                        ? cudaErrorUnknown
                        : item.free_after_device_sync();
                    if (free_error != cudaSuccess) {
                        ++error_count;
                        first_error = first_error == cudaSuccess ? free_error : first_error;
                        legacy_free_ambiguous_for_device = true;
                        ambiguous_pointer = item.ptr;
                        ambiguous_error = free_error;
                        ambiguous_test_synthetic_attempt = injected_free_error;
                    }
                };
                (release_one(buffer), ...);
            },
            buffers
        );
        if (legacy_free_ambiguous_for_device) {
            std::array<PersistentCudaAllocationQuarantineSpec, buffer_slot_count> specs{};
            std::size_t count = 0;
            std::apply(
                [&](auto&... buffer) {
                    auto collect = [&](auto& item) {
                        if (item.ptr != nullptr && item.device_index == device_index) {
                            const bool attempted = item.ptr == ambiguous_pointer;
                            specs[count++] = PersistentCudaAllocationQuarantineSpec{
                                item.ptr, item.device_index,
                                attempted ? ambiguous_error : cudaErrorNotReady,
                                attempted, false, !attempted,
                                attempted && ambiguous_test_synthetic_attempt
                            };
                        }
                    };
                    (collect(buffer), ...);
                },
                buffers
            );
            retain_persistent_cuda_allocation_quarantine_batch(specs.data(), count);
            std::apply(
                [&](auto&... buffer) {
                    auto clear = [&](auto& item) {
                        if (item.ptr != nullptr && item.device_index == device_index) {
                            item.reset();
                        }
                    };
                    (clear(buffer), ...);
                },
                buffers
            );
        }
    }

    if (get_device_error == cudaSuccess && original_device >= 0) {
        const cudaError_t restore_error = cudaSetDevice(original_device);
        if (restore_error != cudaSuccess) {
            ++error_count;
            first_error = first_error == cudaSuccess ? restore_error : first_error;
        }
    }

    reset_metadata();
    if (error_count != 0) {
        destructor_release_errors.fetch_add(error_count, std::memory_order_relaxed);
        std::fprintf(
            stderr,
            "[CuMesh] Destructor release failed for %zu CUDA operation(s); first error %d: %s\n",
            error_count,
            static_cast<int>(first_error),
            cudaGetErrorString(first_error)
        );
    }
}

int CuMesh::num_vertices() const {
    std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
    return vertices.size;
}

int CuMesh::num_faces() const {
    std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
    return faces.size;
}

int CuMesh::num_edges() const {
    std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
    return edges.size;
}

int CuMesh::num_boundaries() const {
    std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
    return boundaries.size;
}

int CuMesh::num_conneted_components() const {
    std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
    return num_conn_comps;
}

int CuMesh::num_boundary_conneted_components() const {
    std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
    return num_bound_conn_comps;
}

int CuMesh::num_boundary_loops() const {
    std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
    return num_bound_loops;
}

void CuMesh::clear_cache() {
    std::unique_lock<std::recursive_mutex> prelock(operation_mutex_);
    require_cuda_allocation_quarantine_clear();
    require_operable_locked();
    auto buffers = cache_buffers();
    const int device_index = require_single_device(buffers, "clear_cache");
    if (device_index < 0) {
        reset_cache_metadata();
        return;
    }

    CuMeshOperationScope operation(*this);
    prelock.unlock();
    TORCH_CHECK(
        operation.device_index() == device_index,
        "[CuMesh] clear_cache: cache buffers do not match canonical CUDA device"
    );
    require_async_release_support(operation.stream(), device_index, "clear_cache");
    preflight_pointer_ownership(buffers, device_index, "clear_cache");
    const ReleaseEnqueueOutcome outcome = release_tuple_async(buffers, operation.stream(), device_index);
    reset_cache_metadata();
    if (outcome.error != cudaSuccess) {
        durable_global_quarantine_ = true;
        quarantine_reason_ = std::string("clear_cache_release_enqueue_failed_quarantined: ")
            + cudaGetErrorString(outcome.error);
        TORCH_CHECK(false, "[CuMesh] ", quarantine_reason_);
    }
    operation.bridge_to_caller();
}

} // namespace cumesh
