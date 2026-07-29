import gc
import os
from pathlib import Path
from threading import Thread
from typing import cast

import pytest
import torch
from torch.utils.cpp_extension import load

from cumesh import CuMesh


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def native_release_module(tmp_path_factory: pytest.TempPathFactory):
    os.environ.setdefault("MAX_JOBS", "1")
    capability = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}")
    build_directory = tmp_path_factory.mktemp("cumesh-release-native")
    return load(
        name="cumesh_release_native_test",
        sources=[
            str(ROOT / "src" / "atlas.cu"),
            str(ROOT / "src" / "clean_up.cu"),
            str(ROOT / "src" / "connectivity.cu"),
            str(ROOT / "src" / "cumesh.cu"),
            str(ROOT / "src" / "geometry.cu"),
            str(ROOT / "src" / "io.cu"),
            str(ROOT / "src" / "shared.cu"),
            str(ROOT / "src" / "simplify.cu"),
            str(ROOT / "tests" / "cumesh_release_native.cu"),
        ],
        extra_include_paths=[str(ROOT / "src")],
        extra_cflags=["-O2", "-std=c++17", "-DCUMESH_ENABLE_TEST_HOOKS"],
        extra_cuda_cflags=["-O2", "-std=c++17", "-DCUMESH_ENABLE_TEST_HOOKS"],
        build_directory=str(build_directory),
        with_cuda=True,
        verbose=True,
    )


def _mesh(vertices: list[list[float]], faces: list[list[int]]) -> CuMesh:
    mesh = CuMesh()
    mesh.init(
        torch.tensor(vertices, dtype=torch.float32, device="cuda"),
        torch.tensor(faces, dtype=torch.int32, device="cuda"),
    )
    return mesh


def test_all_48_buffers_release_on_one_stream_without_per_buffer_sync(native_release_module) -> None:
    stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    with torch.cuda.stream(stream):
        before, released, after, device_index, synchronize_count, output = (
            native_release_module.exercise_explicit_release(True)
        )

    assert before == 48
    assert released == 48
    assert after == 0
    assert device_index == torch.cuda.current_device()
    assert synchronize_count == 1
    assert output.item() == 37


def test_async_release_can_defer_the_single_completion_boundary(native_release_module) -> None:
    stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    with torch.cuda.stream(stream):
        before, released, after, _device_index, synchronize_count, output = (
            native_release_module.exercise_explicit_release(False)
        )
    stream.synchronize()

    assert before == released == 48
    assert after == 0
    assert synchronize_count == 0
    assert output.item() == 37


def test_destructor_fallback_uses_one_device_boundary_for_48_buffers(native_release_module) -> None:
    sync_before, sync_after, errors_before, errors_after = native_release_module.exercise_destructor_fallback()

    assert sync_after - sync_before == 1
    assert errors_after == errors_before


def test_destructor_sync_failure_transfers_every_persistent_pointer_to_global_q(
    native_release_module,
) -> None:
    retained, new_admission_blocked, retry_error, after_retry = (
        native_release_module.exercise_destructor_sync_failure_retains_persistent_quarantine()
    )

    assert retained == 48
    assert new_admission_blocked is True
    assert retry_error == 0
    assert after_retry == 0


def test_persistent_quarantine_retry_releases_global_mutex_for_callback(
    native_release_module,
) -> None:
    owner_retry, concurrent_retry, expected_concurrent, callback_blocked, after_retry = (
        native_release_module.exercise_persistent_quarantine_retry_reentrancy()
    )

    assert owner_retry == 0
    assert concurrent_retry == expected_concurrent
    assert callback_blocked is True
    assert after_retry == 0


def test_destructor_first_free_error_keeps_later_pointers_retryable(
    native_release_module,
) -> None:
    (
        retained_before_retry,
        initial_free_attempts,
        new_admission_blocked,
        first_retry,
        expected_first_retry,
        first_slot_state,
        tail_slot_state,
        retained_after_retry,
        cleanup_retry,
        after_cleanup,
    ) = native_release_module.exercise_destructor_first_free_error_preserves_later_retryable_owners()

    assert retained_before_retry == 48
    assert initial_free_attempts == 1
    assert new_admission_blocked is True
    assert first_retry == expected_first_retry
    assert first_slot_state == 8
    assert tail_slot_state == 4
    assert retained_after_retry == 1
    assert cleanup_retry == 0
    assert after_cleanup == 0


def test_explicit_release_carries_owner_generation_and_avoids_destructor_fallback(
    native_release_module,
) -> None:
    (
        owner,
        generation,
        token_owner,
        token_generation,
        repeated_token_owner,
        repeated_token_generation,
        first_released,
        second_released,
        status,
        sync_delta,
    ) = (
        native_release_module.exercise_explicit_release_identity_and_destructor()
    )

    assert owner > 0
    assert generation == 1
    assert token_owner == owner
    assert token_generation == generation
    assert repeated_token_owner == owner
    assert repeated_token_generation == generation
    assert first_released == 2
    assert second_released == 0
    assert status == "release_completed"
    assert sync_delta == 0


def test_init_failure_releases_partial_async_storage_before_next_generation(
    native_release_module,
) -> None:
    rejected, retained, quarantined, failed_generation, success_generation, vertices, faces, released = (
        native_release_module.exercise_init_failure_rolls_back_and_allows_next_generation()
    )

    assert rejected is True
    assert retained == 0
    assert quarantined == 0
    assert failed_generation == 0
    assert success_generation == 1
    torch.testing.assert_close(
        vertices,
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device="cuda"),
    )
    torch.testing.assert_close(
        faces,
        torch.tensor([[0, 0, 0]], dtype=torch.int32, device="cuda"),
    )
    assert released == 2


def test_init_cleanup_failure_records_only_durable_global_quarantine() -> None:
    source = (ROOT / "src" / "io.cu").read_text(encoding="utf-8")
    rollback = source.split(
        "} catch (const std::exception& primary_error) {", maxsplit=1
    )[1].split("TORCH_CHECK(", maxsplit=1)[0]

    assert "durable_global_quarantine_ = true;" in rollback
    assert "\n            quarantined_ =" not in rollback
    assert "completion_proof_pending_" not in rollback


def test_clear_cache_bulk_releases_all_46_cache_buffers(native_release_module) -> None:
    stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    with torch.cuda.stream(stream):
        before, after, metadata_sum, primary_released = native_release_module.exercise_clear_cache()

    assert before == 48
    assert after == 2
    assert metadata_sum == 0
    assert primary_released == 2


def test_swap_transfers_all_48_buffers_and_metadata_without_copying(native_release_module) -> None:
    (
        pointer_owners_exchanged,
        selected_count,
        retained_count,
        selected_metadata_sum,
        retained_metadata_sum,
        retained_released,
        selected_released,
    ) = native_release_module.exercise_full_swap()

    assert pointer_owners_exchanged is True
    assert selected_count == selected_released == 2
    assert retained_count == retained_released == 48
    assert selected_metadata_sum == 21 + 22 + 23 + 24
    assert retained_metadata_sum == 11 + 12 + 13 + 14


def test_release_is_idempotent_and_reports_exactly_owned_buffers() -> None:
    mesh = _mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
    )
    assert mesh.allocated_buffer_count == 2
    owner_id = mesh.owner_id
    generation = mesh.generation

    first = mesh.release_async()
    second = mesh.release_async()

    assert first.status == "release_enqueued"
    assert first.buffers_released == 2
    assert first.device_index == torch.cuda.current_device()
    assert isinstance(first.stream_id, int)
    assert first.stream_synchronize_count == 0
    assert first.quarantined_buffer_count == 0
    assert first.error_code == 0
    assert first.error_message == ""
    assert first.cleanup_error_code == 0
    assert first.cleanup_error_message == ""
    assert first.owner_id == owner_id
    assert first.generation == generation
    assert first.completion_token is not None
    assert first.completion_token.device_index == first.device_index
    assert first.completion_token.stream_id == first.stream_id
    assert first.completion_token.owner_id == owner_id
    assert first.completion_token.generation == generation
    assert second.status == "release_enqueued"
    assert second.buffers_released == 0
    assert second.device_index == first.device_index
    assert second.stream_id == first.stream_id
    assert second.stream_synchronize_count == 0
    assert second.completion_token is not None
    assert second.owner_id == owner_id
    assert second.generation == generation
    assert second.completion_token.owner_id == owner_id
    assert second.completion_token.generation == generation
    assert second.quarantined_buffer_count == 0
    assert mesh.allocated_buffer_count == 0
    first.completion_token.wait_on_current_stream()
    torch.cuda.current_stream().synchronize()
    assert first.completion_token.is_complete is True
    assert first.completion_status == "release_completed"
    completed = mesh.release_async(synchronize=True)
    assert completed.status == "release_completed"
    assert completed.buffers_released == 0


def test_partial_release_transfers_boundary_failure_and_never_attempted_owners_to_global_q(
    native_release_module,
) -> None:
    (
        partial_status,
        partial_released,
        partial_quarantined,
        partial_error,
        retained_after_partial,
        boundary_state,
        tail_state,
        pending_observation_status,
        global_q_final,
        first_global_retry,
        expected_not_ready,
        retained_after_first_global_retry,
        cleanup_global_retry,
        clean_observation_status,
        final_count,
    ) = native_release_module.exercise_partial_release_quarantine()

    assert partial_status == "release_partially_enqueued_quarantined"
    assert partial_released == 3
    # The Buffer owners are cleared immediately; ownership has moved to Q.
    assert partial_quarantined == retained_after_partial == 0
    assert partial_error != 0
    # Both identities are in Q and remain distinct never-attempted records;
    # the pre-call hook did not manufacture a post-call permanent owner.
    assert boundary_state == tail_state == 4
    assert pending_observation_status == "release_primary_failure_quarantined"
    # This fixture injects before cudaFreeAsync, so all 45 retained slots are
    # never-attempted and become reclaimable only after device completion.
    assert first_global_retry == 0
    assert retained_after_first_global_retry == 0
    assert cleanup_global_retry == 0
    assert clean_observation_status == "release_completed"
    assert global_q_final == final_count == 0


def test_stream_device_mismatch_fails_before_cuda_work(native_release_module) -> None:
    with pytest.raises(RuntimeError, match="stream device does not match expected canonical device"):
        native_release_module.exercise_stream_device_mismatch()


def test_pointer_device_preflight_rejects_before_detaching_any_owner(native_release_module) -> None:
    rejected, retained, released_during_cleanup = native_release_module.exercise_pointer_device_preflight_mismatch()

    assert rejected is True
    assert retained == released_during_cleanup == 48


def test_async_init_and_read_bridge_two_torch_streams_without_host_sync() -> None:
    producer = cast(torch.cuda.Stream, torch.cuda.Stream())
    consumer = cast(torch.cuda.Stream, torch.cuda.Stream())
    mesh = CuMesh()

    with torch.cuda.stream(producer):
        expected_vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            dtype=torch.float32,
            device="cuda",
        )
        expected_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
        torch.cuda._sleep(5_000_000)
        mesh.init(expected_vertices, expected_faces)
    with torch.cuda.stream(consumer):
        vertices, faces = mesh.read()
        checksum = vertices.sum()
    consumer.synchronize()

    assert checksum.item() == 7.0
    torch.testing.assert_close(vertices, expected_vertices)
    torch.testing.assert_close(faces, expected_faces)
    release = mesh.release_async()
    assert release.completion_token is not None
    release.completion_token.synchronize()


def test_mutation_scope_bridges_canonical_work_back_to_the_caller_stream() -> None:
    producer = cast(torch.cuda.Stream, torch.cuda.Stream())
    consumer = cast(torch.cuda.Stream, torch.cuda.Stream())
    mesh = CuMesh()
    with torch.cuda.stream(producer):
        mesh.init(
            torch.tensor(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                dtype=torch.float32,
                device="cuda",
            ),
            torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda"),
        )
        torch.cuda._sleep(5_000_000)

    with torch.cuda.stream(consumer):
        mesh.compute_face_normals()
        normals = mesh.read_face_normals()
        norm_sum = normals.abs().sum()
    consumer.synchronize()

    assert norm_sum.item() == pytest.approx(1.0)
    mesh.release_async(synchronize=True)


def test_swap_joins_both_source_streams_and_advances_each_owner_generation() -> None:
    left_stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    right_stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    consumer = cast(torch.cuda.Stream, torch.cuda.Stream())
    left = CuMesh()
    right = CuMesh()
    with torch.cuda.stream(left_stream):
        left.init(
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"),
            torch.tensor([[0, 0, 0]], dtype=torch.int32, device="cuda"),
        )
        torch.cuda._sleep(5_000_000)
    with torch.cuda.stream(right_stream):
        right.init(
            torch.tensor([[9.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"),
            torch.tensor([[0, 0, 0]], dtype=torch.int32, device="cuda"),
        )
        torch.cuda._sleep(5_000_000)
    left_owner, right_owner = left.owner_id, right.owner_id
    left_generation, right_generation = left.generation, right.generation

    with torch.cuda.stream(consumer):
        left.swap(right)
        left_vertices, _ = left.read()
        right_vertices, _ = right.read()
    consumer.synchronize()

    assert left.owner_id == left_owner
    assert right.owner_id == right_owner
    assert left.generation == left_generation + 1
    assert right.generation == right_generation + 1
    assert left_vertices[0, 0].item() == 9.0
    assert right_vertices[0, 0].item() == 1.0
    left.release_async(synchronize=True)
    right.release_async(synchronize=True)


def test_cross_thread_release_uses_canonical_owner_stream_and_completion_token() -> None:
    producer = cast(torch.cuda.Stream, torch.cuda.Stream())
    release_stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    with torch.cuda.stream(producer):
        mesh = _mesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0, 1, 2]],
        )
        torch.cuda._sleep(5_000_000)
        mesh.compute_face_normals()

    receipts = []

    def release_from_other_thread() -> None:
        with torch.cuda.stream(release_stream):
            receipts.append(mesh.release_async())

    thread = Thread(target=release_from_other_thread)
    thread.start()
    thread.join()

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.status == "release_enqueued"
    assert receipt.buffers_released == 3
    assert receipt.stream_synchronize_count == 0
    assert receipt.completion_token is not None
    assert mesh.allocated_buffer_count == 0
    release_stream.synchronize()
    assert receipt.completion_token.is_complete is True


def test_bulk_clear_cache_preserves_primary_geometry() -> None:
    expected_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    expected_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    mesh = CuMesh()
    mesh.init(expected_vertices, expected_faces)
    mesh.compute_face_normals()
    assert mesh.allocated_buffer_count == 3

    mesh.clear_cache()
    vertices, faces = mesh.read()

    assert mesh.allocated_buffer_count == 2
    torch.testing.assert_close(vertices, expected_vertices)
    torch.testing.assert_close(faces, expected_faces)
    mesh.release_async(synchronize=True)


def test_failed_replacement_creation_leaves_old_owner_readable() -> None:
    old = _mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
    )
    replacement = CuMesh()

    with pytest.raises(AssertionError, match="Input vertices must be of shape"):
        replacement.init(
            torch.zeros((3, 2), dtype=torch.float32, device="cuda"),
            torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda"),
        )

    old_vertices, old_faces = old.read()
    torch.testing.assert_close(
        old_vertices,
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device="cuda",
        ),
    )
    torch.testing.assert_close(old_faces, torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda"))
    old.release_async(synchronize=True)


def test_successful_swap_preserves_both_owners_until_explicit_release() -> None:
    old = _mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
    )
    replacement = _mesh(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        [[0, 1, 2], [0, 1, 3]],
    )

    old.swap(replacement)
    selected_vertices, selected_faces = old.read()
    retained_vertices, retained_faces = replacement.read()

    assert selected_vertices.shape == (4, 3)
    assert selected_faces.shape == (2, 3)
    assert retained_vertices.shape == (3, 3)
    assert retained_faces.shape == (1, 3)
    assert old.allocated_buffer_count == replacement.allocated_buffer_count == 2

    retained_release = replacement.release_async(synchronize=True)
    selected_release = old.release_async(synchronize=True)
    assert retained_release.buffers_released == selected_release.buffers_released == 2
    assert retained_release.status == selected_release.status == "release_completed"
    assert retained_release.stream_synchronize_count == selected_release.stream_synchronize_count == 1
    assert retained_release.completion_status == selected_release.completion_status == "release_completed"


def test_swap_rejects_non_mesh_without_changing_owner() -> None:
    mesh = _mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
    )

    swap_method = vars(type(mesh))["swap"]
    with pytest.raises(TypeError, match="other must be a CuMesh"):
        swap_method(mesh, object())

    assert mesh.num_vertices == 3
    assert mesh.num_faces == 1
    mesh.release_async(synchronize=True)


def test_python_gc_fallback_reports_one_device_boundary_and_no_error() -> None:
    sync_before = CuMesh.destructor_device_synchronize_count()
    errors_before = CuMesh.destructor_release_error_count()
    mesh = _mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
    )

    del mesh
    gc.collect()

    assert CuMesh.destructor_device_synchronize_count() - sync_before == 1
    assert CuMesh.destructor_release_error_count() == errors_before


def test_release_during_graph_capture_fails_before_detaching_buffers() -> None:
    mesh = _mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
    )
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    with pytest.warns(UserWarning, match="CUDA Graph is empty"):
        with pytest.raises(RuntimeError, match="not allowed during CUDA graph capture"):
            with torch.cuda.graph(graph, stream=capture_stream):
                mesh.release_async()

    assert mesh.allocated_buffer_count == 2
    mesh.release_async(synchronize=True)


@pytest.mark.parametrize("failure", ["record", "token", "wait", "sync"])
def test_release_completion_failures_remain_durable_until_retry(
    native_release_module,
    failure: str,
) -> None:
    failed_status, failed_released, quarantined, retained, retry_status, retry_error, final_count = (
        native_release_module.exercise_release_completion_failure(failure)
    )

    if failure == "token":
        assert failed_status == "release_enqueue_failed_quarantined"
        assert failed_released == 0
        assert quarantined == retained == 48
    else:
        assert failed_status == "release_completion_failed_quarantined"
        assert failed_released == 48
        assert quarantined == retained == 0
    assert retry_status == "release_completed"
    assert retry_error == 0
    assert final_count == 0


def test_combined_primary_and_completion_failures_recover_independently(
    native_release_module,
) -> None:
    (
        combined_status,
        combined_released,
        primary_error,
        completion_error,
        combined_facts,
        combined_global_q,
        completion_status,
        after_completion_facts,
        after_completion_global_q,
        global_retry,
        recovered_status,
        recovered_facts,
        recovered_global_q,
        final_buffer_count,
    ) = native_release_module.exercise_combined_primary_and_completion_failure()

    assert combined_status == "release_completion_failed_quarantined"
    assert combined_released == 3
    assert primary_error != 0
    assert completion_error != 0
    assert combined_facts == 0b11
    assert combined_global_q == 45

    assert completion_status == "release_primary_failure_quarantined"
    assert after_completion_facts == 0b10
    assert after_completion_global_q == combined_global_q

    assert global_retry == 0
    assert recovered_status == "release_completed"
    assert recovered_facts == 0
    assert recovered_global_q == final_buffer_count == 0


def test_remove_faces_native_validation_precedes_mutation(native_release_module) -> None:
    shape, dtype, contiguous, cross_device, before, after, _device_count = (
        native_release_module.exercise_remove_faces_validation()
    )

    assert shape is True
    assert dtype is True
    assert contiguous is True
    assert cross_device is True
    assert before == after == 2


def test_temporary_allocation_exception_releases_all_prior_temporaries(
    native_release_module,
) -> None:
    rejected, before, after = native_release_module.exercise_temporary_allocation_failure()

    assert rejected is True
    assert before == after == 0


@pytest.mark.parametrize("failure", ["allocation", "copy", "map"])
def test_torch_output_exception_does_not_allow_allocator_reuse_uaf(
    native_release_module,
    failure: str,
) -> None:
    rejected, reuse_intact, cleanup_error = native_release_module.exercise_output_exception_allocator_reuse(failure)

    assert rejected is True
    assert reuse_intact is True
    assert cleanup_error == 0


def test_event_construction_failure_quarantines_owned_buffers_until_retry(
    native_release_module,
) -> None:
    status, rejected, blocked, retained, retry_status, retry_released, final_count = (
        native_release_module.exercise_event_construction_failure()
    )

    assert status == "release_event_construction_failed_quarantined"
    assert rejected is True
    assert blocked is True
    assert retained == retry_released == 48
    assert retry_status == "release_completed"
    assert final_count == 0


def test_consecutive_temporary_cleanup_failures_are_owned_and_bound_growth(
    native_release_module,
) -> None:
    (
        before,
        quarantined,
        blocked,
        after_blocked,
        after_retry,
        active,
        retry_synchronizations,
        blocked_attempt_allocations,
    ) = (
        native_release_module.exercise_temporary_cleanup_quarantine()
    )

    assert before == 0
    assert quarantined == 2
    assert blocked == 3
    assert after_blocked == quarantined
    assert after_retry == 0
    assert active == 0
    assert retry_synchronizations == 1
    assert blocked_attempt_allocations == 0


def test_buffer_attribute_cleanup_failure_transfers_pointer_to_quarantine(
    native_release_module,
) -> None:
    before, rejected, quarantined, blocked, after_blocked, after_retry = (
        native_release_module.exercise_buffer_attribute_cleanup_quarantine()
    )

    assert before == 0
    assert rejected is True
    assert quarantined == 1
    assert blocked is True
    assert after_blocked == quarantined
    assert after_retry == 0


def test_quarantine_retry_waits_every_owner_event_and_uses_one_device_barrier(
    native_release_module,
) -> None:
    (
        first_error,
        injected_error,
        after_first_retry,
        first_synchronizations,
        first_event_waits,
        second_error,
        after_second_retry,
        second_synchronizations,
        second_event_waits,
    ) = native_release_module.exercise_quarantine_retry_groups_by_device_and_stream()

    assert first_error == injected_error
    assert after_first_retry == 4
    assert first_synchronizations == 1
    assert first_event_waits == 4
    assert second_error == 0
    assert after_second_retry == 0
    assert second_synchronizations == 1
    assert second_event_waits == 4


@pytest.mark.parametrize("destructor_path", [False, True])
def test_sync_failure_never_calls_legacy_free_before_completion_proof(
    native_release_module,
    destructor_path: bool,
) -> None:
    rejected, quarantined, fallback_attempts, retry_error, after_retry = (
        native_release_module.exercise_sync_failure_never_falls_back(destructor_path)
    )

    assert rejected is (not destructor_path)
    assert quarantined == 1
    assert fallback_attempts == 0
    assert retry_error == 0
    assert after_retry == 0


@pytest.mark.parametrize("destructor_path", [False, True])
def test_pre_call_cleanup_failure_is_retryable_without_scoped_legacy_fallback(
    native_release_module,
    destructor_path: bool,
) -> None:
    (
        rejected,
        quarantined,
        fallback_attempts,
        retry_error,
        expected_retry_error,
        retry_free_attempts,
        retained_after_retry,
        cleanup_error,
        after_cleanup,
    ) = native_release_module.exercise_fallback_error_is_never_legacy_freed_twice(
        destructor_path
    )

    assert rejected is (not destructor_path)
    assert quarantined == 1
    assert fallback_attempts == 0
    assert retry_error == 0
    assert expected_retry_error == 0
    assert retry_free_attempts == 1
    assert retained_after_retry == 0
    assert cleanup_error == 0
    assert after_cleanup == 0


@pytest.mark.parametrize("revoke_fails", [False, True])
def test_pending_admission_never_publishes_after_quarantine_epoch_change(
    native_release_module,
    revoke_fails: bool,
) -> None:
    committed, revoke_error, after_revoke, retry_error, after_retry = (
        native_release_module.exercise_pending_admission_epoch_revoke(revoke_fails)
    )

    assert committed is False
    assert revoke_error == (999 if revoke_fails else 0)
    assert after_revoke == (1 if revoke_fails else 0)
    assert retry_error == 0
    assert after_retry == 0


def test_async_free_event_record_failure_recovers_by_exact_device_sync_without_second_free(
    native_release_module,
) -> None:
    (
        revoke_error,
        expected_revoke_error,
        recovery_retry,
        expected_recovery_retry,
        retained_after_recovery_retry,
        legacy_attempts_during_blocked_retry,
        recovery_syncs,
        after_recovery,
    ) = native_release_module.exercise_async_free_event_record_failure_recovers_by_device_sync()

    assert revoke_error == expected_revoke_error
    assert recovery_retry == expected_recovery_retry
    assert retained_after_recovery_retry == 0
    assert legacy_attempts_during_blocked_retry == 0
    assert recovery_syncs == 1
    assert after_recovery == 0


def test_committed_private_allocation_transfers_after_later_q_but_q_before_commit_revokes(
    native_release_module,
) -> None:
    (
        first_commit,
        first_private_publication,
        transferred_after_q,
        transfer_metadata_visible,
        blocked_commit,
        blocked_consumer_publication,
        pending_revoke,
        after_cleanup,
    ) = native_release_module.exercise_committed_transfer_survives_later_quarantine()

    assert first_commit is True
    assert first_private_publication is True
    assert transferred_after_q is True
    assert transfer_metadata_visible is True
    assert blocked_commit is False
    assert blocked_consumer_publication is False
    assert pending_revoke == 0
    assert after_cleanup == 0


def test_external_destroyed_origin_uses_only_durable_completion_event(
    native_release_module,
) -> None:
    (
        destroy_error,
        retry_error,
        replacement_destroy_error,
        quarantined,
        after_retry,
        origin_id,
        replacement_id,
        event_waits,
        barrier_synchronizations,
    ) = native_release_module.exercise_external_destroyed_stream_recovery()

    assert destroy_error == 0
    assert retry_error == 0
    assert replacement_destroy_error == 0
    assert quarantined == 1
    assert after_retry == 0
    assert origin_id != replacement_id
    assert event_waits == 1
    assert barrier_synchronizations == 1


def test_per_thread_origin_uses_owner_managed_event_barrier(
    native_release_module,
) -> None:
    retry_error, quarantined, after_retry, stream_id, event_waits = (
        native_release_module.exercise_per_thread_stream_recovery()
    )

    assert retry_error == 0
    assert quarantined == 1
    assert after_retry == 0
    assert stream_id >= 0
    assert event_waits == 1


def test_completion_event_device_guards_preserve_current_device(
    native_release_module,
) -> None:
    injected_rejected, real_multi_gpu, restored, device_count = (
        native_release_module.exercise_completion_device_guards()
    )

    assert injected_rejected is True
    assert restored is True
    assert real_multi_gpu is (device_count > 1)


def test_compress_ids_allocates_count_output_only_for_inverse(
    native_release_module,
) -> None:
    without_inverse, with_inverse = native_release_module.exercise_compress_ids_allocation_count()

    assert without_inverse == 4
    assert with_inverse == without_inverse + 1


def test_quarantine_retry_releases_mutex_for_callback_and_has_one_owner(
    native_release_module,
) -> None:
    callback_blocked, owner_error, concurrent_error, claimed_error, final_count = (
        native_release_module.exercise_reentrant_concurrent_quarantine_retry()
    )

    assert callback_blocked is True
    assert owner_error == 0
    assert concurrent_error == claimed_error
    assert final_count == 0


def test_completion_record_rejects_same_device_non_origin_stream(
    native_release_module,
) -> None:
    rejected, expected, origin_error, sync_error, device, origin_id, different_id = (
        native_release_module.exercise_completion_origin_stream_identity()
    )

    assert rejected == expected
    assert origin_error == 0
    assert sync_error == 0
    assert device >= 0
    assert origin_id != different_id
