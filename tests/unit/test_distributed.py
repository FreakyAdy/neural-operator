"""Unit tests for distributed training helpers."""

from __future__ import annotations

from operatorlab.training.distributed import barrier, cleanup_distributed, is_main_process


class TestDistributedHelpers:
    def test_single_process_behavior(self) -> None:
        # In non-distributed environment, is_main_process must return True
        assert is_main_process() is True
        # barrier and cleanup must run without raising errors
        barrier()
        cleanup_distributed()
