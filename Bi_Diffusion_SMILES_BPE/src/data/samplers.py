"""Length-aware batch samplers for faster sequence training."""

import math
import random
from typing import Iterator, List, Sequence

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups similar-length samples together.

    The sampler optionally shuffles sample order each epoch, then sorts samples
    inside local buckets by sequence-length proxy. This keeps padding overhead
    low without a full global sort.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        bucket_size_multiplier: int = 50,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if bucket_size_multiplier <= 0:
            raise ValueError(
                f"bucket_size_multiplier must be > 0, got {bucket_size_multiplier}"
            )
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas - 1}], got {rank}")

        self.lengths = [int(v) for v in lengths]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size_multiplier = int(bucket_size_multiplier)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

        self.total_size = len(self.lengths)
        self.samples_per_rank = int(math.ceil(self.total_size / self.num_replicas))

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic epoch-wise shuffling."""
        self.epoch = int(epoch)

    def _partition_for_rank(self, indices: List[int]) -> List[int]:
        """Shard indices across ranks with deterministic padding."""
        if self.num_replicas == 1:
            return indices

        required_total = self.samples_per_rank * self.num_replicas
        if len(indices) < required_total:
            indices = indices + indices[: required_total - len(indices)]
        else:
            indices = indices[:required_total]

        return indices[self.rank:required_total:self.num_replicas]

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(self.total_size))
        rng = random.Random(self.seed + self.epoch)

        if self.shuffle:
            rng.shuffle(indices)

        rank_indices = self._partition_for_rank(indices)
        bucket_size = max(self.batch_size, self.batch_size * self.bucket_size_multiplier)

        for bucket_start in range(0, len(rank_indices), bucket_size):
            bucket = rank_indices[bucket_start:bucket_start + bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx], reverse=True)

            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start:batch_start + self.batch_size]
                if len(batch) == self.batch_size:
                    yield batch
                elif batch and not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.samples_per_rank // self.batch_size
        return int(math.ceil(self.samples_per_rank / self.batch_size))
