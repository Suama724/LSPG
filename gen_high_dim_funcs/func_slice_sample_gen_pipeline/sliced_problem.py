from typing import Protocol, List, Tuple, Iterator
import numpy as np

class ProblemLike(Protocol):
    def eval(self, x: np.ndarray) -> np.ndarray:
        ...

class SliceInfo:
    
    # slots 控制这个类里只允许存在这些属性
    __slots__ = ("slice_index", "start", "end", "slice_dim", "problem")

    def __init__(
        self,
        slice_index: int,
        start: int,
        end: int,
        problem: ProblemLike,
    ):
        self.slice_index = slice_index
        self.start = start
        self.end = end
        self.slice_dim = end - start
        self.problem = problem  # 带 .eval(x) 的封装，x 的列为 [start:end]

    @property
    def dim(self) -> int:
        return self.slice_dim


class _SlicedProblemWrapper:

    def __init__(
        self,
        full_problem: ProblemLike,
        full_dim: int,
        start: int,
        end: int,
        fill_value: float,
    ):
        self.full_problem = full_problem
        self.full_dim = full_dim
        self.start = start
        self.end = end
        self.fill_value = fill_value
        self.slice_dim = end - start

    def eval(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n, d = x.shape
        if d != self.slice_dim:
            raise ValueError(f"Slice expects {self.slice_dim} dims, got {d}")
        x_full = np.full((n, self.full_dim), self.fill_value, dtype=np.float64)
        x_full[:, self.start : self.end] = x
        return np.asarray(self.full_problem.eval(x_full), dtype=np.float64).ravel()


class HighDimSlicedWrapper:

    def __init__(
        self,
        problem: ProblemLike,
        full_dim: int,
        slice_len: int = 50,
        fill_value: float = 0.0,
    ):
        self.problem = problem
        self.full_dim = full_dim
        self.slice_len = min(slice_len, full_dim)
        self.fill_value = fill_value
        self._slices: List[SliceInfo] = []
        self._build_slices()

    def _build_slices(self) -> None:
        self._slices.clear()
        start = 0
        idx = 0
        while start < self.full_dim:
            end = min(start + self.slice_len, self.full_dim)
            wrapper = _SlicedProblemWrapper(
                self.problem,
                self.full_dim,
                start,
                end,
                self.fill_value,
            )
            self._slices.append(
                SliceInfo(slice_index=idx, start=start, end=end, problem=wrapper)
            )
            start = end
            idx += 1

    def num_slices(self) -> int:
        return len(self._slices)

    def get_slice(self, index: int) -> SliceInfo:
        return self._slices[index]

    def iter_slices(self) -> Iterator[SliceInfo]:
        yield from self._slices

    def slices(self) -> List[SliceInfo]:
        return list(self._slices)
