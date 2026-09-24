import os

# Single-threaded BLAS by default (measured slightly faster for the codec's small
# matmuls, and parallel experiments must not oversubscribe cores). Must be set
# before NumPy loads; an explicit environment setting always wins.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from .cli import main  # noqa: E402

if __name__ == "__main__":
    main()
