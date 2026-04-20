from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


PACKAGE_ROOT = Path(__file__).resolve().parent

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")


setup(
    name="mempalace-cuda-exact-kernel",
    version="0.1.0",
    description="Optional CUDA exact scoring and deterministic top-k kernels for MemPalace",
    packages=["mempalace", "mempalace.backends"],
    ext_modules=[
        CUDAExtension(
            name="mempalace.backends._cuda_exact_kernel",
            sources=[
                str(PACKAGE_ROOT / "mempalace" / "backends" / "cuda_exact_kernel_bind.cpp"),
                str(PACKAGE_ROOT / "mempalace" / "backends" / "cuda_exact_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": ["/O2"] if os.name == "nt" else ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
