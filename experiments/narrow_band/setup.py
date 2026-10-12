"""Build only the experimental hash/contour extension, never production CuMesh."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "cumesh_remesh_experiment" / "_src"

setup(
    name="cumesh-remesh-experiment",
    version="0.0.1",
    packages=["cumesh_remesh_experiment"],
    package_data={
        "cumesh_remesh_experiment": [
            "_src/*.h",
            "_src/*.cpp",
            "_src/hash/*",
            "_src/remesh/*",
        ]
    },
    ext_modules=[
        CUDAExtension(
            name="cumesh_remesh_experiment._C",
            sources=[
                str(SRC / path)
                for path in (
                    "bindings.cpp",
                    "hash/hash.cu",
                    "remesh/simple_dual_contour.cu",
                    "remesh/svox2vert.cu",
                )
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
