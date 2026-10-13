from setuptools import setup

setup(
    name="cumesh-wrap",
    version="0.0.3",
    packages=["cumesh_wrap"],
    python_requires=">=3.10",
    install_requires=["torch>=2.5", "warp-lang==1.13.0"],
)
