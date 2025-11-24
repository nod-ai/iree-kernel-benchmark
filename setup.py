from setuptools import find_namespace_packages, setup


#### KERNEL BENCHMARK SUITE SETUP ####

setup(
    name=f"iree_kernel_benchmark",
    version="1.0.0",
    author="AMD-SHARK Authors",
    author_email="esaimana@amd.com",
    description="Kernel Perf Tools",
    packages=find_namespace_packages(
        include=[
            "iree_kernel_benchmark.*",
        ],
    ),
    package_dir={"": "."},
)
