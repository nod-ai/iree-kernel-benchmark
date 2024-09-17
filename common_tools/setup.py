from setuptools import find_namespace_packages, setup


#### KERNEL BENCHMARK SUITE SETUP ####

setup(
    name=f"utils",
    version="1.0.0",
    author="SHARK Authors",
    author_email="esaimana@amd.com",
    description="Kernel Perf Tools",
    packages=find_namespace_packages(
        include=[
            "utils",
            "utils.*",
        ],
    ),
)
