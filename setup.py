import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kim-python-utils",
    version="0.2.0",
    description=(
        "Helper routines for writing KIM Tests and Verification Checks"
    ),
    author=["Ellad B. Tadmor", "Daniel S. Karls"],
    url="https://github.com/openkim/kim-python-utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license="CDDL",
    install_requires=["numpy >= 1.13.1", "scipy >= 1.3.0", "jinja2 >= 2.7.2", "ase >= 3.19.0b1"],
    classifiers=[
        "Development Status :: 4 - Beta"
        "License :: OSI Approved :: Common Development and Distribution License 1.0 (CDDL-1.0)",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
