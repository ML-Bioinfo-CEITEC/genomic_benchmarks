from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="genomic_benchmarks",
    version="0.0.7",
    description="Genomic Benchmarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RBP Bioinformatics",
    author_email="ML.Bioinfo.CEITEC@gmail.com",
    license="Apache License 2.0",
    keywords=["bioinformatics", "genomics", "data"],
    url="https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks",
    packages=find_packages("src"),
    package_dir={"": "src"},
    scripts=[],
    # setup_requires=['pytest-runner'], setup_requires deprecated in v58.3.0
    install_requires=[
        "biopython>=1.79",
        "requests>=2.23.0",
        "pip>=20.0.1",
        "numpy>=1.17.0",
        "pandas>=1.1.4",
        "tqdm>=4.41.1",
        "pyyaml>=5.3.1",
        'gdown>=4.2.0',
        "yarl",
        #'papermill>=2.3.0',
        #'tensorflow>=2.6.0',
        #'jupyter>=1.0.0',
        #'torch>=1.9.0',
    ],
    tests_require=["pytest"],
    include_package_data=True,
    package_data={},
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Development Status :: 3 - Alpha",
        # Define that your audience are developers
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        # Specify which pyhton versions that you want to support
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
