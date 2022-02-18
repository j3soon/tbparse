import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

github_url = "https://github.com/j3soon/tbparse"

setuptools.setup(
    name="tbparse",
    version="0.0.6",
    author="Johnson",
    author_email="j3.soon@msa.hinet.net",
    description="Load tensorboard event logs as pandas DataFrames; " + \
    "Read, parse, and plot tensorboard event logs with ease!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=github_url,
    project_urls={
        "Changelog": f"{github_url}/blob/master/docs/pages/changelog.rst",
        "Issues": f"{github_url}/issues",
        "Source Code": github_url,
    },
    keywords=(
        "package, parser, plot, python, pytorch, reader, tensorboard, tensorboardx, tensorflow"
    ),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where=".", exclude=("tests*",)),
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "tensorflow>=2.0.0",
    ],
    extras_require={
        "testing": ["pytest", "mypy", "flake8", "pylint", "sphinx",
                    "sphinx-rtd-theme", "torch", "tensorboardX", "seaborn",
                    "torchvision", "soundfile", "pytest-cov", "sphinx-tabs", 
                    "nbsphinx"],
    },
)

# Note: PyPI seems to only recognize double-quoted strings