import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tbparse",
    version="0.0.1",
    author="Johnson",
    author_email="j3.soon@msa.hinet.net",
    description="A simple parser for reading tensorboard logs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/j3soon/tbparse",
    project_urls={
        "Bug Tracker": "https://github.com/j3soon/tbparse/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "tbparse"},
    packages=setuptools.find_packages(where="tbparse"),
    python_requires=">=3.7",
    install_requires=[
        'pandas>=1.3.0',
        'tensorflow>=2.0.0',
    ],
    extras_require={
        'testing': ['pytest', 'mypy', 'flake8', 'pylint', 'sphinx',
                    'sphinx-rtd-theme', 'torch', 'tensorboardX', 'seaborn',
                    'pytest-cov'],
    },
)