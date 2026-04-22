from setuptools import setup, find_packages

setup(
    name="dice-nine",
    version="0.9.4",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.23",
        "rich>=13",
    ],
    python_requires=">=3.12",
)
