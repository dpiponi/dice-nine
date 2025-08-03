from setuptools import setup, find_packages

setup(
    name="dice9",       # Name on PyPI (if published)
    version="0.1.0",          # Version
    package_dir={"": "src"},  # Tell setuptools to look in src/
    packages=find_packages(where="src"),  # Find packages in src/
    install_requires=[],      # Dependencies (e.g., ["numpy>=1.20"])
    python_requires=">=3.6",  # Python version compatibility
)
