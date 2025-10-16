from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="arabic-ocr-system",
    version="1.0.0",
    author="Mohamed Elhaj Suliman",
    author_email="your.email@example.com",
    description="Real-Time Arabic OCR System with Vision Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Real-Time-Arabic-OCR-System",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "arabic-ocr=src.cli:main",
        ],
    },
)
