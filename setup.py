from setuptools import setup, find_packages

setup(
    name="outfitting",
    version="0.1.0",
    description="Outfit recommendation using CLIP embeddings and Transformers",
    author="Your Name",
    author_email="you@example.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "clip": ["open_clip_torch>=2.20.0"],
        "tuning": ["optuna>=3.0.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "isort>=5.10.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
