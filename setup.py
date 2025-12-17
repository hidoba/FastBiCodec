from setuptools import setup, find_packages

setup(
    name="ncodec",
    version="0.0.1", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    packages=find_packages(), # Generally good practice to include
)
