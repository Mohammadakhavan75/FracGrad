from setuptools import setup, find_packages

setup(
    name="FracGrad",
    version="0.1.0",
    author="Mohammad Akhavan Anvari",
    author_email="mohammad.akhavan75@gmail.com",
    description="Fractional derivatives based gradient descent optimizers for PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Mohammadakhavan75/FracGrad",  # Update with your repo URL
    packages=find_packages(),
    install_requires=[
        "torch",  # Add any other dependencies if needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
