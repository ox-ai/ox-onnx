
from setuptools import setup, find_packages

setup(
    name="ox-onnx",
    version="0.0.01",
    description="ox-onnx a clean interface lib to work with onnx models",
    author="Lokeshwaran M",
    author_email="lokeshwaran.m23072003@gmail.com",
    url="https://github.com/ox-ai/ox-onnx.git",
    license="MIT",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
    # entry_points={
    #     "console_scripts": [
    #         "ox-onnx=ox_onnx.run:main",
          
    #     ],
    # },
    package_data={
        "": ["requirements.txt", "README.md"]
    },
    include_package_data=True,
    python_requires=">=3.6",
    keywords="ox-onnx ox-ai",
)