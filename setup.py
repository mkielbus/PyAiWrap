from setuptools import setup, find_packages
import os


def read_dependencies(file_path: str):
    dependencies = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    dependencies.append(line)
    return dependencies


setup(
    name="PyAiWrap",
    version="1.0.0",
    description="Neural network skeleton with dynamic layer construction from JSON",
    author="Mateusz Kiełbus",
    author_email="mateusz.kielbus.mk@gmail.com",
    url="https://github.com/mkielbus/AI-in-Computer-Graphics-Labs",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=read_dependencies("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Data Science :: Deep Learning",
        "Programming :: Developers"
    ],
    include_package_data=True
)
