from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-stepper",
    version="0.1.3",
    author="alfredwallace7",
    author_email="alfred.wallace@netcraft.fr",
    description="A flexible Python framework for creating step-by-step AI workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alfredwallace7/ai-stepper",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "ai_stepper": ["schema/*.py", "yaml/*.yaml"],
    },
)
