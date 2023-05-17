from setuptools import setup, find_packages

# get description from readme file
with open("README.md", "r") as f:
    long_description = f.read()

# setup
setup(
    name="SkillsRefining",
    version="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="A, B, C",
    author_email="abs@def",
    maintainer=" ",
    maintainer_email="",
    license=" ",
    url=" ",
    platforms=["Linux Ubuntu"],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
