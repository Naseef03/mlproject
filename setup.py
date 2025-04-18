from setuptools import setup, find_packages

def get_requirements(file_path: str) -> list[str]:
    """
    This function will return the list of requirements from the file_path.
    """
    requirements=[]
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Naseef",
    author_email="muhammednaseef03@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)