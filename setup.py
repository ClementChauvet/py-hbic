import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='py_hbic',
    version='1.0',
    author='Adán José-García and Clément Chauvet',
    author_email='clement.chauvet@univ-lille.fr',
    description='Official implementation of the hbic biclustering algorithm',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ClementChauvet/py_hbic',
    license='MIT',
    packages=['hbic', 'hbic.utils'],
    install_requires=['numpy', 'tqdm', 'scipy'],
)