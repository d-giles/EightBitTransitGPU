# to run: python setup.py build_ext --inplace
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ["matplotlib", "numpy", "scipy",
                "imageio", "numba", "jupyterlab"]

test_requirements = []

setup(name='EightBitTransit',
      description='Shadow imaging of transiting objects',
      python_requires='>=3.6',
      author='Daniel Giles (Orig. Emily Sandford)',
      author_email='daniel.k.giles@gmail.com',
      url='https://github.com/d-giles/EightBitTransitGPU',
      license='MIT',
      packages=find_packages(include=['EightBitTransit', 'EightBitTransit.*']),
      install_requires=requirements,
      long_description=readme,
      long_description_content_type='text/markdown',
      include_package_data=True,
      test_suite='tests',
      tests_require=test_requirements,
      version='2.0.7',
      zip_safe=False,
      )
