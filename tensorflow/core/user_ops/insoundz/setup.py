"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution


__version__ = '0.0.1'
REQUIRED_PACKAGES = [
    'tensorflow >= 1.12.0',
]

project_name = 'tensorflow-insoundz-ops'

class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True

setup(
    name=project_name,
    version=__version__,
    description=('tensorflow-insoundz-ops is insoundz ops for TensorFlow'),
    author='InSoundz LTD.',
    author_email='emil.winebrand@insoundz.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    #ext_modules=[Extension('_foo', ['stub.cc'])],
    zip_safe=False,
    distclass=BinaryDistribution,
    # PyPI package information.
    keywords='tensorflow insoundz ops',
)
