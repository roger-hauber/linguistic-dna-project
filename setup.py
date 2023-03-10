from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='linguistic-dna',
      version="1.0", # TO DO: check correct version
      description="Linguistic DNA Model English aceent prediction",
      # url=https://github.com/roger-hauber/linguistic-dna-project
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
