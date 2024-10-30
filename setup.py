from setuptools import setup, find_packages

# # Read dependencies from requirements.txt
# required = []
# with open('requirements.txt') as f:
# 	required = f.read().splitlines()

setup(name='sss',
	  version='0.0.1',
	  packages=find_packages(),
	  description='Stochastic Sparse Sampling (SSS)',
      # requires=required,
	  author='Anonymous Authors',
	  # author_email='',
	  # license='MIT',install_requires=required,classifiers=['Programming Language :: Python :: 3','License :: OSI Approved :: MIT License','Operating System :: OS Independent',],
	  python_requires='>=3.10')
