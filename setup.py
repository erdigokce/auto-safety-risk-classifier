from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='auto-safety-risk-classifier',
      version='0.1',
      description='A Gaussian Naive Bayes approach for classification.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Intended Audience :: Financial and Insurance Industry'
        'Environment :: Console'
      ],
      url='https://github.com/erdigokce/auto-safety-risk-classifier',
      author='Erdi Gokce',
      author_email='erdi.gokce@gmail.com',
      license='BSD',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'sklearn',
          'pandas',
          'matplotlib',
          'requests',
          'seaborn',
          'nose',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      scripts=['bin/app.py'],
      zip_safe=False)
