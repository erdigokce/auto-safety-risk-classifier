from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='pgi-ins-auto-riskestimator-classifier',
      version='0.1',
      description='A Naive Bayesian approach for classification.',
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
      url='https://git.planetgroupint.com/erdi.gokce/pgi-ins-auto-riskestimator-classifier',
      author='Erdi Gokce',
      author_email='erdi.gokce@planetgroupint.com',
      license='BSD',
      packages=['pgi_ins_auto_riskestimator_classifier'],
      install_requires=[
          'numpy',
          'sklearn',
          'pandas',
          'matplotlib',
          'requests',
          'nose',
          'sklearn_pandas'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      scripts=['bin/pgi-ins-auto-riskestimator-classifier'],
      zip_safe=False)
