from setuptools import setup, find_packages

setup(
    name='BOMLIP-CSP', 
    version='0.1',
    author='Chengxi Zhao, Zhaojia Ma, Dingrui Fan',
    author_email='chengxi_zhao@ustc.edu.cn, zhaojia_ma@foxmail.com',
    description='Integrating machine learning interatomic potentials with batched optimization for crystal structure prediction',
    url='https://github.com/pic-ai-robotic-chemistry/BOMLIP-CSP',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    python_requires='>=3.10',
    package_dir={'': 'src'},
    packages=find_packages('src'),
)