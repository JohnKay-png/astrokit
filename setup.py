from setuptools import setup, find_packages

setup(
    name="astrokit",
    version="0.1.0",
    author="zhy",
    author_email="zhy.email@example.com",
    description="Advanced astrodynamics toolkit for Python",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/astrokit",
    packages=find_packages(include=['astrokit*']),
    package_data={
        'astrokit': [
            '**/*.dat',
            '**/*.py'
        ]
    },
    install_requires=[
        'numpy>=1.26.0',
        'scipy>=1.12.0',
        'astropy>=6.0',
        'matplotlib>=3.5.0',
        'plotly>=5.5.0',
        'jplephem>=2.16'
    ],
    python_requires='>=3.12',
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
