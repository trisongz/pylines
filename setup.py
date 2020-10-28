import os
import os.path
from setuptools import setup, find_packages

root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, 'README.md'), 'rb') as readme:
    long_description = readme.read().decode('utf-8')

setup(
    name='pylines',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    version='0.0.1',
    description='work with large jsonline files with ease',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Tri Songz',
    author_email='ts@scontentengine.ai',
    url='http://github.com/trisongz/pylines',
    keywords=['json', 'json lines', 'jsonlines'],
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
    ],
    python_requires='>3.4',
    install_requires=[
        "pysimdjson"
    ],
    extras_require={
        'gcp': [
            'tensorflow>=2.1.0',
            'google-cloud-storage'
        ],
        'cloud': [
            'tensorflow>=2.1.0',
            'google-cloud-storage',
            'smart_open[all]'
        ],
        'nlp': [
            'transformers',
        ],
        'all': [
            'tensorflow>=2.1.0',
            'torch>=1.0',
            'google-cloud-storage',
            'smart_open[all]',
            'transformers',
        ]
    }
)