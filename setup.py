from distutils.core import setup as CySetup
from distutils.core import Extension
from Cython.Build import cythonize
import numpy

CySetup(
    name='datamining',
    ext_modules=cythonize([
        Extension(
            'datamining.clustering',
            [
                'src/cython/gpu/clustering.pyx',
            ],
            libraries=['clustering'],
            library_dirs=[
                './libs',
                'cython',
                '.',
                '..',
            ],
            language='c++',
            include_dirs=[
                numpy.get_include(),
            ],
            runtime_library_dirs=[
                '.',
            ]
        ),
        Extension(
            'datamining.subspace_clustering',
            [
                'src/cython/gpu/subspace_clustering.pyx',
            ],
            libraries=['subspaceclustering'],
            library_dirs=[
                './libs',
                'cython',
                '.',
                '..',
            ],
            language='c++',
            include_dirs=[
                numpy.get_include(),
            ],
            runtime_library_dirs=[
                '.',
            ]
        )
    ])
)
