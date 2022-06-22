from distutils.core import setup
from distutils.core import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

ext_modules = [
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
        'datamining.projected_clustering',
        [
            'src/cython/gpu/projected_clustering.pyx',
        ],
        libraries=['projectedclustering'],
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
]

for e in ext_modules:
    e.cython_directives = {"embedsignature": True}

setup(
    name='datamining',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules)
)
