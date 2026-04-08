# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)

with open('README.md', 'r') as fh:
    long_description = fh.read()


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[],
                  extra_libraries=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        libraries=extra_libraries,
    )


setup(
    name='fluxvla',
    version='0.0.1',
    author='liyinhao',
    author_email='liyinhao0413@qq.com',
    description='Codebase of fluxvla',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=None,
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[],
    ext_modules=[
        make_cuda_ext(
            name='gemma_rotary_embedding_ext',
            module='fluxvla.ops.cuda.gemma_rotary_embedding',
            sources=['src/gemma_rotary_embedding_forward.cpp'],
            sources_cuda=['src/gemma_rotary_embedding_forward_cuda.cu'],
        ),
        make_cuda_ext(
            name='rotary_pos_embedding_ext',
            module='fluxvla.ops.cuda.rotary_pos_embedding',
            sources=['src/rotary_pos_embedding_forward.cpp'],
            sources_cuda=['src/rotary_pos_embedding_forward_cuda.cu'],
        ),
        make_cuda_ext(
            name='matmul_bias_ext',
            module='fluxvla.ops.cuda.matmul_bias',
            sources=['src/matmul_bias_forward.cpp'],
            sources_cuda=['src/matmul_bias_forward_cuda.cu'],
            extra_libraries=['cublasLt'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
