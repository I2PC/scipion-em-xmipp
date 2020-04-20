# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Laura del Cano (ldelcano@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
"""
This module contains utils functions for Xmipp protocols
"""

from os.path import join
import numpy as np

from pyworkflow import Config
from pwem import emlib


def validateXmippGpuBins():
    pass


BAD_IMPORT_TENSORFLOW_KERAS_MSG='''
Error, tensorflow/keras is probably not installed. Install it with:\n  ./scipion installb deepLearningToolkit
If gpu version of tensorflow desired, install cuda 8.0 or cuda 9.0
We will try to automatically install cudnn, if unsucesfully, install cudnn and add to LD_LIBRARY_PATH
add to {0}
CUDA = True
CUDA_VERSION = 8.0 or 9.0
CUDA_HOME = /path/to/cuda-%(CUDA_VERSION)s
CUDA_BIN = %(CUDA_HOME)s/bin
CUDA_LIB = %(CUDA_HOME)s/lib64
CUDNN_VERSION = 6 or 7
'''.format(Config.SCIPION_CONFIG)


def copy_image(imag):
    ''' Return a copy of a xmipp_image
    '''
    new_imag = emlib.Image()
    new_imag.setData(imag.getData())
    return new_imag


def matmul_serie(mat_list, size=4):
    '''Return the matmul of several numpy arrays'''
    #Return the identity matrix if te list is empty
    if len(mat_list) > 0:
        res = np.identity(len(mat_list[0]))
        for i in range(len(mat_list)):
            res = np.matmul(res, mat_list[i])
    else:
        res = np.identity(size)
    return res


def normalize_array(ar):
    '''Normalize values in an array with mean 0 and std deviation 1
    '''
    ar -= np.mean(ar)
    ar /= np.std(ar)
    return ar


def surrounding_values(a,ii,jj,depth=1):
    '''Return a list with the surrounding elements, given the indexs of the center, from an 2D numpy array
    '''
    values=[]
    for i in range(ii-depth,ii+depth+1):
        for j in range(jj-depth,jj+depth+1):
            if i>=0 and j>=0 and i<a.shape[0] and j<a.shape[1]:
                if i!=ii or j!=jj:
                    values+=[a[i][j]]
    return values
