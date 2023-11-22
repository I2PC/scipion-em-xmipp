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

from os.path import join, basename, isfile
import numpy as np
import math
from pyworkflow import Config
import pyworkflow.utils as pwutils
from pwem import emlib
from pyworkflow.object import Object


def validateXmippGpuBins():
    pass


BAD_IMPORT_TENSORFLOW_KERAS_MSG = '''
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


def writeImageFromArray(array, fn):
    img = emlib.Image()
    img.setData(array)
    img.write(fn)


def readImage(fn):
    img = emlib.Image()
    img.read(fn)
    return img


def applyTransform(imag_array, M, shape):
    ''' Apply a transformation(M) to a np array(imag) and return it in a given shape
  '''
    imag = emlib.Image()
    imag.setData(imag_array)
    imag = imag.applyWarpAffine(list(M.flatten()), shape, True)
    return imag.getData()


def rotation(imag, angle, shape, P):
    '''Rotate a np.array and return also the transformation matrix
  #imag: np.array
  #angle: angle in degrees
  #shape: output shape
  #P: transform matrix (further transformation in addition to the rotation)'''
    (hsrc, wsrc) = imag.shape
    angle *= math.pi / 180
    T = np.asarray([[1, 0, -wsrc / 2], [0, 1, -hsrc / 2], [0, 0, 1]])
    R = np.asarray([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    M = np.matmul(np.matmul(np.linalg.inv(T), np.matmul(R, T)), P)

    transformed = applyTransform(imag, M, shape)
    return transformed, M


def flipYImage(inFn, outFn=None, outDir=None):
    '''Flips an image in the Y axis'''
    if outFn == None:
        if not '_flipped' in basename(inFn):
            ext = pwutils.getExt(inFn)
            outFn = inFn.replace(ext, '_flipped' + ext)
        else:
            outFn = inFn.replace('_flipped', '')
    if outDir != None:
        outFn = outDir + '/' + basename(outFn)
    if isfile(outFn):
        return outFn 
    gainImg = readImage(inFn)
    imag_array = np.asarray(gainImg.getData(), dtype=np.float64)

    # Flipped Y matrix
    M, angle = np.asarray([[1, 0, 0], [0, -1, imag_array.shape[0]], [0, 0, 1]]), 0
    flipped_array, M = rotation(imag_array, angle, imag_array.shape, M)
    writeImageFromArray(flipped_array, outFn)
    return outFn


def copy_image(imag):
    ''' Return a copy of a xmipp_image
    '''
    new_imag = emlib.Image()
    new_imag.setData(imag.getData())
    return new_imag


def matmul_serie(mat_list, size=4):
    '''Return the matmul of several numpy arrays'''
    # Return the identity matrix if te list is empty
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


def surrounding_values(a, ii, jj, depth=1):
    '''Return a list with the surrounding elements, given the indexs of the center, from an 2D numpy array
    '''
    values = []
    for i in range(ii - depth, ii + depth + 1):
        for j in range(jj - depth, jj + depth + 1):
            if i >= 0 and j >= 0 and i < a.shape[0] and j < a.shape[1]:
                if i != ii or j != jj:
                    values += [a[i][j]]
    return values


class Point:
    """ Return x, y 2d coordinates and some other properties
    such as weight and state.
    """
    # Selection states
    DISCARDED = -1
    NORMAL = 0
    SELECTED = 1

    def __init__(self, pointId, data, weight, state=0):
        self._id = pointId
        self._data = data
        self._weight = weight
        self._state = state
        self._container = None

    def getId(self):
        return self._id

    def getX(self):
        return self._data[self._container.XIND]

    def setX(self, value):
        self._data[self._container.XIND] = value

    def getY(self):
        return self._data[self._container.YIND]

    def setY(self, value):
        self._data[self._container.YIND] = value

    def getZ(self):
        return self._data[self._container.ZIND]

    def setZ(self, value):
        self._data[self._container.ZIND] = value

    def getWeight(self):
        return self._weight

    def getState(self):
        return self._state

    def setState(self, newState):
        self._state = newState

    def eval(self, expression):
        localDict = {}
        for i, x in enumerate(self._data):
            localDict['x%d' % (i + 1)] = x
        return eval(expression, {"__builtins__": None}, localDict)

    def setSelected(self):
        self.setState(Point.SELECTED)

    def isSelected(self):
        return self.getState() == Point.SELECTED

    def setDiscarded(self):
        self.setState(Point.DISCARDED)

    def isDiscarded(self):
        return self.getState() == Point.DISCARDED

    def getData(self):
        return self._data


class Data():
    """ Store data points. """

    def __init__(self, **kwargs):
        # Indexes of data
        self._dim = kwargs.get('dim')  # The points dimensions
        self.clear()

    def addPoint(self, point, position=None):
        point._container = self
        if position is None:
            self._points.append(point)
        else:
            self._points.insert(position, point)

    def getPoint(self, index):
        return self._points[index]

    def __iter__(self):
        for point in self._points:
            if not point.isDiscarded():
                yield point

    def iterAll(self):
        """ Iterate over all points, including the discarded ones."""
        return iter(self._points)

    def getXData(self):
        return [p.getX() for p in self]

    def getYData(self):
        return [p.getY() for p in self]

    def getZData(self):
        return [p.getZ() for p in self]

    def getWeights(self):
        return [p.getWeight() for p in self]

    def getSize(self):
        return len(self._points)

    def getSelectedSize(self):
        return len([p for p in self if p.isSelected()])

    def getDiscardedSize(self):
        return len([p for p in self.iterAll() if p.isDiscarded()])

    def clear(self):
        self.XIND = 0
        self.YIND = 1
        self.ZIND = 2
        self._points = []


class PathData(Data):
    """ Just contains two list of x and y coordinates. """

    def __init__(self, **kwargs):
        Data.__init__(self, **kwargs)

    def splitLongestSegment(self):
        """ Split the longest segment by adding the midpoint. """
        maxDist = 0
        n = self.getSize()
        # Find the longest segment and its index
        for i in range(n - 1):
            p1 = self.getPoint(i)
            x1, y1 = p1.getX(), p1.getY()
            p2 = self.getPoint(i + 1)
            x2, y2 = p2.getX(), p2.getY()
            dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if dist > maxDist:
                maxDist = dist
                maxIndex = i + 1
                midX = (x1 + x2) / 2
                midY = (y1 + y2) / 2
        # Add a midpoint to it
        point = self.createEmptyPoint()
        point.setX(midX)
        point.setY(midY)
        self.addPoint(point, position=maxIndex)

    def createEmptyPoint(self):
        data = [0.] * self._dim  # create 0, 0...0 point
        point = Point(0, data, 0)
        point._container = self

        return point

    def removeLastPoint(self):
        del self._points[-1]

