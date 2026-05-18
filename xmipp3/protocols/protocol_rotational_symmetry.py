# **************************************************************************
# *
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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

import pyworkflow
import pyworkflow.object as pwobj
from pwem.emlib.image import ImageHandler
from pwem.objects import Volume

from pwem.protocols import (ProtPreprocessVolumes)
from pyworkflow.protocol import (PointerParam, IntParam, EnumParam, FloatParam)

from pwem.emlib import MetaData, MDL_ANGLE_ROT, MDL_ANGLE_TILT
from xmipp3.convert import getImageLocation


class XmippProtRotationalSymmetry(ProtPreprocessVolumes):
    """
    Estimate the orientation of a rotational axis and symmetrize.
    The user should know the order of the axis (two-fold, three-fold, ...)
    If this is unknown you may try several and see the most consistent results.

    AI Generated

    ## Overview

    The Rotational Symmetry protocol estimates the orientation of a rotational
    symmetry axis in a 3D volume and then symmetrizes the volume according to that
    axis.

    Many cryo-EM structures have rotational symmetry, such as two-fold, three-fold,
    or higher-order axes. If the symmetry order is known, this protocol can search
    for the axis orientation that best explains the symmetry present in the map.
    After the axis has been estimated, the protocol rotates the volume into the
    corresponding orientation and applies cyclic symmetrization.

    The protocol is useful when the user knows the expected order of the symmetry
    axis but does not know its exact orientation in the volume.

    The main output is a symmetrized volume. The protocol also outputs the
    estimated rotational and tilt angles of the symmetry axis.

    ## Inputs and General Workflow

    The input is a single 3D volume.

    The user provides the expected rotational symmetry order and chooses how the
    axis orientation should be searched. The search can be global, local, or a
    combination of global and local.

    The protocol first copies the input volume to a working file. It then searches
    for the symmetry axis using the selected mode. Once the best axis orientation
    has been found, the volume is rotated according to the estimated angles and
    symmetrized with the selected cyclic order.

    Finally, the symmetrized volume is registered as the output volume, and the
    estimated axis angles are stored as additional outputs.

    ## Input Volume

    The **Input volume** parameter defines the map in which the rotational symmetry
    axis will be searched.

    The volume should already be reasonably centered and should contain the
    structure whose symmetry is being analyzed. Strong miscentering, artifacts,
    large background regions, or severe asymmetry may make the symmetry-axis
    estimation less reliable.

    The protocol does not refine the reconstruction. It works only on the provided
    map and produces a symmetrized version of it.

    ## Symmetry Order

    The **Symmetry order** parameter defines the order of the rotational symmetry
    axis.

    For example:

    - 2 means a two-fold rotational axis;
    - 3 means a three-fold rotational axis;
    - 4 means a four-fold rotational axis.

    The user is expected to know or hypothesize this order before running the
    protocol. If the order is unknown, the user can try several possible values and
    compare which results are most consistent with the map and with the biological
    structure.

    Using an incorrect symmetry order may produce an artificially symmetrized
    volume and can obscure real asymmetric features.

    ## Search Mode

    The **Search mode** parameter controls how the orientation of the symmetry axis
    is estimated.

    There are three options:

    **Global** performs a coarse search over the specified rotational and tilt
    angle ranges.

    **Local** starts from user-provided initial rotational and tilt angles and
    refines the axis orientation locally.

    **Global+Local** first performs a global coarse search and then refines the
    best result locally. This is the default and usually the safest option when the
    axis orientation is not known.

    The search mode should be chosen according to how much the user already knows
    about the axis orientation.

    ## Global Search

    In global search mode, the protocol scans a range of possible symmetry-axis
    orientations.

    The user defines the minimum, maximum, and step values for:

    - rotational angle;
    - tilt angle.

    The protocol evaluates possible axis orientations over this grid and writes
    the best result to an intermediate metadata file.

    Global search is useful when the symmetry axis could be anywhere in the volume.
    However, the accuracy of the initial result depends on the angular step size. A
    smaller step gives a finer search but increases computation time.

    ## Local Search

    In local search mode, the user provides an initial estimate of the symmetry
    axis orientation.

    The parameters are:

    - **Initial rotational angle**;
    - **Initial tilt angle**.

    The protocol then performs a local refinement around this starting orientation.

    Local search is useful when the user already has a good approximate axis
    orientation, for example from previous analysis, visual inspection, or a
    related map. It is faster than a full global search but can fail if the initial
    orientation is too far from the correct axis.

    ## Global+Local Search

    In Global+Local mode, the protocol first performs a coarse global search and
    then refines the best candidate with a local search.

    This mode combines the robustness of global exploration with the accuracy of
    local refinement. It is recommended when the symmetry order is known but the
    axis orientation is not known precisely.

    The global step reduces the risk of starting the local refinement from a poor
    orientation.

    ## Rotational Angle Range

    The **Rotational angle** parameters define the angular range explored during
    global or Global+Local search.

    The parameters are:

    - minimum rotational angle;
    - maximum rotational angle;
    - rotational step.

    All values are expressed in degrees.

    A broad range such as 0 to 360 degrees explores the full rotational space. A
    smaller range can be used when the approximate direction is already known.

    ## Tilt Angle Range

    The **Tilt angle** parameters define the tilt range explored during global or
    Global+Local search.

    In this convention, tilt = 0 degrees corresponds to a top axis, while tilt = 90
    degrees corresponds to a side axis.

    The parameters are:

    - minimum tilt angle;
    - maximum tilt angle;
    - tilt step.

    As with the rotational range, smaller steps give a finer search but increase
    computation time.

    ## Symmetrization Step

    After estimating the symmetry-axis orientation, the protocol symmetrizes the
    volume.

    It first rotates the volume using the estimated Euler orientation of the
    symmetry axis. It then applies cyclic symmetry of the selected order.

    For example, if the symmetry order is 3, the protocol applies C3
    symmetrization. If the order is 4, it applies C4 symmetrization.

    The result is a new volume in which density has been averaged according to the
    estimated rotational symmetry.

    ## Output Volume

    The main output is **outputVolume**.

    This volume is the symmetrized version of the input map. It is written as
    `volume_symmetrized.mrc` and keeps the sampling rate and metadata copied from
    the input volume.

    The original input volume is not modified.

    The output can be used for visualization, further processing, or comparison
    with the unsymmetrized input map.

    ## Estimated Symmetry-Axis Angles

    The protocol also outputs two scalar values:

    - **rotSym**, the estimated rotational angle of the symmetry axis;
    - **tiltSym**, the estimated tilt angle of the symmetry axis.

    These values describe the orientation of the detected symmetry axis in degrees.

    They are also shown in the protocol summary, helping the user document the
    axis orientation found by the search.

    ## Interpreting the Result

    The symmetrized output should be interpreted carefully.

    If the symmetry order and axis are correct, symmetrization can improve the
    apparent signal by averaging equivalent density. It can make symmetric features
    clearer and reduce asymmetric noise.

    However, if the symmetry order is wrong, if the axis is incorrectly estimated,
    or if the biological structure is only approximately symmetric, the output may
    contain artificial density or may erase real asymmetric features.

    The symmetrized volume should therefore be compared with the original input
    map.

    ## Practical Recommendations

    Use this protocol when the expected rotational symmetry order is known.

    Use **Global+Local** search when the axis orientation is unknown. Use **Local**
    search only when a reliable approximate orientation is already available.

    Start with moderate angular steps for global search. If the result is
    promising, repeat with a finer search or use local refinement.

    Try different symmetry orders if the correct order is uncertain, but interpret
    the results biologically rather than selecting only the visually sharpest map.

    Always compare the symmetrized volume with the original volume to check
    whether real asymmetric features have been lost.

    Do not use this protocol as evidence that a structure truly has a given
    symmetry. It is a tool for estimating and applying a hypothesized rotational
    symmetry.

    ## Final Perspective

    Rotational Symmetry is a volume-processing protocol for detecting the
    orientation of a known-order rotational axis and applying the corresponding
    cyclic symmetrization.

    For biological users, it is useful when a map is expected to contain rotational
    symmetry but the axis is not aligned with the standard coordinate axes.

    The protocol provides both a symmetrized map and the estimated axis angles,
    making it useful for reorientation, symmetry validation, visualization, and
    preparation of maps for downstream workflows that assume a particular symmetry
    axis.
    """
    _label = 'rotational symmetry'
    _version = pyworkflow.VERSION_1_1
    
    GLOBAL_SEARCH = 0
    LOCAL_SEARCH = 1
    GLOBAL_LOCAL_SEARCH = 2

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('inputVolume', PointerParam, pointerClass="Volume",
                      label='Input volume')
        form.addParam('symOrder',IntParam, default=3,
                      label='Symmetry order',
                      help="3 for a three-fold symmetry axis, "
                           "4 for a four-fold symmetry axis, ...")
        form.addParam('searchMode', EnumParam,label='Search mode',
                      choices=['Global','Local','Global+Local'],
                      default=self.GLOBAL_LOCAL_SEARCH)

        form.addParam('rot', FloatParam, default=0,
                      condition='searchMode==%d' % self.LOCAL_SEARCH,
                      label='Initial rotational angle', help="In degrees")
        form.addParam('tilt', FloatParam, default=0,
                      condition='searchMode==%d' % self.LOCAL_SEARCH,
                      label='Initial tilt angle',
                      help="In degrees. tilt=0 is a top axis while "
                           "tilt=90 defines a side axis")

        rot = form.addLine('Rotational angle',
                           condition='searchMode!=%d' % self.LOCAL_SEARCH,
                           help='Minimum, maximum and step values for '
                                'rotational angle range, all in degrees.')
        rot.addParam('rot0', FloatParam, default=0, label='Min')
        rot.addParam('rotF', FloatParam, default=360, label='Max')
        rot.addParam('rotStep', FloatParam, default=5, label='Step')

        tilt = form.addLine('Tilt angle',
                            condition='searchMode!=%d' % self.LOCAL_SEARCH,
                            help='In degrees. tilt=0 is a top axis while '
                                 'tilt=90 defines a side axis')
        tilt.addParam('tilt0', FloatParam, default=0, label='Min')
        tilt.addParam('tiltF', FloatParam, default=180, label='Max')
        tilt.addParam('tiltStep', FloatParam, default=5, label='Step')

        self.rotSym = pwobj.Float()
        self.tiltSym = pwobj.Float()

        form.addParallelSection(threads=4, mpi=0)

    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('copyInput')
        if self.searchMode.get()!=self.LOCAL_SEARCH:
            self._insertFunctionStep('coarseSearch')
        if self.searchMode.get()!=self.GLOBAL_SEARCH:
            self._insertFunctionStep('fineSearch')
        self._insertFunctionStep('symmetrize')
        self._insertFunctionStep('createOutput')
        self.fnVol = getImageLocation(self.inputVolume.get())
        self.fnVolSym=self._getPath('volume_symmetrized.mrc')
        [self.height,_,_]=self.inputVolume.get().getDim()
    
    #--------------------------- STEPS functions -------------------------------
    def copyInput(self):
        ImageHandler().convert(self.inputVolume.get(), self.fnVolSym)
                        
    def coarseSearch(self):
        self.runJob("xmipp_volume_find_symmetry",
                    "-i %s -o %s --rot %f %f %f --tilt %f %f %f --sym rot %d --thr %d" %
                    (self.fnVolSym, self._getExtraPath('coarse.xmd'),
                     self.rot0, self.rotF, self.rotStep,
                     self.tilt0, self.tiltF, self.tiltStep,
                     self.symOrder, self.numberOfThreads))

    def getAngles(self, fnAngles=""):
        if fnAngles=="":
            if self.searchMode.get()==self.GLOBAL_SEARCH:
                fnAngles = self._getExtraPath("coarse.xmd")
            else:
                fnAngles = self._getExtraPath("fine.xmd")
        md = MetaData(fnAngles)
        objId = md.firstObject()
        rot0 = md.getValue(MDL_ANGLE_ROT, objId)
        tilt0 = md.getValue(MDL_ANGLE_TILT, objId)
        return (rot0,tilt0)

    def fineSearch(self):
        if self.searchMode == self.LOCAL_SEARCH:
            rot0 = self.rot.get()
            tilt0 = self.tilt.get()
        else:
            rot0, tilt0 = self.getAngles(self._getExtraPath('coarse.xmd'))
        self.runJob("xmipp_volume_find_symmetry",
                    "-i %s -o %s --localRot %f %f --sym rot %d"%
                    (self.fnVolSym,self._getExtraPath('fine.xmd'),
                     rot0, tilt0, self.symOrder.get()))

    def symmetrize(self):
        rot0, tilt0 = self.getAngles()
        self.runJob("xmipp_transform_geometry",
                    "-i %s --rotate_volume euler %f %f 0 --dont_wrap" %
                    (self.fnVolSym,rot0,tilt0))
        self.runJob("xmipp_transform_symmetrize",
                    "-i %s --sym c%d --dont_wrap" %
                    (self.fnVolSym,self.symOrder.get()))

    def createOutput(self):
        Ts = self.inputVolume.get().getSamplingRate()
        self.runJob("xmipp_image_header","-i %s --sampling_rate %f" %(self.fnVolSym,Ts))

        volume = Volume()
        volume.setFileName(self.fnVolSym)
        volume.copyInfo(self.inputVolume.get())
        self._defineOutputs(outputVolume=volume)
        self._defineTransformRelation(self.inputVolume, self.outputVolume)
        
        rot0, tilt0 = self.getAngles()
        self._defineOutputs(rotSym=pwobj.Float(rot0),
                            tiltSym=pwobj.Float(tilt0))

    #--------------------------- INFO functions --------------------------------
    def _summary(self):
        messages = []
        if self.rotSym.hasValue():
            messages.append('Rot. Angle of Symmetry axis=%0.2f (degrees)' %
                            self.rotSym.get())
            messages.append('Tilt.Angle of Symmetry axis=%0.2f (degrees)' %
                            self.tiltSym.get())
        return messages

    def _methods(self):
        messages = []      
        messages.append('We looked for the %d-fold rotational axis of the '
                        'volume %s using Xmipp [delaRosaTrevin2013]. '
                        'We found it to be with an orientation given by a '
                        'rotational angle of %f and a tilt angle of %f degrees.'
                        % (self.symOrder, self.getObjectTag('inputVolume'),
                          self.rotSym, self.tiltSym))
        return messages

