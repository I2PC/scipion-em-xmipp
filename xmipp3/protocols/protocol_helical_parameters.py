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

import pyworkflow.object as pwobj
from pwem.objects import Volume
from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtPreprocessVolumes
import pyworkflow.protocol.params as params
from pyworkflow.protocol.constants import LEVEL_ADVANCED

from pwem.emlib import MetaData, MDL_ANGLE_ROT, MDL_SHIFT_Z
from xmipp3.base import HelicalFinder
from xmipp3.convert import getImageLocation


class XmippProtHelicalParameters(ProtPreprocessVolumes, HelicalFinder):
    """ Estimate helical parameters and symmetrize.
    
    Helical symmetry is defined as V(r,rot,z)=V(r,rot+k*DeltaRot,z+k*Deltaz).
    """
    _label = 'helical symmetry'
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('inputVolume', params.PointerParam, pointerClass="Volume", label='Input volume')
        form.addParam('cylinderInnerRadius', params.IntParam,label='Cylinder inner radius', default=-1,
                      help="The helix is supposed to occupy this radius in voxels around the Z axis. Leave it as -1 for symmetrizing the whole volume")
        form.addParam('cylinderOuterRadius',params.IntParam,label='Cylinder outer radius', default=-1,
                      help="The helix is supposed to occupy this radius in voxels around the Z axis. Leave it as -1 for symmetrizing the whole volume")
        form.addParam('dihedral',params.BooleanParam,default=False,label='Apply dihedral symmetry')
        form.addParam('forceDihedralX',params.BooleanParam,default=False,expertLevel=LEVEL_ADVANCED, label='Force the dihedral axis to be in X',
                      help="If this option is chosen, then the dihedral axis is not searched and it is assumed that it is around X.")
        form.addParam('additionalCn',params.BooleanParam,default=False,label='Apply Cn symmetry')
        form.addParam('Cn',params.StringParam,default="C2",condition="additionalCn",label='Cn symmetry')

        form.addSection(label='Search limits')
        form.addParam('heightFraction',params.FloatParam,default=0.9,label='Height fraction',
                      help="The helical parameters are only sought using the fraction indicated by this number. "\
                           "In this way, you can avoid including planes that are poorly resolved at the extremes of the volume. " \
                           "However, note that the algorithm can perfectly work with a fraction of 1.")
        form.addParam('rot0',params.FloatParam,default=0,label='Minimum rotational angle',help="In degrees")
        form.addParam('rotF',params.FloatParam,default=360,label='Maximum rotational angle',help="In degrees")
        form.addParam('rotStep',params.FloatParam,default=5,label='Angular step',help="In degrees")
        form.addParam('z0',params.FloatParam,default=1,label='Minimum shift Z',help="In Angstroms")
        form.addParam('zF',params.FloatParam,default=10,label='Maximum shift Z',help="In Angstroms")
        form.addParam('zStep',params.FloatParam,default=0.5,label='Shift step',help="In Angstroms")
        self.deltaZ= params.Float()
        self.deltaRot=params.Float()
        form.addParallelSection(threads=4, mpi=0)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('copyInput')
        self._insertFunctionStep('coarseSearch')
        self._insertFunctionStep('fineSearch')
        self._insertFunctionStep('symmetrize')
        self._insertFunctionStep('createOutput')
        self.fnVol = getImageLocation(self.inputVolume.get())
        self.fnVolSym=self._getPath('volume_symmetrized.mrc')
        [self.height,_,_]=self.inputVolume.get().getDim()

    def _getFileName(self, key, **kwargs):
        if key=="fine":
            return self._getExtraPath('fineParams.xmd')
        elif key=="coarse":
            return self._getExtraPath('coarseParams.xmd')
        else:
            return ""
    
    #--------------------------- STEPS functions --------------------------------------------
    def copyInput(self):
        if self.dihedral:
            if not self.forceDihedralX:
                self.runJob("xmipp_transform_symmetrize","-i %s -o %s --sym dihedral --dont_wrap" % (self.fnVol, self.fnVolSym))
            else:
                self.runJob("xmipp_transform_geometry","-i %s -o %s --rotate_volume axis 180 1 0 0" % (self.fnVol, self.fnVolSym))
                self.runJob("xmipp_image_operate","-i %s --plus %s -o %s" % (self.fnVol, self.fnVolSym, self.fnVolSym))
                self.runJob("xmipp_image_operate","-i %s --mult 0.5" % self.fnVolSym)
        else:
            ImageHandler().convert(self.inputVolume.get(), self.fnVolSym)
                        
    def coarseSearch(self):
        Cn = "c1"
        if self.additionalCn:
            Cn=self.Cn.get()
        self.runCoarseSearch(self.fnVolSym,self.dihedral.get(),float(self.heightFraction.get()),
                             float(self.z0.get()),float(self.zF.get()),float(self.zStep.get()),
                             float(self.rot0.get()),float(self.rotF.get()),float(self.rotStep.get()),
                             self.numberOfThreads.get(),self._getFileName('coarse'),
                             int(self.cylinderInnerRadius.get()),int(self.cylinderOuterRadius.get()),int(self.height),
                             self.inputVolume.get().getSamplingRate(), Cn)

    def fineSearch(self):
        Cn = "c1"
        if self.additionalCn:
            Cn=self.Cn.get()
        self.runFineSearch(self.fnVolSym, self.dihedral.get(), self._getFileName('coarse'),
                           self._getFileName('fine'),
                           float(self.heightFraction.get()),float(self.z0.get()),float(self.zF.get()),
                           float(self.rot0.get()),float(self.rotF.get()),
                           int(self.cylinderInnerRadius.get()),int(self.cylinderOuterRadius.get()),int(self.height),
                           self.inputVolume.get().getSamplingRate(), Cn)

    def symmetrize(self):
        Cn = "c1"
        if self.additionalCn:
            Cn=self.Cn.get()
        self.runSymmetrize(self.fnVolSym, self.dihedral.get(), self._getFileName('fine'), self.fnVolSym,
                           float(self.heightFraction.get()),
                           self.cylinderInnerRadius.get(), self.cylinderOuterRadius.get(), self.height,
                           self.inputVolume.get().getSamplingRate(), Cn)

    def createOutput(self):
        volume = Volume()
        volume.setFileName(self.fnVolSym)
        Ts = self.inputVolume.get().getSamplingRate()
        self.runJob("xmipp_image_header","-i %s --sampling_rate %f"%(self.fnVolSym,Ts))
        volume.copyInfo(self.inputVolume.get())
        self._defineOutputs(outputVolume=volume)
        self._defineTransformRelation(self.inputVolume, self.outputVolume)
        
        md = MetaData(self._getFileName('fine'))
        objId = md.firstObject()
        self._defineOutputs(deltaRot=pwobj.Float(md.getValue(MDL_ANGLE_ROT, objId)),
                            deltaZ=pwobj.Float(md.getValue(MDL_SHIFT_Z, objId)))

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        messages = []
        if self.deltaZ.hasValue():
            messages.append('DeltaZ=%f (voxels) %f (Angstroms)'%(self.deltaZ.get()/self.inputVolume.get().getSamplingRate(),self.deltaZ.get()))
            messages.append('DeltaRot=%f (degrees)'%self.deltaRot.get())      
        return messages

    def _citations(self):
        papers=[]
        return papers

    def _validate(self):
        messages=[]
        if float(self.z0.get())<=0:
            messages.append("z0 should not be negative or zero")
        return messages

    def _methods(self):
        messages = []      
        messages.append('We looked for the helical symmetry parameters of the volume %s using Xmipp [delaRosaTrevin2013].' % self.getObjectTag('inputVolume'))
        if self.deltaZ.hasValue():
            messages.append('We found them to be %f Angstroms and %f degrees.'%(self.deltaZ.get(),self.deltaRot.get()))
            messages.append('We symmetrized %s with these parameters and produced the volume %s.'%(self.getObjectTag('inputVolume'),
                                                                                                  self.getObjectTag('outputVolume')))
            if self.dihedral.get():
                messages.append('We applied dihedral symmetry.')
        return messages

