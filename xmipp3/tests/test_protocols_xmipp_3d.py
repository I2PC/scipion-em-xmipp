# **************************************************************************
# *
# * Authors:    Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
# *             Jose Gutierrez (jose.gutierrez@cnb.csic.es)
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

from pwem.protocols import (ProtImportVolumes, ProtImportMask,
                            ProtImportParticles, ProtImportAverages,
                            ProtImportPdb, ProtSubSet)
from xmipp3.protocols.protocol_align_volume_and_particles import AlignVolPartOutputs

try:
    from itertools import izip
except ImportError:
    izip = zip

from pyworkflow.utils import greenStr, magentaStr
from pyworkflow.tests import *
from xmipp3.base import *
from xmipp3.convert import *
from xmipp3.constants import *
from xmipp3.protocols import *
from xmipp3.protocols import (XmippFilterHelper as xfh,
                              XmippResizeHelper as xrh,
                              OP_COLUNM, OP_DOTPRODUCT, OP_MULTIPLY,
                              OP_SQRT, OP_RADIAL, OP_ROW)
from xmipp3.protocols.protocol_align_volume import (ALIGN_ALGORITHM_EXHAUSTIVE,
                                               ALIGN_ALGORITHM_EXHAUSTIVE_LOCAL,
                                               ALIGN_ALGORITHM_LOCAL)

# Global variables
db_xmipp_tutorial = 'xmipp_tutorial'
db_general = 'general'
db_model_building_tutorial = 'model_building_tutorial'
vol_coot1 = 'volumes/coot1.mrc'
vol1_iter2 = 'volumes/volume_1_iter_002.mrc'
vol2_iter2 = 'volumes/volume_2_iter_002.mrc'
helix = 'volumes/helix_59_4__6_7.vol'
pdb_coot1 = 'PDBx_mmCIF/coot1.pdb'


# Output error messages
MSG_WRONG_SAMPLING = "There was a problem with the sampling rate value of the output "
MSG_WRONG_SIZE = "There was a problem with the size of the output "
MSG_WRONG_DIM = "There was a problem with the dimensions of the output "
MSG_WRONG_MASK = "There was a problem with create mask from volume"
MSG_WRONG_ALIGNMENT = "There was a problem with the alignment of the output "
MSG_WRONG_SHIFT = "There was a problem with output shift "
MSG_WRONG_GALLERY = "There was a problem with the gallery creation"
MSG_WRONG_ROTATION = "There was a problem with the rotation"
MSG_WRONG_IMPORT = "There was a problem with the import of "
MSG_WRONG_PROTOCOL = "There was a problem with the protocol: "
MSG_WRONG_MAP = "There was a problem with the map creation"

class TestXmippBase(BaseTest):
    """ Some utility functions to import volumes that are used in several tests."""

    @classmethod
    def setData(cls, dataProject=db_xmipp_tutorial):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.volumes = cls.dataset.getFile('volumes')
        cls.vol1 = cls.dataset.getFile('vol1')
        cls.vol2 = cls.dataset.getFile('vol2')
        cls.vol3 = cls.dataset.getFile('vol3')
        cls.vol4 = cls.dataset.getFile('vol4')

    @classmethod
    def runImportVolumes(cls, pattern, samplingRate):
        """ Run an Import particles protocol. """
        cls.protImport = cls.newProtocol(ProtImportVolumes,
                                         filesPath=pattern,
                                         samplingRate=samplingRate)
        cls.launchProtocol(cls.protImport)
        return cls.protImport

    @classmethod
    def runImportMask(cls, pattern, samplingRate):
        """ Run an Import particles protocol. """
        cls.protImportMask = cls.newProtocol(ProtImportMask,
                                         maskPath=pattern,
                                             samplingRate=samplingRate)
        cls.launchProtocol(cls.protImportMask)
        return cls.protImportMask

    @classmethod
    def runImportParticles(cls, pattern, samplingRate, checkStack=False):
        """ Run an Import particles protocol. """
        cls.protImport = cls.newProtocol(ProtImportParticles,
                                         filesPath=pattern,
                                         samplingRate=samplingRate,
                                         checkStack=checkStack)
        cls.launchProtocol(cls.protImport)
        # check that input images have been imported (a better way to do this?)
        if cls.protImport.outputParticles is None:
            raise Exception('Import of images: %s, failed. outputParticles '
                            'is None.' % pattern)
        return cls.protImport

    @classmethod
    def runClassify(cls, particles):
        cls.ProtClassify = cls.newProtocol(XmippProtML2D,
                                           numberOfClasses=8, maxIters=4,
                                           doMlf=False,
                                           numberOfMpi=2, numberOfThreads=2)
        cls.ProtClassify.inputParticles.set(particles)
        cls.launchProtocol(cls.ProtClassify)
        return cls.ProtClassify


class TestXmippCreateMask3D(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData()
        cls.protImport = cls.runImportVolumes(cls.vol1, 9.896)

    def testCreateMask(self):
        print("Run create threshold mask from volume")
        protMask1 = self.newProtocol(XmippProtCreateMask3D,
                                     source=0, volumeOperation=0,
                                     threshold=0.4)
        protMask1.inputVolume.set(self.protImport.outputVolume)
        protMask1.setObjLabel('threshold mask')
        self.launchProtocol(protMask1)
        self.assertIsNotNone(protMask1.outputMask,
                             MSG_WRONG_MASK)


        print("Run create segment mask from volume")
        protMask2 = self.newProtocol(XmippProtCreateMask3D,
                                     source=0, volumeOperation=1,
                                     segmentationType=3)
        protMask2.inputVolume.set(self.protImport.outputVolume)
        protMask2.setObjLabel('segmentation automatic')
        self.launchProtocol(protMask2)
        self.assertIsNotNone(protMask2.outputMask,
                             MSG_WRONG_MASK)

    
        print("Run create mask from another mask")
        protMask3 = self.newProtocol(XmippProtCreateMask3D,
                                     source=0, volumeOperation=2,
                                     doMorphological=True, elementSize=3)
        protMask3.inputVolume.set(protMask1.outputMask)
        protMask3.setObjLabel('dilation mask')
        self.launchProtocol(protMask3)
        self.assertIsNotNone(protMask3.outputMask,
                             "There was a problem with mask from another mask")

        print("Run create mask from geometry")
        protMask4 = self.newProtocol(XmippProtCreateMask3D,
                                     source=1, size=64, samplingRate=9.89,
                                     geo=6, innerRadius=10, outerRadius=25,
                                     borderDecay=2)
        protMask4.setObjLabel('crown mask')
        self.launchProtocol(protMask4)
        self.assertIsNotNone(protMask4.outputMask,
                             "There was a problem with create mask from geometry")
         
        print("Run create mask from feature file")
        #create feature file
        featFileName = "/tmp/kk.feat" 
        f=open(featFileName,"w")
        f.write("""# XMIPP_STAR_1 *
# Type of feature (sph, blo, gau, Cyl, dcy, cub, ell, con)(Required)
# The operation after adding the feature to the phantom (+/=) (Required)
# The feature density (Required)
# The feature center (Required)
# The vector for special parameters of each vector (Required)
# Sphere: [radius] 
# Blob : [radius alpha m] Gaussian : [sigma]
# Cylinder : [xradius yradius height rot tilt psi]
# DCylinder : [radius height separation rot tilt psi]
# Cube : [xdim ydim zdim rot tilt psi]
# Ellipsoid : [xradius yradius zradius rot tilt psi]
# Cone : [radius height rot tilt psi]
data_block1
 _dimensions3D  '34 34 34' 
 _phantomBGDensity  0.
 _scale  1.
data_block2
loop_
 _featureType
 _featureOperation
 _featureDensity
 _featureCenter
 _featureSpecificVector
sph + 1 '3.03623188  0.02318841 -5.04130435' '7'
""")
        f.close()
        protMask5 = self.newProtocol(XmippProtCreateMask3D,
                                     featureFilePath='/tmp/kk.feat',
                                     source=2, samplingRate=9.89)
        protMask5.setObjLabel('feat mask')
        self.launchProtocol(protMask5)
        self.assertIsNotNone(protMask5.outputMask,
                             "There was a problem with create mask from feature")

class TestXmippApplyMask3D(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData()
        cls.protImport = cls.runImportVolumes(cls.vol1, 9.896)

    def testApplyCircularMask(self):
        print("Run apply circular mask for volumes")
        protMask1 = self.newProtocol(XmippProtMaskVolumes,
                                     source=0, geo=0, radius=-1,
                                     fillType=0, fillValue=5 )
        protMask1.inputVolumes.set(self.protImport.outputVolume)
        protMask1.setObjLabel('circular mask')
        self.launchProtocol(protMask1)
        self.assertAlmostEquals(protMask1.outputVol.getSamplingRate(), 
                                self.protImport.outputVolume.getSamplingRate(),
                                "There was a problem with the sampling rate value for the apply user custom mask for Volumes")
        self.assertIsNotNone(protMask1.outputVol,
                             "There was a problem with apply circular mask for Volumes")
    
    def testApplyBoxMask(self):
        print("Run apply box mask for Volumes")
        protMask2 = self.newProtocol(XmippProtMaskVolumes,
                                     source=0, geo=1, boxSize=-1,
                                     fillType=1 )
        protMask2.inputVolumes.set(self.protImport.outputVolume)
        protMask2.setObjLabel('box mask')
        self.launchProtocol(protMask2)
        self.assertAlmostEquals(protMask2.outputVol.getSamplingRate(), 
                                self.protImport.outputVolume.getSamplingRate(),
                                "There was a problem with the sampling rate value for the apply user custom mask for Volumes")
        self.assertIsNotNone(protMask2.outputVol,
                             "There was a problem with apply boxed mask for Volumes")
    
    def testapplyCrownMask(self):
        print("Run apply crown mask for Volumes")
        protMask3 = self.newProtocol(XmippProtMaskVolumes,
                                     source=0, geo=2, innerRadius=2, outerRadius=12,
                                     fillType=2 )
        protMask3.inputVolumes.set(self.protImport.outputVolume)
        protMask3.setObjLabel('crown mask')
        self.launchProtocol(protMask3)
        self.assertAlmostEquals(protMask3.outputVol.getSamplingRate(), 
                                self.protImport.outputVolume.getSamplingRate(),
                                "There was a problem with the sampling rate value for the apply user custom mask for Volumes")
        self.assertIsNotNone(protMask3.outputVol, "There was a problem with apply crown mask for Volumes")
        
    def testApplyGaussianMask(self):
        print("Run apply gaussian mask for Volumes")
        protMask4 = self.newProtocol(XmippProtMaskVolumes,
                                     source=0, geo=3, sigma=-1,
                                     fillType=3 )
        protMask4.inputVolumes.set(self.protImport.outputVolume)
        protMask4.setObjLabel('gaussian mask')
        self.launchProtocol(protMask4)
        self.assertAlmostEquals(protMask4.outputVol.getSamplingRate(), 
                                self.protImport.outputVolume.getSamplingRate(),
                                "There was a problem with the sampling rate value for the apply user custom mask for Volumes")
        self.assertIsNotNone(protMask4.outputVol, "There was a problem with apply gaussian mask for Volumes")
        
    def testApplyRaisedCosineMask(self):
        print("Run apply raised cosine mask for Volumes")
        protMask5 = self.newProtocol(XmippProtMaskVolumes,
                                     source=0, geo=4, innerRadius=2, outerRadius=12,
                                     fillType=0, fillValue=5 )
        protMask5.inputVolumes.set(self.protImport.outputVolume)
        protMask5.setObjLabel('raised cosine mask')
        self.launchProtocol(protMask5)
        self.assertAlmostEquals(protMask5.outputVol.getSamplingRate(), 
                                self.protImport.outputVolume.getSamplingRate(),
                                "There was a problem with the sampling rate value for the apply user custom mask for Volumes")
        self.assertIsNotNone(protMask5.outputVol, "There was a problem with apply raised cosine mask for Volumes")
        
    def testApplyRaisedCrownMask(self):
        print("Run apply raised crown mask for Volumes")
        protMask6 = self.newProtocol(XmippProtMaskVolumes,
                                     source=0, geo=5, innerRadius=2, outerRadius=12, borderDecay=2,
                                     fillType=1 )
        protMask6.inputVolumes.set(self.protImport.outputVolume)
        protMask6.setObjLabel('raised crown mask')
        self.launchProtocol(protMask6)
        self.assertAlmostEquals(protMask6.outputVol.getSamplingRate(), 
                                self.protImport.outputVolume.getSamplingRate(),
                                "There was a problem with the sampling rate value for the apply user custom mask for Volumes")
        
        self.assertIsNotNone(protMask6.outputVol, "There was a problem with apply raised crown mask for Volumes")
        
    def testApplyUserMask(self):
        print("Run apply user mask for Volumes")
        # Create MASK
        protMask01 = self.newProtocol(XmippProtCreateMask3D,
                                     source=1, size=64, samplingRate=9.89,
                                     geo=6, innerRadius=10, outerRadius=25, borderDecay=2)
        protMask01.setObjLabel('crown mask')
        self.launchProtocol(protMask01)
        self.assertIsNotNone(protMask01.outputMask,
                             "There was a problem with create mask from geometry")
        # Apply MASK
        protMask02 = self.newProtocol(XmippProtMaskVolumes,
                                     source=1,
                                     fillType=1 )
        protMask02.inputVolumes.set(self.protImport.outputVolume)
        protMask02.inputMask.set(protMask01.outputMask)
        protMask02.setObjLabel('user custom mask')
        self.launchProtocol(protMask02)
        self.assertAlmostEquals(protMask02.outputVol.getSamplingRate(), 
                                self.protImport.outputVolume.getSamplingRate(),
                                "There was a problem with the sampling rate value for the apply user custom mask for Volumes")
         
        self.assertIsNotNone(protMask02.outputVol,
                             "There was a problem with apply user custom mask for Volumes")
  

class TestXmippPreprocessVolumes(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData()
        cls.protImport1 = cls.runImportVolumes(cls.volumes, 9.896)
        cls.protImport2 = cls.runImportVolumes(cls.vol1, 9.896)
        cls.protImport2 = cls.runImportVolumes(cls.vol1, 9.896)
        #test symmetryze with mask
        dataProject='SymVirus'
        dataset = DataSet.getDataSet(dataProject)
        virusVol  = dataset.getFile('whole_vol_half')
        virusMaskCapsid = dataset.getFile('large_vol_half_th')
        virusMaskPenton = dataset.getFile('small_vol_half_th')
        cls.protImportVirus = cls.runImportVolumes(virusVol, 1)
        cls.protImportvirusMaskCapsid = cls.runImportMask(virusMaskCapsid, 1)
        cls.protImportvirusMaskPenton = cls.runImportMask(virusMaskPenton, 1)


    def testPreprocessVolumes(self):
        print("Run preprocess a volume")
        protPreprocessVol1 = XmippProtPreprocessVolumes(doChangeHand=True, doRandomize=True, doSymmetrize=True, symmetryGroup='d6',
                                                        doSegment=True, doNormalize=True, backRadius=20, doInvert=True,
                                                        doThreshold=True, thresholdType=1)
        protPreprocessVol1.inputVolumes.set(self.protImport2.outputVolume)
        self.proj.launchProtocol(protPreprocessVol1, wait=True)
        self.assertIsNotNone(protPreprocessVol1.outputVol, "There was a problem with a volume")

        print("Run preprocess a SetOfVolumes")
        protPreprocessVol2 = XmippProtPreprocessVolumes(doChangeHand=True, doRandomize=True, doSymmetrize=True, symmetryGroup='d6',
                                                        doSegment=True, doNormalize=True, backRadius=20, doInvert=True,
                                                        doThreshold=True, thresholdType=1)
        protPreprocessVol2.inputVolumes.set(self.protImport1.outputVolumes)
        self.proj.launchProtocol(protPreprocessVol2, wait=True)
        self.assertIsNotNone(protPreprocessVol2.outputVol, "There was a problem with preprocess a SetOfVolumes")

        print("Run preprocess a volume using mask_1 in the symmetrization")
        protPreprocessVol3 = XmippProtPreprocessVolumes(doChangeHand=False, doRandomize=False,
                                                        doRotateIco=True, rotateFromIco=0, rotateToIco=2,
                                                        doSymmetrize=True, symmetryGroup='i3',
                                                        doSegment=False, doNormalize=False,
                                                        doInvert=False, doThreshold=False,
                                                        doVolumeMask=True
                                                        )
        protPreprocessVol3.inputVolumes.set(self.protImportVirus.outputVolume)
        protPreprocessVol3.volumeMask.set(self.protImportvirusMaskCapsid.outputMask)
        self.proj.launchProtocol(protPreprocessVol3, wait=True)
        self.assertIsNotNone(protPreprocessVol3.outputVol, "There was a problem with a volume")

        print("Run preprocess a volume using mask_2 in the symmetrization")
        protPreprocessVol4 = XmippProtPreprocessVolumes(doChangeHand=False, doRandomize=False,
                                                        doSymmetrize=True, symmetryGroup='c7',
                                                        doSegment=False, doNormalize=False,
                                                        doInvert=False, doThreshold=False,
                                                        doVolumeMask=True
                                                        )
        protPreprocessVol4.inputVolumes.set(self.protImportVirus.outputVolume)
        protPreprocessVol4.volumeMask.set(self.protImportvirusMaskPenton.outputMask)
        self.proj.launchProtocol(protPreprocessVol4, wait=True)
        self.assertIsNotNone(protPreprocessVol4.outputVol, "There was a problem with a volume")


class TestXmippResolution3D(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData()
        cls.protImport1 = cls.runImportVolumes(cls.vol2, 9.896)
        cls.protImport2 = cls.runImportVolumes(cls.vol3, 9.896)

    def testCalculateResolution(self):
        print("Run resolution 3D")
        protResol3D = XmippProtResolution3D(doSSNR=False)
        protResol3D.inputVolume.set(self.protImport1.outputVolume)
        protResol3D.referenceVolume.set(self.protImport2.outputVolume)
        self.proj.launchProtocol(protResol3D, wait=True)
        self.assertIsNotNone(protResol3D._defineFscName(), "There was a problem with fsc")
        self.assertIsNotNone(protResol3D._defineStructFactorName(), "There was a problem with structure factor")


class TestXmippFilterVolumes(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        print("\n", greenStr(" Filter Volumes Set Up - Collect data ".center(75, '-')))
        setupTestProject(cls)
        TestXmippBase.setData()
        cls.protImport1 = cls.runImportVolumes(cls.volumes, 9.896)
        cls.protImport2 = cls.runImportVolumes(cls.vol1, 9.896)

    # Tests with single volume as input.
    def launchAndTestSingle(self, **kwargs):
        "Launch XmippProtFilterVolumes on single volume and check results."
        print(magentaStr("\n==> Filter singe volume input params: %s" % kwargs))
        prot = XmippProtFilterVolumes(**kwargs)
        prot.inputVolumes.set(self.protImport2.outputVolume)
        self.proj.launchProtocol(prot, wait=True)
        self.assertTrue(hasattr(prot, "outputVol") and prot.outputVol is not None,
                        "There was a problem with filter single volume")
        self.assertTrue(prot.outputVol.equalAttributes(
            self.protImport2.outputVolume, ignore=['_index', '_filename'],
            verbose=True))

    def testSingleFourier(self):
        self.launchAndTestSingle(filterSpace=FILTER_SPACE_FOURIER,
                                 lowFreq=0.1, highFreq=0.25)

    def testSingleMedian(self):
        self.launchAndTestSingle(filterSpace=FILTER_SPACE_REAL,
                                 filterModeReal=xfh.FM_MEDIAN)

    def testSingleWavelets(self):
        self.launchAndTestSingle(filterSpace=FILTER_SPACE_WAVELET,
                                 filterModeWavelets=xfh.FM_DAUB12,
                                 waveletMode=xfh.FM_REMOVE_SCALE)

    # Tests with multiple volumes as input.
    def launchAndTestSet(self, **kwargs):
        "Launch XmippProtFilterVolumes on set of volumes and check results."
        print(magentaStr("\n==> Filter multiple volumes input params: %s" % kwargs))
        prot = XmippProtFilterVolumes(**kwargs)
        vIn = self.protImport1.outputVolumes  # short notation
        prot.inputVolumes.set(vIn)
        self.proj.launchProtocol(prot, wait=True)
        self.assertTrue(hasattr(prot, "outputVol") and prot.outputVol is not None,
                        "There was a problem with filter multiple volumes")
        self.assertTrue(prot.outputVol.equalAttributes(
            self.protImport1.outputVolumes, ignore=['_mapperPath'],
            verbose=True))
        # Compare the individual volumes too.
        self.assertTrue(prot.outputVol.equalItemAttributes(
            self.protImport1.outputVolumes, ignore=['_index', '_filename'],
            verbose=True))

    def testSetFourier(self):
        self.launchAndTestSet(filterSpace=FILTER_SPACE_FOURIER,
                              lowFreq=0.1, highFreq=0.25)

    def testSetMedian(self):
        self.launchAndTestSet(filterSpace=FILTER_SPACE_REAL,
                              filterModeReal=xfh.FM_MEDIAN)

    def testSetWavelets(self):
        self.launchAndTestSet(filterSpace=FILTER_SPACE_WAVELET,
                              filterModeWavelets=xfh.FM_DAUB12,
                              waveletMode=xfh.FM_REMOVE_SCALE)


class TestXmippMaskVolumes(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData()
        cls.protImport1 = cls.runImportVolumes(cls.volumes, 9.896)
        cls.protImport2 = cls.runImportVolumes(cls.vol1, 9.896)

    def testMaskVolumes(self):
        print("Run mask single volume")
        protMaskVolume = self.newProtocol(XmippProtMaskVolumes,
                                          radius=23)
        protMaskVolume.inputVolumes.set(self.protImport2.outputVolume)
        self.launchProtocol(protMaskVolume)
        self.assertIsNotNone(protMaskVolume.outputVol, "There was a problem with applying mask to a volume")

        print("Run mask SetOfVolumes")
        protMaskVolumes = self.newProtocol(XmippProtMaskVolumes,
                                           geo=MASK3D_CROWN, innerRadius=18, outerRadius=23)
        protMaskVolumes.inputVolumes.set(self.protImport1.outputVolumes)
        self.proj.launchProtocol(protMaskVolumes, wait=True)
        self.assertIsNotNone(protMaskVolumes.outputVol, "There was a problem with applying mask to SetOfVolumes")


class TestXmippCropResizeVolumes(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        print("\n", greenStr(" Crop/Resize Volumes Set Up - Collect data ".center(75, '-')))
        setupTestProject(cls)
        TestXmippBase.setData()
        cls.protImport1 = cls.runImportVolumes(cls.volumes, 9.896)
        cls.protImport2 = cls.runImportVolumes(cls.vol1, 9.896)

    # Tests with single volume as input.
    def launchSingle(self, **kwargs):
        "Launch XmippProtCropResizeVolumes and return output volume."
        print(magentaStr("\n==> Crop/Resize single volume input params: %s" % kwargs))
        prot = XmippProtCropResizeVolumes(**kwargs)
        prot.inputVolumes.set(self.protImport2.outputVolume)
        self.proj.launchProtocol(prot, wait=True)
        self.assertTrue(hasattr(prot, "outputVol") and prot.outputVol is not None,
                        "There was a problem with applying resize/crop to a volume")
        return prot.outputVol

    def testSingleResizeDimensions(self):
        inV = self.protImport2.outputVolume  # short notation
        newSize = 128
        outV = self.launchSingle(doResize=True,
                                 resizeOption=xrh.RESIZE_DIMENSIONS,
                                 resizeDim=newSize, doWindow=True,
                                 windowOperation=xrh.WINDOW_OP_WINDOW,
                                 windowSize=newSize*2)

        self.assertEqual(newSize * 2, outV.getDim()[0])
        self.assertAlmostEqual(outV.getSamplingRate(),
                               inV.getSamplingRate() * (inV.getDim()[0] / float(newSize)))
        self.assertTrue(outV.equalAttributes(
            inV, ignore=['_index', '_filename', '_samplingRate', '_origin', '_matrix'], verbose=True))

    def testSingleFactorAndCrop(self):
        inV = self.protImport2.outputVolume  # short notation
        outV = self.launchSingle(doResize=True,
                                 resizeOption=xrh.RESIZE_FACTOR,
                                 resizeFactor=0.5,
                                 doWindow=True,
                                 windowOperation=xrh.WINDOW_OP_CROP)

        self.assertEqual(inV.getDim()[0] * 0.5, outV.getDim()[0])
        self.assertAlmostEqual(outV.getSamplingRate(), inV.getSamplingRate() * 2)
        self.assertTrue(outV.equalAttributes(
            inV, ignore=['_index', '_filename', '_samplingRate'], verbose=True))

    def testSinglePyramid(self):
        inV = self.protImport2.outputVolume  # short notation
        outV = self.launchSingle(doResize=True, resizeOption=xrh.RESIZE_PYRAMID,
                                 resizeLevel=1)

        # Since the images were expanded by 2**resizeLevel (=2) the new
        # pixel size (painfully called "sampling rate") should be 0.5x.
        self.assertEqual(inV.getDim()[0] * 2, outV.getDim()[0])
        self.assertAlmostEqual(outV.getSamplingRate(), inV.getSamplingRate() * 0.5)
        self.assertTrue(outV.equalAttributes(
            inV, ignore=['_index', '_filename', '_samplingRate'], verbose=True))

    # Tests with multiple volumes as input.
    def launchSet(self, **kwargs):
        "Launch XmippProtCropResizeVolumes and return output volumes."
        print(magentaStr("\n==> Crop/Resize single set of volumes input params: %s" % kwargs))
        prot = XmippProtCropResizeVolumes(**kwargs)
        prot.inputVolumes.set(self.protImport1.outputVolumes)
        self.proj.launchProtocol(prot, wait=True)
        self.assertTrue(hasattr(prot, "outputVol") and prot.outputVol is not None,
                        "There was a problem with applying resize/crop to a set of volumes")
        return prot.outputVol

    def testSetResizeDimensions(self):
        inV = self.protImport1.outputVolumes  # short notation
        newSize = 128
        outV = self.launchSet(doResize=True,
                              resizeOption=xrh.RESIZE_DIMENSIONS,
                              resizeDim=newSize, doWindow=True,
                              windowOperation=xrh.WINDOW_OP_WINDOW,
                              windowSize=newSize*2)

        self.assertEqual(newSize * 2, outV.getDim()[0])
        self.assertAlmostEqual(outV.getSamplingRate(),
                               inV.getSamplingRate() * (inV.getDim()[0] / float(newSize)))
        self.assertTrue(outV.equalAttributes(
            inV, ignore=['_mapperPath', '_samplingRate', '_firstDim'], verbose=True))
        # Compare the individual volumes too.
        self.assertTrue(outV.equalItemAttributes(
            inV, ignore=['_index', '_filename', '_samplingRate'], verbose=True))

    def testSetFactorAndCrop(self):
        inV = self.protImport1.outputVolumes  # short notation
        outV = self.launchSet(doResize=True,
                              resizeOption=xrh.RESIZE_FACTOR,
                              resizeFactor=0.5,
                              doWindow=True,
                              windowOperation=xrh.WINDOW_OP_CROP)

        self.assertEqual(inV.getDim()[0] * 0.5, outV.getDim()[0])
        self.assertAlmostEqual(outV.getSamplingRate(), inV.getSamplingRate() * 2)
        self.assertTrue(outV.equalAttributes(
            inV, ignore=['_mapperPath', '_samplingRate', '_firstDim'], verbose=True))
        # Compare the individual volumes too.
        self.assertTrue(outV.equalItemAttributes(
            inV, ignore=['_index', '_filename', '_samplingRate'], verbose=True))

    def testSetPyramid(self):
        inV = self.protImport1.outputVolumes  # short notation
        outV = self.launchSet(doResize=True, resizeOption=xrh.RESIZE_PYRAMID,
                              resizeLevel=1)

        # Since the images were expanded by 2**resizeLevel (=2) the new
        # pixel size (painfully called "sampling rate") should be 0.5x.
        self.assertEqual(inV.getDim()[0] * 2, outV.getDim()[0])
        self.assertAlmostEqual(outV.getSamplingRate(), inV.getSamplingRate() * 0.5)
        self.assertTrue(outV.equalAttributes(
            inV, ignore=['_mapperPath', '_samplingRate', '_firstDim'], verbose=True))
        # Compare the individual volumes too.
        self.assertTrue(outV.equalItemAttributes(
            inV, ignore=['_index', '_filename', '_samplingRate'], verbose=True))


class TestXmippOperateVolumes(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData()
        cls.protImport1 = cls.runImportVolumes(cls.vol1, 9.896)
        cls.protImport2 = cls.runImportVolumes(cls.vol2, 9.896)
        cls.protImport3 = cls.runImportVolumes(cls.volumes, 9.896)

    # Tests with single volume as input.
    def launchSingle(self, **kwargs):
        "Launch XmippProtImageOperateVolumes and return output volume."
        print(magentaStr("\n==> Operate single volume input params: %s" % kwargs))
        prot = XmippProtImageOperateVolumes()
        prot.operation.set(kwargs.get('operation', 1))
        prot.inputVolumes.set(self.protImport1.outputVolume)
        prot.setObjLabel(kwargs.get('objLabel', None))
        prot.isValue.set(kwargs.get('isValue', False))
        prot.inputVolumes2.set(kwargs.get('volumes2', None))
        prot.value.set(kwargs.get('value', None))
        prot.intValue.set(kwargs.get('intValue', None))
        
        self.proj.launchProtocol(prot, wait=True)
        self.assertTrue(hasattr(prot, "outputVol") and prot.outputVol is not None,
                        "There was a problem producing the output")
        return prot.outputVol

    def testMultiplyVolumes(self):
        vol2 = self.protImport2.outputVolume  # short notation
        prot1 = self.launchSingle(operation=OP_MULTIPLY,
                                  objLabel='Multiply two Volumes',
                                  volumes2=vol2)

    def testMultiplyValue(self):
        prot2 = self.launchSingle(operation=OP_MULTIPLY,
                                  isValue=True,
                                  objLabel='Multiply by a Value',
                                  value=2.5)
    
    def testDotProduct(self):
        vol2 = self.protImport2.outputVolume  # short notation
        prot3 = self.launchSingle(operation=OP_DOTPRODUCT,
                                  objLabel='Dot Product',
                                  volumes2=vol2)

    def testSqrt(self):
        prot4 = self.launchSingle(operation=OP_SQRT,
                                  objLabel='Sqrt')

    def testRadial(self):
        prot5 = self.launchSingle(operation=OP_RADIAL,
                                  objLabel='Radial Average')

    def testColumn(self):
        prot6 = self.launchSingle(operation=OP_COLUNM,
                                  objLabel='Column',
                                  intValue  =7)

    def testRow(self):
        prot6 = self.launchSingle(operation=OP_ROW,
                                  objLabel='Row',
                                  intValue  =8)

#     # Tests with multiple volumes as input.
    def launchSet(self, **kwargs):
        "Launch XmippProtImageOperateVolumes and return output volumes."
        print(magentaStr("\n==> Operate set of volumes input params: %s" % kwargs))
        prot = XmippProtImageOperateVolumes()
        prot.operation.set(kwargs.get('operation', 1))
        prot.inputVolumes.set(self.protImport3.outputVolumes)
        prot.setObjLabel(kwargs.get('objLabel', None))
        prot.isValue.set(kwargs.get('isValue', False))
        prot.inputVolumes2.set(kwargs.get('volumes2', None))
        prot.value.set(kwargs.get('value', None))
        prot.intValue.set(kwargs.get('intValue', None))
        
        self.proj.launchProtocol(prot, wait=True)
        self.assertTrue(hasattr(prot, "outputVol") and prot.outputVol is not None,
                        "There was a problem producing the output")
        return prot.outputVol
        
    def testMultiplyVolSets(self):
        vol2 = self.protImport3.outputVolumes  # short notation
        prot6 = self.launchSet(operation=OP_MULTIPLY,
                                  objLabel='Multiply two SetOfVolumes',
                                  volumes2=vol2)

    def testMultiplyValue2(self):
        prot7 = self.launchSet(operation=OP_MULTIPLY,
                               isValue=True,
                               objLabel='Multiply by a Value 2',
                               value=2.5)
    
    def testDotProduct2(self):
        vol2 = self.protImport3.outputVolumes  # short notation
        prot8 = self.launchSet(operation=OP_DOTPRODUCT,
                               objLabel='Dot Product 2',
                               volumes2=vol2)

    def testSqrt2(self):
        prot9 = self.launchSet(operation=OP_SQRT,
                               objLabel='Sqrt 2')


class TestXmippProtAlignVolume(TestXmippBase):
    @classmethod
    def setData(cls, dataProject=db_xmipp_tutorial):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.volumes = cls.dataset.getFile('volumes')
        cls.vol1 = cls.dataset.getFile('vol1')
        cls.vol2 = cls.dataset.getFile('vol2')
        cls.vol3 = cls.dataset.getFile('vol3')

    @classmethod
    def runImportVolumes(cls, pattern, samplingRate):
        """ Run an Import particles protocol. """
        return cls.protImport
    
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('relion_tutorial')

        cls.protImport1 = cls.newProtocol(ProtImportVolumes,
                                         filesPath=cls.ds.getFile('volumes/reference_rotated.vol'), 
                                         samplingRate=1.0)
        cls.launchProtocol(cls.protImport1)
        
        cls.protImport2 = cls.newProtocol(ProtImportVolumes,
                                         filesPath=cls.ds.getFile('volumes/reference.mrc'), 
                                         samplingRate=1.0)
        cls.launchProtocol(cls.protImport2)
        
        # Rotate that volume rot=90 tilt=90 to create 
        # a gold rotated volume
        os.system('')

    def testExhaustive(self):
        protAlign = self.newProtocol(XmippProtAlignVolume,
                                     inputReference=self.protImport1.outputVolume,
                                     alignmentAlgorithm=ALIGN_ALGORITHM_EXHAUSTIVE,
                                     minRotationalAngle=65, 
                                     maxRotationalAngle=100,
                                     stepRotationalAngle=10,
                                     minTiltAngle=65,
                                     maxTiltAngle=100,
                                     stepTiltAngle=10,
                                     minInplaneAngle=0,
                                     maxInplaneAngle=0,
                                     stepInplaneAngle=0,
                                     numberOfMpi=1, numberOfThreads=1                               
                                     )
        protAlign.inputVolumes.append(self.protImport2.outputVolume)
        self.launchProtocol(protAlign)
        
    def testLocal(self):
        protAlign = self.newProtocol(XmippProtAlignVolume,
                                     inputReference=self.protImport1.outputVolume,
                                     alignmentAlgorithm=ALIGN_ALGORITHM_LOCAL,
                                     initialRotAngle=0,
                                     initialTiltAngle=0,
                                     initialInplaneAngle=0,
                                     initialShiftX=0,
                                     initialShiftY=0,
                                     initialShiftZ=0,
                                     initialScale=1,
                                     optimizeScale=True,
                                     numberOfMpi=1, numberOfThreads=1                               
                                     )
        protAlign.inputVolumes.append(self.protImport2.outputVolume)
        self.launchProtocol(protAlign)
        
    def testExhaustiveLocal(self):
        protAlign = self.newProtocol(XmippProtAlignVolume,
                                     inputReference=self.protImport1.outputVolume,
                                     alignmentAlgorithm=ALIGN_ALGORITHM_EXHAUSTIVE_LOCAL,
                                     minRotationalAngle=65,
                                     maxRotationalAngle=100,
                                     stepRotationalAngle=10,
                                     minTiltAngle=65,
                                     maxTiltAngle=100,
                                     stepTiltAngle=10,
                                     minInplaneAngle=0,
                                     maxInplaneAngle=0,
                                     stepInplaneAngle=0,
                                     numberOfMpi=1, numberOfThreads=1                               
                                     )
        protAlign.inputVolumes.append(self.protImport2.outputVolume)
        self.launchProtocol(protAlign)

    def testExhaustiveCircularMask(self):
        protAlign = self.newProtocol(XmippProtAlignVolume,
                                     inputReference=self.protImport1.outputVolume,
                                     alignmentAlgorithm=ALIGN_ALGORITHM_EXHAUSTIVE,
                                     minRotationalAngle=65,
                                     maxRotationalAngle=100,
                                     stepRotationalAngle=10,
                                     minTiltAngle=65,
                                     maxTiltAngle=100,
                                     stepTiltAngle=10,
                                     minInplaneAngle=0,
                                     maxInplaneAngle=0,
                                     stepInplaneAngle=0,
                                     applyMask=True,
                                     maskType=0,
                                     maskRadius=22,
                                     numberOfMpi=1, numberOfThreads=1
                                     )
        protAlign.inputVolumes.append(self.protImport2.outputVolume)
        self.launchProtocol(protAlign)

    def testExhaustiveBinaryMask(self):
        protMask = self.newProtocol(XmippProtCreateMask3D,
                                    source=0,
                                    volumeOperation=0,
                                    threshold=0.0)
        protMask.inputVolume.set(self.protImport1.outputVolume)
        self.launchProtocol(protMask)
        self.assertIsNotNone(protMask.outputMask,
                             MSG_WRONG_MASK)
        protAlign = self.newProtocol(XmippProtAlignVolume,
                                     inputReference=self.protImport1.outputVolume,
                                     alignmentAlgorithm=ALIGN_ALGORITHM_EXHAUSTIVE,
                                     minRotationalAngle=65,
                                     maxRotationalAngle=100,
                                     stepRotationalAngle=10,
                                     minTiltAngle=65,
                                     maxTiltAngle=100,
                                     stepTiltAngle=10,
                                     minInplaneAngle=0,
                                     maxInplaneAngle=0,
                                     stepInplaneAngle=0,
                                     applyMask=True,
                                     maskType=1,
                                     numberOfMpi=1, numberOfThreads=1
                                     )
        protAlign.inputVolumes.append(self.protImport2.outputVolume)
        protAlign.maskFile.set(protMask.outputMask)
        self.launchProtocol(protAlign)


class TestXmippProtHelicalParameters(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('general')
        cls.vol = cls.ds.getFile('vol_helix')
        cls.protImport = cls.runImportVolumes(cls.vol, 1.0)

    def testHelicalParameters(self):
        print("Run symmetrize helical")
        protHelical = XmippProtHelicalParameters(cylinderOuterRadius=20,dihedral=True,rot0=50,rotF=70,rotStep=5,z0=5,zF=10,zStep=0.5)
        protHelical.inputVolume.set(self.protImport.outputVolume)
        self.proj.launchProtocol(protHelical, wait=True)

        self.assertIsNotNone(protHelical.outputVolume, "There was a problem with Helical output volume")
        self.assertIsNotNone(protHelical.deltaRot.get(), "Output delta rot is None")
        self.assertIsNotNone(protHelical.deltaZ.get(), "Output delta Z is None")
        print("protHelical.deltaRot.get() ", protHelical.deltaRot.get())
        self.assertAlmostEqual(protHelical.deltaRot.get(), 59.59, delta=1, msg="Output delta rot is wrong")
        self.assertAlmostEqual(protHelical.deltaZ.get(), 6.628, delta=0.2, msg="Output delta Z is wrong")


class TestXmippRansacMda(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('mda')
        cls.averages = cls.dataset.getFile('averages')
        cls.samplingRate = 3.5
        cls.symmetryGroup = 'd6'
        cls.angularSampling = 15
        cls.nRansac = 25
        cls.numSamples = 5
        cls.dimRed = False
        cls.numVolumes = 2
        cls.maxFreq = 30

    def test_ransac(self):
        #Import a set of averages
        print("Import Set of averages")
        protImportAvg = self.newProtocol(ProtImportAverages, 
                                         filesPath=self.averages, 
                                         checkStack=True,
                                         samplingRate=self.samplingRate)
        self.launchProtocol(protImportAvg)
        self.assertIsNotNone(protImportAvg.getFiles(), "There was a problem with the import")
        
        print("Run Ransac")
        protRansac = self.newProtocol(XmippProtRansac,
                                      symmetryGroup=self.symmetryGroup, 
                                      angularSampling=self.angularSampling,
                                      nRansac=self.nRansac, 
                                      numSamples=self.numSamples, 
                                      dimRed=self.dimRed,
                                      numVolumes=self.numVolumes, 
                                      maxFreq=self.maxFreq, useAll=True, 
                                      numberOfThreads=4)
        protRansac.inputSet.set(protImportAvg.outputAverages)
        self.launchProtocol(protRansac)
        self.assertIsNotNone(protRansac.outputVolumes, "There was a problem with ransac protocol")


class TestXmippRansacGroel(TestXmippRansacMda):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('groel')
        cls.averages = cls.dataset.getFile('averages')
        cls.samplingRate = 2.1
        cls.symmetryGroup = 'd7'
        cls.angularSampling = 7
        cls.nRansac = 25
        cls.numSamples = 5
        cls.dimRed = True
        cls.numVolumes = 2
        cls.maxFreq = 12


class TestXmippSwarmMda(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('mda')
        cls.averages = cls.dataset.getFile('averages')
        cls.particles = cls.dataset.getFile('particles')
        cls.samplingRate = 3.5
        cls.symmetryGroup = 'd6'
        cls.angularSampling = 15
        cls.nRansac = 25
        cls.numSamples = 5
        cls.dimRed = False
        cls.numVolumes = 2
        cls.maxFreq = 30

    def test_swarm(self):
        #Import a set of averages
        print("Import Set of averages")
        protImportAvg = self.newProtocol(ProtImportAverages, 
                                         filesPath=self.averages, 
                                         checkStack=True,
                                         samplingRate=self.samplingRate)
        self.launchProtocol(protImportAvg)
        self.assertIsNotNone(protImportAvg.getFiles(), "There was a problem with the import")
        
        #Import a set of particles
        print("Import Set of particles")
        protImportParticles = self.newProtocol(ProtImportParticles, 
                                         filesPath=self.particles, 
                                         checkStack=True,
                                         samplingRate=self.samplingRate)
        self.launchProtocol(protImportParticles)
        self.assertIsNotNone(protImportParticles.getFiles(), "There was a problem with the import")

        print("Run Ransac")
        protRansac = self.newProtocol(XmippProtRansac,
                                      symmetryGroup=self.symmetryGroup, 
                                      angularSampling=self.angularSampling,
                                      nRansac=self.nRansac, 
                                      numSamples=self.numSamples, 
                                      dimRed=self.dimRed,
                                      numVolumes=self.numVolumes, 
                                      maxFreq=self.maxFreq, useAll=True, 
                                      numberOfThreads=4)
        protRansac.inputSet.set(protImportAvg.outputAverages)
        self.launchProtocol(protRansac)
        self.assertIsNotNone(protRansac.outputVolumes, "There was a problem with ransac protocol")

        print("Run Swarm")
        protSwarm = self.newProtocol(XmippProtReconstructSwarm,
                                      symmetryGroup=self.symmetryGroup,
                                      numberOfIterations=5,
                                      targetResolution=15,
                                      NimgTrain=20,
                                      NimgTest=10,
                                      numberOfMpi=4)
        protSwarm.inputParticles.set(protImportParticles.outputParticles)
        protSwarm.inputVolumes.set(protRansac.outputVolumes)
        self.launchProtocol(protSwarm)
        self.assertIsNotNone(protSwarm.outputVolumes, "There was a problem with ransac protocol")

class TestXmippRotationalSymmetry(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet(db_xmipp_tutorial)
        cls.vol = cls.dataset.getFile('vol110')

    def test_rotsym(self):
        print("Import Volume")
        protImportVol = self.newProtocol(ProtImportVolumes,
                                         objLabel='Volume',
                                         filesPath=self.vol,
                                         samplingRate=7.08)
        self.launchProtocol(protImportVol)
        self.assertIsNotNone(protImportVol.getFiles(),
                             "There was a problem with the import")
        
        print("Run find rotational symmetry axis")
        protRotSym = self.newProtocol(XmippProtRotationalSymmetry,
                                         symOrder=2,
                                         searchMode=2,
                                         tilt0=70,
                                         tiltF=110)
        protRotSym.inputVolume.set(protImportVol.outputVolume)
        self.launchProtocol(protRotSym)
        self.assertIsNotNone(protRotSym.outputVolume,
                             "There was a problem with Rotational Symmetry")


class TestXmippProjMatching(TestXmippBase):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('relion_tutorial')
        cls.vol = cls.dataset.getFile('volume')

    def testXmippProjMatching(self):
        print("Import Particles")
        protImportParts = self.newProtocol(ProtImportParticles,
                                 objLabel='Particles from scipion',
                                 importFrom=ProtImportParticles.IMPORT_FROM_SCIPION,
                                 sqliteFile=self.dataset.getFile('import/case2/particles.sqlite'),
                                 magnification=50000,
                                 samplingRate=7.08,
                                 haveDataBeenPhaseFlipped=True
                                 )
        self.launchProtocol(protImportParts)
        self.assertIsNotNone(protImportParts.getFiles(), "There was a problem with the import")
        
        print("Get a Subset of particles")
        protSubset = self.newProtocol(ProtSubSet,
                                         objLabel='100 Particles',
                                         chooseAtRandom=True,
                                         nElements=100)
        protSubset.inputFullSet.set(protImportParts.outputParticles)
        self.launchProtocol(protSubset)
        
        print("Import Volume")
        protImportVol = self.newProtocol(ProtImportVolumes,
                                         objLabel='Volume',
                                         filesPath=self.vol,
                                         samplingRate=7.08)
        self.launchProtocol(protImportVol)
        self.assertIsNotNone(protImportVol.getFiles(), "There was a problem with the import")
        
        print("Run Projection Matching")
        protProjMatch = self.newProtocol(XmippProtProjMatch,
                                         ctfGroupMaxDiff=0.00001,
                                         mpiJobSize=10,
                                         numberOfIterations=2,
                                         numberOfThreads=2,
                                         numberOfMpi=3)
        protProjMatch.inputParticles.set(protSubset.outputParticles)
        protProjMatch.input3DReferences.set(protImportVol.outputVolume)
        self.launchProtocol(protProjMatch)
        self.assertIsNotNone(protProjMatch.outputVolume, "There was a problem with Projection Matching")


class TestPdbImport(TestXmippBase):
    
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('nma')
        cls.pdb = cls.dataset.getFile('pdb')
    
    def testImportPdbFromId(self):
        print("Run convert a pdb from database")
        protConvert = self.newProtocol(ProtImportPdb, pdbId="3j3i")
        self.launchProtocol(protConvert)
        self.assertIsNotNone(protConvert.outputPdb.getFileName(), 
                             "There was a problem with the import")
        
    def testImportPdbFromFn(self):
        print("Run convert a pdb from file")
        protConvert = self.newProtocol(ProtImportPdb, 
                                       inputPdbData=ProtImportPdb.IMPORT_FROM_FILES, 
                                       pdbFile=self.pdb)
        self.launchProtocol(protConvert)
        self.assertIsNotNone(protConvert.outputPdb.getFileName(), 
                             "There was a problem with the import")


class TestXmippPdbConvert(TestXmippBase):
    
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('nma')
        cls.pdb = cls.dataset.getFile('pdb')
    
    def testXmippPdbConvertFromDb(self):
        print("Run convert a pdb from database")
        protConvert = self.newProtocol(XmippProtConvertPdb, pdbId="3j3i", sampling=4, setSize=True,
                                       size_z=100, size_y=100, size_x=100)
        self.launchProtocol(protConvert)
        self.assertIsNotNone(protConvert.outputVolume.getFileName(), "There was a problem with the conversion")
        self.assertAlmostEqual(protConvert.outputVolume.getSamplingRate(), protConvert.sampling.get(), places=1,
                               msg=(MSG_WRONG_SAMPLING, "volume"))
        self.assertAlmostEqual(protConvert.outputVolume.getDim()[0], protConvert.size_z.get(), places=1,
                               msg=(MSG_WRONG_SIZE, "volume"))
        
    def testXmippPdbConvertFromObj(self):
        print("Run convert a pdb from import")
        protImport = self.newProtocol(ProtImportPdb, 
                                      inputPdbData=ProtImportPdb.IMPORT_FROM_FILES, 
                                      pdbFile=self.pdb)
        self.launchProtocol(protImport)
        self.assertIsNotNone(protImport.outputPdb.getFileName(), "There was a problem with the import")
        
        protConvert = self.newProtocol(XmippProtConvertPdb, 
                                       inputPdbData=XmippProtConvertPdb.IMPORT_OBJ, 
                                       sampling=3, setSize=True, size_z=20, size_y=20, size_x=20)
        protConvert.pdbObj.set(protImport.outputPdb)
        self.launchProtocol(protConvert)
        self.assertIsNotNone(protConvert.outputVolume.getFileName(), "There was a problem with the conversion")
        self.assertAlmostEqual(protConvert.outputVolume.getSamplingRate(), protConvert.sampling.get(), places=1,
                               msg=(MSG_WRONG_SAMPLING, "volume"))
        self.assertAlmostEqual(protConvert.outputVolume.getDim()[0], protConvert.size_z.get(), places=1,
                               msg=(MSG_WRONG_SIZE, "volume"))

    def testXmippPdbConvertFromFn(self):
        print("Run convert a pdb from file")
        protConvert = self.newProtocol(XmippProtConvertPdb,inputPdbData=2, pdbFile=self.pdb, sampling=2, setSize=False)
        self.launchProtocol(protConvert)
        self.assertIsNotNone(protConvert.outputVolume.getFileName(), "There was a problem with the conversion")
        self.assertAlmostEqual(protConvert.outputVolume.getSamplingRate(), protConvert.sampling.get(), places=1,
                               msg=(MSG_WRONG_SAMPLING, "volume"))
        self.assertAlmostEqual(protConvert.outputVolume.getDim()[0], 48, places=1, msg=(MSG_WRONG_SIZE, "volume"))


class TestXmippValidateNonTilt(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('relion_tutorial')
        cls.vol = cls.dataset.getFile('volume')

    def testXmippValidateNonTilt(self):
        print("Import Particles")
        protImportParts = self.newProtocol(ProtImportParticles,
                                 objLabel='Particles from scipion',
                                 importFrom=ProtImportParticles.IMPORT_FROM_SCIPION,
                                 sqliteFile=self.dataset.getFile('import/case2/particles.sqlite'),
                                 magnification=50000,
                                 samplingRate=7.08,
                                 haveDataBeenPhaseFlipped=True
                                 )
        self.launchProtocol(protImportParts)
        self.assertIsNotNone(protImportParts.getFiles(), "There was a problem with the import")
        
        print("Get a Subset of particles")
        protSubset = self.newProtocol(ProtSubSet,
                                         objLabel='100 Particles',
                                         chooseAtRandom=True,
                                         nElements=100)
        protSubset.inputFullSet.set(protImportParts.outputParticles)
        self.launchProtocol(protSubset)
        
        print("Import Volume")
        protImportVol = self.newProtocol(ProtImportVolumes,
                                         objLabel='Volume',
                                         filesPath=self.vol,
                                         samplingRate=7.08)
        self.launchProtocol(protImportVol)
        self.assertIsNotNone(protImportVol.getFiles(), "There was a problem with the import")
        
        print("Run Validate Non-Tilt significant GPU")
        protValidate = self.newProtocol(XmippProtValidateNonTilt)
        protValidate.inputParticles.set(protSubset.outputParticles)
        protValidate.inputVolumes.set(protImportVol.outputVolume)
        protValidate.setObjLabel('Validate Non-Tilt significant GPU')
        self.launchProtocol(protValidate)
        self.assertIsNotNone(protValidate.outputVolumes, "There was a problem with Validate Non-Tilt GPU")

        print("Run Validate Non-Tilt significant CPU")
        protValidate = self.newProtocol(XmippProtValidateNonTilt)
        protValidate.inputParticles.set(protSubset.outputParticles)
        protValidate.inputVolumes.set(protImportVol.outputVolume)
        protValidate.useGpu.set(False)
        protValidate.setObjLabel('Validate Non-Tilt significant CPU')
        self.launchProtocol(protValidate)
        self.assertIsNotNone(protValidate.outputVolumes, "There was a problem with Validate Non-Tilt CPU")
        
        print("Run Validate Non-Tilt projection matching")
        protValidate = self.newProtocol(XmippProtValidateNonTilt, alignmentMethod=1)
        protValidate.inputParticles.set(protSubset.outputParticles)
        protValidate.inputVolumes.set(protImportVol.outputVolume)
        protValidate.setObjLabel('Validate Non-Tilt projection matching')
        self.launchProtocol(protValidate)
        self.assertIsNotNone(protValidate.outputVolumes, "There was a problem with Validate Non-Tilt")


class TestXmippVolSubtraction(TestXmippBase):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet(db_xmipp_tutorial)
        cls.vol1 = cls.dataset.getFile(vol1_iter2)
        cls.vol2 = cls.dataset.getFile(vol2_iter2)

    def testXmippVolSub(self):
        print("Import Volume 1")
        protImportVol1 = self.newProtocol(ProtImportVolumes,
                                         objLabel='Volume',
                                         filesPath=self.vol1,
                                         samplingRate=7.08)
        self.launchProtocol(protImportVol1)
        self.assertIsNotNone(protImportVol1.getFiles(),
                             "There was a problem with the import 1")

        print("Import Volume 2")
        protImportVol2 = self.newProtocol(ProtImportVolumes,
                                          objLabel='Volume',
                                          filesPath=self.vol2,
                                          samplingRate=7.08)
        self.launchProtocol(protImportVol2)
        self.assertIsNotNone(protImportVol2.getFiles(),
                             "There was a problem with the import 2")

        print("Import atomic structure")
        protImportPdb = self.newProtocol(ProtImportPdb,
                                          pdbId='6Z9F',
                                          inputVolume=protImportVol1.outputVolume)
        self.launchProtocol(protImportPdb)
        self.assertIsNotNone(protImportPdb.getFiles(),
                             "There was a problem with the atomic structure")

        print("Create mask")
        protCreateMask = self.newProtocol(XmippProtCreateMask3D,
                                          inputVolume=protImportVol1.outputVolume,
                                          threshold=0.28)
        self.launchProtocol(protCreateMask)
        self.assertIsNotNone(protCreateMask.getFiles(),
                             "There was a problem with the 3D mask")

        print("Run volume adjust")
        protVolAdj = self.newProtocol(XmippProtVolAdjust,
                                      vol1=protImportVol1.outputVolume,
                                      vol2=protImportVol2.outputVolume,
                                      masks=False)
        self.launchProtocol(protVolAdj)
        self.assertIsNotNone(protVolAdj.outputVolume, "There was a problem with Volumes adjust")
        self.assertEqual(protVolAdj.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protVolAdj.outputVolume.getDim(), (64, 64, 64), (MSG_WRONG_DIM, "volume"))
        protVolAdjNoE = self.newProtocol(XmippProtVolAdjust,
                                         vol1=protImportVol1.outputVolume,
                                         vol2=protImportVol2.outputVolume,
                                         computeE=False,
                                         masks=False)
        self.launchProtocol(protVolAdjNoE)
        self.assertIsNotNone(protVolAdjNoE.outputVolume,
                             "There was a problem with Volumes adjust without computing energy")
        self.assertEqual(protVolAdj.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protVolAdj.outputVolume.getDim(), (64, 64, 64), (MSG_WRONG_DIM, "volume"))

        protVolAdjNoRadAvg = self.newProtocol(XmippProtVolAdjust,
                                              vol1=protImportVol1.outputVolume,
                                              vol2=protImportVol2.outputVolume,
                                              masks=False,
                                              radavg=False)
        self.launchProtocol(protVolAdjNoRadAvg)
        self.assertIsNotNone(protVolAdjNoRadAvg.outputVolume,
                             "There was a problem with Volumes adjust without radial average")
        self.assertEqual(protVolAdj.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protVolAdj.outputVolume.getDim(), (64, 64, 64), (MSG_WRONG_DIM, "volume"))

        print("Run volume subtraction")
        protVolSub = self.newProtocol(XmippProtVolSubtraction,
                                      vol1=protImportVol1.outputVolume,
                                      vol2=protImportVol2.outputVolume,
                                      masks=False,
                                      radavg=False)
        self.launchProtocol(protVolSub)
        self.assertIsNotNone(protVolSub.outputVolume,
                             "There was a problem with Volumes subtraction")
        self.assertEqual(protVolAdj.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protVolAdj.outputVolume.getDim(), (64, 64, 64), (MSG_WRONG_DIM, "volume"))

        protVolSubNoE = self.newProtocol(XmippProtVolSubtraction,
                                         vol1=protImportVol1.outputVolume,
                                         vol2=protImportVol2.outputVolume,
                                         computeE=False,
                                         masks=False,
                                         radavg=False)
        self.launchProtocol(protVolSubNoE)
        self.assertIsNotNone(protVolSubNoE.outputVolume,
                             "There was a problem with Volumes subtraction without computing energy")
        self.assertEqual(protVolAdj.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protVolAdj.outputVolume.getDim(), (64, 64, 64), (MSG_WRONG_DIM, "volume"))

        protVolSubMask = self.newProtocol(XmippProtVolSubtraction,
                                      vol1=protImportVol1.outputVolume,
                                      vol2=protImportVol2.outputVolume,
                                      radavg=False,
                                      mask1=protCreateMask.outputMask,
                                      mask2=protCreateMask.outputMask)
        self.launchProtocol(protVolSubMask)
        self.assertIsNotNone(protVolSubMask.outputVolume,
                             "There was a problem with Volumes subtraction with masks")
        self.assertEqual(protVolAdj.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protVolAdj.outputVolume.getDim(), (64, 64, 64), (MSG_WRONG_DIM, "volume"))

        protVolSubRadAvg = self.newProtocol(XmippProtVolSubtraction,
                                      vol1=protImportVol1.outputVolume,
                                      vol2=protImportVol2.outputVolume,
                                      masks=False)
        self.launchProtocol(protVolSubRadAvg)
        self.assertIsNotNone(protVolSubRadAvg.outputVolume,
                             "There was a problem with Volumes subtraction radial average")
        self.assertEqual(protVolAdj.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protVolAdj.outputVolume.getDim(), (64, 64, 64), (MSG_WRONG_DIM, "volume"))

        protVolSubPdb = self.newProtocol(XmippProtVolSubtraction,
                                      vol1=protImportVol1.outputVolume,
                                      pdb=True,
                                      pdbObj=protImportPdb.outputPdb,
                                      masks=False)
        self.launchProtocol(protVolSubPdb)
        self.assertIsNotNone(protVolSubPdb.outputVolume, "There was a problem with Volumes subtraction pdb")
        self.assertEqual(protVolAdj.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protVolAdj.outputVolume.getDim(), (64, 64, 64), (MSG_WRONG_DIM, "volume"))

        print("Run volume consensus")
        protVolConsensus = self.newProtocol(XmippProtVolConsensus,
                                            vols=[protImportVol1.outputVolume, protImportVol2.outputVolume,
                                                  protVolAdj.outputVolume])
        self.launchProtocol(protVolConsensus)
        self.assertIsNotNone(protVolConsensus.outputVolume, "There was a problem with Volumes consensus")
        self.assertEqual(protVolAdj.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protVolAdj.outputVolume.getDim(), (64, 64, 64), (MSG_WRONG_DIM, "volume"))


class TestXmippVolPhantom(TestXmippBase):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testXmippPhantomVol(self):
        protCreatePhantom = self.newProtocol(XmippProtPhantom)
        self.launchProtocol(protCreatePhantom)
        self.assertIsNotNone(protCreatePhantom.getFiles(),  "There was a problem with phantom creation")
        self.assertEqual(protCreatePhantom.outputVolume.getSamplingRate(), 4,
                         "There was a problem with the sampling rate value of output phantom")
        self.assertEqual(protCreatePhantom.outputVolume.getDim(), (40, 40, 40),
                         "There was a problem with the dimensions of output phantom")


class TestXmippShiftParticlesAndVolume(TestXmippBase):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet(db_xmipp_tutorial)
        cls.vol1 = cls.dataset.getFile(vol1_iter2)

    def testXmippShiftParticlesAndVolume(self):
        protImportVol = self.newProtocol(ProtImportVolumes,
                                          objLabel='Volume',
                                          filesPath=self.vol1,
                                          samplingRate=7.08)
        self.launchProtocol(protImportVol)
        self.assertIsNotNone(protImportVol.getFiles(),
                             "There was a problem with the volume import")

        protCreateGallery = self.newProtocol(XmippProtCreateGallery,
                                             inputVolume=protImportVol.outputVolume,
                                             rotStep=15.0,
                                             tiltStep=90.0)
        self.launchProtocol(protCreateGallery)
        self.assertIsNotNone(protCreateGallery.getFiles(),
                             MSG_WRONG_GALLERY)

        protShiftParticles = self.newProtocol(XmippProtShiftParticles,
                                              inputParticles=protCreateGallery.outputReprojections,
                                              xin=2.0, yin=3.0, zin=4.0)
        self.launchProtocol(protShiftParticles)
        self.assertIsNotNone(protShiftParticles.getFiles(),
                             "There was a problem with shift particles")
        self.assertEqual(protShiftParticles.outputParticles.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(protShiftParticles.outputParticles.getFirstItem().getDim(), (64, 64, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(protShiftParticles.outputParticles.getSize(), 181, (MSG_WRONG_SIZE, "particles"))
        self.assertEqual(protShiftParticles.shiftX.get(), 2.0, (MSG_WRONG_SHIFT, "x"))
        self.assertEqual(protShiftParticles.shiftY.get(), 3.0, (MSG_WRONG_SHIFT, "y"))
        self.assertEqual(protShiftParticles.shiftZ.get(), 4.0, (MSG_WRONG_SHIFT, "z"))

        protCreateMask = self.newProtocol(XmippProtCreateMask3D,
                                          inputVolume=protImportVol.outputVolume,
                                          threshold=0.1)
        self.launchProtocol(protCreateMask)
        self.assertIsNotNone(protCreateMask.getFiles(),
                             "There was a problem with the 3D mask ")

        protShiftParticlesCenterOfMass = self.newProtocol(XmippProtShiftParticles,
                                                          inputParticles=protCreateGallery.outputReprojections,
                                                          inputMask=protCreateMask.outputMask,
                                                          option=False)
        self.launchProtocol(protShiftParticlesCenterOfMass)
        self.assertIsNotNone(protShiftParticlesCenterOfMass.getFiles(),
                             "There was a problem with shift particles to center of mass")
        self.assertEqual(protShiftParticlesCenterOfMass.outputParticles.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(protShiftParticlesCenterOfMass.outputParticles.getFirstItem().getDim(), (64, 64, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(protShiftParticlesCenterOfMass.outputParticles.getSize(), 181, (MSG_WRONG_SIZE, "particles"))
        self.assertEqual(protShiftParticlesCenterOfMass.shiftX.get(), 32.0, (MSG_WRONG_SHIFT, "x"))
        self.assertEqual(protShiftParticlesCenterOfMass.shiftY.get(), 32.0, (MSG_WRONG_SHIFT, "y"))
        self.assertEqual(protShiftParticlesCenterOfMass.shiftZ.get(), 32.0, (MSG_WRONG_SHIFT, "z"))

        protShiftVolPart = self.newProtocol(XmippProtShiftVolume,
                                            inputVol=protImportVol.outputVolume,
                                            inputProtocol=protShiftParticles)
        self.launchProtocol(protShiftVolPart)
        self.assertIsNotNone(protShiftVolPart.getFiles(),
                             "There was a problem with shift volume with particle shifts")
        self.assertEqual(protShiftVolPart.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protShiftVolPart.outputVolume.getDim(), (64, 64, 64), (MSG_WRONG_DIM, "volume"))
        self.assertEqual(protShiftVolPart.shiftX.get(), 2.0, (MSG_WRONG_SHIFT, "x"))
        self.assertEqual(protShiftVolPart.shiftY.get(), 3.0, (MSG_WRONG_SHIFT, "y"))
        self.assertEqual(protShiftVolPart.shiftZ.get(), 4.0, (MSG_WRONG_SHIFT, "z"))

        protShiftVolCrop = self.newProtocol(XmippProtShiftVolume,
                                            inputVol=protImportVol.outputVolume,
                                            shiftBool=False,
                                            x=5, y=5, z=5,
                                            boxSizeBool=False,
                                            boxSize=32)
        self.launchProtocol(protShiftVolCrop)
        self.assertIsNotNone(protShiftVolCrop.getFiles(),
                             "There was a problem with shift crop volume")
        self.assertEqual(protShiftVolCrop.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protShiftVolCrop.outputVolume.getDim(), (32, 32, 32), (MSG_WRONG_DIM, "volume"))
        self.assertEqual(protShiftVolCrop.shiftX.get(), 5.0, (MSG_WRONG_SHIFT, "x"))
        self.assertEqual(protShiftVolCrop.shiftY.get(), 5.0, (MSG_WRONG_SHIFT, "y"))
        self.assertEqual(protShiftVolCrop.shiftZ.get(), 5.0, (MSG_WRONG_SHIFT, "z"))

        protShiftVolPad = self.newProtocol(XmippProtShiftVolume,
                                           inputVol=protImportVol.outputVolume,
                                           shiftBool=False,
                                           x=5, y=5, z=5,
                                           boxSizeBool=False,
                                           boxSize=80)
        self.launchProtocol(protShiftVolPad)
        self.assertIsNotNone(protShiftVolPad.getFiles(),
                             "There was a problem with shift pad volume")
        self.assertEqual(protShiftVolCrop.outputVolume.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        self.assertEqual(protShiftVolPad.outputVolume.getDim(), (80, 80, 80), (MSG_WRONG_DIM, "volume"))
        self.assertEqual(protShiftVolPad.shiftX.get(), 5.0, (MSG_WRONG_SHIFT, "x"))
        self.assertEqual(protShiftVolPad.shiftY.get(), 5.0, (MSG_WRONG_SHIFT, "y"))
        self.assertEqual(protShiftVolPad.shiftZ.get(), 5.0, (MSG_WRONG_SHIFT, "z"))


class TestXmippProjSubtractionAndBoostParticles(TestXmippBase):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testXmippProjSub(self):
        # Create input data: phantom with two spheres and its projections (particles), phantom with one sphere
        # (reference volume) and its mask
        protCreatePhantom2items = self.newProtocol(XmippProtPhantom,
                                               desc='80 80 80 0\nsph + 1 15 15 0 10\nsph + 5 -15 -15 0 10',
                                               sampling=1.0)
        self.launchProtocol(protCreatePhantom2items)
        self.assertIsNotNone(protCreatePhantom2items.getFiles(),
                             "There was a problem with phantom with 2 items creation")
        protCreateGallery = self.newProtocol(XmippProtCreateGallery,
                                             inputVolume=protCreatePhantom2items.outputVolume,
                                             rotStep=15.0,
                                             tiltStep=90.0)
        self.launchProtocol(protCreateGallery)
        self.assertIsNotNone(protCreateGallery.getFiles(),
                             MSG_WRONG_GALLERY)
        protCreatePhantom1item = self.newProtocol(XmippProtPhantom,
                                                  desc='80 80 80 0\nsph + 1 -15 -15 0 10',
                                                  sampling=1.0)
        self.launchProtocol(protCreatePhantom1item)
        self.assertIsNotNone(protCreatePhantom1item.getFiles(),
                             "There was a problem with phantom with 1 item creation")
        protCreateMaskKeep = self.newProtocol(XmippProtCreateMask3D,
                                              inputVolume=protCreatePhantom1item.outputVolume,
                                              threshold=0.1)
        self.launchProtocol(protCreateMaskKeep)
        self.assertIsNotNone(protCreateMaskKeep.getFiles(),
                             "There was a problem with the 3D mask of the 1 item phantom")

        # Subtraction of particles - reference volume with and without mask
        protSubtractProj = self.newProtocol(XmippProtSubtractProjection,
                                            inputParticles=protCreateGallery.outputReprojections,
                                            vol=protCreatePhantom1item.outputVolume)
        self.launchProtocol(protSubtractProj)
        self.assertIsNotNone(protSubtractProj.outputParticles,
                             "There was a problem with projection subtraction")
        self.assertEqual(protSubtractProj.outputParticles.getSamplingRate(), 1.0, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(protSubtractProj.outputParticles.getFirstItem().getDim(), (80, 80, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(protSubtractProj.outputParticles.getSize(), 181, (MSG_WRONG_SIZE, "particles"))
        protSubtractProjMask = self.newProtocol(XmippProtSubtractProjection,
                                                inputParticles=protCreateGallery.outputReprojections,
                                                vol=protCreatePhantom2items.outputVolume,
                                                mask=protCreateMaskKeep.outputMask)
        self.launchProtocol(protSubtractProjMask)
        self.assertIsNotNone(protSubtractProjMask.outputParticles,
                             "There was a problem with projection subtraction with mask")
        self.assertEqual(protSubtractProjMask.outputParticles.getSamplingRate(), 1.0, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(protSubtractProjMask.outputParticles.getFirstItem().getDim(), (80, 80, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(protSubtractProjMask.outputParticles.getSize(), 181, (MSG_WRONG_SIZE, "particles"))

        # Add CTF and noise to particles (projections of the two spheres phantom) and perform the subtraction
        protSimulateCTF = self.newProtocol(XmippProtSimulateCTF,
                                           inputParticles=protCreateGallery.outputReprojections)
        self.launchProtocol(protSimulateCTF)
        self.assertIsNotNone(protSimulateCTF.outputParticles,
                             "There was a problem with CTF simulation")
        protSubtractProjCTF = self.newProtocol(XmippProtSubtractProjection,
                                               inputParticles=protSimulateCTF.outputParticles,
                                               vol=protCreatePhantom1item.outputVolume)
        self.launchProtocol(protSubtractProjCTF)
        self.assertIsNotNone(protSubtractProjCTF.outputParticles,
                             "There was a problem with projection subtraction CTF")
        self.assertEqual(protSubtractProjCTF.outputParticles.getSamplingRate(), 1.0, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(protSubtractProjCTF.outputParticles.getFirstItem().getDim(), (80, 80, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(protSubtractProjCTF.outputParticles.getSize(), 181, (MSG_WRONG_SIZE, "particles"))
        protAddNoise = self.newProtocol(XmippProtAddNoiseParticles,
                                        input=protCreateGallery.outputReprojections,
                                        gaussianStd=15.0)
        self.launchProtocol(protAddNoise)
        self.assertIsNotNone(protAddNoise.outputParticles,
                             "There was a problem with add noise protocol")
        protSubtractProjNoise = self.newProtocol(XmippProtSubtractProjection,
                                                 inputParticles=protAddNoise.outputParticles,
                                                 vol=protCreatePhantom1item.outputVolume)
        self.launchProtocol(protSubtractProjNoise)
        self.assertIsNotNone(protSubtractProjNoise.outputParticles,
                             "There was a problem with projection subtraction with noise")
        self.assertEqual(protSubtractProjNoise.outputParticles.getSamplingRate(), 1.0, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(protSubtractProjNoise.outputParticles.getFirstItem().getDim(), (80, 80, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(protSubtractProjNoise.outputParticles.getSize(), 181, (MSG_WRONG_SIZE, "particles"))
        protAddNoiseCTF = self.newProtocol(XmippProtAddNoiseParticles,
                                           input=protSimulateCTF.outputParticles,
                                           gaussianStd=15.0)
        self.launchProtocol(protAddNoiseCTF)
        self.assertIsNotNone(protAddNoiseCTF.outputParticles,
                             "There was a problem with add noise to ctf particles protocol")
        protSubtractProjNoiseCTF = self.newProtocol(XmippProtSubtractProjection,
                                                    inputParticles=protAddNoiseCTF.outputParticles,
                                                    vol=protCreatePhantom1item.outputVolume)
        self.launchProtocol(protSubtractProjNoiseCTF)
        self.assertIsNotNone(protSubtractProjNoiseCTF.outputParticles,
                             "There was a problem with projection subtraction with noise and CTF")
        self.assertEqual(protSubtractProjNoiseCTF.outputParticles.getSamplingRate(), 1.0, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(protSubtractProjNoiseCTF.outputParticles.getFirstItem().getDim(), (80, 80, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(protSubtractProjNoiseCTF.outputParticles.getSize(), 181, (MSG_WRONG_SIZE, "particles"))

        # Create a new phantom with two *overlapping* spheres and project it to create the particles
        protCreatePhantom2Over = self.newProtocol(XmippProtPhantom,
                                                  desc='80 80 80 0\nsph + 1 5 5 0 10\nsph + 5 -5 -5 0 10',
                                                  sampling=1.0)
        self.launchProtocol(protCreatePhantom2Over)
        self.assertIsNotNone(protCreatePhantom2Over.getFiles(),
                             "There was a problem with phantom with 2 items overlap creation")
        protCreateGalleryOver = self.newProtocol(XmippProtCreateGallery,
                                                 inputVolume=protCreatePhantom2Over.outputVolume)
        self.launchProtocol(protCreateGalleryOver)
        self.assertIsNotNone(protCreateGalleryOver.getFiles(),
                             "There was a problem with create gallery overlap")

        # Create phantom with just one of the two overlapping spheres to use it as reference volume and mask
        protCreatePhantom1Over = self.newProtocol(XmippProtPhantom,
                                                  desc='80 80 80 0\nsph + 1 -5 -5 0 10',
                                                  sampling=1.0)
        self.launchProtocol(protCreatePhantom1Over)
        self.assertIsNotNone(protCreatePhantom1Over.getFiles(),
                             "There was a problem with phantom with 1 item overlap creation")
        protCreateMaskKeepOver = self.newProtocol(XmippProtCreateMask3D,
                                                  inputVolume=protCreatePhantom1Over.outputVolume,
                                                  threshold=0.1)
        self.launchProtocol(protCreateMaskKeepOver)
        self.assertIsNotNone(protCreateMaskKeepOver.getFiles(),
                             "There was a problem with the 3D mask of the 1 item overlap phantom")

        # Perform subtraction of overlapping particles with and without mask, noise and CTF
        protSubtractProjOver = self.newProtocol(XmippProtSubtractProjection,
                                                inputParticles=protCreateGalleryOver.outputReprojections,
                                                vol=protCreatePhantom2Over.outputVolume,
                                                mask=protCreateMaskKeepOver.outputMask)
        self.launchProtocol(protSubtractProjOver)
        self.assertIsNotNone(protSubtractProjOver.outputParticles,
                             "There was a problem with projection subtraction with overlap")
        self.assertEqual(protSubtractProjOver.outputParticles.getSamplingRate(), 1.0, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(protSubtractProjOver.outputParticles.getFirstItem().getDim(), (80, 80, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(protSubtractProjOver.outputParticles.getSize(), 1647, (MSG_WRONG_SIZE, "particles"))
        protSubtractProjOverNoMask = self.newProtocol(XmippProtSubtractProjection,
                                                      inputParticles=protCreateGalleryOver.outputReprojections,
                                                      vol=protCreatePhantom1Over.outputVolume)
        self.launchProtocol(protSubtractProjOverNoMask)
        self.assertIsNotNone(protSubtractProjOverNoMask.outputParticles,
                             "There was a problem with projection subtraction with overlap")
        self.assertEqual(protSubtractProjOverNoMask.outputParticles.getSamplingRate(), 1.0, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(protSubtractProjOverNoMask.outputParticles.getFirstItem().getDim(), (80, 80, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(protSubtractProjOverNoMask.outputParticles.getSize(), 1647, (MSG_WRONG_SIZE, "particles"))
        protSimulateCTFOver = self.newProtocol(XmippProtSimulateCTF,
                                               inputParticles=protCreateGalleryOver.outputReprojections)
        self.launchProtocol(protSimulateCTFOver)
        self.assertIsNotNone(protSimulateCTFOver.outputParticles,
                             "There was a problem with overlap CTF simulation")
        protAddNoiseCTFOver = self.newProtocol(XmippProtAddNoiseParticles,
                                               input=protSimulateCTFOver.outputParticles,
                                               gaussianStd=15.0)
        self.launchProtocol(protAddNoiseCTFOver)
        self.assertIsNotNone(protAddNoiseCTFOver.outputParticles,
                             "There was a problem with add noise to ctf overlap particles protocol")
        protSubtractProjNoiseCTFOver = self.newProtocol(XmippProtSubtractProjection,
                                                        inputParticles=protAddNoiseCTFOver.outputParticles,
                                                        vol=protCreatePhantom1Over.outputVolume)
        self.launchProtocol(protSubtractProjNoiseCTFOver)
        self.assertIsNotNone(protSubtractProjNoiseCTFOver.outputParticles,
                             "There was a problem with projection subtraction with noise and CTF overlap")
        self.assertEqual(protSubtractProjNoiseCTFOver.outputParticles.getSamplingRate(), 1.0, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(protSubtractProjNoiseCTFOver.outputParticles.getFirstItem().getDim(), (80, 80, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(protSubtractProjNoiseCTFOver.outputParticles.getSize(), 1647, (MSG_WRONG_SIZE, "particles"))

        # Boost particles with reference volume
        protBoostPart = self.newProtocol(XmippProtBoostParticles,
                                         inputParticles=protCreateGallery.outputReprojections,
                                         vol=protCreatePhantom1item.outputVolume)
        self.launchProtocol(protBoostPart)
        self.assertIsNotNone(protBoostPart.outputParticles,
                             "There was a problem with projection subtraction")
        self.assertEqual(protBoostPart.outputParticles.getSamplingRate(), 1.0, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(protBoostPart.outputParticles.getFirstItem().getDim(), (80, 80, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(protBoostPart.outputParticles.getSize(), 181, (MSG_WRONG_SIZE, "particles"))


class TestXmippAlignVolumeAndParticles(TestXmippBase):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testXmippAlignVolumeAndParticles(self):

        # Create input data: phantom with two cylinders and its projections (particles)
        protPhantomRef = self.newProtocol(XmippProtPhantom,
                                               desc='80 80 80 0\ncyl + 1 0 0 0 5 5 10 0 0 0',
                                               sampling=1.0,
                                               objLabel="Reference phantom")
        self.launchProtocol(protPhantomRef)
        self.assertIsNotNone(protPhantomRef.getFiles(),
                             "There was a problem with the first phantom creation")

        protPhantomToAlign = self.newProtocol(XmippProtPhantom,
                                               desc='80 80 80 0\ncyl + 5 0 0 0 5 5 10 45 90 0',
                                               sampling=1.0,
                                               objLabel="Phantom to align")
        self.launchProtocol(protPhantomToAlign)
        self.assertIsNotNone(protPhantomToAlign.getFiles(),
                             "There was a problem with the second phantom creation")

        protCreateGallery = self.newProtocol(XmippProtCreateGallery,
                                             inputVolume=protPhantomToAlign.outputVolume,
                                             rotStep=15.0,
                                             tiltStep=90.0)
        self.launchProtocol(protCreateGallery)
        self.assertIsNotNone(protCreateGallery.getFiles(),
                             MSG_WRONG_GALLERY)

        protAlignVolumeParticles = self.newProtocol(XmippProtAlignVolumeParticles,
                                                    inputReference=protPhantomRef.outputVolume,
                                                    inputVolume=protPhantomToAlign.outputVolume,
                                                    inputParticles=protCreateGallery.outputReprojections)

        self.launchProtocol(protAlignVolumeParticles)

        volume = getattr(protAlignVolumeParticles, AlignVolPartOutputs.Volume.name, None)
        particles = getattr(protAlignVolumeParticles, AlignVolPartOutputs.Particles.name, None)

        self.assertIsNotNone(volume,
                             "There was a problem with the alignment of the volume")
        self.assertIsNotNone(particles,
                             "There was a problem with the alignment of the particles")
        self.assertEqual(particles.getSamplingRate(), 1.0, (MSG_WRONG_SAMPLING, "particles"))
        self.assertEqual(particles.getFirstItem().getDim(), (80, 80, 1),
                         (MSG_WRONG_DIM, "particles"))
        self.assertEqual(particles.getSize(), 181, (MSG_WRONG_SIZE, "particles"))


        matrices = self.getExpectedMatrices()

        for particle in particles:

            partMatrix = particle.getTransform().getMatrix()
            expectedMatrix = matrices[particle.getObjId() - 1]

            self.assertTrue(np.allclose(partMatrix,expectedMatrix, atol=1.e-4), (MSG_WRONG_ALIGNMENT, "particles"))



    def getExpectedMatrices(self):
        matrices = [
            [[-1.29893531e-16, 4.32974355e-17, -1.00000000e+00, -0.00000000e+00],
             [7.07108780e-01, -7.07104782e-01, -1.22464680e-16, 0.00000000e+00],
             [-7.07104782e-01, -7.07108780e-01, 6.12323400e-17, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818972e-01, -6.33160306e-09, -9.65925846e-01, -0.00000000e+00],
             [6.83014657e-01, -7.07104771e-01, 1.83013175e-01, 0.00000000e+00],
             [-6.83010775e-01, -7.07108792e-01, -1.83012126e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818969e-01, 1.35951160e-09, -9.65925847e-01, -0.00000000e+00],
             [-2.49997373e-01, -9.65926556e-01, -6.69865745e-02, -0.00000000e+00],
             [-9.33013427e-01, 2.58816320e-01, -2.50000115e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818972e-01, 3.02503078e-09, -9.65925846e-01, -0.00000000e+00],
             [-9.33012015e-01, -2.58821773e-01, -2.49999741e-01, -0.00000000e+00],
             [-2.50002640e-01, 9.65925095e-01, -6.69879822e-02, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818972e-01, -6.33160300e-09, -9.65925846e-01, -0.00000000e+00],
             [-6.83014657e-01, 7.07104771e-01, -1.83013175e-01, -0.00000000e+00],
             [6.83010775e-01, 7.07108792e-01, 1.83012126e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818969e-01, 1.35951158e-09, -9.65925847e-01, -0.00000000e+00],
             [2.49997373e-01, 9.65926556e-01, 6.69865745e-02, 0.00000000e+00],
             [9.33013427e-01, -2.58816320e-01, 2.50000115e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818972e-01, 3.02503071e-09, -9.65925846e-01, -0.00000000e+00],
             [9.33012015e-01, 2.58821773e-01, 2.49999741e-01, 0.00000000e+00],
             [2.50002640e-01, -9.65925095e-01, 6.69879822e-02, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999888e-01, 2.14734474e-08, -8.66025468e-01, -0.00000000e+00],
             [6.12374243e-01, -7.07104756e-01, 3.53554311e-01, 0.00000000e+00],
             [-6.12370720e-01, -7.07108806e-01, -3.53552312e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999890e-01, 5.88497022e-09, -8.66025467e-01, -0.00000000e+00],
             [2.24146271e-01, -9.65925089e-01, 1.29410866e-01, -0.00000000e+00],
             [-8.36515726e-01, -2.58821798e-01, -4.82962439e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999890e-01, -1.03496165e-08, -8.66025467e-01, -0.00000000e+00],
             [-2.24141542e-01, -9.65926552e-01, -1.29408130e-01, -0.00000000e+00],
             [-8.36516992e-01, 2.58816335e-01, -4.82963173e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999877e-01, -2.10888133e-08, -8.66025475e-01, -0.00000000e+00],
             [-6.12370783e-01, -7.07108756e-01, -3.53552303e-01, -0.00000000e+00],
             [-6.12374189e-01, 7.07104806e-01, -3.53554304e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999879e-01, -4.88024898e-09, -8.66025474e-01, -0.00000000e+00],
             [-8.36515743e-01, -2.58821757e-01, -4.82962432e-01, -0.00000000e+00],
             [-2.24146232e-01, 9.65925100e-01, -1.29410851e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999883e-01, 1.32033358e-08, -8.66025472e-01, -0.00000000e+00],
             [-8.36517006e-01, 2.58816307e-01, -4.82963164e-01, -0.00000000e+00],
             [2.24141508e-01, 9.65926560e-01, 1.29408134e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999888e-01, 2.14734474e-08, -8.66025468e-01, -0.00000000e+00],
             [-6.12374243e-01, 7.07104756e-01, -3.53554311e-01, -0.00000000e+00],
             [6.12370720e-01, 7.07108806e-01, 3.53552312e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999890e-01, 5.88497019e-09, -8.66025467e-01, -0.00000000e+00],
             [-2.24146271e-01, 9.65925089e-01, -1.29410866e-01, 0.00000000e+00],
             [8.36515726e-01, 2.58821798e-01, 4.82962439e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999890e-01, -1.03496168e-08, -8.66025467e-01, -0.00000000e+00],
             [2.24141542e-01, 9.65926552e-01, 1.29408130e-01, 0.00000000e+00],
             [8.36516992e-01, -2.58816335e-01, 4.82963173e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999877e-01, -2.10888136e-08, -8.66025475e-01, -0.00000000e+00],
             [6.12370783e-01, 7.07108756e-01, 3.53552303e-01, 0.00000000e+00],
             [6.12374189e-01, -7.07104806e-01, 3.53554304e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999879e-01, -4.88024898e-09, -8.66025474e-01, -0.00000000e+00],
             [8.36515743e-01, 2.58821757e-01, 4.82962432e-01, 0.00000000e+00],
             [2.24146232e-01, -9.65925100e-01, 1.29410851e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999883e-01, 1.32033356e-08, -8.66025472e-01, -0.00000000e+00],
             [8.36517006e-01, -2.58816307e-01, 4.82963164e-01, 0.00000000e+00],
             [-2.24141508e-01, -9.65926560e-01, -1.29408134e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106672e-01, 4.35939631e-08, -7.07106890e-01, -0.00000000e+00],
             [5.00001540e-01, -7.07104743e-01, 5.00001342e-01, 0.00000000e+00],
             [-4.99998614e-01, -7.07108819e-01, -4.99998504e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106670e-01, 5.52544747e-08, -7.07106893e-01, -0.00000000e+00],
             [2.85617224e-01, -9.14792706e-01, 2.85617062e-01, -0.00000000e+00],
             [-6.46856212e-01, -4.03923637e-01, -6.46856040e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106669e-01, 1.68940571e-08, -7.07106893e-01, -0.00000000e+00],
             [3.26586630e-02, -9.98932843e-01, 3.26586288e-02, -0.00000000e+00],
             [-7.06352299e-01, -4.61862999e-02, -7.06352076e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106672e-01, -2.97394772e-08, -7.07106890e-01, -0.00000000e+00],
             [-2.24710654e-01, -9.48161533e-01, -2.24710545e-01, -0.00000000e+00],
             [-6.70451547e-01, 3.17788777e-01, -6.70451353e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106673e-01, -5.99903706e-08, -7.07106890e-01, -0.00000000e+00],
             [-4.51731499e-01, -7.69335746e-01, -4.51731295e-01, -0.00000000e+00],
             [-5.44002579e-01, 6.38844668e-01, -5.44002466e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106675e-01, -3.06564823e-08, -7.07106888e-01, -0.00000000e+00],
             [-6.17743458e-01, -4.86606921e-01, -6.17743251e-01, -0.00000000e+00],
             [-3.44083086e-01, 8.73621030e-01, -3.44083021e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106677e-01, -1.08103433e-09, -7.07106885e-01, -0.00000000e+00],
             [-7.00325760e-01, -1.38159139e-01, -7.00325554e-01, -0.00000000e+00],
             [-9.76932779e-02, 9.90410042e-01, -9.76932507e-02, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106671e-01, 1.64166702e-08, -7.07106891e-01, -0.00000000e+00],
             [-6.88325222e-01, 2.28947769e-01, -6.88325002e-01, -0.00000000e+00],
             [1.61890534e-01, 9.73438709e-01, 1.61890506e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106673e-01, 4.23610224e-08, -7.07106890e-01, -0.00000000e+00],
             [-5.83362562e-01, 5.65134047e-01, -5.83362349e-01, -0.00000000e+00],
             [3.99610154e-01, 8.24999096e-01, 3.99610081e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106676e-01, 4.44615930e-08, -7.07106886e-01, -0.00000000e+00],
             [-3.99613558e-01, 8.24995845e-01, -3.99613388e-01, 0.00000000e+00],
             [5.83360226e-01, 5.65138793e-01, 5.83360087e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106675e-01, 3.88472993e-08, -7.07106888e-01, -0.00000000e+00],
             [-1.61894504e-01, 9.73437400e-01, -1.61894402e-01, 0.00000000e+00],
             [6.88324284e-01, 2.28953331e-01, 6.88324089e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106672e-01, -3.23226656e-08, -7.07106891e-01, -0.00000000e+00],
             [9.76893638e-02, 9.90410819e-01, 9.76892882e-02, 0.00000000e+00],
             [7.00326312e-01, -1.38153570e-01, 7.00326101e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106670e-01, -6.50865464e-08, -7.07106893e-01, -0.00000000e+00],
             [3.44079709e-01, 8.73623739e-01, 3.44079520e-01, 0.00000000e+00],
             [6.17745345e-01, -4.86602058e-01, 6.17745195e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106669e-01, -4.82454159e-08, -7.07106894e-01, -0.00000000e+00],
             [5.44000123e-01, 6.38848940e-01, 5.43999906e-01, 0.00000000e+00],
             [4.51734463e-01, -7.69332199e-01, 4.51734372e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106667e-01, -2.29245414e-08, -7.07106896e-01, -0.00000000e+00],
             [6.70450308e-01, 3.17794075e-01, 6.70450080e-01, 0.00000000e+00],
             [2.24714367e-01, -9.48159757e-01, 2.24714325e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106677e-01, -7.22923877e-09, -7.07106885e-01, -0.00000000e+00],
             [7.06352476e-01, -4.61806309e-02, 7.06352269e-01, 0.00000000e+00],
             [-3.26546472e-02, -9.98933106e-01, -3.26546274e-02, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106668e-01, 3.31554107e-08, -7.07106894e-01, -0.00000000e+00],
             [6.46857880e-01, -4.03918380e-01, 6.46857655e-01, 0.00000000e+00],
             [-2.85613449e-01, -9.14795028e-01, -2.85613401e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025336e-01, 7.64847654e-08, -5.00000118e-01, 0.00000000e+00],
             [3.53554536e-01, -7.07104752e-01, 6.12374118e-01, 0.00000000e+00],
             [-3.53552413e-01, -7.07108811e-01, -6.12370657e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025333e-01, 1.10193108e-07, -5.00000123e-01, 0.00000000e+00],
             [2.26996679e-01, -8.91005211e-01, 3.93169457e-01, -0.00000000e+00],
             [-4.45502671e-01, -4.53993078e-01, -7.71633109e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025334e-01, 4.76845001e-08, -5.00000121e-01, 0.00000000e+00],
             [7.82187053e-02, -9.87687893e-01, 1.35478634e-01, -0.00000000e+00],
             [-4.93844059e-01, -1.56437291e-01, -8.55362741e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025335e-01, -5.44865966e-08, -5.00000118e-01, 0.00000000e+00],
             [-7.82159161e-02, -9.87688778e-01, -1.35473790e-01, -0.00000000e+00],
             [-4.93844499e-01, 1.56431702e-01, -8.55363510e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025337e-01, -1.06291172e-07, -5.00000115e-01, 0.00000000e+00],
             [-2.26994148e-01, -8.91007784e-01, -3.93165087e-01, -0.00000000e+00],
             [-4.45503953e-01, 4.53988027e-01, -7.71635341e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025334e-01, -8.31178995e-08, -5.00000121e-01, 0.00000000e+00],
             [-3.53552543e-01, -7.07108749e-01, -6.12370652e-01, -0.00000000e+00],
             [-3.53554409e-01, 7.07104813e-01, -6.12374120e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025336e-01, -5.47984173e-08, -5.00000118e-01, 0.00000000e+00],
             [-4.45502752e-01, -4.53992999e-01, -7.71633109e-01, -0.00000000e+00],
             [-2.26996511e-01, 8.91005251e-01, -3.93169464e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025341e-01, -1.49156698e-08, -5.00000109e-01, 0.00000000e+00],
             [-4.93844058e-01, -1.56437257e-01, -8.55362748e-01, -0.00000000e+00],
             [-7.82186327e-02, 9.87687898e-01, -1.35478636e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025338e-01, 1.29877133e-08, -5.00000114e-01, 0.00000000e+00],
             [-4.93844506e-01, 1.56431668e-01, -8.55363512e-01, -0.00000000e+00],
             [7.82158406e-02, 9.87688784e-01, 1.35473794e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025341e-01, 5.45050422e-08, -5.00000109e-01, 0.00000000e+00],
             [-4.45504029e-01, 4.53987956e-01, -7.71635339e-01, -0.00000000e+00],
             [2.26993986e-01, 8.91007820e-01, 3.93165099e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025336e-01, 7.64847656e-08, -5.00000118e-01, 0.00000000e+00],
             [-3.53554536e-01, 7.07104752e-01, -6.12374118e-01, -0.00000000e+00],
             [3.53552413e-01, 7.07108811e-01, 6.12370657e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025333e-01, 1.10193109e-07, -5.00000123e-01, 0.00000000e+00],
             [-2.26996679e-01, 8.91005211e-01, -3.93169457e-01, 0.00000000e+00],
             [4.45502671e-01, 4.53993078e-01, 7.71633109e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025334e-01, 4.76845005e-08, -5.00000121e-01, 0.00000000e+00],
             [-7.82187053e-02, 9.87687893e-01, -1.35478634e-01, 0.00000000e+00],
             [4.93844059e-01, 1.56437291e-01, 8.55362741e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025335e-01, -5.44865966e-08, -5.00000118e-01, 0.00000000e+00],
             [7.82159161e-02, 9.87688778e-01, 1.35473790e-01, 0.00000000e+00],
             [4.93844499e-01, -1.56431702e-01, 8.55363510e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025337e-01, -1.06291172e-07, -5.00000115e-01, 0.00000000e+00],
             [2.26994148e-01, 8.91007784e-01, 3.93165087e-01, 0.00000000e+00],
             [4.45503953e-01, -4.53988027e-01, 7.71635341e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025334e-01, -8.31178997e-08, -5.00000121e-01, 0.00000000e+00],
             [3.53552543e-01, 7.07108749e-01, 6.12370652e-01, 0.00000000e+00],
             [3.53554409e-01, -7.07104813e-01, 6.12374120e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025336e-01, -5.47984172e-08, -5.00000118e-01, 0.00000000e+00],
             [4.45502752e-01, 4.53992999e-01, 7.71633109e-01, 0.00000000e+00],
             [2.26996511e-01, -8.91005251e-01, 3.93169464e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025341e-01, -1.49156700e-08, -5.00000109e-01, 0.00000000e+00],
             [4.93844058e-01, 1.56437257e-01, 8.55362748e-01, 0.00000000e+00],
             [7.82186327e-02, -9.87687898e-01, 1.35478636e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025338e-01, 1.29877134e-08, -5.00000114e-01, 0.00000000e+00],
             [4.93844506e-01, -1.56431668e-01, 8.55363512e-01, 0.00000000e+00],
             [-7.82158406e-02, -9.87688784e-01, -1.35473794e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025341e-01, 5.45050421e-08, -5.00000109e-01, 0.00000000e+00],
             [4.45504029e-01, -4.53987956e-01, 7.71635339e-01, 0.00000000e+00],
             [-2.26993986e-01, -8.91007820e-01, -3.93165099e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, 6.48608573e-08, -2.58819120e-01, 0.00000000e+00],
             [1.83013322e-01, -7.07104762e-01, 6.83014626e-01, 0.00000000e+00],
             [-1.83012188e-01, -7.07108800e-01, -6.83010750e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, 1.02567524e-07, -2.58819121e-01, 0.00000000e+00],
             [1.26850627e-01, -8.71659069e-01, 4.73412490e-01, -0.00000000e+00],
             [-2.25601986e-01, -4.90112709e-01, -8.41958002e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925807e-01, 1.24808279e-07, -2.58819118e-01, 0.00000000e+00],
             [6.12799952e-02, -9.71566412e-01, 2.28699518e-01, -0.00000000e+00],
             [-2.51459934e-01, -2.36767200e-01, -9.38461078e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925804e-01, -2.74816234e-08, -2.58819127e-01, 0.00000000e+00],
             [-8.83567101e-03, -9.99417118e-01, -3.29750558e-02, -0.00000000e+00],
             [-2.58668265e-01, 3.41382980e-02, -9.65362784e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, -1.41380484e-07, -2.58819120e-01, 0.00000000e+00],
             [-7.82959938e-02, -9.53145608e-01, -2.92204015e-01, -0.00000000e+00],
             [-2.46692267e-01, 3.02511899e-01, -9.20667951e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, -1.03750915e-07, -2.58819123e-01, 0.00000000e+00],
             [-1.41949311e-01, -8.36183656e-01, -5.29761537e-01, -0.00000000e+00],
             [-2.16420265e-01, 5.48449536e-01, -8.07691386e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925805e-01, -6.38452202e-08, -2.58819125e-01, 0.00000000e+00],
             [-1.95074905e-01, -6.57205811e-01, -7.28029054e-01, -0.00000000e+00],
             [-1.70097386e-01, 7.53711167e-01, -6.34812064e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925807e-01, -2.65380312e-08, -2.58819118e-01, 0.00000000e+00],
             [-2.33732703e-01, -4.29485992e-01, -8.72302015e-01, -0.00000000e+00],
             [-1.11159163e-01, 9.03073520e-01, -4.14851610e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925807e-01, -1.38442081e-08, -2.58819118e-01, 0.00000000e+00],
             [-2.55055641e-01, -1.69913175e-01, -9.51880315e-01, -0.00000000e+00],
             [-4.39767648e-02, 9.85459037e-01, -1.64123524e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925805e-01, 7.46517604e-09, -2.58819123e-01, 0.00000000e+00],
             [-2.57462282e-01, 1.02261331e-01, -9.60862005e-01, -0.00000000e+00],
             [2.64671807e-02, 9.94757569e-01, 9.87768601e-02, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925804e-01, 2.48804253e-08, -2.58819127e-01, 0.00000000e+00],
             [-2.40774133e-01, 3.66851576e-01, -8.98580958e-01, -0.00000000e+00],
             [9.49481822e-02, 9.30279486e-01, 3.54351409e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, 4.58456376e-08, -2.58819120e-01, 0.00000000e+00],
             [-2.06228870e-01, 6.04234121e-01, -7.69656275e-01, -0.00000000e+00],
             [1.56387308e-01, 7.96806832e-01, 5.83645339e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925805e-01, 9.19733392e-08, -2.58819125e-01, 0.00000000e+00],
             [-1.56388592e-01, 7.96803389e-01, -5.83649696e-01, 0.00000000e+00],
             [2.06227902e-01, 6.04238661e-01, 7.69652969e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, 1.17212953e-07, -2.58819121e-01, 0.00000000e+00],
             [-9.49496808e-02, 9.30277394e-01, -3.54356500e-01, 0.00000000e+00],
             [2.40773536e-01, 3.66856881e-01, 8.98578952e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925805e-01, 9.55871518e-08, -2.58819124e-01, 0.00000000e+00],
             [-2.64687443e-02, 9.94756987e-01, -9.87822987e-02, 0.00000000e+00],
             [2.57462123e-01, 1.02266989e-01, 9.60861446e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925805e-01, -1.15973981e-07, -2.58819124e-01, 0.00000000e+00],
             [4.39754569e-02, 9.85459991e-01, 1.64118144e-01, 0.00000000e+00],
             [2.55055873e-01, -1.69907640e-01, 9.51881240e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, -1.14862176e-07, -2.58819122e-01, 0.00000000e+00],
             [1.11157976e-01, 9.03075929e-01, 4.14846683e-01, 0.00000000e+00],
             [2.33733271e-01, -4.29480926e-01, 8.72304357e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925808e-01, -7.90071770e-08, -2.58819115e-01, 0.00000000e+00],
             [1.70096386e-01, 7.53714865e-01, 6.34807941e-01, 0.00000000e+00],
             [1.95075764e-01, -6.57201569e-01, 7.28032653e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925807e-01, -4.45608176e-08, -2.58819117e-01, 0.00000000e+00],
             [2.16419542e-01, 5.48454233e-01, 8.07688391e-01, 0.00000000e+00],
             [1.41950404e-01, -8.36180575e-01, 5.29766107e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925808e-01, -1.91493226e-08, -2.58819113e-01, 0.00000000e+00],
             [2.46691868e-01, 3.02517243e-01, 9.20666302e-01, 0.00000000e+00],
             [7.82972268e-02, -9.53143912e-01, 2.92209217e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, -2.00565933e-10, -2.58819121e-01, 0.00000000e+00],
             [2.58668211e-01, 3.41439414e-02, 9.65362599e-01, 0.00000000e+00],
             [8.83710472e-03, -9.99416926e-01, 3.29805141e-02, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, 1.01961976e-08, -2.58819119e-01, 0.00000000e+00],
             [2.51460315e-01, -2.36761659e-01, 9.38462374e-01, 0.00000000e+00],
             [-6.12784346e-02, -9.71567762e-01, -2.28694199e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925808e-01, 4.32637529e-08, -2.58819112e-01, 0.00000000e+00],
             [2.25602769e-01, -4.90107745e-01, 8.41960681e-01, 0.00000000e+00],
             [-1.26849215e-01, -8.71661860e-01, -4.73407730e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, -1.04530020e-16, 4.32979252e-17, 0.00000000e+00],
             [-1.04530020e-16, -7.07104782e-01, 7.07108780e-01, 0.00000000e+00],
             [-4.32979252e-17, -7.07108780e-01, -7.07104782e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, -1.14606065e-16, 3.00107463e-17, 0.00000000e+00],
             [-1.14606065e-16, -8.71659085e-01, 4.90112680e-01, -0.00000000e+00],
             [-3.00107463e-17, -4.90112680e-01, -8.71659085e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, -1.20723625e-16, 1.44978075e-17, 0.00000000e+00],
             [-1.20723625e-16, -9.71566421e-01, 2.36767165e-01, -0.00000000e+00],
             [-1.44978075e-17, -2.36767165e-01, -9.71566421e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 1.22428989e-16, 2.09036731e-18, 0.00000000e+00],
             [1.22428989e-16, -9.99417119e-01, -3.41382890e-02, -0.00000000e+00],
             [-2.09036731e-18, 3.41382890e-02, -9.99417119e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 1.19595677e-16, 1.85235092e-17, 0.00000000e+00],
             [1.19595677e-16, -9.53145620e-01, -3.02511862e-01, -0.00000000e+00],
             [-1.85235092e-17, 3.02511862e-01, -9.53145620e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 1.12433823e-16, 3.35828471e-17, 0.00000000e+00],
             [1.12433823e-16, -8.36183671e-01, -5.48449513e-01, -0.00000000e+00],
             [-3.35828471e-17, 5.48449513e-01, -8.36183671e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 1.01474590e-16, 4.61514981e-17, 0.00000000e+00],
             [1.01474590e-16, -6.57205816e-01, -7.53711162e-01, -0.00000000e+00],
             [-4.61514981e-17, 7.53711162e-01, -6.57205816e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 8.75307729e-17, 5.52973045e-17, 0.00000000e+00],
             [8.75307729e-17, -4.29486003e-01, -9.03073515e-01, -0.00000000e+00],
             [-5.52973045e-17, 9.03073515e-01, -4.29486003e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 7.16365213e-17, 6.03419627e-17, 0.00000000e+00],
             [7.16365213e-17, -1.69913177e-01, -9.85459036e-01, -0.00000000e+00],
             [-6.03419627e-17, 9.85459036e-01, -1.69913177e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 5.49706393e-17, 6.09113336e-17, 0.00000000e+00],
             [5.49706393e-17, 1.02261333e-01, -9.94757568e-01, -0.00000000e+00],
             [-6.09113336e-17, 9.94757568e-01, 1.02261333e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 3.87691589e-17, 5.69631895e-17, 0.00000000e+00],
             [3.87691589e-17, 3.66851586e-01, -9.30279481e-01, -0.00000000e+00],
             [-5.69631895e-17, 9.30279481e-01, 3.66851586e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 2.42336699e-17, 4.87903461e-17, 0.00000000e+00],
             [2.42336699e-17, 6.04234136e-01, -7.96806820e-01, -0.00000000e+00],
             [-4.87903461e-17, 7.96806820e-01, 6.04234136e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 1.24422031e-17, 3.69989459e-17, 0.00000000e+00],
             [1.24422031e-17, 7.96803403e-01, -6.04238642e-01, 0.00000000e+00],
             [-3.69989459e-17, 6.04238642e-01, 7.96803403e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 4.26927752e-18, 2.24635032e-17, 0.00000000e+00],
             [4.26927752e-18, 9.30277407e-01, -3.66856847e-01, 0.00000000e+00],
             [-2.24635032e-17, 3.66856847e-01, 9.30277407e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, 3.21041765e-19, 6.26204513e-18, 0.00000000e+00],
             [3.21041765e-19, 9.94756990e-01, -1.02266958e-01, 0.00000000e+00],
             [-6.26204513e-18, 1.02266958e-01, 9.94756990e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, -8.90318393e-19, 1.04038402e-17, 0.00000000e+00],
             [-8.90318393e-19, 9.85459997e-01, 1.69907604e-01, 0.00000000e+00],
             [-1.04038402e-17, -1.69907604e-01, 9.85459997e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, -5.93488676e-18, 2.62981202e-17, 0.00000000e+00],
             [-5.93488676e-18, 9.03075944e-01, 4.29480896e-01, 0.00000000e+00],
             [-2.62981202e-17, -4.29480896e-01, 9.03075944e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, -1.50806143e-17, 4.02419890e-17, 0.00000000e+00],
             [-1.50806143e-17, 7.53714878e-01, 6.57201554e-01, 0.00000000e+00],
             [-4.02419890e-17, -6.57201554e-01, 7.53714878e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, -2.76492034e-17, 5.12012929e-17, 0.00000000e+00],
             [-2.76492034e-17, 5.48454242e-01, 8.36180569e-01, 0.00000000e+00],
             [-5.12012929e-17, -8.36180569e-01, 5.48454242e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, -4.27085008e-17, 5.83632319e-17, 0.00000000e+00],
             [-4.27085008e-17, 3.02517252e-01, 9.53143910e-01, 0.00000000e+00],
             [-5.83632319e-17, -9.53143910e-01, 3.02517252e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, -5.91416266e-17, 6.11966370e-17, 0.00000000e+00],
             [-5.91416266e-17, 3.41439405e-02, 9.99416926e-01, 0.00000000e+00],
             [-6.11966370e-17, -9.99416926e-01, 3.41439405e-02, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, -7.57298111e-17, 5.94913673e-17, 0.00000000e+00],
             [-7.57298111e-17, -2.36761670e-01, 9.71567760e-01, 0.00000000e+00],
             [-5.94913673e-17, -9.71567760e-01, -2.36761670e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[1.00000000e+00, -9.12427844e-17, 5.33738951e-17, 0.00000000e+00],
             [-9.12427844e-17, -4.90107751e-01, 8.71661857e-01, 0.00000000e+00],
             [-5.33738951e-17, -8.71661857e-01, -4.90107751e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, -6.48608573e-08, 2.58819120e-01, 0.00000000e+00],
             [-1.83013322e-01, -7.07104762e-01, 6.83014626e-01, 0.00000000e+00],
             [1.83012188e-01, -7.07108800e-01, -6.83010750e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, -1.02567525e-07, 2.58819121e-01, 0.00000000e+00],
             [-1.26850627e-01, -8.71659069e-01, 4.73412490e-01, -0.00000000e+00],
             [2.25601986e-01, -4.90112709e-01, -8.41958002e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925807e-01, -1.24808280e-07, 2.58819118e-01, 0.00000000e+00],
             [-6.12799952e-02, -9.71566412e-01, 2.28699518e-01, -0.00000000e+00],
             [2.51459934e-01, -2.36767200e-01, -9.38461078e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925804e-01, 2.74816234e-08, 2.58819127e-01, 0.00000000e+00],
             [8.83567101e-03, -9.99417118e-01, -3.29750558e-02, -0.00000000e+00],
             [2.58668265e-01, 3.41382980e-02, -9.65362784e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, 1.41380484e-07, 2.58819120e-01, 0.00000000e+00],
             [7.82959938e-02, -9.53145608e-01, -2.92204015e-01, -0.00000000e+00],
             [2.46692267e-01, 3.02511899e-01, -9.20667951e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, 1.03750915e-07, 2.58819123e-01, 0.00000000e+00],
             [1.41949311e-01, -8.36183656e-01, -5.29761537e-01, -0.00000000e+00],
             [2.16420265e-01, 5.48449536e-01, -8.07691386e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925805e-01, 6.38452204e-08, 2.58819125e-01, 0.00000000e+00],
             [1.95074905e-01, -6.57205811e-01, -7.28029054e-01, -0.00000000e+00],
             [1.70097386e-01, 7.53711167e-01, -6.34812064e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925807e-01, 2.65380313e-08, 2.58819118e-01, 0.00000000e+00],
             [2.33732703e-01, -4.29485992e-01, -8.72302015e-01, -0.00000000e+00],
             [1.11159163e-01, 9.03073520e-01, -4.14851610e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925807e-01, 1.38442080e-08, 2.58819118e-01, 0.00000000e+00],
             [2.55055641e-01, -1.69913175e-01, -9.51880315e-01, -0.00000000e+00],
             [4.39767648e-02, 9.85459037e-01, -1.64123524e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925805e-01, -7.46517612e-09, 2.58819123e-01, 0.00000000e+00],
             [2.57462282e-01, 1.02261331e-01, -9.60862005e-01, -0.00000000e+00],
             [-2.64671807e-02, 9.94757569e-01, 9.87768601e-02, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925804e-01, -2.48804251e-08, 2.58819127e-01, 0.00000000e+00],
             [2.40774133e-01, 3.66851576e-01, -8.98580958e-01, -0.00000000e+00],
             [-9.49481822e-02, 9.30279486e-01, 3.54351409e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, -4.58456374e-08, 2.58819120e-01, 0.00000000e+00],
             [2.06228870e-01, 6.04234121e-01, -7.69656275e-01, -0.00000000e+00],
             [-1.56387308e-01, 7.96806832e-01, 5.83645339e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925805e-01, -9.19733390e-08, 2.58819125e-01, 0.00000000e+00],
             [1.56388592e-01, 7.96803389e-01, -5.83649696e-01, 0.00000000e+00],
             [-2.06227902e-01, 6.04238661e-01, 7.69652969e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, -1.17212953e-07, 2.58819121e-01, 0.00000000e+00],
             [9.49496808e-02, 9.30277394e-01, -3.54356500e-01, 0.00000000e+00],
             [-2.40773536e-01, 3.66856881e-01, 8.98578952e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925805e-01, -9.55871517e-08, 2.58819124e-01, 0.00000000e+00],
             [2.64687443e-02, 9.94756987e-01, -9.87822987e-02, 0.00000000e+00],
             [-2.57462123e-01, 1.02266989e-01, 9.60861446e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925805e-01, 1.15973980e-07, 2.58819124e-01, 0.00000000e+00],
             [-4.39754569e-02, 9.85459991e-01, 1.64118144e-01, 0.00000000e+00],
             [-2.55055873e-01, -1.69907640e-01, 9.51881240e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, 1.14862176e-07, 2.58819122e-01, 0.00000000e+00],
             [-1.11157976e-01, 9.03075929e-01, 4.14846683e-01, 0.00000000e+00],
             [-2.33733271e-01, -4.29480926e-01, 8.72304357e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925808e-01, 7.90071770e-08, 2.58819115e-01, 0.00000000e+00],
             [-1.70096386e-01, 7.53714865e-01, 6.34807941e-01, 0.00000000e+00],
             [-1.95075764e-01, -6.57201569e-01, 7.28032653e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925807e-01, 4.45608175e-08, 2.58819117e-01, 0.00000000e+00],
             [-2.16419542e-01, 5.48454233e-01, 8.07688391e-01, 0.00000000e+00],
             [-1.41950404e-01, -8.36180575e-01, 5.29766107e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925808e-01, 1.91493227e-08, 2.58819113e-01, 0.00000000e+00],
             [-2.46691868e-01, 3.02517243e-01, 9.20666302e-01, 0.00000000e+00],
             [-7.82972268e-02, -9.53143912e-01, 2.92209217e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, 2.00565819e-10, 2.58819121e-01, 0.00000000e+00],
             [-2.58668211e-01, 3.41439414e-02, 9.65362599e-01, 0.00000000e+00],
             [-8.83710472e-03, -9.99416926e-01, 3.29805141e-02, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925806e-01, -1.01961977e-08, 2.58819119e-01, 0.00000000e+00],
             [-2.51460315e-01, -2.36761659e-01, 9.38462374e-01, 0.00000000e+00],
             [6.12784346e-02, -9.71567762e-01, -2.28694199e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[9.65925808e-01, -4.32637531e-08, 2.58819112e-01, 0.00000000e+00],
             [-2.25602769e-01, -4.90107745e-01, 8.41960681e-01, 0.00000000e+00],
             [1.26849215e-01, -8.71661860e-01, -4.73407730e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025336e-01, -7.64847658e-08, 5.00000118e-01, 0.00000000e+00],
             [-3.53554536e-01, -7.07104752e-01, 6.12374118e-01, 0.00000000e+00],
             [3.53552413e-01, -7.07108811e-01, -6.12370657e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025333e-01, -1.10193108e-07, 5.00000123e-01, 0.00000000e+00],
             [-2.26996679e-01, -8.91005211e-01, 3.93169457e-01, -0.00000000e+00],
             [4.45502671e-01, -4.53993078e-01, -7.71633109e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025334e-01, -4.76845009e-08, 5.00000121e-01, 0.00000000e+00],
             [-7.82187053e-02, -9.87687893e-01, 1.35478634e-01, -0.00000000e+00],
             [4.93844059e-01, -1.56437291e-01, -8.55362741e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025335e-01, 5.44865968e-08, 5.00000118e-01, 0.00000000e+00],
             [7.82159161e-02, -9.87688778e-01, -1.35473790e-01, -0.00000000e+00],
             [4.93844499e-01, 1.56431702e-01, -8.55363510e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025337e-01, 1.06291172e-07, 5.00000115e-01, 0.00000000e+00],
             [2.26994148e-01, -8.91007784e-01, -3.93165087e-01, -0.00000000e+00],
             [4.45503953e-01, 4.53988027e-01, -7.71635341e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025334e-01, 8.31178996e-08, 5.00000121e-01, 0.00000000e+00],
             [3.53552543e-01, -7.07108749e-01, -6.12370652e-01, -0.00000000e+00],
             [3.53554409e-01, 7.07104813e-01, -6.12374120e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025336e-01, 5.47984172e-08, 5.00000118e-01, 0.00000000e+00],
             [4.45502752e-01, -4.53992999e-01, -7.71633109e-01, -0.00000000e+00],
             [2.26996511e-01, 8.91005251e-01, -3.93169464e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025341e-01, 1.49156699e-08, 5.00000109e-01, 0.00000000e+00],
             [4.93844058e-01, -1.56437257e-01, -8.55362748e-01, -0.00000000e+00],
             [7.82186327e-02, 9.87687898e-01, -1.35478636e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025338e-01, -1.29877134e-08, 5.00000114e-01, 0.00000000e+00],
             [4.93844506e-01, 1.56431668e-01, -8.55363512e-01, -0.00000000e+00],
             [-7.82158406e-02, 9.87688784e-01, 1.35473794e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025341e-01, -5.45050421e-08, 5.00000109e-01, 0.00000000e+00],
             [4.45504029e-01, 4.53987956e-01, -7.71635339e-01, -0.00000000e+00],
             [-2.26993986e-01, 8.91007820e-01, 3.93165099e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025336e-01, -7.64847657e-08, 5.00000118e-01, 0.00000000e+00],
             [3.53554536e-01, 7.07104752e-01, -6.12374118e-01, -0.00000000e+00],
             [-3.53552413e-01, 7.07108811e-01, 6.12370657e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025333e-01, -1.10193108e-07, 5.00000123e-01, 0.00000000e+00],
             [2.26996679e-01, 8.91005211e-01, -3.93169457e-01, 0.00000000e+00],
             [-4.45502671e-01, 4.53993078e-01, 7.71633109e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025334e-01, -4.76845005e-08, 5.00000121e-01, 0.00000000e+00],
             [7.82187053e-02, 9.87687893e-01, -1.35478634e-01, 0.00000000e+00],
             [-4.93844059e-01, 1.56437291e-01, 8.55362741e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025335e-01, 5.44865968e-08, 5.00000118e-01, 0.00000000e+00],
             [-7.82159161e-02, 9.87688778e-01, 1.35473790e-01, 0.00000000e+00],
             [-4.93844499e-01, -1.56431702e-01, 8.55363510e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025337e-01, 1.06291172e-07, 5.00000115e-01, 0.00000000e+00],
             [-2.26994148e-01, 8.91007784e-01, 3.93165087e-01, 0.00000000e+00],
             [-4.45503953e-01, -4.53988027e-01, 7.71635341e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025334e-01, 8.31178996e-08, 5.00000121e-01, 0.00000000e+00],
             [-3.53552543e-01, 7.07108749e-01, 6.12370652e-01, 0.00000000e+00],
             [-3.53554409e-01, -7.07104813e-01, 6.12374120e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025336e-01, 5.47984172e-08, 5.00000118e-01, 0.00000000e+00],
             [-4.45502752e-01, 4.53992999e-01, 7.71633109e-01, 0.00000000e+00],
             [-2.26996511e-01, -8.91005251e-01, 3.93169464e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025341e-01, 1.49156699e-08, 5.00000109e-01, 0.00000000e+00],
             [-4.93844058e-01, 1.56437257e-01, 8.55362748e-01, 0.00000000e+00],
             [-7.82186327e-02, -9.87687898e-01, 1.35478636e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025338e-01, -1.29877133e-08, 5.00000114e-01, 0.00000000e+00],
             [-4.93844506e-01, -1.56431668e-01, 8.55363512e-01, 0.00000000e+00],
             [7.82158406e-02, -9.87688784e-01, -1.35473794e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[8.66025341e-01, -5.45050422e-08, 5.00000109e-01, 0.00000000e+00],
             [-4.45504029e-01, -4.53987956e-01, 7.71635339e-01, 0.00000000e+00],
             [2.26993986e-01, -8.91007820e-01, -3.93165099e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106672e-01, -4.35939631e-08, 7.07106890e-01, 0.00000000e+00],
             [-5.00001540e-01, -7.07104743e-01, 5.00001342e-01, -0.00000000e+00],
             [4.99998614e-01, -7.07108819e-01, -4.99998504e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106670e-01, -5.52544745e-08, 7.07106893e-01, 0.00000000e+00],
             [-2.85617224e-01, -9.14792706e-01, 2.85617062e-01, -0.00000000e+00],
             [6.46856212e-01, -4.03923637e-01, -6.46856040e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106669e-01, -1.68940567e-08, 7.07106893e-01, 0.00000000e+00],
             [-3.26586630e-02, -9.98932843e-01, 3.26586288e-02, -0.00000000e+00],
             [7.06352299e-01, -4.61862999e-02, -7.06352076e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106672e-01, 2.97394765e-08, 7.07106890e-01, 0.00000000e+00],
             [2.24710654e-01, -9.48161533e-01, -2.24710545e-01, -0.00000000e+00],
             [6.70451547e-01, 3.17788777e-01, -6.70451353e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106673e-01, 5.99903707e-08, 7.07106890e-01, 0.00000000e+00],
             [4.51731499e-01, -7.69335746e-01, -4.51731295e-01, 0.00000000e+00],
             [5.44002579e-01, 6.38844668e-01, -5.44002466e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106675e-01, 3.06564824e-08, 7.07106888e-01, 0.00000000e+00],
             [6.17743458e-01, -4.86606921e-01, -6.17743251e-01, 0.00000000e+00],
             [3.44083086e-01, 8.73621030e-01, -3.44083021e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106677e-01, 1.08103441e-09, 7.07106885e-01, 0.00000000e+00],
             [7.00325760e-01, -1.38159139e-01, -7.00325554e-01, 0.00000000e+00],
             [9.76932779e-02, 9.90410042e-01, -9.76932507e-02, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106671e-01, -1.64166701e-08, 7.07106891e-01, 0.00000000e+00],
             [6.88325222e-01, 2.28947769e-01, -6.88325002e-01, 0.00000000e+00],
             [-1.61890534e-01, 9.73438709e-01, 1.61890506e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106673e-01, -4.23610226e-08, 7.07106890e-01, 0.00000000e+00],
             [5.83362562e-01, 5.65134047e-01, -5.83362349e-01, 0.00000000e+00],
             [-3.99610154e-01, 8.24999096e-01, 3.99610081e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106676e-01, -4.44615927e-08, 7.07106886e-01, 0.00000000e+00],
             [3.99613558e-01, 8.24995845e-01, -3.99613388e-01, 0.00000000e+00],
             [-5.83360226e-01, 5.65138793e-01, 5.83360087e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106675e-01, -3.88472995e-08, 7.07106888e-01, 0.00000000e+00],
             [1.61894504e-01, 9.73437400e-01, -1.61894402e-01, 0.00000000e+00],
             [-6.88324284e-01, 2.28953331e-01, 6.88324089e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106672e-01, 3.23226657e-08, 7.07106891e-01, 0.00000000e+00],
             [-9.76893638e-02, 9.90410819e-01, 9.76892882e-02, 0.00000000e+00],
             [-7.00326312e-01, -1.38153570e-01, 7.00326101e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106670e-01, 6.50865465e-08, 7.07106893e-01, 0.00000000e+00],
             [-3.44079709e-01, 8.73623739e-01, 3.44079520e-01, 0.00000000e+00],
             [-6.17745345e-01, -4.86602058e-01, 6.17745195e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106669e-01, 4.82454161e-08, 7.07106894e-01, 0.00000000e+00],
             [-5.44000123e-01, 6.38848940e-01, 5.43999906e-01, -0.00000000e+00],
             [-4.51734463e-01, -7.69332199e-01, 4.51734372e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106667e-01, 2.29245413e-08, 7.07106896e-01, 0.00000000e+00],
             [-6.70450308e-01, 3.17794075e-01, 6.70450080e-01, -0.00000000e+00],
             [-2.24714367e-01, -9.48159757e-01, 2.24714325e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106677e-01, 7.22923882e-09, 7.07106885e-01, 0.00000000e+00],
             [-7.06352476e-01, -4.61806309e-02, 7.06352269e-01, -0.00000000e+00],
             [3.26546472e-02, -9.98933106e-01, -3.26546274e-02, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.07106668e-01, -3.31554106e-08, 7.07106894e-01, 0.00000000e+00],
             [-6.46857880e-01, -4.03918380e-01, 6.46857655e-01, -0.00000000e+00],
             [2.85613449e-01, -9.14795028e-01, -2.85613401e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999888e-01, -2.14734474e-08, 8.66025468e-01, 0.00000000e+00],
             [-6.12374243e-01, -7.07104756e-01, 3.53554311e-01, -0.00000000e+00],
             [6.12370720e-01, -7.07108806e-01, -3.53552312e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999890e-01, -5.88496993e-09, 8.66025467e-01, 0.00000000e+00],
             [-2.24146271e-01, -9.65925089e-01, 1.29410866e-01, -0.00000000e+00],
             [8.36515726e-01, -2.58821798e-01, -4.82962439e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999890e-01, 1.03496171e-08, 8.66025467e-01, 0.00000000e+00],
             [2.24141542e-01, -9.65926552e-01, -1.29408130e-01, -0.00000000e+00],
             [8.36516992e-01, 2.58816335e-01, -4.82963173e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999877e-01, 2.10888136e-08, 8.66025475e-01, 0.00000000e+00],
             [6.12370783e-01, -7.07108756e-01, -3.53552303e-01, 0.00000000e+00],
             [6.12374189e-01, 7.07104806e-01, -3.53554304e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999879e-01, 4.88024892e-09, 8.66025474e-01, 0.00000000e+00],
             [8.36515743e-01, -2.58821757e-01, -4.82962432e-01, 0.00000000e+00],
             [2.24146232e-01, 9.65925100e-01, -1.29410851e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999883e-01, -1.32033357e-08, 8.66025472e-01, 0.00000000e+00],
             [8.36517006e-01, 2.58816307e-01, -4.82963164e-01, 0.00000000e+00],
             [-2.24141508e-01, 9.65926560e-01, 1.29408134e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999888e-01, -2.14734476e-08, 8.66025468e-01, 0.00000000e+00],
             [6.12374243e-01, 7.07104756e-01, -3.53554311e-01, 0.00000000e+00],
             [-6.12370720e-01, 7.07108806e-01, 3.53552312e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999890e-01, -5.88497003e-09, 8.66025467e-01, 0.00000000e+00],
             [2.24146271e-01, 9.65925089e-01, -1.29410866e-01, 0.00000000e+00],
             [-8.36515726e-01, 2.58821798e-01, 4.82962439e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999890e-01, 1.03496170e-08, 8.66025467e-01, 0.00000000e+00],
             [-2.24141542e-01, 9.65926552e-01, 1.29408130e-01, 0.00000000e+00],
             [-8.36516992e-01, -2.58816335e-01, 4.82963173e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999877e-01, 2.10888135e-08, 8.66025475e-01, 0.00000000e+00],
             [-6.12370783e-01, 7.07108756e-01, 3.53552303e-01, -0.00000000e+00],
             [-6.12374189e-01, -7.07104806e-01, 3.53554304e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999879e-01, 4.88024905e-09, 8.66025474e-01, 0.00000000e+00],
             [-8.36515743e-01, 2.58821757e-01, 4.82962432e-01, -0.00000000e+00],
             [-2.24146232e-01, -9.65925100e-01, 1.29410851e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[4.99999883e-01, -1.32033356e-08, 8.66025472e-01, 0.00000000e+00],
             [-8.36517006e-01, -2.58816307e-01, 4.82963164e-01, -0.00000000e+00],
             [2.24141508e-01, -9.65926560e-01, -1.29408134e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818972e-01, 6.33160309e-09, 9.65925846e-01, 0.00000000e+00],
             [-6.83014657e-01, -7.07104771e-01, 1.83013175e-01, -0.00000000e+00],
             [6.83010775e-01, -7.07108792e-01, -1.83012126e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818969e-01, -1.35951157e-09, 9.65925847e-01, 0.00000000e+00],
             [2.49997373e-01, -9.65926556e-01, -6.69865745e-02, -0.00000000e+00],
             [9.33013427e-01, 2.58816320e-01, -2.50000115e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818972e-01, -3.02503072e-09, 9.65925846e-01, 0.00000000e+00],
             [9.33012015e-01, -2.58821773e-01, -2.49999741e-01, 0.00000000e+00],
             [2.50002640e-01, 9.65925095e-01, -6.69879822e-02, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818972e-01, 6.33160300e-09, 9.65925846e-01, 0.00000000e+00],
             [6.83014657e-01, 7.07104771e-01, -1.83013175e-01, 0.00000000e+00],
             [-6.83010775e-01, 7.07108792e-01, 1.83012126e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818969e-01, -1.35951156e-09, 9.65925847e-01, 0.00000000e+00],
             [-2.49997373e-01, 9.65926556e-01, 6.69865745e-02, 0.00000000e+00],
             [-9.33013427e-01, -2.58816320e-01, 2.50000115e-01, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[2.58818972e-01, -3.02503058e-09, 9.65925846e-01, 0.00000000e+00],
             [-9.33012015e-01, 2.58821773e-01, 2.49999741e-01, -0.00000000e+00],
             [-2.50002640e-01, -9.65925095e-01, 6.69879822e-02, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
            [[7.91668712e-17, 1.65761784e-16, 1.00000000e+00, 0.00000000e+00],
             [-7.07108780e-01, -7.07104782e-01, 1.73190540e-16, -0.00000000e+00],
             [7.07104782e-01, -7.07108780e-01, 6.12323400e-17, -0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
        ]
        
        npArrays = []

        for matrix in matrices:
            
            npArrays.append(np.array(matrix))

        return npArrays

class TestXmippRotateVolume(TestXmippBase):
    """This class checks if the protocol rotate volume in Xmipp works properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testXmippAlignVolumeAndParticles(self):
        # Create input data: phantom with one cylinder
        protCreatePhantomRotated = self.newProtocol(XmippProtPhantom,
                                               desc='80 80 80 0\ncyl + 5 0 0 0 5 5 10 0 90 0',
                                               sampling=1.0)
        self.launchProtocol(protCreatePhantomRotated)
        self.assertIsNotNone(protCreatePhantomRotated.getFiles(),
                             "There was a problem with the rotated phantom creation")

        # First type of rotation (Align with Z)
        protRotateVolume = self.newProtocol(XmippProtRotateVolume,
                                            vol=protCreatePhantomRotated.outputVolume,
                                            rotType=0,
                                            dirParam=0)
        self.launchProtocol(protRotateVolume)
        self.assertIsNotNone(protRotateVolume.getFiles(),
                             MSG_WRONG_ROTATION)

        # Second type of rotation (rotate)
        protRotateVolume2 = self.newProtocol(XmippProtRotateVolume,
                                            vol=protCreatePhantomRotated.outputVolume,
                                            rotType=1,
                                            dirParam=1,
                                            deg=90)
        self.launchProtocol(protRotateVolume2)
        self.assertIsNotNone(protRotateVolume2.getFiles(),
                             MSG_WRONG_ROTATION)

        # Create the referenced cylinder (without rotation)
        # This new cylinder is the reference, meaning that it is created to check visually that
        # the results of volume rotation protocols should look like this one
        protCreatePhantomReference = self.newProtocol(XmippProtPhantom,
                                                      desc='80 80 80 0\ncyl + 5 0 0 0 5 5 10 0 0 0',
                                                      sampling=1.0)
        self.launchProtocol(protCreatePhantomReference)
        self.assertIsNotNone(protCreatePhantomReference.getFiles(),
                             "There was a problem with the referenced phantom creation")
        # First type of rotation checked (Align with Z)
        self.assertEqual(protRotateVolume.rotType.get(), 0, "The phantom is not aligning with the Z axis")
        self.assertEqual(protRotateVolume.dirParam.get(), 0, "The phantom is rotating in the wrong axis")
        self.assertEqual(protRotateVolume.outputVolume.getDim(), protCreatePhantomReference.outputVolume.getDim(),
                         (MSG_WRONG_DIM, "phantom (initially rotated)"))
        self.assertEqual(protRotateVolume.outputVolume.getSamplingRate(),
                         protCreatePhantomReference.outputVolume.getSamplingRate(),
                         (MSG_WRONG_SAMPLING, "rotated phantom"))
        # Second type of rotation checked (rotate)
        self.assertEqual(protRotateVolume2.rotType.get(), 1, "The phantom is not rotating")
        self.assertEqual(protRotateVolume2.dirParam.get(), 1, "The phantom is rotating in the wrong axis")
        self.assertEqual(protRotateVolume2.deg.get(), 90, "The degree of rotation is wrong")
        self.assertEqual(protRotateVolume2.outputVolume.getDim(), protCreatePhantomReference.outputVolume.getDim(),
                         (MSG_WRONG_DIM, "phantom (initially rotated)"))
        self.assertEqual(protRotateVolume2.outputVolume.getSamplingRate(),
                         protCreatePhantomReference.outputVolume.getSamplingRate(),
                         (MSG_WRONG_SAMPLING, "rotated phantom"))

class TestXmippDeepHand(TestXmippBase):
    """This class checks if the protocol deep hand in Xmipp works properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet(db_general)
        cls.vol1 = cls.dataset.getFile(helix)

    def testXmippDeepHand(self):
        # Import input data
        protImportVol = self.newProtocol(ProtImportVolumes,
                                         objLabel='Volume',
                                         filesPath=self.vol1,
                                         samplingRate=7.08)
        self.launchProtocol(protImportVol)
        # Check if there is an output
        self.assertIsNotNone(protImportVol.getFiles(),
                             "There was a problem with the volume import")

        # Creation of the mask
        protDeepHand = self.newProtocol(XmippProtDeepHand,
                                        inputVolume=protImportVol.outputVolume,
                                        threshold=0.05)
        self.launchProtocol(protDeepHand)
        # Check if there is an output
        self.assertIsNotNone(protDeepHand.getFiles(), "There was a problem with the mask creation")

        # Check if the sampling rate is right
        self.assertEqual(protDeepHand.outputVol.getSamplingRate(), 7.08, (MSG_WRONG_SAMPLING, "volume"))
        # Check if the input threshold is the same as the density of the volume
        self.assertEqual(protDeepHand.threshold.get(), 0.05, "There was a problem with the density value")
        # Check if the thresholdAlpha and thresholdHand match the default values
        self.assertEqual(protDeepHand.thresholdAlpha.get(), 0.7, "There was a problem with the thresholdAlpha value")
        self.assertEqual(protDeepHand.thresholdHand.get(), 0.6, "There was a problem with the thresholdHand value")
        # Check if the dimension of the volume does not vary
        self.assertEqual(protDeepHand.outputVol.getDim(), protImportVol.outputVolume.getDim(),
                         (MSG_WRONG_DIM, "volume"))
        # Check if the hand value is right
        self.assertAlmostEquals(protDeepHand.outputHand.get(), 0.380511, 6,"There was a problem with the hand value")
        # Check if the flip is right
        self.assertTrue(protDeepHand.outputHand.get()<protDeepHand.thresholdHand.get(), "There was a problem with the flip")

class TestXmippResolutionBfactor(TestXmippBase):
    """This class checks if the protocol resolution b factor in Xmipp works properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet(db_model_building_tutorial)
        cls.vol1 = cls.dataset.getFile(vol_coot1)
        cls.pdb = cls.dataset.getFile(pdb_coot1)

    def testXmippResolutionBfactor(self):
        # Import input volume
        print("Import input volume")
        protImportVol = self.newProtocol(ProtImportVolumes,
                                         objLabel='Volume',
                                         filesPath=self.vol1,
                                         samplingRate=4)
        self.launchProtocol(protImportVol)
        # Check if there is an output volume
        self.assertIsNotNone(protImportVol.getFiles(),
                             (MSG_WRONG_IMPORT, "volume"))

        # Create a mask from the volume
        print("Create a mask from the volume")
        protCreateMask = self.newProtocol(XmippProtCreateMask3D,
                                          inputVolume=protImportVol.outputVolume,
                                          threshold=0.09)
        self.launchProtocol(protCreateMask)
        # Check if there is an output 3D mask
        self.assertIsNotNone(protCreateMask.getFiles(),
                             MSG_WRONG_MASK)

        # Create a map
        print("Create a map")
        protCreateMap = self.newProtocol(XmippProtMonoRes,
                                         fullMap=protImportVol.outputVolume,
                                         mask=protCreateMask.outputMask,
                                         maxRes=10)
        self.launchProtocol(protCreateMap)
        # Check if there is an output map
        self.assertIsNotNone(protCreateMap.getFiles(),
                             MSG_WRONG_MAP)

        # Import atomic structure
        print("Import atomic structure")
        protImportPdb = self.newProtocol(ProtImportPdb,
                                         inputPdbData=ProtImportPdb.IMPORT_FROM_FILES,
                                         pdbFile=self.pdb,
                                         inputVolume=protImportVol.outputVolume)
        self.launchProtocol(protImportPdb)
        # Check if there is an output atomic structure
        self.assertIsNotNone(protImportPdb.getFiles(),
                             (MSG_WRONG_IMPORT, "atomic structure"))

        # Protocol local resolution/local bfactor
        print("Protocol local resolution/local bfactor")
        protbfactorResolution = self.newProtocol(XmippProtbfactorResolution,
                                                 pdbfile=protImportPdb.outputPdb,
                                                 localResolutionMap=protCreateMap.resolution_Volume,
                                                 fscResolution=8.35)
        self.launchProtocol(protbfactorResolution)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        className = sys.argv[1]
        cls = globals().get(className, None)
        if cls:
            suite = unittest.TestLoader().loadTestsFromTestCase(cls)
            unittest.TextTestRunner(verbosity=2).run(suite)
        else:
            print("Test: '%s' not found." % className)
    else:
        unittest.main()
