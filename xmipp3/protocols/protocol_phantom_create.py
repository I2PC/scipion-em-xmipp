# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *
# *  BCU, Centro Nacional de Biotecnologia, CSIC
# *
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
from pwem.objects.data import Volume
from pwem.protocols import EMProtocol
from pyworkflow.protocol.params import FloatParam, TextParam


class XmippProtPhantom(EMProtocol):
    """ Creates a phantom volume based on a feature description file by using
    the xmipp_phantom_create tool. This synthetic volume can be used to test
    and validate algorithms or processing pipelines under controlled conditions.

    AI Generated

    ## Overview

    The Phantom Volume protocol creates a synthetic 3D volume from a textual
    description of simple geometric objects.

    A phantom is an artificial volume designed by the user. Instead of coming from
    experimental cryo-EM data, it is generated from known shapes such as spheres,
    cylinders, or other supported geometric primitives. Because the contents of the
    volume are known in advance, phantoms are useful for testing, debugging,
    teaching, and validating image-processing workflows.

    This protocol uses the Xmipp phantom-generation tool to convert a phantom
    description into a 3D volume. The resulting volume can then be used in Scipion
    like any other volume object.

    ## Inputs and General Workflow

    The protocol requires two inputs:

    - a phantom description;
    - a sampling rate.

    The phantom description is written as text. It defines the size of the volume,
    the background value, and the geometric objects that should be inserted into
    the volume.

    When the protocol runs, it writes this description to a temporary description
    file and calls the Xmipp phantom-creation program. The resulting volume is then
    registered in Scipion as the output volume, with the sampling rate specified by
    the user.

    ## Phantom Description

    The **Create phantom** field contains the textual description of the synthetic
    volume.

    The first line defines the dimensions of the volume and the background value.
    For example, a line such as:

    \[
    40\; 40\; 40\; 0
    \]

    defines a volume of size 40 by 40 by 40 voxels with a background value of 0.

    The following lines define the objects inserted into the volume. Each object
    line describes a geometric primitive, its density, position, size, and
    orientation.

    The default example contains cylinders and spheres. These objects are combined
    to create a simple synthetic structure.

    ## Geometric Objects

    The phantom description can include geometric primitives such as cylinders and
    spheres.

    Each object line specifies what type of object is created and how it is placed
    inside the volume. The parameters include information such as:

    - object type;
    - whether it is added to the existing density;
    - density value;
    - origin or position;
    - radius or radii;
    - height when relevant;
    - rotation angles.

    The exact interpretation depends on the type of geometric object. Users who
    want to create complex phantoms should follow the Xmipp phantom description
    format.

    ## Density Values

    Each object in the phantom description has a density value.

    This value determines the intensity assigned to that object in the synthetic
    volume. Objects can be superimposed on the background or on other objects,
    depending on the operation specified in the description.

    Because the user controls the density values, phantoms can be designed to test
    specific situations, such as high-contrast objects, weak densities, overlapping
    features, or simple geometric arrangements.

    ## Object Position and Orientation

    The phantom description defines where each object is placed and how it is
    oriented.

    For example, spheres are mainly controlled by their center and radius, whereas
    cylinders also require orientation information. Rotations can be used to place
    objects at different angles inside the volume.

    This makes it possible to build simple synthetic structures with known
    geometry. Such structures are useful for testing projection, alignment,
    filtering, reconstruction, and visualization protocols.

    ## Sampling Rate

    The **Sampling rate** parameter assigns a physical pixel size to the generated
    volume.

    The phantom itself is created in voxels, but the sampling rate tells Scipion
    how many angstroms each voxel represents. For example, a sampling rate of 4
    means that each voxel corresponds to 4 Å.

    This value is important if the phantom will be used in protocols that depend
    on physical units, such as resolution filtering, projection generation, or
    comparison with other volumes.

    The sampling rate does not change the generated voxel values or dimensions. It
    only defines their physical scale.

    ## Output Volume

    The main output is **outputVolume**.

    This output is the synthetic phantom volume generated from the description
    provided by the user. It is stored as a Scipion Volume object with the selected
    sampling rate.

    The output volume can be visualized, projected, filtered, compared, or used as
    a test reference in other workflows.

    Because the phantom is synthetic, its structure and density distribution are
    known by construction. This makes it useful for controlled experiments.

    ## Typical Uses

    Phantom volumes are useful in several situations.

    They can be used to test whether a protocol behaves as expected on a known
    input. For example, one can project a phantom, reconstruct it, and check
    whether the recovered volume matches the original.

    They can also be used for teaching, because simple geometric structures are
    easier to understand than noisy experimental cryo-EM maps.

    In method development, phantoms are useful for debugging algorithms under
    controlled conditions before moving to real data.

    They can also help illustrate the effect of filtering, masking, projection,
    alignment, or reconstruction parameters.

    ## Practical Recommendations

    Use simple phantoms when testing a workflow for the first time. A small number
    of spheres or cylinders is easier to interpret than a complex synthetic model.

    Choose the volume size large enough to contain all objects without clipping
    them at the borders.

    Use meaningful density values. Very low contrast may be useful for testing
    difficult cases, but high-contrast objects are better for initial debugging.

    Set the sampling rate according to the physical scale you want the phantom to
    represent. This is especially important if later protocols use resolution in
    angstroms.

    Inspect the generated volume before using it in a longer workflow. This helps
    confirm that the description was interpreted as intended.

    ## Final Perspective

    The Phantom Volume protocol creates controlled synthetic 3D data.

    For biological users, it is not intended to represent an experimental
    structure directly. Its value lies in testing and understanding processing
    steps under known conditions.

    For developers, advanced users, and educators, phantom volumes provide a simple
    way to generate reproducible test data with known geometry, density, and
    sampling.
    """

    _label = 'phantom volume'

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('desc', TextParam, label='Create phantom',
                      default='40 40 40 0\ncyl + 1 0 0 0 15 15 2 0 0 0\nsph + 1 0 0 5 2\ncyl + 1 0 0 -5 2 2 10 0 90 0\n'
                              'sph + 1 0 -5 5 2',
                      help="create a phantom description: x y z backgroundValue geometry(cyl, sph...) +(superimpose) "
                           "desnsityValue origin radius height rot tilt psi. See more information in "
                           "https://web.archive.org/web/20180813105422/http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/FileFormats#Phantom_metadata_file")
        form.addParam('sampling', FloatParam, label='Sampling rate', default=4)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('createPhantomsStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def createPhantomsStep(self):
        desc = self.desc.get()
        fnDescr = self._getExtraPath("phantom.descr")
        fhDescr = open(fnDescr, 'w')
        fhDescr.write(desc)
        fhDescr.close()
        self.runJob("xmipp_phantom_create", " -i %s -o %s" % (fnDescr, self._getExtraPath("phantom.vol")))

    def createOutputStep(self):
        outputVol = Volume()
        outputVol.setLocation(self._getExtraPath("phantom.vol"))
        outputVol.setSamplingRate(self.sampling.get())
        self._defineOutputs(outputVolume=outputVol)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputVolume'):
            summary.append("Output phantom not ready yet.")
        else:
            summary.append("Phantoms created")
        return summary
