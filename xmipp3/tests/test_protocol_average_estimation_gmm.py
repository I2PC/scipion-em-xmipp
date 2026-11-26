# ***************************************************************************
# * Authors:     Andres Contreras (andres.contreras@cnb.csic.es)
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
# ***************************************************************************/

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import mrcfile
import numpy as np
from pwem.objects import (
    CTFModel,
    Particle,
    SetOfClasses2D,
    SetOfParticles,
    Transform,
)

from xmipp3.protocols.protocol_average_estimation_gmm import (
    XmippProtAverageEstimationGmm,
)


class TestAverageEstimationGmm(TestCase):
    def test_outputs_keep_original_particle_location_ctf_and_alignment(self):
        with TemporaryDirectory() as tmpdir, ExitStack() as cleanup:
            root = Path(tmpdir)
            protocol = XmippProtAverageEstimationGmm(gmmReweighting=False)
            protocol.setWorkingDir(str(root / "run"))
            protocol.makeWorkingDir()

            original_stack = root / "original.mrcs"
            mrcfile.write(original_stack, np.zeros((2, 8, 8), dtype=np.float32))
            location = (2, str(original_stack))
            matrix = np.array(
                [[0, -1, 0, 2.5], [1, 0, 0, -1.25], [0, 0, 1, 0], [0, 0, 0, 1]],
            )
            defocus = (15000.0, 13500.0, 23.0)
            phase_shift = 0.25
            particle = Particle(objId=42, location=location)
            particle.setClassId(9)
            particle.setTransform(Transform(matrix=matrix.copy()))
            ctf = CTFModel()
            ctf.setStandardDefocus(*defocus)
            ctf.setPhaseShift(phase_shift)
            particle.setCTF(ctf)

            particles = SetOfParticles(filename=str(root / "input_particles.sqlite"))
            cleanup.callback(particles.close)
            particles.setSamplingRate(1.5)
            particles.setAlignment2D()
            particles.setHasCTF(True)
            particles.append(particle)
            particles.write()

            classes = SetOfClasses2D(filename=str(root / "input_classes.sqlite"))
            cleanup.callback(classes.close)
            classes.setImages(particles)
            classes.classifyItems()
            classes.write()
            protocol.inputClasses.set(classes)

            # Model the metadata after CTF correction and applying the alignment.
            # It points to the temporary stack and has a zero residual transform.
            Path(protocol._getPreprocessedMetadataPath()).write_text(
                "data_\n\nloop_\n_itemId\n_ref\n_image\n_anglePsi\n_shiftX\n_shiftY\n"
                f"42 9 1@{protocol._getPreprocessedParticlesPath()} 0 0 0\n"
            )
            Path(protocol._getOutputStarPath()).write_text(
                "data_\n\nloop_\n_itemId\n_ref\n_wRobust\n_wRobustStd\n42 9 0.8 0\n"
            )

            with patch.object(protocol, "_defineSourceRelation"):
                protocol.createOutputStep()

            for output_name in (
                "outputClasses_robust",
                "outputClasses_standard",
                "outputParticles",
            ):
                with self.subTest(output=output_name):
                    output = getattr(protocol, output_name)
                    output.write()
                    output.close()
                    reopened = type(output)(filename=output.getFileName())
                    cleanup.callback(reopened.close)
                    reopened.loadAllProperties()
                    if isinstance(reopened, SetOfClasses2D):
                        result = reopened[9][42]
                    else:
                        result = reopened[42]
                    self.assertEqual(result.getLocation(), location)
                    np.testing.assert_array_equal(
                        result.getTransform().getMatrix(), matrix
                    )
                    self.assertEqual(result.getCTF().getDefocus(), defocus)
                    self.assertEqual(result.getCTF().getPhaseShift(), phase_shift)
