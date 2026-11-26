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
from pwem.constants import ALIGN_2D
from pwem.objects import (
    Class2D,
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
    def test_outputs_keep_original_particle_metadata_for_each_membership(self):
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
            matrices = {2: matrix, 9: matrix.copy()}
            matrices[9][0, 3] = -2.0
            weights = {2: (0.8, 0.5), 9: (0.3, -0.75)}
            defocus = (15000.0, 13500.0, 23.0)
            phase_shift = 0.25
            particle = Particle(objId=42, location=location)
            ctf = CTFModel()
            ctf.setStandardDefocus(*defocus)
            ctf.setPhaseShift(phase_shift)
            particle.setCTF(ctf)

            particles = SetOfParticles(filename=str(root / "input_particles.sqlite"))
            cleanup.callback(particles.close)
            particles.setSamplingRate(1.5)
            particles.setHasCTF(True)
            particles.append(particle)
            particles.write()

            classes = SetOfClasses2D(filename=str(root / "input_classes.sqlite"))
            cleanup.callback(classes.close)
            classes.setImages(particles)
            for class_id in (2, 5, 9):
                input_class = Class2D(objId=class_id)
                input_class.copyInfo(particles)
                input_class.setAlignment2D()
                classes.append(input_class)
                if class_id in matrices:
                    member = particle.clone()
                    member.setClassId(class_id)
                    member.setTransform(Transform(matrix=matrices[class_id].copy()))
                    input_class.append(member)
                    classes.update(input_class)
            classes.write()
            protocol.inputClasses.set(classes)

            # Model the metadata after CTF correction and applying the alignment.
            # It points to the temporary stack and has a zero residual transform.
            Path(protocol._getPreprocessedMetadataPath()).write_text(
                "data_\n\nloop_\n_itemId\n_ref\n_image\n_anglePsi\n_shiftX\n_shiftY\n"
                f"42 2 1@{protocol._getPreprocessedParticlesPath()} 0 0 0\n"
                f"42 9 2@{protocol._getPreprocessedParticlesPath()} 0 0 0\n"
            )
            Path(protocol._getOutputStarPath()).write_text(
                "data_\n\nloop_\n_itemId\n_ref\n_wRobust\n_wRobustStd\n"
                "42 9 0.3 -0.75\n42 2 0.8 0.5\n"
            )

            with patch.object(protocol, "_defineSourceRelation"):
                protocol.createOutputStep()

            for output_name, averages_path in (
                ("outputClasses_robust", protocol._getCorrectedAveragesPath()),
                ("outputClasses_standard", protocol._getOriginalAveragesPath()),
            ):
                with self.subTest(output=output_name):
                    output = getattr(protocol, output_name)
                    output.close()
                    reopened = SetOfClasses2D(filename=output.getFileName())
                    cleanup.callback(reopened.close)
                    reopened.loadAllProperties()
                    self.assertEqual(reopened.getIdSet(), {2, 9})
                    for stack_index, class_id in enumerate((2, 9), start=1):
                        with self.subTest(class_id=class_id):
                            output_class = reopened[class_id]
                            self.assertEqual(output_class.getSize(), 1)
                            self.assertEqual(output_class.getIdSet(), {42})
                            self.assertEqual(
                                output_class.getRepresentative().getLocation(),
                                (stack_index, averages_path),
                            )
                            result = output_class[42]
                            self.assertEqual(result.getClassId(), class_id)
                            self.assertEqual(result.getLocation(), location)
                            np.testing.assert_array_equal(
                                result.getTransform().getMatrix(), matrices[class_id]
                            )
                            result_ctf = result.getCTF()
                            self.assertEqual(result_ctf.getDefocus(), defocus)
                            self.assertEqual(result_ctf.getPhaseShift(), phase_shift)
                            robust, standardized = weights[class_id]
                            self.assertAlmostEqual(
                                result._xmippRobustWeight.get(), robust
                            )
                            self.assertAlmostEqual(
                                result._xmippRobustWeightStandardized.get(),
                                standardized,
                            )

            protocol.outputParticles.close()
            flat = SetOfParticles(filename=protocol.outputParticles.getFileName())
            cleanup.callback(flat.close)
            flat.loadAllProperties()
            self.assertEqual(flat.getAlignment(), ALIGN_2D)
            self.assertEqual(flat.getIdSet(), {42})
            winner = flat[42]
            self.assertEqual(winner.getClassId(), 2)
            self.assertEqual(winner.getLocation(), location)
            np.testing.assert_array_equal(winner.getTransform().getMatrix(), matrices[2])
            self.assertEqual(winner.getCTF().getDefocus(), defocus)
            self.assertEqual(winner.getCTF().getPhaseShift(), phase_shift)
            robust, standardized = weights[2]
            self.assertAlmostEqual(winner._xmippRobustWeight.get(), robust)
            self.assertAlmostEqual(winner._xmippRobustWeightStandardized.get(), standardized)
