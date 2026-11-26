from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import mrcfile
import numpy as np
import pwem.emlib.metadata as md
from pwem.objects import CTFModel, Particle, SetOfClasses2D, SetOfParticles, Transform

from xmipp3.convert import xmippToLocation
from xmipp3.protocols.protocol_average_estimation_gmm import XmippProtAverageEstimationGmm
from xmipp3.protocols.protocol_cone_averaging import GroupingModes, XmippProtConeGrouping


class TestConeGrouping(TestCase):
    def test_overlapping_cone_memberships_feed_robust_averaging(self):
        with TemporaryDirectory() as tmpdir, ExitStack() as cleanup:
            root = Path(tmpdir)
            protocol = XmippProtConeGrouping(groupingMode=GroupingModes.THRESHOLD.value)
            protocol.setWorkingDir(str(root / "grouping"))
            protocol.makeWorkingDir()

            stack = root / "original.mrcs"
            mrcfile.write(stack, np.zeros((2, 8, 8), dtype=np.float32))
            location = (2, str(stack))
            defocus = (15000.0, 13500.0, 23.0)
            phaseShift = 0.25
            particles = SetOfParticles(filename=str(root / "input.sqlite"))
            cleanup.callback(particles.close)
            particles.setSamplingRate(1.5)
            particles.setAlignmentProj()
            particles.setHasCTF(True)
            particle = Particle(objId=42, location=location)
            particle.setTransform(Transform())
            ctf = CTFModel()
            ctf.setStandardDefocus(*defocus)
            ctf.setResolution(4.0)
            ctf.setFitQuality(0.8)
            ctf.setPhaseShift(phaseShift)
            particle.setCTF(ctf)
            particles.append(particle)
            particles.write()
            protocol.inputParticles.set(particles)

            Path(protocol._getGroupingOutputStarPath()).write_text(
                "data_\n\nloop_\n_itemId\n_cone_group\n_angleRot\n_angleTilt\n"
                "_anglePsi\n_shiftX\n_shiftY\n_flip\n"
                "42 3 70 25 -90 -2.0 3.0 1\n"
                "42 1 70 25 90 2.5 -1.25 0\n"
            )
            Path(protocol._getGroupingReferenceMetadataPath()).write_text(
                "data_\n\nloop_\n_cone_group\n_n_particles\n1 1\n2 0\n3 1\n"
            )

            with patch.object(protocol, "_defineSourceRelation") as sourceRelation:
                protocol.createOutputStep()
            output = protocol.outputClasses
            sourceRelation.assert_called_once_with(protocol.inputParticles, output)
            self.assertEqual(
                [name for name, _ in protocol.iterOutputAttributes()], ["outputClasses"]
            )
            self.assertEqual(output.getImages().getFileName(), particles.getFileName())
            output.close()

            classes = SetOfClasses2D(filename=output.getFileName())
            cleanup.callback(classes.close)
            classes.loadAllProperties()
            self.assertEqual(classes.getIdSet(), {1, 3})
            self.assertFalse(classes.hasRepresentatives())
            expectedMatrices = {
                1: [[0, 1, 0, 2.5], [-1, 0, 0, -1.25], [0, 0, 1, 0], [0, 0, 0, 1]],
                3: [[0, 1, 0, -2], [1, 0, 0, 3], [0, 0, -1, 0], [0, 0, 0, 1]],
            }
            for cone in classes:
                with self.subTest(cone=cone.getObjId()):
                    self.assertTrue(cone.hasAlignment2D())
                    self.assertFalse(cone.hasRepresentative())
                    self.assertEqual(cone.getIdSet(), {42})
                    member = cone[42]
                    self.assertEqual(member.getClassId(), cone.getObjId())
                    self.assertEqual(member.getLocation(), location)
                    self.assertEqual(member.getCTF().getDefocus(), defocus)
                    self.assertAlmostEqual(member.getCTF().getPhaseShift(), phaseShift)
                    np.testing.assert_allclose(
                        member.getTransform().getMatrix(),
                        expectedMatrices[cone.getObjId()],
                        atol=1e-12,
                    )

            robust = XmippProtAverageEstimationGmm()
            robust.setWorkingDir(str(root / "robust"))
            robust.makeWorkingDir()
            robust.inputClasses.set(classes)
            robust.convertInputStep()

            metadata = md.MetaData(str(robust._getInputParticlesPath()))
            self.assertEqual(metadata.size(), 2)
            expectedGeometry = {1: (90, 2.5, -1.25, False), 3: (-90, -2, 3, True)}
            exportedMemberships = set()
            for row in md.iterRows(metadata):
                coneId = row.getValue(md.MDL_REF)
                itemId = row.getValue(md.MDL_ITEM_ID)
                exportedMemberships.add((itemId, coneId))
                with self.subTest(exportedCone=coneId):
                    psi, shiftX, shiftY, flip = expectedGeometry[coneId]
                    self.assertEqual(xmippToLocation(row.getValue(md.MDL_IMAGE)), location)
                    self.assertEqual(
                        (
                            row.getValue(md.MDL_CTF_DEFOCUSU),
                            row.getValue(md.MDL_CTF_DEFOCUSV),
                            row.getValue(md.MDL_CTF_DEFOCUS_ANGLE),
                        ),
                        defocus,
                    )
                    self.assertAlmostEqual(row.getValue(md.MDL_CTF_PHASE_SHIFT), phaseShift)
                    self.assertAlmostEqual(row.getValue(md.MDL_ANGLE_PSI), psi)
                    self.assertAlmostEqual(row.getValue(md.MDL_SHIFT_X), shiftX)
                    self.assertAlmostEqual(row.getValue(md.MDL_SHIFT_Y), shiftY)
                    self.assertEqual(row.getValue(md.MDL_FLIP), flip)
            self.assertEqual(exportedMemberships, {(42, 1), (42, 3)})
