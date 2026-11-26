from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import mrcfile
import numpy as np
from pwem.objects import Particle, SetOfClasses2D, SetOfParticles, Transform

from xmipp3.protocols.protocol_cone_averaging import XmippProtConeAveraging


class TestConeAveraging(TestCase):
    def test_outputs_use_winning_cone_memberships_and_fitted_stack_indices(self):
        for use_gmm in (False, True):
            with self.subTest(use_gmm=use_gmm):
                self._check_outputs(use_gmm)

    def _check_outputs(self, use_gmm):
        with TemporaryDirectory() as tmpdir, ExitStack() as cleanup:
            root = Path(tmpdir)
            protocol = XmippProtConeAveraging(gmmReweighting=use_gmm)
            protocol.setWorkingDir(str(root / "run"))
            protocol.makeWorkingDir()

            stack = root / "original.mrcs"
            mrcfile.write(stack, np.zeros((3, 8, 8), dtype=np.float32))
            particles = SetOfParticles(filename=str(root / "input.sqlite"))
            cleanup.callback(particles.close)
            particles.setSamplingRate(1.5)
            particles.setAlignmentProj()
            for index, item_id in enumerate((42, 43, 44), start=1):
                particle = Particle(objId=item_id, location=(index, str(stack)))
                transform = Transform()
                transform.setShifts(float(index), 0.0, 0.0)
                particle.setTransform(transform)
                particles.append(particle)
            particles.write()
            protocol.inputParticles.set(particles)

            # Cone 5 was fitted but has no winners. Particle 43 ties in either mode.
            columns = ["itemId", "cone_group", "wRobust", "wRobustStd"]
            rows = [
                ("42", "9", "0.3", "-0.75", "0.9"),
                ("43", "9", "0.5", "0.25", "8e-1"),
                ("42", "5", "0.1", "-1.0", "0.05"),
                ("44", "9", "0.7", "1.0", "0.7"),
                ("42", "2", "0.8", "0.5", "1e-1"),
                ("43", "2", "0.5", "-0.25", "0.8"),
            ]
            if use_gmm:
                columns.append("wRobustGmm")
            metadata = "data_\n\nloop_\n" + "".join(f"_{col}\n" for col in columns)
            metadata += "".join(f"{' '.join(row[:len(columns)])}\n" for row in rows)
            Path(protocol._getAveragingOutputStarPath()).write_text(metadata)

            with patch.object(protocol, "_defineSourceRelation"):
                protocol.createOutputStep()

            output = protocol.outputParticles
            output.close()
            flat = SetOfParticles(filename=output.getFileName())
            cleanup.callback(flat.close)
            flat.loadAllProperties()
            self.assertEqual(flat.getIdSet(), {42, 43, 44})
            expected = {
                42: (2, 0.8, 0.5, 0.1),
                43: (2, 0.5, -0.25, 0.8),
                44: (9, 0.7, 1.0, 0.7),
            }
            if use_gmm:
                expected[42] = (9, 0.3, -0.75, 0.9)
            for item_id, (group, robust, standardized, gmm) in expected.items():
                winner = flat[item_id]
                original = particles[item_id]
                self.assertEqual(winner.getClassId(), group)
                self.assertEqual(winner.getLocation(), original.getLocation())
                np.testing.assert_array_equal(
                    winner.getTransform().getMatrix(),
                    original.getTransform().getMatrix(),
                )
                self.assertAlmostEqual(winner._xmippRobustWeight.get(), robust)
                self.assertAlmostEqual(
                    winner._xmippRobustWeightStandardized.get(), standardized
                )
                if use_gmm:
                    self.assertAlmostEqual(winner._xmippRobustWeightGmm.get(), gmm)

            for output, averages_path in (
                (
                    protocol.outputClasses_robust,
                    protocol._getCorrectedConeAveragesPath(),
                ),
                (protocol.outputClasses_standard, protocol._getRawConeAveragesPath()),
            ):
                output.close()
                classes = SetOfClasses2D(filename=output.getFileName())
                cleanup.callback(classes.close)
                classes.loadAllProperties()
                self.assertEqual(classes.getIdSet(), {2, 9})
                for group, stack_index in ((2, 1), (9, 3)):
                    output_class = classes[group]
                    expected_ids = {
                        item_id
                        for item_id, record in expected.items()
                        if record[0] == group
                    }
                    self.assertEqual(output_class.getIdSet(), expected_ids)
                    self.assertEqual(
                        output_class.getRepresentative().getLocation(),
                        (stack_index, averages_path),
                    )
