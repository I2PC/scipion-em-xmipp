# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

import unittest

from xmipp3.protocols.protocol_movie_dose_analysis import XmippProtMovieDoseAnalysis


class TestMovieDoseAnalysisRegression(unittest.TestCase):
    """Regression tests for Movie Dose Analysis streaming/resume handling."""

    def testGetNewDoneIdsDoesNotStallOnOutOfOrderParallelCompletion(self):
        # Regression test: this protocol runs with parallel step execution
        # (STEPS_PARALLEL, batched), so movies do not necessarily finish
        # processing in id order. A single movie that is still pending (or
        # permanently stuck, e.g. a batch step that never completes) must
        # not block every other, already-processed, higher-id movie from
        # ever being counted as done - otherwise the protocol stalls
        # forever, never reaching allDone == maxMicSize even though almost
        # everything actually finished.
        prot = XmippProtMovieDoseAnalysis()
        prot.insertedIds = [1, 2, 3, 4, 5]
        prot.processedIds = [1, 2, 4, 5]  # 3 is still pending/stuck

        newDone = prot._getNewDoneIds(doneListIds=[])

        self.assertEqual(
            [1, 2, 4, 5], newDone,
            "Movies processed out of id order must still be reported as "
            "done instead of being permanently blocked by a pending one.",
        )

    def testGetNewDoneIdsPicksUpPreviouslyStuckIdOnceProcessed(self):
        # Once the previously pending movie finishes and is skipped via the
        # doneIds set on a later check, it must be reported too - no item is
        # permanently lost, it is just deferred to the round where it
        # actually completes.
        prot = XmippProtMovieDoseAnalysis()
        prot.insertedIds = [1, 2, 3, 4, 5]
        prot.processedIds = [1, 2, 3, 4, 5]

        newDone = prot._getNewDoneIds(doneListIds=[1, 2, 4, 5])

        self.assertEqual([3], newDone)


if __name__ == "__main__":
    unittest.main()
