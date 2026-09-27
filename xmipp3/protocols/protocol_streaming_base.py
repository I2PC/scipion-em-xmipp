# **************************************************************************
# *
# * Authors: Yunior C. Fonseca Reyna    (cfonseca@cnb.csic.es)
# *
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


class XmippStreamingBase:
    """Backend-independent helpers shared by Xmipp streaming protocols."""

    def _loadLogicalSet(self, pointer):
        inputSet = pointer.get()
        inputSet.loadAllProperties()
        return inputSet

    def _discoverIdsAfter(self, inputSet, lastId):
        ids = list(
            inputSet.getUniqueValues(
                'id',
                where='id > %d' % lastId,
            )
        )

        if ids:
            lastId = max(ids)

        return ids, lastId

    def _reconcileClosedStreamIds(
            self,
            inputSet,
            discoveredIds,
            knownIds,
            producerClosed,
    ):
        """Recover late-visible rows only after producer stream closure."""
        discoveredIds = list(discoveredIds)

        if not producerClosed:
            return discoveredIds, True

        expectedSize = inputSet.getSize()
        knownIds = set(knownIds)
        visibleKnownIds = knownIds.union(discoveredIds)

        if len(visibleKnownIds) >= expectedSize:
            return discoveredIds, True

        reconciledIds = list(
            inputSet.getUniqueValues('id')
        )

        if reconciledIds:
            self._lastInputId = max(
                getattr(self, '_lastInputId', 0),
                max(reconciledIds),
            )

        visibleIds = set(discoveredIds)
        visibleIds.update(reconciledIds)

        newIds = [
            itemId
            for itemId in sorted(visibleIds)
            if itemId not in knownIds
        ]

        terminalConsistent = (
            len(knownIds.union(visibleIds)) >= expectedSize
        )

        return newIds, terminalConsistent

    def _getPersistedOutputIds(self, outputName):
        outputSet = getattr(self, outputName, None)
        if outputSet is None:
            return set()

        return set(outputSet.getIdSet())

    def _getPersistedOutputSize(self, outputName):
        outputSet = getattr(self, outputName, None)
        if outputSet is None:
            return 0

        return outputSet.getSize()

    def _restorePersistedOutputIds(self, outputName):
        cache = getattr(self, '_persistedOutputIds', None)

        if cache is None:
            cache = {}
            self._persistedOutputIds = cache

        if outputName not in cache:
            outputSet = getattr(self, outputName, None)
            cache[outputName] = (
                set(outputSet.getIdSet())
                if outputSet is not None
                else set()
            )

        return set(cache[outputName])

    def _markOutputIdsPersisted(self, outputName, itemIds):
        persistedIds = self._restorePersistedOutputIds(
            outputName,
        )

        cache = self._persistedOutputIds
        cache[outputName].update(itemIds)

        return set(cache[outputName])

    def _getKnownPersistedOutputIds(self, outputName):
        return self._restorePersistedOutputIds(outputName)
