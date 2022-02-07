# **************************************************************************
# *
# * Authors:     Jorge Garcia Condado (jorgegarciacondado@gmail.com)
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

from pwem.protocols import EMProtocol

class XmippProtDeepHand(EMProtocol):

    _label ="deep hand"

    def __init__(self, *args, **kwargs):
        EMProtocol.__init__(self, *args, **kwargs)

    def _defineParams(self, form):

        form.addSection('Input')

# --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        print('Hey')
        pass

    def _validate(self):
        pass

# --------------------------- INFO functions -------------------------------
    def _summary(self):
        summary = []

    def _methods(self):
        methods = []
        return methods

    def _validate(self):
        errors = []
        return error
