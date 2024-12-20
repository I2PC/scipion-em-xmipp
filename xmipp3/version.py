# **************************************************************************
# *
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
# *              Alberto García (alberto.garcia@cnb.csic.es)
# *				 Martín Salinas (martin.salinas@cnb.csic.es)
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

type_of_version = 'devel' #'release'
_logo = "xmipp_logo" + ("" if type_of_version == 'release' else '_devel') + '.png'
_current_xmipp_tag = 'devel'
_currentBinVersion = '3.24.06.0'
_currentDepVersion = '1.0'
__version__ = _currentBinVersion[2:] + ".0"  # Set this to ".0" on each xmipp binary release, otherwise increase it --> ".1", ".2", ...
