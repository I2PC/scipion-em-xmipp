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

type_of_version = 'devel' #'release' 'devel'
_logo = "xmipp_logo" + ("" if type_of_version == 'release' else '_devel') + '.png'

_binVersion = 'v3' # Increase it if major release is generated in xmipp
# Increase according to SemVer rules:
# Rules with initial package version of vX.Y.Z
# - If the change consists of fixing a bug (that does not change at all
#   how the protocols are used), increase Z by 1.
# - If the change is adding new functionality (extra params for a protocol,
#   or new protocols), increase Y by 1.
# - If the change deprecates existing functionality (remove a protocol,
#   or a param), increase X by 1
# - If several of the above are true, only change the biggest one applicable (
#   for example, if a fix is made and a new protocol are included in the same
#   pull request, increase the one related to the new protocol).
__version__ = 'v24.23.0'

_binTagVersion = _binVersion + '-Poseidon' #'devel' or _binVersion + '-Poseidon'
_pluginTagVersion = __version__ + '-Poseidon'  #'devel' or _pluginVersion + '-Poseidon'

_currentDepVersion = '1.0'
