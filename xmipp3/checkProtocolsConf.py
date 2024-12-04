# ***************************************************************************
# * Authors:		 Carlos Oscar Sorzano (coss@cnb.csic.es)
# *
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307 USA
# *
# * All comments concerning this program package may be sent to the
# * e-mail address 'scipion@cnb.csic.es'
# ***************************************************************************/
'''
Description: This script checks for any protocols missing in the file 
             protocols.conf and lists them for further action.

Usage: python check_missing_protocols.py
'''

import ast
import glob

def get_classes_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        tree = ast.parse(file_content)
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        return class_names

with open("protocols.conf", 'r', encoding='utf-8') as file:
    protocolsConf = file.read()

blackList = ['ScatterImageMarker','XMIPPCOLUMNS', 'XmippProtVolAdjBase', 'AlignVolPartOutputs','ProtPickingConsensusOutput',
             'XmippProtEliminateEmptyBase', 'XmippProtDeepConsSubSet', 'XmippProtWriteTestP', 'NoOutputGenerated','XmippProtSubtractProjectionBase',
             'XmippProtWriteTestC','KendersomBaseClassify','XmippProtAlignVolumeForWeb']

missingCounter = 0
classCounter = 0
for py_file in glob.glob("protocols/*.py"):
    for class_name in get_classes_from_file(py_file):
        classCounter+=1
        if not class_name in blackList:
            if not class_name in protocolsConf:
                print(f"Missing: {class_name} from {py_file}")
                missingCounter +=1
print(f"Missing classes: {missingCounter}/{classCounter}")
