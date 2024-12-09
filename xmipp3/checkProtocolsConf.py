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
import os

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
             'XmippProtWriteTestC','KendersomBaseClassify','XmippProtAlignVolumeForWeb', 'XMIPPCOLUMNS', 'NoOutputGenerated', 'ScatterImageMarker',
             'AlignVolPartOutputs']


#######Missing classes
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



#######Missing tests

def extract_protocols_from_file(file_path):
	"""Extrae las llamadas a self.newProtocol en un archivo."""
	protocols = []
	with open(file_path, 'r', encoding='utf-8') as f:
		source_code = f.read()
	tree = ast.parse(source_code)
	for node in ast.walk(tree):
		if isinstance(node, ast.Call):
			if (isinstance(node.func, ast.Attribute) and
					node.func.attr == "newProtocol" and
					isinstance(node.func.value, ast.Name) and
					node.func.value.id == "self"):
				if node.args:
					protocol_node = node.args[0]
					if isinstance(protocol_node, ast.Name):
						protocol = protocol_node.id
					elif isinstance(protocol_node, ast.Attribute):
						protocol = protocol_node.attr
					elif isinstance(protocol_node, ast.Constant):
						protocol = protocol_node.value
					if protocol not in protocols:
						protocols.append(protocol)
	return protocols


def extract_protocols_from_folder(folder_path):
	"""Extrae todos los protocolos de los archivos .py en una carpeta."""
	protocolsFull= []
	for root, _, files in os.walk(folder_path):
		for file in files:
			if file.endswith(".py"):
				file_path = os.path.join(root, file)
				protocols = extract_protocols_from_file(file_path)
				for protocol in protocols:
					if protocol not in protocolsFull:
						protocolsFull.append(protocol)
	return protocolsFull

folder_path = "tests/"
protocols = extract_protocols_from_folder(folder_path)
partProt = protocols.index('xpsp')
protocols[partProt] = 'XmippProtScreenParticles'

protocolsWithoutTest = []
for py_file in glob.glob("protocols/*.py"):
    for class_name in get_classes_from_file(py_file):
	    if not class_name in blackList:
		    if class_name not in protocols:
			    protocolsWithoutTest.append(class_name)
		    
print(f"\nProtocolos without test ({len(protocolsWithoutTest)}/{len(protocols)})")
print('\n'.join(protocolsWithoutTest))
