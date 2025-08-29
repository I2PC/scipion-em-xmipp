# *
# * Authors:     Alberto Garcia Mena     alberto.garcia@cnb.csic.es
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

import os
import re
import ast
from datetime import date
import subprocess
from pathlib import Path
import json


SITE_PACKAGES ='/home/agarcia/miniconda/envs/scipionProtocolRecomender/lib/python3.8/site-packages'
PATH_SCIPION_INSTALLED = '/home/agarcia/scipion3'
JSON_PROTOCOLS = 'protocolsInfo.json'

def readingProtocols():
    #### LIST PROTOCOLS
    protocol_dict = {}
    result = subprocess.run(f'./scipion3 protocols', shell=True, check=True,
                            cwd=PATH_SCIPION_INSTALLED, capture_output=True,
                            text=True)
    protocolsStr = result.stdout
    protocolsStr = protocolsStr[protocolsStr.find('LABEL') + 5:]

    for line in protocolsStr.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            package = parts[0]
            protocol = parts[1]
            if package not in protocol_dict:
                protocol_dict[package] = []
            protocol_dict[package].append(protocol)

    protocol_dict.pop("Scipion", None)
    protocol_dict["chimera"] = protocol_dict.pop("chimerax")
    blackList = ['pyworkflowtests', 'xmipp2']
    for p in blackList:
        protocol_dict.pop(p, None)

    dictProtocolFile = {}
    BASE_DIR = Path(__file__).resolve().parent

    dictProtocolFile = {}
    listFolders = [
	    BASE_DIR / "protocols",
	    BASE_DIR / "protocols/protocol_preprocess",
	    BASE_DIR / "protocols/protocol_projmatch"
    ]
    for path in listFolders:
        for file in path.iterdir():
            if file.is_file() and file.suffix == ".py" and file.name != "__init__.py":
                with open(file, "r", encoding="utf-8") as f:
                    scriptTexted = f.read()
                    tree = ast.parse(scriptTexted)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if node.name in protocol_dict['xmipp3']:
                                #print(f"Clase encontrada: {node.name}")
                                dictProtocolFile.update({node.name: file})

    protocol_dict.pop("Scipion", None)
    print(f'Registred: {len(dictProtocolFile)} protocols')
    return dictProtocolFile



def requestDSFillMap(dictProtocolFile):
    dictVectors = {}
    index = 0
    dictProtocolsInfo = {}
    for protocol in dictProtocolFile.keys():
        index+=1
        indexProt = 0
        dictVectors[protocol] = {}
        indexProt += 1
        dictVectors[protocol] = {}
        file = dictProtocolFile[protocol]
        with open(file, "r", encoding="utf-8") as f:
            scriptTexted = f.read()
            protocolString = classTexted(scriptTexted, protocol)
            helpProtocol = removeJumpLine(helpProtocolStr(protocolString))
            labelProtocol = extract_label_protocol(protocolString, protocol)
            dictProtocolsInfo['protocol']= {'name':labelProtocol,  'Description': helpProtocol}
    return dictProtocolsInfo


def classTexted(scriptTexted, protocol):
    tree = ast.parse(scriptTexted)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name in protocol:
                start_line = node.lineno - 1  # ast use 1-based indexing
                end_line = node.end_lineno if hasattr(node,"end_lineno") else None
                if end_line is None:
                    for child_node in ast.walk(node):
                        if hasattr(child_node,
                                   "lineno") and child_node.lineno > start_line:
                            end_line = child_node.lineno
                return "\n".join(
                    scriptTexted.splitlines()[start_line:end_line])
    print(f'classTexted ERROR on protocol: {protocol} ')
    return ' '


def removeJumpLine(string):
    return string.replace('\n', ' ')


def extract_label_protocol(scriptTexted, protocol):
    tree = ast.parse(scriptTexted)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):  #
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == '_label':
                            if isinstance(stmt.value, ast.Str) or isinstance(stmt.value, ast.Constant):
                                return stmt.value.s
    print(f'extract_label_protocol ERROR on script: {scriptTexted[:30]}... ')
    return ' '.join(re.sub(r'([a-z])([A-Z])', r'\1 \2', protocol).split()).lower()


def helpProtocolStr(scriptTexted):
    tree = ast.parse(scriptTexted)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.body and isinstance(node.body[0], ast.Expr):
                docstring_node = node.body[0].value
                if isinstance(docstring_node, ast.Str) or isinstance(docstring_node, ast.Constant):
                    return docstring_node.s
    return ''

def writeJson(dict):
    with open(JSON_PROTOCOLS, "w", encoding="utf-8") as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    dictProtocolFile = readingProtocols()
    dictProtocolsInfo = requestDSFillMap(dictProtocolFile)
    writeJson(dictProtocolsInfo)