# **************************************************************************
# *
# * Authors:     Alberto Garcia Mena (alberto.garcia@cnb.csic.es)
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

'''
Script to collect all failled attempts from aspecific release of Xmipp and return a json with just the logs reported
Requires a json generated on https://xmipp.i2pc.es/api/attempts/filters/?user__userId=&xmipp__branch=v3.25.06.0-Rhea&returnCode=&returnCode_not=0 (get-> json save)
After this scxript, ask any IA:
ChatGPT Question: This is a JSON file with the logs of the installation failures of Xmipp. Can you make a summary of the most common errors?
Ommit the Matlab errors and warnings, report the percent of each kind of error
'''

import json
import random

jsonpath = '/home/agarcia/Downloads/Rhea.json'
jsonGenerated = '/home/agarcia/Downloads/RheaJustLogs.json'
with open(jsonpath, "r") as f:
    data = json.load(f)

# data is now a list of dicts
print(f"Number of entries: {len(data)}")

logs = [entry["logTail"] for entry in data if "logTail" in entry]

# pick a random subset of 100 entries
subset = random.sample(logs, min(400, len(logs)))

# write to new JSON file
with open(jsonGenerated, "w") as f:
    json.dump(subset, f, indent=4, ensure_ascii=False)