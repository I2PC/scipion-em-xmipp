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

blackList = ['ScatterImageMarker','XMIPPCOLUMNS']

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