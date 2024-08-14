'''
from pyworkflow.object import Float, Integer, Boolean, String
from pwem.objects.data import EMObject


class ModelLol (EMObject):

    def __init__(self, fileName = None):
        self.fileName = String(fileName)
'''