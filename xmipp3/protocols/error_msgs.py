

BAD_IMPORT_TENSORFLOW_KERAS_MSG='''
Error, tensorflow/keras is probably not installed. Install it with:\n  ./scipion installb deepLearningToolkit
If gpu version of tensorflow desired, install cuda 8.0 or cuda 9.0
We will try to automatically install cudnn, if unsucesfully, install cudnn and add to LD_LIBRARY_PATH
add to SCIPION_DIR/config/scipion.conf
CUDA = True
CUDA_VERSION = 8.0 or 9.0
CUDA_HOME = /path/to/cuda-%(CUDA_VERSION)
CUDA_BIN = %(CUDA_HOME)s/bin
CUDA_LIB = %(CUDA_HOME)s/lib64
CUDNN_VERSION = 6 or 7
'''
