


import os

from OpenGL.GL import *



def loadShader(shaderType, shaderFile):
    # check if file exists, get full path name
    strFilename = findFileOrThrow(shaderFile)
    shaderData = None
    with open(strFilename, 'r') as f:
        shaderData = f.read()

    shader = glCreateShader(shaderType)
    glShaderSource(shader, shaderData)  # note that this is a simpler function call than in C


    glCompileShader(shader)

    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if status == GL_FALSE:

        strInfoLog = glGetShaderInfoLog(shader)
        strShaderType = ""
        if shaderType is GL_VERTEX_SHADER:
            strShaderType = "vertex"
        elif shaderType is GL_GEOMETRY_SHADER:
            strShaderType = "geometry"
        elif shaderType is GL_FRAGMENT_SHADER:
            strShaderType = "fragment"

        print("Compilation failure for " + strShaderType + " shader:\n" + str(strInfoLog))

    return shader



def createProgram(shaderList):
    program = glCreateProgram()

    for shader in shaderList:
        glAttachShader(program, shader)

    glLinkProgram(program)

    status = glGetProgramiv(program, GL_LINK_STATUS)
    if status == GL_FALSE:

        strInfoLog = glGetProgramInfoLog(program)
        print("Linker failure: \n" + str(strInfoLog))

    for shader in shaderList:
        glDetachShader(program, shader)

    return program



def findFileOrThrow(strBasename):

    if os.path.isfile(strBasename):
        return strBasename

    LOCAL_FILE_DIR = "data" + os.sep
    GLOBAL_FILE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep + "data" + os.sep

    strFilename = LOCAL_FILE_DIR + strBasename
    if os.path.isfile(strFilename):
        return strFilename

    strFilename = GLOBAL_FILE_DIR + strBasename
    if os.path.isfile(strFilename):
        return strFilename

    raise IOError('Could not find target file ' + strBasename)
