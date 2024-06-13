
import numpy as np

from .framework import *
from .cam_render import CamRender


class NormalRender(CamRender):
    def __init__(self, width=1600, height=1200, name='Normal Renderer'):
        CamRender.__init__(self, width, height, name, program_files=['normal.vs', 'normal.fs'])

        self.norm_buffer = glGenBuffers(1)

        self.norm_data = None

    def set_normal_mesh(self, vertices, faces, norms, face_normals):
        CamRender.set_mesh(self, vertices, faces)

        self.norm_data = norms[face_normals.reshape([-1])]

        glBindBuffer(GL_ARRAY_BUFFER, self.norm_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.norm_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self):
        self.draw_init()

        glUseProgram(self.program)
        glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE, self.model_view_matrix.transpose())
        glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE, self.projection_matrix.transpose())

        # Handle vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, self.vertex_dim, GL_DOUBLE, GL_FALSE, 0, None)

        # Handle normal buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.norm_buffer)

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, self.n_vertices)

        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glUseProgram(0)

        self.draw_end()
