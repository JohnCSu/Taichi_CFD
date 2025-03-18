import taichi as ti
import pyvista as pv
import numpy as np
from pyvista import CellType
from src.preprocess import Mesh

ti.init(ti.cpu)
# mesh = pv.read('1_0.vtm')['internal']

# mesh = mesh.extract_cells([0,1,2,3,4,5])
points = np.array([
    [0, 0, 0],  # 0
    [1, 0, 0],  # 1
    [1, 1, 0],  # 2
    [0, 0, 1],  # 4
])
# Define the hexahedral cells with shared face
cells = np.array([
    [4,0,1,2,3]
], dtype=np.int64).flatten()

# Create the mesh
mesh = pv.UnstructuredGrid(cells, [pv.CellType.TETRA], points)
m = Mesh(mesh)

f_norms = m.faces['normal']
f_cent = m.faces['centroid']

plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=True, opacity=0.7)

# Add arrows to indicate face normals
for i in range(f_cent.shape[0]):

    centers = f_cent[i]
    f_norm = f_norms[i]
    for j,(center,normal) in enumerate(zip(centers,f_norm)):
        # r = np.random.rand()*0.1
        arrow = pv.Arrow(start=center, direction=normal * 0.1, scale=0.2)


        plotter.add_mesh(arrow, color='red',label = 'Hi')
        # plotter.add_text(str(j),position = center,font_size = 100,color = 'black')

        plotter.add_point_labels([center],[str(j)])
plotter.show()