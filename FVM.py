import taichi as ti
import pyvista as pv
import numpy as np
from pyvista import CellType
from taichi.math import vec3,mat3
from src.preprocess import Mesh

ti.init(ti.cuda,debug  = True)

'''
LETS FIRST SOLVE EXPLICIT METHOD AS IT IS EASY AS FUCK


'''


VALID_FP_DTYPES = {ti.f32,ti.f16,ti.f64}
def create_field_output_stuct(*,return_grad = False,**kwargs) -> ti.Struct:
    if not kwargs:
        kwargs = {'u':ti.math.vec3,'p':ti.f32}
    
    field_struct = ti.types.struct(**kwargs)


    if return_grad:
        gradients = {}
        for key,val in kwargs.items():
            name = f'grad_{key}'
            if hasattr(val,'n'):
                gradients[name] =  ti.types.matrix(val.n,3,dtype=val.dtype)
            else:
                assert val in VALID_FP_DTYPES
                gradients[name] = ti.types.matrix(1,3,dtype=val)
                

        gradient_struct = ti.types.struct(**gradients)
        return field_struct,gradient_struct



    return field_struct

fields = create_field_output_stuct(return_grad=True)


def create_FVM_cell(num_nodes:int,num_faces:int,float_dtype = ti.f32):
    face_array_ids = ti.types.vector(num_faces,int)
    face_array_float = ti.types.vector(num_faces,float_dtype)
    face_array_vector = ti.types.matrix(num_faces,3,float_dtype)

    node_array_ids = ti.types.vector(num_nodes,dtype=int)
     
    centroid_vector = ti.types.vector(3,dtype=float_dtype)


    @ti.dataclass
    class Cell():
        '''
        Stores predefined information such as neighbors, nodes, face normals etc into a convienent struct
        '''
        id:int
        nodes: node_array_ids
        '''(N,) Vector'''
        centroid:centroid_vector # (3,)
        '''(D,) Vector'''
        faces: face_array_ids
        '''(F,) Vector'''
        face_normals: face_array_vector
        '''(F,D) Vector'''
        faces_area: face_array_float
        '''(F,) Vector'''
        neighbor_ids: face_array_ids # Use -1 for no neighbor
        '''(F,) Vector'''
        neighbor_distance: face_array_float
        '''(F,) Vector'''
        volume:float_dtype
        '''Float'''
        cell_centroid_to_face_centroid: face_array_vector
        '''(F,D) vector containg the distance from the cell centroid to the corresponding face centroid'''

    return Cell


FVM_CELL = create_FVM_cell(8,6)

def set_Vectorfield_from_numpy(arr:np.ndarray):
    dtype = getattr(ti,arr.dtype.name)
    if len(arr.shape) == 1:
        f:ti.Field = ti.field(dtype,shape = arr.shape)
        f.from_numpy(arr)
    else:
        f:ti.MatrixField = ti.Vector.field(n = arr.shape[-1],dtype=dtype,shape = arr.shape[:-1])
        f.from_numpy(arr)
    return f



@ti.data_oriented
class FVM():
    def __init__(self,mesh:Mesh,density:float = 1000,viscosity:float = 1e-3,dtype = ti.f32):
        self.density = density
        self.viscosity = viscosity
        self.gridType:str = mesh.gridType
        self.dimension:int = mesh.dimension
        self.dtype = dtype
        self.cellType = mesh.cellType


        self.faces = mesh.faces.to_taichi()
        self.nodes:ti.MatrixField = set_Vectorfield_from_numpy(mesh.nodes)
        '''(N,3)  Vector Field'''
        self.cells:ti.MatrixField = set_Vectorfield_from_numpy(mesh.cells)
        '''(C,F)  Vector Field'''
        self.cell_centroids:ti.MatrixField = set_Vectorfield_from_numpy(mesh.cell_centroids)
        '''(C,3) Vector Field'''

        self.cell_volumes:ti.MatrixField = set_Vectorfield_from_numpy(mesh.cell_volumes)
        '''(C,) Vector Field'''
        
        # Output for each cell: T,C,4 (u,v,w,p)
        num_outputs = self.dimension + 1
        
        t = 10
        self.time_points = ti.ndarray(dtype,(t,))
        self.time_points.from_numpy(np.linspace(0,1,10))


        self.field_structType,self.fieldGrad_structType = create_field_output_stuct(return_grad=True)

        self.num_nodes = mesh.nodes.shape[0]
        self.num_cells = mesh.cells.shape[0]
        self.num_faces = mesh.faces.unique_faces.shape[0]
        self.num_outputs = self.get_num_outputs(**self.field_structType.members)

        self.FVM_cell_structType = create_FVM_cell(self.num_nodes,self.num_faces)



        self.FVM_cells = self.FVM_cell_structType.field(shape= self.num_cells)
        '''Store Cells Struct which contains cell info'''        


        self.init_analysis_fields()



    

    @staticmethod
    def get_num_outputs(**kwargs):
        num_outputs = 0

        for key,val in kwargs.items():
            if hasattr(val,'n'):
                num_outputs+= val.n*val.m
            else:
                assert val in VALID_FP_DTYPES, 'values given must be taichi types vector,matrix or floating point primitive (e.g. ti.f32) '
                num_outputs+= 1


    def init_analysis_fields(self):
        '''
        For now we assume we have at most 2 time steps (t_i, t_(i+1) )

        '''
        n = 2

        self.face_values:ti.Field = self.field_structType.field(shape = (self.num_faces))
        ''' Structfield containig u:vec3 and p:scalar'''

        self.mass_fluxes:ti.Field = ti.field(dtype = self.dtype,shape = (self.num_cells,self.num_faces) )
        

        self.nodal_values:ti.Field = self.field_structType.field(shape = (self.num_nodes))
        self.cell_values:ti.Field =  self.field_structType.field(shape = (self.num_cells))
        self.cell_gradients:ti.Field = self.fieldGrad_structType.field(shape=(self.num_cells))

        self.face_values.fill(0.)
        self.nodal_values.fill(0.)
        self.cell_values.fill(0.)
        self.cell_gradients.fill(0.)


    @ti.kernel
    def init_cell_structs(self):
        '''
        
        '''
        for i in self.FVM_cells:
            cell = self.FVM_cells[i] 
            cell.id = i
            print(self.cells[i])
            cell.nodes = self.cells[i]
            cell.centroid = self.cell_centroids[i]
            '''(D,) Vector'''
            cell.faces = self.faces.id[i]
            '''(F,) Vector'''
            
            cell.neighbor_distance = self.faces.distance[i]
            '''(F,) Vector'''
            cell.volume = self.cell_volumes[i]
            '''Float'''

            cell.faces_area = self.faces.area[i]

            # for j in range(self.num_faces):
            cell.cell_centroid_to_face_centroid = ti.Matrix.rows([ self.faces.cell_centroid_to_face_centroid[i,j] for j in range(self.num_faces)])
            cell.face_normals = ti.Matrix.rows([ self.faces.normal[i,j] for j in range(self.num_faces)])
            '''(F,D) Vector'''
            
            '''(F,) Vector'''
            cell.neighbor_ids =ti.Matrix.rows([ self.faces.neighbors[i,j] for j in range(self.num_faces)])
        
    def init_step(self):
        '''
        Performs a single iteration. We need to use central Difference for the first face estimation
        '''
        self.init_cell_structs()
        self.apply_BC()
        self.u_initial_face_interpolation()
        self.calculate_mass_flux()
        self.get_gradients()
        
    def step(self):
        self.apply_BC()
        self.u_face_interpolation() # Use Upwind Linear
        self.calculate_mass_flux()
        self.get_gradients()
    
    @ti.kernel
    def apply_BC(self):
        for i in ti.grouped(self.face_values):
            for j in ti.static(range(3)):
                if self.faces.boundary_value_is_fixed[i][j]:
                    self.face_values[i]['u'][j] = self.faces.boundary_value[i][j]
            if self.faces.boundary_value_is_fixed[i][3]:
                self.face_values[i]['p'] = self.faces.boundary_value[i][3]

    @ti.kernel
    def calculate_mass_flux(self):
        # Lets just use central difference for now         
        for i,j in (self.mass_fluxes): # (C,K)
            normal:vec3 =    self.faces.normal[i,j]
            area:float =      self.faces.area[i][j]
            face_u = self.face_values[i].u

            self.mass_fluxes[i,j] = self.density*area*normal*face_u


    @ti.kernel
    def u_initial_face_interpolation(self):
        '''
        Face Interpolation. Initialise using central difference
        '''
        # Lets just use central difference for now 
        
        for i in ti.grouped(self.face_values): # (K)
            #Given a Face, find the neighbors and calulate velocity 
            # Check if a boundary face
            adj_vec = self.faces.adjacent_cells[i]
            C_n,C_p = adj_vec
            centroid_n = self.cell_centroids[C_n]
            centroid_p = self.cell_centroids[C_p]
            face_centroid = self.faces.centroid[i]
            u_n,u_p = self.cell_values[C_n].u,self.cell_values[C_p].u
            u = central_difference(centroid_p,centroid_n,face_centroid,u_p,u_n)
            for j in ti.static(range(3)):
                if not self.faces.boundary_value_is_fixed[i][j]:
                    self.face_values[i].u[j] = u[j]

    @ti.kernel
    def u_face_interpolation(self):
        '''
        Face Interpolation, using regular upwind linear scheme
        '''
        # Lets just use central difference for now 
        
        for i in ti.grouped(self.face_values): # (K)
            #Given a Face, find the neighbors and calulate velocity 
            # Check if a boundary face
            adj_vec = self.faces.adjacent_cells[i]
            C_n,C_p = adj_vec
            centroid_n = self.cell_centroids[C_n]
            centroid_p = self.cell_centroids[C_p]
            face_centroid = self.faces.centroid[i]
            u_n,u_p = self.cell_values[C_n].u,self.cell_values[C_p].u
            u = upwind_linear(cell_r_i,cell_r_j,)
            for j in ti.static(range(3)):
                if not self.faces.boundary_value_is_fixed[i][j]:
                    self.face_values[i].u[j] = u[j]


           
    @ti.kernel
    def get_gradients(self):
        for i in self.cells:
            flux = ti.Matrix([[0.,0.,0,],[0.,0.,0,],[0.,0.,0,]],dt = self.dtype)
            for j in range(self.cells.n):
                face_id = self.cells[i][j]
                u = self.face_values[face_id].u
                face_normal = self.faces.normal[i,j]
                face_area = self.faces.area[i][j] #Scalar
                flux += get_flux_vector(u,face_normal,face_area)
            vol = self.cell_volumes[i]
            self.cell_gradients[i].grad_u = flux/vol
        
            
    
    
@ti.func
def central_difference(cell_centroid_i:ti.math.vec3,cell_centroid_j:ti.math.vec3,face_centroid:ti.math.vec3,u_i,u_j):
    '''Face Interpolation Scheme Via Central Difference'''
    psi = ti.math.length(face_centroid - cell_centroid_i)/ti.math.length(cell_centroid_j - cell_centroid_i)
    return u_i*psi + (1-psi)*u_j

@ti.func
def upwind_linear(cell_r_i,cell_r_j,u_i,u_j,grad_u_i,grad_u_j,mass_flux:float):
    '''
    Face Interpolation Scheme Via Upwind Linear

    Set grad_u_i and grad_uj to zero for first order upwind
    
    '''
    if mass_flux > 0: # Coming out
        return u_i + grad_u_i*cell_r_i
    else:
        return u_j + grad_u_j*cell_r_j



@ti.func
def get_flux_vector(face_value:ti.math.vec3,face_normal,face_area):
    return face_value.outer_product(face_normal)*face_area


if __name__ == '__main__':
    print('hi')
    mesh = pv.read('1_0.vtm')['internal']

    mesh = mesh.extract_cells([0,1,2,3,4,5])
    m = Mesh(mesh)
    m.set_boundary_value(0,u = 1,v = 2,w = 3.3,p = 2)
    model = FVM(m)
    print(model.cell_centroids.shape,model.nodes.shape, model.cells.shape)
    model.init_step()
    # model.face_interpolate('hello WORLD')