import taichi as ti
import pyvista as pv
import numpy as np
from pyvista import CellType


class Mesh():
    def __init__(self,pyvista_mesh: pv.UnstructuredGrid |pv.StructuredGrid,get_face_info = True,dtype = np.float32) -> None:
        assert isinstance(pyvista_mesh,(pv.UnstructuredGrid,pv.StructuredGrid))
        assert len(list(pyvista_mesh.cells_dict.keys())) == 1, 'Meshes can only contain a single cell type, Got multiple'
        
        self.pyvista_mesh = pyvista_mesh
        self.nodes = np.array(pyvista_mesh.points)

        _celltype = list(pyvista_mesh.cells_dict.keys())[0]
        self.dtype = dtype
        self.cellType = (CellType(_celltype).name,_celltype)
        self.cells = np.array(pyvista_mesh.cells_dict[self.cellType[1]])
        self.cell_centroids = np.array(pyvista_mesh.cell_centers().points)
        self.cell_volumes = np.array(pyvista_mesh.compute_cell_sizes()['Volume'])
        self.groups = {}
        self.cell_check()

        if get_face_info:
            self.faces = self.get_face_attr()
        
        
        self.gridType = 'Unstructured' if isinstance(pyvista_mesh,pv.UnstructuredGrid) else 'Structured'
    @property
    def dimension(self):
        assert self.nodes.shape[-1] == 2 or self.nodes.shape[-1] == 3 
        return self.nodes.shape[-1]


    def cell_check(self):
        if self.dimension == 3:
            assert self.cellType[1] == CellType.HEXAHEDRON or self.cellType[1] == CellType.TETRA, 'Currently only Tetra and Hexahedron elements are supported' 


    def get_faces(self):
        if self.dimension == 2:
            #In 2D Faces are edges
            self.faces = self.pyvista_mesh.extract_all_edges(clear_data=True).lines
        elif self.dimension == 3:
            hex_faces = np.array([
                [3, 2, 1, 0],  # Bottom face
                [4, 5, 6, 7],  # Top face
                [0, 1, 5, 4],  # Front face
                [1, 2, 6, 5],  # Right face
                [3, 7, 6, 2],  # Back face
                [3, 0, 4, 7]   # Left face
            ])

            tet_faces = np.array([
                [2, 1, 0],
                [0, 1, 3],
                [3, 2, 0],
                [1, 2, 3]
            ])



            face_idx = tet_faces if self.cellType[1] == CellType.TETRA else hex_faces
            # self.face_idx = face_idx

            faces = self.cells[:,face_idx] # (C,F,N) C number of cells, F number of faces per cell, N number of nodes per cell
            faces_coords = self.nodes[faces] # (C,F,N,3)
            faces_sorted = np.sort(faces, axis=2)

            # Step 3: Reshape the array to have shape (num_cells*4, 3), where each row is a face.
            faces_reshaped = faces_sorted.reshape(-1, face_idx.shape[-1])

            # Step 4: Use np.unique to get the unique faces along axis=0.
            unique_faces, face_ids = np.unique(faces_reshaped, axis=0, return_inverse=True)

            face_ids = face_ids.reshape(self.cells.shape[0], face_idx.shape[0])

            return faces,faces_coords,unique_faces,face_ids

    @staticmethod
    def calculate_face_normal_and_area(faces,nodes):

        faces_coords = nodes[faces]
        # Anchor
        if faces.shape[-1] < 3:
            raise ValueError('The number of nodes defined for each face must be at least 3 or more nodes')
        # (C,F,N,3)
        v0 = faces_coords[:,:,0,:] # C,F,3 Array
        
        areas = []
        normal_vectors = []
        for i in range(1,faces.shape[-1]-1):
            vi = faces_coords[:,:,i,:]
            vii = faces_coords[:,:,i+1,:]
            normal_vector = 1/2*np.cross((vi - v0),(vii-v0)) # Default uses last axis
            area =  np.linalg.norm(normal_vector,axis = -1)
            areas.append(area)
            normal_vectors.append(normal_vector)
        face_areas = sum(areas)
        face_normals = sum(normal_vectors)/face_areas[:,:,np.newaxis]

        return face_areas,face_normals
    
    
    def get_neighbors(self) -> tuple[np.ndarray]:
        '''
        Return all the neighbors of the centroids of each cell. Assumes that all cells are connected to at least another cell



        Returns: `(D,2) np.array` where D is the number of unique neighbors describing face connectivity in the mesh.
          the 3 elements in each row are `(cell_1,cell_2,dist)` where cell 1 and cell 2 are connected cells and 
          dist is the euclidean distance between the centroids of the 2 cells

          The edge (cell_1,cell_2) is undirected and is sorted such that the lower cell id is always the first element 
        '''

        connections = 'faces' if self.dimension == 3 else 'edges'



        cell_neighbors = np.array([ [i,j] for i in range(len(self.cells)) for j in self.pyvista_mesh.cell_neighbors(i,connections=connections)])
        # unique_cell_neighbors = np.unique(np.sort(cell_neighbors,axis = 1),axis = 1)
        # points = self.cell_centroids[unique_cell_neighbors]

        # distance = np.linalg.norm(points[:,0,:]- points[:,1,:],axis =-1)
        # cell_dist = np.concat([unique_cell_neighbors,distance[:,np.newaxis]],axis = -1)

        return cell_neighbors
    

    def get_face_attr(self):
        cell_neighbors = self.get_neighbors()
        faces,faces_coords,unique_faces,face_ids = self.get_faces()

        face_areas,face_normals = self.calculate_face_normal_and_area(faces,self.nodes)
        # Face Areas and normals is (C,F) and (C,F,D)
        # Cell Neightbors is (C,k_i) list (variable)

        #We should extend the list to also have: Face ID, Face Normal, Face Area, Cell-connected_to, Distance
        C,F,N = faces.shape
        K = unique_faces.shape[0]
        D = self.dimension
        faces = faces_data()
        #We can populate all FaceID, Face Normal And Face Area first
        faces.dimension = self.dimension
        faces.unique_faces = unique_faces # (K,N)
        faces.face_coords = faces_coords # (C,F,N,D)
        faces.centroid = self.nodes[faces.unique_faces].mean(axis=1)
        # faces.centroid = faces_coords[face_ids]
        faces.id = face_ids
        faces.normal = face_normals
        faces.area = face_areas
        faces.neighbors = np.ones((C,F,2),dtype = np.int64)*(-1) # --1 for no neighbot (external face), otherwise give cell neighbor ID and faceID
        faces.distance= np.ones((C,F))*(-1)
        faces.is_boundary = np.ones((K,),dtype=np.uint8) # (K,) bool 1 if BC 0 otherwise for each face
        faces.boundary_value_is_fixed = np.zeros((K,D+1),dtype=np.uint8) 
        faces.boundary_value = np.zeros((K,D+1)) # (K, dim+1) dim+1 number of output variables (u,v,w,p)
        faces.cell_centroid_to_face_centroid = faces.centroid[faces.id] - self.cell_centroids[:,np.newaxis,:]

        faces.adjacent_cells = np.ones((K,2),dtype = np.int64)*(-1) # -1 means no neighbors


        for i,j in cell_neighbors: # Get Cell i and j ID
            faces_i,faces_j = face_ids[i],face_ids[j]
            intersect_face_id,face_i_ind, _ = np.intersect1d(faces_i,faces_j,return_indices= True)
            
            assert face_i_ind.shape[0] == 1, 'If this is raised then a cell shares a face with multiple cells'
            #Neighbor we need to store: cellID and then the associated face ID
            faces.neighbors[i,face_i_ind[0]] = np.array([j,intersect_face_id[0]],dtype = np.int64)
            faces.distance[i,face_i_ind[0]] = np.linalg.norm(self.cell_centroids[i] - self.cell_centroids[j])
            faces.is_boundary[intersect_face_id] = 0
            faces.adjacent_cells[intersect_face_id] = [i,j]

        return faces


    def set_boundary_value(self,face_ids:str | int|list|tuple|np.ndarray,u = None,v=None,w=None,p=None):

        if isinstance(face_ids,str):
            group_face_ids = self.groups[face_ids]
            self.faces.set_boundary_value(group_face_ids,u,v,w,p)

        elif isinstance(face_ids,(int,list,tuple,np.ndarray)):
            self.faces.set_boundary_value(face_ids,u,v,w,p)

        else:
            raise ValueError(f'face_ids can be type string,int,list,tuple, or np.ndarray got {type(face_ids)} instead')


class faces_data():
    '''
    Stores Info on faces:
        shape Dimensions:
            - C - Number of cells in Mesh
            - N - Number of nodes PER face
            - D - Dimension of model (usually 3)
            - K - Number of unique faces in mesh
            - F - Number of faces PER cell

        Keys:
    - `unique_faces` - a (K,N,D) where K is the number of unique faces in the mesh and N is the number of nodes per face. Returns the nodal coordinates of each face    
    - `face coords` a (C,F,N,D) array containing the nodal coordinates of each face. For convience it is over all Cells
    - 'centroid' a (C,F,D) array containing the centroid point of each face
    - `id` - a (C,F) array where C is the number of Cells and F is the number of faces per cell. Each element value is an integer corresponding to the face found in `face coords`
    - `normal` - a (C,F,D) array giving the normal of each face for each Cell.
    - `area` - a (C,F) array giving the area of each face. This should be reduces to K,3 but for convience the first 2 axis are the same shape as `normal`
    - `neighbors` - a (C,F,2) array that describes if a face is connected to another Cell. for a given Cell and Face, the 2 values are (Cell neighbor ID, FaceID). If the face is not connected to any neighbors both values are set to -1
    - `distance` - a (C,F) array that stores the centroid distance between 2 neighboring cells. If a face is not connected to a neighboring cell, distance is set to -1
    - `is_boundary` a (K,) array that stores a boolean on whether a unique face is a boundary/external face (i.e. no neighbors) or internal face
    - `boundary_value` a (K,D+1) specifying the values (u,v,w,p in 3D) given at each face. Default all 0
    '''
    
    unique_faces : np.ndarray | ti.Field
    '''(K,N,D) where K is the number of unique faces in the mesh and N is the number of nodes per face. Returns the nodal coordinates of each face '''
    face_coords:  np.ndarray | ti.Field
    '''(C,F,N,D) array containing the nodal coordinates of each face. For convience it is over all Cells'''
    centroid: np.ndarray | ti.Field
    '''(K,D) array containing the centroid point of each face'''
    id: np.ndarray| ti.Field
    '''(C,F) array where C is the number of Cells and F is the number of faces per cell. Each element value is an integer corresponding to the face found in `unique_faces`'''
    normal: np.ndarray| ti.Field
    '''(C,F,D) array giving the normal of each face for each Cell.'''
    area: np.ndarray| ti.Field
    '''(C,F) array giving the area of each face. This should be reduces to K,3 but for convience the first 2 axis are the same shape as `normal`'''
    neighbors: np.ndarray| ti.Field
    '''(C,F,2) array that describes if a face is connected to another Cell. for a given Cell and Face, the 2 values are (Cell neighbor ID, FaceID). If the face is not connected to any neighbors both values are set to -1'''
    distance: np.ndarray| ti.Field
    '''(C,F) array that stores the centroid distance between 2 neighboring cells. If a face is not connected to a neighboring cell, distance is set to -1'''
    is_boundary: np.ndarray| ti.Field
    '''(K,D+1 array that stores a boolean on whether a unique face is a boundary/external face (i.e. no neighbors) or internal face and if that variable at that face is a BC'''
    boundary_value: np.ndarray| ti.Field
    ''' (K,D+1) specifying the values (u,v,w,p in 3D) given at each face. Default all 0'''
    boundary_value_is_fixed : np.ndarray| ti.Field
    ''' (K,D+1) array specifying if the variable (u,v,w,p) is free or fixed. set to True if value id determined and fixed'''
    adjacent_cells: np.ndarray| ti.Field
    '''(K,2) array for a given face, return the adjacent cells'''
    cell_centroid_to_face_centroid: np.ndarray| ti.Field
    '''(C,F,D) vector containg the distance from the cell centroid to the corresponding face centroid'''
    dimension: int
    ''' Dimension of mesh'''
    array_type: str = 'numpy'
    ''' String indicating whether the attributes are `numpy` or `taichi`'''
    dtype: np.dtype = np.float32
    '''dtype used for floating point arrays. Default float32'''


    def set_boundary_value(self,face_ids: int | list |tuple |np.ndarray,u = None,v=None,w=None,p=None):
        assert isinstance(face_ids,(int,list,tuple,np.ndarray))
        
        if not isinstance(face_ids,(list,tuple,np.ndarray)):
            face_ids = np.array([face_ids])

        assert np.all(self.is_boundary[face_ids]), 'One of the provided face id is not an external face valid for applying boundary conditions'

        vars_mapping = {
            'u': 0, 
            'v': 1,
            'w': 2,
            'p': 3,
        }


        self.vars = vars_mapping
        ''' Mapping of variable names to order'''
        values_to_set = {name:val for name,val in zip(vars_mapping.keys(),[u,v,w,p]) if val is not None}
        vars_to_fix = [idx for key,idx in vars_mapping.items() if key in values_to_set.keys() ]

        self.boundary_value[face_ids,vars_to_fix] = list(values_to_set.values())
        self.boundary_value_is_fixed[face_ids,vars_to_fix] = True

    def to_taichi(self,init=ti.cpu):
        ti.init(init)
        
        faces = faces_data()
        faces.array_type = 'taichi'

        C = self.id.shape[0]

        for key,val in self.__dict__.items():
            if isinstance(val,np.ndarray):
                # The last array is the vector lenght
                
                shape = val.shape[0:-1]
                n = val.shape[-1]

                dtype = getattr(ti,val.dtype.name)
                # f = ti.ndarray(dtype,shape = val.shape)
                f = ti.Vector.field(n=n,dtype=dtype,shape = shape)
                setattr(faces,key,f)
                getattr(faces,key).from_numpy(val)
            else:
                setattr(faces,key,val)
        return faces
    




    # @ti.dataclass
    # class Cell():
    #     '''
    #     Stores predefined information such as neighbors, nodes, face normals etc into a convienent struct
    #     '''
    #     id:int
    #     nodes: node_array_ids
    #     centroid:centroid_vector # (3,)
    #     faces: face_array_ids
    #     face_normals: face_array_vector
    #     faces_area: face_array_float
    #     neighbor_ids: face_array_ids # Use -1 for no neighbor
    #     neighbor_distance: face_array_float
    #     volume:float_dtype
    # return Cell
