function info_extract(){
    input: 
    1. mask mrc file. matrix (X,Y,Z). 1 is mask 0 is background
    output: 
    1.ndarray with shape (X,Y,Z), element is 0 or 1. only extract the surface voxels of mask that adjacent with background voxels.
    2.ndarray with shape (3,X,Y,Z), element is a vector with 3 float. record the orientation of all voxels. the orientation is generate by the relationship between the surface voxels of mask and its faces which is adjacent with background voxels. for example, for 1 mask voxel, if all faces is buried by mask voxels, it should return (0,0,0). If only one face is adjacent with backgrond voxel, the vector should be with an orientation from the center of this mask voxel to the center of that background voxel, with a length of unit 1. if more than 1 faces, then should return averages of the normal of those fases with a length of unit 1.
}

function sample(){
    input:
    1. min_distance: minimum Euclidean distance between selected centers (in pixels)
    2. edge_distance: edge distance margin from the volume boundary (in pixels); do not sample
       within this distance of any boundary to avoid half particles
    3. surface_mask: ndarray with shape (X,Y,Z)
    4. orientations: ndarray with shape (3,X,Y,Z) = info_extract()

    output:
    1. ndarray with shape (6,X,Y,Z), each selected surface voxel stores a vector with
       coordinates and orientation (x,y,z,vx,vy,vz), unselected voxels are zeros
}


