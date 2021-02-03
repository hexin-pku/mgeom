# mgeom

A **vec3d** python script for manipulation of molecular geometry. Conversion between zmatice and cartesian coordinates. Supporting gauss format, mndo format, etc... (used in mndo-MRCI interface for nonadiabatic dynamics simulation)

## function

1. read cartesian coordinates (from mndo input or gauss input)
2. calculate bonds, angle and dihedral angle
3. transition, rotation analysis
4. read zmatrice coordinates (from mndo input or gauss input)
5. convert between cartesian coordinates (xyz) and zmatrice coordinates (internal local axis)
6. build smooth transition path from zmatrice 1 to zmatrice 2
7. supported format: gauss format, mndo format, .xyz format

## features

1. it's fast. Built in numpy with vectorization programming.
2. lightweight.
3. it's easy to be extened or embedded.

## TODO
 - [ ] topology analysis
 - [ ] RMSD analysis and provide more trajectory analysis
 - [ ] provide more IO format, such as .pdb etc.
