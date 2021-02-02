# mgeom
a python script for manipulation of molecular geometry.

## function
1. read cartesian coordinates (from mndo input or gauss input)
2. calculate bonds, angle and dihedral angle
3. transition, rotation analysis **TO FIX** 
4. read zmatrice coordinates (from mndo input or gauss input)
5. convert between cartesian coordinates (xyz) and zmatrice coordinates (internal locale axes)
6. build smooth transition path from zmatrice 1 to zmatrice 2
7. support format: gauss format, mndo format, .xyz format

## features
1. it's fast. Built in numpy with vectorization programming.
2. lightweight.
3. and it's easy to extened.

## TODO
 - [ ] topology analysis
 - [ ] RMSD analysis and provide more trajectory analysis
 - [ ] provide more IO format, such as .pdb etc.
