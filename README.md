This is an addon for blender.  As of 2015-May it works with blender
2.74.

This addon is used to create spherical meshes that are based on
decomposition of a tetrahedron into a series of quads and triangles.
I originally wrote it so I could create
http://gfycat.com/SentimentalSpiffyAnteater which is based on a
pysanky by Halyna Mudryj.  The addon's automatic calculation of UVs
and assignment of materials greatly streamlines the creation of
similar egg models.

To use this addon with your blender installation you'll have to bring
up the User Preferences, select the Addons tab, and use the Install
from File button at the very bottom to pick the tetrahedral-sphere.py
file from this package.  Finally, find the addon in the list and check
the box on the right to activate it. ( see
http://blender.stackexchange.com/questions/1688/installing-an-addon#1689
for screenshots )

Once the addon is installed you should be able to press Spacebar in
the 3D view (used to invoke an arbitrary operator by name) and type in
Tetrahedral Sphere (although you probably only have to type in the
first few letters to narrow down the list to something clickable) and
invoke that operator.  Once you have invoked it, the default
parameters of the operator will appear at the bottom of the 3D view's
t-panel and you can adjust them to suit your purposes.

The latest version of the blender 3D modeling and animation suite can
be downloaded from http://www.blender.org/
