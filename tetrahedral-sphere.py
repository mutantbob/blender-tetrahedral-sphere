__author__ = 'thoth'
bl_info = {
    "name": "tetrahedral sphere",
    "description": "construct a sphere based on a decomposed tetrahedron including UV maps",
    "author": "Robert Forsman <blender@thoth.purplefrog.com>",
    "version": (0, 1),
    "blender": (2, 71, 0),
    "location": "3D View > something",
    "warning": "",
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.6/Py/Scripts/something",
    "category": "Mesh",
}


import bpy
from mathutils import *
from math import *

#

class VertexAccumulator:
    # use this class when you can't come up with a good deterministic numbering scheme for your vertices.

    def __init__(self):
        self.verts_ = []
        self.vertIdxs = {}

    def keyFor(v):
        return "%f,%f,%f"%(v[0], v[1], v[2])

    def idxFor(self, v):
        key = VertexAccumulator.keyFor(v)
        rval = self.vertIdxs.get(key)
        if None==rval:
            rval = len(self.verts_)
            self.vertIdxs[key] = rval
            self.verts_.append(v)
        return rval

    def verts(self):
        return self.verts_

#

class TetrahedralSphereArbitrary:

    @classmethod
    def triangle_interp(cls, va, f1, vb, f2, vc):
        return (vb - va) * (f1) + (vc - va) * (f2) + va

    @classmethod
    def build_face(cls, accum, faces, u_res, va, vb, vc):
        for u in range(u_res):

            for v in range(u_res - u):

                f1 = u / u_res
                f2 = v / u_res
                v5 = cls.triangle_interp(va, f1, vb, f2, vc).normalized()
                v6 = cls.triangle_interp(va, (u + 1) / u_res, vb, v / u_res, vc).normalized()
                v7 = cls.triangle_interp(va, (u + 1) / u_res, vb, (v + 1) / u_res, vc).normalized()
                v8 = cls.triangle_interp(va, f1, vb, (v + 1) / u_res, vc).normalized()
                print([v5, v6, v7])

                faces.append([accum.idxFor(v) for v in [v5, v6, v8]])
                if (v + 1 < u_res - u):
                    faces.append([accum.idxFor(v) for v in [v6, v7, v8]])

    @classmethod
    def make_mesh(cls, name, len1, u_res):

        accum = VertexAccumulator()
        faces=[]

        len2 = sqrt(1-len1*len1)

        v1 = Vector([len1, 0, len2])
        v2 = Vector([-len1, 0, len2])
        v3 = Vector([0, len1, -len2])
        v4 = Vector([0, -len1, -len2])

        va = v2
        vb = v1
        vc = v3
        cls.build_face(accum, faces, u_res, va, vb, vc)
        cls.build_face(accum, faces, u_res, v1, v2, v4)
        #cls.build_face(accum, faces, u_res, v1, v4, v3)
        #cls.build_face(accum, faces, u_res, v2, v3, v4)

        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(accum.verts(), [], faces)

        mesh.show_normal_face = True

        return mesh

    @classmethod
    def letterbox_arbitrary_op(cls, scn, len1, u_res):
        name = "tetrahedral sphere"
        mesh = cls.make_mesh(name, len1, u_res)
        obj = bpy.data.objects.new(name, mesh)
        scn.objects.link(obj)
        return obj

#

class TetrahedralSphere(bpy.types.Operator):
    """Add or adjust the transform effect on a strip to put letterboxes on the left and right or top and bottom to center the content and preserve its original aspect ratio"""
    bl_idname = "mesh.tetrahedral_sphere"
    bl_label = "Tetrahedral Sphere"
    bl_options = {'REGISTER', 'UNDO'}
    bl_region_type='UI'

    len1 = bpy.props.FloatProperty(name="Anchor Edge Length", default=1, min=0, max=2.0,
                                      subtype='FACTOR', precision=4, step=100,
                                      description="length of the anchor edge (the edge perpendicular to the poles")
    u_res = bpy.props.IntProperty(name="U resolution", default=12, min=1,
                                      description="number of subdivisions of the edges")

    def execute(self, ctx):
        try:
            obj = TetrahedralSphereArbitrary.letterbox_arbitrary_op(ctx.scene, self.len1/2.0, self.u_res)
            obj.select = True
            ctx.scene.objects.active = obj
            return {'FINISHED'}
        except ValueError as e:
            self.report({'ERROR'}, e.args[0])
            return {'CANCELLED'}
        except AttributeError as e:
            self.report({'ERROR'}, e.args[0])
            return {'CANCELLED'}

#
#

def menu_func(self, ctx):
    self.layout.operator(TetrahedralSphere.bl_idname, text=TetrahedralSphere.bl_label)


def register():
    bpy.utils.register_module(__name__)
    bpy.types.VIEW3D_PT_tools_add_mesh_edit.append(menu_func)


def unregister():
    bpy.types.VIEW3D_PT_tools_add_mesh_edit.remove(menu_func)
    bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
    register()
