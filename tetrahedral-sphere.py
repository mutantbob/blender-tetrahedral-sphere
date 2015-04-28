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

class GreatCircleArc:
    def __init__(self, p1, p2):
        self.p1 = p1.normalized()
        self.p2 = p2.normalized()

        normal_ = self.p1.cross(self.p2)
        if (normal_.magnitude>0):
            self.normal = normal_.normalized()
        else:
            self.normal = Vector([0,0,1])
        self.axis2 = self.normal.cross(self.p1)
        self.cos_theta = self.p1.dot(self.p2)
        if (self.cos_theta>=1):
            self.theta = 0
        elif (self.cos_theta<=-1):
            self.theta = pi
        else:
            self.theta = acos(self.cos_theta)
        #print("%f\t%r,%r,%r"%(self.theta, self.p1, self.axis2, self.normal))

    def point_for(self, t):
        """
        :param t: in the range[0..1] .  0 corresponds to p1.  1 corresponds to p2
        :return:
        """
        theta_2 = t * self.theta
        return self.p1 * cos(theta_2) + self.axis2 * sin(theta_2)


#

class TetrahedralSphereArbitrary:

    @classmethod
    def triangle_interp(cls, va, f1, vb, f2, vc):
        return (vb - va) * (f1) + (vc - va) * (f2) + va

    @classmethod
    def artillery(cls, circle_j, v1, u_res, u, v):
        if u + v > 0:
            v2 = circle_j.point_for(u / (u + v))
            circle_k = GreatCircleArc(v1, v2)
            return circle_k.point_for((u + v) / u_res)
        else:
            return v1


    @classmethod
    def build_face(cls, accum, faces, u_res, va, vb, vc):

        # this strategy is not always pretty for arbitrary spherical triangles,
        # but for the ones we are constructing this decomposition is the least atrocious.

        circle_j = GreatCircleArc(va, vb)

        for u in range(u_res):

            v_res2 = u_res-u
            for v in range(v_res2):

                v1 = vc
                v5 = cls.artillery(circle_j, v1, u_res, u, v)
                v6 = cls.artillery(circle_j, v1, u_res, u+1, v)
                v8 = cls.artillery(circle_j, v1, u_res, u, v+1)

                faces.append([accum.idxFor(v) for v in [v5, v6, v8]])
                if (v < v_res2-1):
                    v7 = cls.artillery(circle_j, v1, u_res, u+1, v+1)
                    faces.append([accum.idxFor(v) for v in [v6, v7, v8]])



    @classmethod
    def build_face_GC(cls, accum, faces, u_res, va, vb, vc):

        circle_k = GreatCircleArc(va, vc)
        circle_m = GreatCircleArc(vb, vc)
        circle1 = GreatCircleArc(circle_k.point_for(0), circle_m.point_for(0))
        #print("%r \n%r\n%r"%(circle_k.point_for(1), circle_m.point_for(1), vc))
        for u in range(u_res):

            circle2 = GreatCircleArc(circle_k.point_for((u+1)/u_res), circle_m.point_for((u+1)/u_res))

            v_res2 = u_res-u
            for v in range(v_res2):

                v5 = circle1.point_for(v/v_res2)
                if v_res2>1:
                    f2 = v / (v_res2 - 1)
                else:
                    f2 = 0
                v6 = circle2.point_for(f2)
                v8 = circle1.point_for((v+1)/v_res2)

                #print([v5, v6, v8])

                faces.append([accum.idxFor(v) for v in [v5, v6, v8]])
                if (v < v_res2-1):
                    v7 = circle2.point_for((v+1)/(v_res2-1))
                    faces.append([accum.idxFor(v) for v in [v6, v7, v8]])

            circle1 = circle2

    @classmethod
    def make_mesh(cls, name, len1, u_res):

        accum = VertexAccumulator()
        faces=[]

        len2 = sqrt(1-len1*len1)

        v1 = Vector([len1, 0, len2])
        v2 = Vector([-len1, 0, len2])
        v3 = Vector([0, len1, -len2])
        v4 = Vector([0, -len1, -len2])

        if True:
            accum.idxFor(v1)
            accum.idxFor(v2)
            accum.idxFor(v3)
            accum.idxFor(v4)

        cls.build_face(accum, faces, u_res, v2, v1, v3)
        cls.build_face(accum, faces, u_res, v1, v2, v4)
        cls.build_face(accum, faces, u_res, v4, v3, v1)
        cls.build_face(accum, faces, u_res, v3, v4, v2)

        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(accum.verts(), [], faces)

        mesh.show_normal_face = True

        return mesh

    @classmethod
    def tet_sphere_arbitrary_op(cls, scn, len1, u_res):
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
            obj = TetrahedralSphereArbitrary.tet_sphere_arbitrary_op(ctx.scene, self.len1/2.0, self.u_res)
            obj.select = True
            ctx.scene.objects.active = obj
            return {'FINISHED'}
        except ValueError as e:
#            print(traceback.format_exc())
            self.report({'ERROR'}, e.args[0])
            return {'CANCELLED'}
        except AttributeError as e:
#            print(traceback.format_exc())
            self.report({'ERROR'}, e.args[0])
            return {'CANCELLED'}
#        except Exception as e:
#            print(traceback.format_exc())
#            self.report({'ERROR'}, e.args[0])
#            return {'CANCELLED'}

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
