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
import bmesh
from mathutils import *
from math import *

#

class VertexAccumulator:
    # use this class when you can't come up with a good deterministic numbering scheme for your vertices.

    def __init__(self):
        self.verts_ = []
        self.vertIdxs = {}

    def idxFor(self, v):
        for i in range(len(self.verts_)):
            if (Vector(self.verts_[i])-Vector(v)).magnitude < 0.0001:
                return i
        rval = len(self.verts_)
        self.verts_.append(v)
        return rval

    def verts(self):
        return self.verts_

#

def invent_coordinate_system(a, b):
    a = a.normalized()
    b = b.normalized()

    c = a.cross(b)
    if (c.magnitude>0):
        c = c.normalized()
    else:
        c = Vector( [0,1,0] )
    b2 = c.cross(a)

    return [a,b2,c]


def acos_(x):
    if x>=1:
        return 0
    if x<=-1:
        return pi

    return acos(x)


class GreatCircleArc:
    def __init__(self, p1, p2):
        p2 = p2.normalized()

        self.p1, self.axis2, self.normal = invent_coordinate_system(p1, p2)

        self.cos_theta = self.p1.dot(p2)
        if (self.cos_theta>=1):
            self.theta = 0
        elif (self.cos_theta<=-1):
            self.theta = pi
        else:
            self.theta = acos(self.cos_theta)
        #print("%f\t%r,%r,%r"%(self.theta, self.p1, self.axis2, self.normal))

    def interpolate(self, t):
        """
        :param t: in the range[0..1] .  0 corresponds to p1.  1 corresponds to p2
        :return:
        """
        theta_2 = t * self.theta
        return self.p1 * cos(theta_2) + self.axis2 * sin(theta_2)

    def point_for_raw(self, ascension):
        return self.p1 * cos(ascension) + self.axis2 * sin(ascension)

    class Inset:
        def __init__(self, base, erosion):
            self.base = base
            self.erosion = erosion
            self.e2 = erosion / base.theta

        def interpolate(self, t):
            t2 = t*(1-2*self.e2) + self.e2

            return self.base.interpolate(t2)

        def theta_span(self):
            return self.base.theta - 2*self.erosion

    def inset(self, erosion):
        return GreatCircleArc.Inset(self, erosion)

class LesserCircle:
    """
    this defines the subset of the sphere whose points satisfy v.dot(normal) == offset
    """
    def __init__(self, normal, offset):
        self.normal = normal.normalized()
        self.offset = offset


    def intersect(self, lc2):

        c = self.normal.cross(lc2.normal) .normalized()

        # these if statements only work because c is normalized
        if (abs(c[2])>0.1):
            x,y = self.solve(self.normal[0], self.normal[1], self.offset,
            lc2.normal[0], lc2.normal[1], lc2.offset)
            xyz = Vector( [x,y,0] )
        elif (abs(c[0])>0.1):
            z,y = self.solve(self.normal[2], self.normal[1], self.offset,
            lc2.normal[2], lc2.normal[1], lc2.offset)
            xyz = Vector( [0,y,z] )
        else:
            x,z = self.solve(self.normal[0], self.normal[2], self.offset,
            lc2.normal[0], lc2.normal[2], lc2.offset)
            xyz = Vector( [x,0,z] )

        #print( [ xyz.dot(self.normal), self.offset ])
        #print( [ xyz.dot(lc2.normal), lc2.offset ])

        d0 = c .dot(xyz)

        v1 = xyz - d0*c

        #print("%r = %r + %f*%r"%(xyz, v1, d0, c))

        b = sqrt(max( 0, 1-v1.dot(v1)))

        v2 = v1 + b*c
        v3 = v1 - b*c

        #print("%r and %r" % (v2, v3))

        return (v2, v3)

    def solve(self, a, b, d, e, f, h):
        det = a*f - e*b
        if (det==0):
            print("tetrahedral-sphere.py: determinant = 0 ; %r"%[a,b,d,e,f,h])
        x = (d*f - h*b) / det
        y = (a*h - e*d) / det

        return [ x, y ]

class LesserCircleArc:
    def __init__(self, normal, p1, p2):
        normal = normal.normalized()
        #p2 = p2.normalized

        self.normal, self.axis1, self.axis2 = invent_coordinate_system(normal, p1)

        self.offset = p1.dot(normal)
        if False:
            o2 = p2.dot(normal)
            print(" %f = %f-%f"%(o2-self.offset, o2, self.offset))
        self.r = sqrt(1-self.offset*self.offset)

        self.center = self.offset*self.normal

        p2b = (p2 - self.center)
        #print(p2b.dot(self.normal))
        cos_theta = p2b.normalized() .dot( self.axis1)
        self.theta = acos(cos_theta)

    def interpolate(self, t):
        theta2 = t*self.theta
        return self.r * ( self.axis1 * cos(theta2) + self.axis2 * sin(theta2) ) + self.center

#

class UVAngleStabilizer:
    base=None

    def stabilize(self, theta):

        if self.base is None:
            self.base = theta
        else:
            adj = floor( 0.5 + (self.base-theta)/(2*pi))
            theta = theta + 2*pi*adj

        return theta

def closer_point(v_master, alternatives):
    d=None
    rval = None
    for v in alternatives:
        d2 = (v-v_master).magnitude
        if d is None or d2<d:
            d = d2
            rval = v
    return rval

class TetrahedralSphereArbitrary:

    def __init__(self):
        self.faces = []
        self.material_indices = []
        self.uv_maps = []
        self.accum = VertexAccumulator()

    @classmethod
    def triangle_interp(cls, va, f1, vb, f2, vc):
        return (vb - va) * (f1) + (vc - va) * (f2) + va

    @classmethod
    def artillery(cls, circle_j, v1, u_res, u, v):
        if u + v > 0:
            v2 = circle_j.interpolate(u / (u + v))
            circle_k = GreatCircleArc(v1, v2)
            return circle_k.interpolate((u + v) / u_res)
        else:
            return v1

    def add_face_v(self, mat_index, verts, uv_maps=None):
        self.add_face_i(mat_index, [ self.accum.idxFor(v) for v in verts] , uv_maps)

    def add_face_i(self, mat_index, indices, uv_maps = None):
        self.faces.append(indices)
        self.material_indices.append(mat_index)
        self.uv_maps.append(uv_maps)

    def bridge_edges(self, material_index, ring1, ring2, uv_method, inside_out=True):
        """

        :param ring1: the "bottom" ring, from left to right
        :param ring2: the "top" ring. from left to right
        :param inside_out:
        :return:
        """
        i=0
        j=0

        old_normal = None

        verts = self.accum.verts()

        while i+1 < len(ring1) or j+1 < len(ring2):
            if i+1 < len(ring1):
                if j+1 < len(ring2):

                    va = verts[ring1[i]]
                    vb = verts[ring2[j]]
                    vc = verts[ring1[i+1]]
                    vd = verts[ring2[j+1]]

                    normal_uno = (vc-va).cross(vb-va) .normalized()
                    normal_dos = (vc-vb).cross(vd-vb) .normalized()
                    convexity_espanol = normal_uno.cross(normal_dos)

                    normal_alpha = (vc-va).cross(vd-va) .normalized()
                    normal_beta = (vd-va).cross(vb-va) .normalized()
                    convexity_greek = normal_alpha.cross(normal_beta)

                    if (convexity_greek > convexity_espanol):
                        advance_ring1=True
                    else:
                        advance_ring1 = False
                else:
                        advance_ring1=True
            else:
                advance_ring1 = False

            if advance_ring1:
                idxs = [ring2[j], ring1[i], ring1[i + 1]]
                i = i + 1
            else:
                if len(ring2)>1 or j==0:
                    idxs = [ring2[j], ring1[i], ring2[j + 1]]
                    j = j + 1
                else:
                    continue
            if inside_out:
                idxs.reverse()

            self.add_face_i(material_index, idxs, uv_method(idxs))


    def lattitudish_corner(self, va, vb, vc, border_res, border_dz):
        if border_res<1:
            return
        n1 = va.cross(vb)
        n2 = vc.cross(va)
        lc_y1 = LesserCircle(n1, 0)
        for v in range(border_res):
            lc_y2 = LesserCircle(n1, (v+1)/border_res *border_dz)

            lc_x1 = LesserCircle(n2, 0)
            for u in range(border_res):
                lc_x2 = LesserCircle(n2, (u+1)/border_res *border_dz)

                v1 = closer_point(va, lc_y1.intersect(lc_x1))
                v2 = closer_point(va, lc_y1.intersect(lc_x2))
                v3 = closer_point(va, lc_y2.intersect(lc_x2))
                v4 = closer_point(va, lc_y2.intersect(lc_x1))

                self.add_face_v(1, [v1, v2, v3, v4])

                lc_x1 = lc_x2
            lc_y1 = lc_y2

        return closer_point(va, lc_y1.intersect(lc_x1))


    def grid_interpolator(self, u_res, v_res, v1, gca_12, v2, gca_23, v3, gca_43, v4, gca_14):
        verts = []
        row = []
        verts.append(row)
        row.append(v1)
        for u in range(1, u_res):
            row.append(gca_12.interpolate(u / u_res))
        row.append(v2)
        for v in range(1, v_res):
            row = []
            verts.append(row)
            v14 = gca_14.interpolate(v / v_res)
            v23 = gca_23.interpolate(v / v_res)
            row.append(v14)
            for u in range(1, u_res):
                gc = GreatCircleArc(v14, v23)
                row.append(gc.interpolate(u / u_res))
            row.append(v23)
        row = []
        verts.append(row)
        row.append(v4)
        for u in range(1, u_res):
            row.append(gca_43.interpolate(u / u_res))
        row.append(v3)

        return verts


    def diamond_corner(self, va, vb, vc, border_res, border_dz, material_index):
        """
        Build the kite-shaped corner at va which points at vb and vc
        """
        if border_res<1:
            return
        n1 = va.cross(vb)
        n2 = vc.cross(va)
        lc_y1 = LesserCircle(n1, 0)
        lc_y2 = LesserCircle(n1, border_dz)

        lc_x1 = LesserCircle(n2, 0)
        lc_x2 = LesserCircle(n2, border_dz)

        ascension = asin(border_dz)
        v1 = closer_point(va, lc_y1.intersect(lc_x1))
        v2 = GreatCircleArc(va, vb).point_for_raw(ascension)
        v3 = closer_point(va, lc_y2.intersect(lc_x2))
        v4 = GreatCircleArc(va, vc).point_for_raw(ascension)

        gca_12 = GreatCircleArc(v1, v2)
        gca_14 = GreatCircleArc(v1, v4)
        gca_23 = GreatCircleArc(v2, v3)
        gca_43 = GreatCircleArc(v4, v3)
        verts = self.grid_interpolator(border_res, border_res, v1, gca_12, v2, gca_23, v3, gca_43, v4, gca_14)

        indices = [  [ self.accum.idxFor(v) for v in row ] for row in verts]

        for v in range(border_res):
            for u in range(border_res):
                uv_maps = {
                    "default" :
                        #[ [a/border_res, b/border_res] for a in [v,v+1] for b in [u,u+1]]
                        [ [ q/border_res for q in uv] for uv in self.uv_unit_square(u,v) ]
                }
                self.add_face_i(material_index, [indices[v][u], indices[v][u+1], indices[v+1][u+1], indices[v+1][u]], uv_maps)

        return v3

    def uv_unit_square(self, u, v):
        """
        :return: a 1x1 square based at u,v.  You probably want to scale this down based on the U and V resolution
        """
        return [[u,v], [u+1, v], [u+1,v+1], [u,v+1]]


    def lattitudish_edge(self, va, vb, vc, va_2, vb_2, vc_2, arc_res, border_res, border_dz, material_index, stripe_aspect):

        gca_12 = GreatCircleArc(vb, vc).inset(asin(border_dz))

        n1 = vb.cross(vc)
        lc_y1 = LesserCircleArc(n1, vb_2, vc_2)

        v1 = gca_12.interpolate(0)
        v2 = gca_12.interpolate(1)

        verts = self.grid_interpolator(border_res, arc_res, vb_2, GreatCircleArc(vb_2, v1), v1, gca_12, v2,
                               GreatCircleArc(vc_2, v2), vc_2, lc_y1 )

        indices = [  [ self.accum.idxFor(v) for v in row ] for row in verts]

        ascension = asin(border_dz)

        v_scale = ceil(gca_12.theta_span() /ascension / stripe_aspect) / arc_res

        for v in range(len(indices)-1):
            for u in range(len(indices[v])-1):
                uv_maps = {
                    "default":
                        [ [uv[0]/border_res, uv[1] * v_scale] for uv in self.uv_unit_square(u,v)]
                }
                self.add_face_i( material_index, [indices[v][u], indices[v][u+1], indices[v+1][u+1], indices[v+1][u]], uv_maps )


    def lattitudish_edge2(self, va, vb, vc, va_2, vb_2, vc_2, u_res, border_res, border_dz):

        n1 = vb.cross(vc)
        lc_y1 = LesserCircle(n1, 0)

        n_ab = va.cross(vb).normalized()
        n_ac = va.cross(vc).normalized()
        r_interp = GreatCircleArc(n_ab, n_ac)

        lca_bc = LesserCircleArc(n1, vb_2, vc_2)

        for v in range(border_res):
            lc_y2 = LesserCircle(n1, (v+1)/border_res *border_dz)

            n2 = r_interp.interpolate(0)
            lc_x1 = LesserCircle(n2, vb_2.dot(n2))

            for u in range(u_res):
                vbct = lca_bc.interpolate(  (u+1)/u_res )
                n2 = r_interp.interpolate( (u+1)/u_res)
                if False and (u+1==u_res):
                    print("%f  ;  %r =? %r" % ((n2-n_ac).magnitude, n2, n_ac))
                    print("%f  ;  %r =? %r" % ((vbct-vc_2).magnitude, vbct, vc_2))
                lc_x2 = LesserCircle(n2, vbct.dot(n2))

                v1 = closer_point(vbct, lc_y1.intersect(lc_x1))
                v2 = closer_point(vbct, lc_y1.intersect(lc_x2))
                v3 = closer_point(vbct, lc_y2.intersect(lc_x2))
                v4 = closer_point(vbct, lc_y2.intersect(lc_x1))

                self.add_face_v( 1, [v1, v2, v3, v4] )

                lc_x1 = lc_x2
            lc_y1 = lc_y2



    def panel_uvs(self, indices):

        cylindrical_uvs = []
        spherical_uvs = []
        panel_uvs = []
        panel_cyl_uvs = []
        panel_sphere_uvs = []

        stabilizer1 = UVAngleStabilizer()
        stabilizer2 = UVAngleStabilizer()
        for i in range(len(indices)):
            v = self.accum.verts()[indices[i]]

            y = v[1]
            x = v[0]
            z = v[2]
            theta = stabilizer1.stabilize(atan2(y, x))
            phi = acos(z /v.magnitude)

            cylindrical_uvs .append( [ theta/pi, (z +1)/2])
            spherical_uvs.append( [ theta/pi, 1-phi/pi])

            y2 = self.panel_context['axis'][0].dot(v)
            z2 = self.panel_context['axis'][1].dot(v)
            x2 = self.panel_context['axis'][2].dot(v)

            theta2 = stabilizer2.stabilize(atan2(y2, x2))
            phi2 = acos_(z2/v.magnitude)

            if self.panel_context is not None:
                panel_uvs.append( [ y2/2 + 0.5, z2/2 +0.5 ] )
                panel_cyl_uvs.append( [ theta2/pi +0.5, z2/2+0.5 ] )
                panel_sphere_uvs.append( [ theta2/pi +0.5, 1-phi2/pi ] )

        rval = {
            "cylindrical" : cylindrical_uvs,
            "spherical" : spherical_uvs
        }

        if self.panel_context is not None:
            rval['panel'] = panel_uvs
            rval['panel_cyl'] = panel_cyl_uvs
            rval['panel_sphere'] = panel_sphere_uvs

        return rval


    def inset_panel(self, va, vb, vc, va_2, vb_2, vc_2, u_res, material_index):

        lca_bc = LesserCircleArc(vb.cross(vc), vb_2, vc_2)
        normal_interp = GreatCircleArc(va.cross(vb), va.cross(vc))

        x_axis = vc-vb
        y_axis = (va_2-lca_bc.interpolate(0.5))
        self.panel_context = {
            "axis": invent_coordinate_system(x_axis, y_axis)
        }

        #print(self.panel_context)

        ring1 = [ self.accum.idxFor(va_2)]
        for m in range(1, u_res+1):

            ring2 = []
            for k in range(m+1):
                v5 = lca_bc.interpolate(k/m)
                lca_a5 = LesserCircleArc(-normal_interp.interpolate(k/m), v5, va_2)
                vert = lca_a5.interpolate(1-m/u_res)
                ring2.append(self.accum.idxFor(vert))

            self.bridge_edges(material_index, ring1, ring2, self.panel_uvs)

            ring1 = ring2

        self.panel_context = None


    def build_face(self, u_res, va, vb, vc, border_res, border_dz, material_index, stripe_aspect):

        va_2 = self.diamond_corner(va, vb, vc, border_res, border_dz,0)
        vb_2 = self.diamond_corner(vb, vc, va, border_res, border_dz,0)
        vc_2 = self.diamond_corner(vc, va, vb, border_res, border_dz,0)

        n_ab = va.cross(vb).normalized()
        n_bc = vb.cross(vc).normalized()
        n_ca = vc.cross(va).normalized()

        self.lattitudish_edge(va, vb, vc, va_2, vb_2, vc_2, u_res, border_res, border_dz, 1, stripe_aspect)
        self.lattitudish_edge(vb, vc, va, vb_2, vc_2, va_2, u_res, border_res, border_dz, 1, stripe_aspect)
        self.lattitudish_edge(vc, va, vb, vc_2, va_2, vb_2, u_res, border_res, border_dz, 1, stripe_aspect)

        self.inset_panel(vc, va, vb, vc_2, va_2, vb_2, u_res, material_index)

    @classmethod
    def build_face_TS(cls, accum, faces, u_res, va, vb, vc):
        # Triangular Sweep

        # this strategy is not always pretty for arbitrary spherical triangles,
        # but for the ones we are constructing (isocelese) this decomposition is the least atrocious.

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
        circle1 = GreatCircleArc(circle_k.interpolate(0), circle_m.interpolate(0))
        #print("%r \n%r\n%r"%(circle_k.point_for(1), circle_m.point_for(1), vc))
        for u in range(u_res):

            circle2 = GreatCircleArc(circle_k.interpolate((u+1)/u_res), circle_m.interpolate((u+1)/u_res))

            v_res2 = u_res-u
            for v in range(v_res2):

                v5 = circle1.interpolate(v/v_res2)
                if v_res2>1:
                    f2 = v / (v_res2 - 1)
                else:
                    f2 = 0
                v6 = circle2.interpolate(f2)
                v8 = circle1.interpolate((v+1)/v_res2)

                #print([v5, v6, v8])

                faces.append([accum.idxFor(v) for v in [v5, v6, v8]])
                if (v < v_res2-1):
                    v7 = circle2.interpolate((v+1)/(v_res2-1))
                    faces.append([accum.idxFor(v) for v in [v6, v7, v8]])

            circle1 = circle2

    def copy_uv_information(self, bm, uv_layer):
        #print(uv_layer.name)
        for fi in range(len(bm.faces)):
            face = bm.faces[fi]
            uvs = self.uv_for(fi, uv_layer.name)
            if uvs is not None:
                # print(uvs)
                for i in range(len(face.loops)):
                    face.loops[i][uv_layer].uv = uvs[i]

    @classmethod
    def make_mesh(cls, name, len1, u_res, border_res, border_dz, stripe_aspect):

        tsa = TetrahedralSphereArbitrary()

        len2 = sqrt(1-len1*len1)

        v1 = Vector([len1, 0, len2])
        v2 = Vector([-len1, 0, len2])
        v3 = Vector([0, len1, -len2])
        v4 = Vector([0, -len1, -len2])

        tsa.build_face(u_res, v2, v1, v3, border_res, border_dz, 2, stripe_aspect)
        tsa.build_face(u_res, v1, v2, v4, border_res, border_dz, 3, stripe_aspect)
        tsa.build_face(u_res, v4, v3, v1, border_res, border_dz, 4, stripe_aspect)
        tsa.build_face(u_res, v3, v4, v2, border_res, border_dz, 5, stripe_aspect)

        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(tsa.accum.verts(), [], tsa.faces)

        mesh.validate(False)

        #mesh.show_normal_face = True
        while len(mesh.materials) < 6:
            mesh.materials.append(None)

        for i in range(len(tsa.material_indices)):
            mesh.polygons[i].material_index = tsa.material_indices[i]

        mesh.uv_textures.new("cylindrical")
        mesh.uv_textures.new("spherical")
        mesh.uv_textures.new("panel")
        mesh.uv_textures.new("panel_cyl")
        mesh.uv_textures.new("panel_sphere")

        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()

        for uvt in mesh.uv_textures:
            uv_layer = bm.loops.layers.uv[uvt.name]
            tsa.copy_uv_information(bm, uv_layer)

        bm.to_mesh(mesh)

        return mesh

    @classmethod
    def tet_sphere_arbitrary_op(cls, scn, len1, u_res, border_res, border_dz, stripe_aspect):
        name = "tetrahedral sphere"
        mesh = cls.make_mesh(name, len1, u_res, border_res, border_dz, stripe_aspect)
        obj = bpy.data.objects.new(name, mesh)
        scn.objects.link(obj)
        obj.location = scn.cursor_location
        return obj

    def uv_for(self, face_index, uv_name):
        uv_map = self.uv_maps[face_index]
        if uv_map is None:
            return None

        rval = uv_map.get(uv_name)

        if (rval is None):
            rval = uv_map.get("default")

        return rval


#

class TetrahedralSphere(bpy.types.Operator):
    """Add or adjust the transform effect on a strip to put letterboxes on the left and right or top and bottom to center the content and preserve its original aspect ratio"""
    bl_idname = "mesh.tetrahedral_sphere"
    bl_label = "Tetrahedral Sphere"
    bl_options = {'REGISTER', 'UNDO'}
    bl_region_type='UI'

    len1 = bpy.props.FloatProperty(name="Anchor Edge Length", default=1, min=0.001, max=1.999,
                                      subtype='FACTOR', precision=4, step=100,
                                      description="length of the anchor edge (the edge perpendicular to the poles")
    u_res = bpy.props.IntProperty(name="inner resolution", default=12, min=1, max=40,
                                      description="number of subdivisions of the edges")
    border_res = bpy.props.IntProperty(name="border resolution", default=3, min=1,
                                      description="number of stripes in the border")
    border_dz = bpy.props.FloatProperty(name="border size", default=0.05, min=0.001, max=0.7,
                                        precision=4, step=10,
                                        description="thickness of the edge border strips measured along each local polar axis")
    stripe_aspect = bpy.props.FloatProperty(name="border aspect", default=1, min=0.01, max=100,
                                        precision=3, step=10,
                                        description="aspect ratio of the texture used in the edge border strips")

    def execute(self, ctx):
        try:
            obj = TetrahedralSphereArbitrary.tet_sphere_arbitrary_op(ctx.scene, self.len1/2.0, self.u_res, self.border_res, self.border_dz, self.stripe_aspect)
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


def testing():
    lc1 = LesserCircle(Vector( [1,1,1] ), 0.1)
    lc2 = LesserCircle(Vector( [1,2,1] ), 0.2)

    lc1.intersect(lc2)

    lc1 = LesserCircle(Vector( [ 0,1,0]), 0.15)
    lc1.intersect(lc2)

    lc2 = LesserCircle(Vector( [ 1,0,0]), 0.1)
    lc1.intersect(lc2)


    lc1 = LesserCircle(Vector( [ 0,0,1]), 0.17)
    lc1.intersect(lc2)


    print()

def test3():
    lc1 = LesserCircle(Vector( [1,0,0] ), 0.1)
    lc2 = LesserCircle(Vector( [0, 1,0] ), 0.1)
    lc3 = LesserCircle(Vector( [0,0, 1] ), 0.1)

    va, discard = lc1.intersect(lc2)
    vb, discard = lc2.intersect(lc3)
    vc, discard = lc3.intersect(lc1)

    print(va)
    print(vb)
    print(vc)

    lca = LesserCircleArc(vb.cross(vc), vb, vc)

    print("%r =? %r" % (vb, lca.interpolate(0)))
    print("%r =? %r" % (vc, lca.interpolate(1)))


def test2():
    obj = TetrahedralSphereArbitrary.tet_sphere_arbitrary_op(bpy.context.scene, 0.5, 12, 3, 0.1)
    obj.data.materials[0] = bpy.data.materials[0]
    obj.data.materials[1] = bpy.data.materials[1]

if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
    register()
