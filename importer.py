# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2021 Tommi Hyppänen
#
# Modified 2026 by Peak-Design:
#   - Updated to pythonocc-core 7.9.3 (OpenCASCADE 7.9.3)
#   - Fixed tessellation race conditions and re-tessellation fallback
#   - Improved corrupt STEP file handling
#   - Added failed parts diagnostics
#   - Added ShapeFix healing and per-face entity recovery for broken shapes
#   - Scoped face recovery to only unmapped entities (prevents geometry bleed)

import sys
from os.path import dirname

file_dirname = dirname(__file__)
if file_dirname not in sys.path:
    sys.path.append(file_dirname)

import importlib
import os
import random
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field

import numpy as np

# import trimesh works in dev, but not in deploy
from . import trimesh
from . import nurbs

importlib.reload(trimesh)
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_Transform
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import breptools
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomConvert import geomconvert_SurfaceToBSplineSurface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp, gp_Dir, gp_Pln, gp_Pnt, gp_Pnt2d, gp_Trsf, gp_Vec, gp_XYZ

# from OCC.Core.Standard import Standard_Real
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.IMeshTools import IMeshTools_Parameters
from OCC.Core.Interface import Interface_Static
from OCC.Core.Poly import poly
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.STEPControl import STEPControl_Reader

# from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.TColStd import TColStd_SequenceOfAsciiString
from OCC.Core.TDF import TDF_Label, TDF_LabelSequence
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_INTERNAL,
    TopAbs_REVERSED,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
    topabs_ShapeTypeToString,
)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location

# from OCC.Core.TopExp import topexp_MapShapes
# from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, topods
from OCC.Core.XCAFApp import XCAFApp_Application_GetApplication
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_ColorCurv
from OCC.Core.XSControl import XSControl_WorkSession
from OCC.Core.StepBasic import StepBasic_ProductDefinition
from OCC.Core.StepRepr import (
    StepRepr_ProductDefinitionShape,
    StepRepr_ShapeRepresentationRelationship,
)
from OCC.Core.StepShape import (
    StepShape_AdvancedBrepShapeRepresentation,
    StepShape_ManifoldSolidBrep,
    StepShape_ManifoldSurfaceShapeRepresentation,
    StepShape_ShapeDefinitionRepresentation,
    StepShape_ShellBasedSurfaceModel,
)

import OCC

print("--> STEPper OpenCASCADE version:", OCC.VERSION)


def b_colorname(col):
    return Quantity_Color.StringName(Quantity_Color.Name(col))


def b_XYZ(v):
    x = v.XYZ()
    return (x.X(), x.Y(), x.Z())


def b_RGB(c):
    return (c.Red(), c.Green(), c.Blue())


def trsf_matrix(shp):
    trsf = shp.Location().Transformation()
    matrix = np.zeros((3, 4), dtype=np.float32)
    for row in range(1, 4):
        for col in range(1, 5):
            matrix[row - 1, col - 1] = trsf.Value(row, col)
    return matrix


def _test_shape(sh):
    tmr = trsf_matrix(sh)
    if np.any(tmr != np.eye(4, dtype=np.float32)[:3, :4]):
        print(tmr)


def nurbs_parse(current_face):
    """Get NURBS points for a TopAbs_FACE"""

    _test_shape(current_face)
    nurbs_converter = BRepBuilderAPI_NurbsConvert(current_face)
    nurbs_converter.Perform(current_face)
    result_shape = nurbs_converter.Shape()
    _test_shape(result_shape)
    brep_face = BRep_Tool.Surface(topods.Face(result_shape))
    occ_face = geomconvert_SurfaceToBSplineSurface(brep_face)
    # _test_shape(occ_face)

    # extract the Control Points of each face
    n_poles_u = occ_face.NbUPoles()
    n_poles_v = occ_face.NbVPoles()

    # cycle over the poles to get their coordinates
    points = []
    for pole_u_direction in range(n_poles_u):
        points.append([])
        for pole_v_direction in range(n_poles_v):
            pos = (pole_u_direction + 1, pole_v_direction + 1)
            coords = occ_face.Pole(*pos)
            np_coords = np.array((coords.X(), coords.Y(), coords.Z()))
            weight = occ_face.Weight(*pos)
            pt = nurbs.NurbsPoint((*np_coords, weight))
            points[-1].append(pt)

    # Get surface data (closed, periodic, degree)
    assert len(points) > 1
    assert len(points[0]) > 1
    nbd = nurbs.NurbsData(points)

    nbd.u_closed = occ_face.IsUClosed()
    nbd.v_closed = occ_face.IsVClosed()
    nbd.u_periodic = occ_face.IsUPeriodic()
    nbd.v_periodic = occ_face.IsVPeriodic()
    nbd.u_degree = occ_face.UDegree()
    nbd.v_degree = occ_face.VDegree()

    return nbd


def force_ascii(i_file):
    from pathlib import Path

    print("Attempting to format STEP file as ASCII 7-bit")
    p = Path(i_file)
    print(p.stat().st_size // 1024, "kB")
    import tempfile

    with tempfile.NamedTemporaryFile("w", encoding="ASCII") as fo:
        temp_name = fo.name
        print(temp_name)
        with p.open("rb") as f:
            while il := f.readline():
                fo.write(il.decode("ASCII"))
    print("done ASCII conversion.")
    return temp_name


# TODO: proper parametrization
def equalize_2d_points(pts):
    """Equalize aspect ratio of 2D point dimensions"""
    x_a, x_b = 1.0, 0.0
    y_a, y_b = 1.0, 0.0

    for i, uv in enumerate(pts):
        if uv[0] < x_a:
            x_a = uv[0]
        if uv[0] > x_b:
            x_b = uv[0]
        if uv[1] < y_a:
            y_a = uv[1]
        if uv[1] > y_b:
            y_b = uv[1]

    rx = abs(x_b - x_a)
    ry = abs(y_b - y_a)
    if rx != 0.0 and ry != 0.0:
        ratio = rx / ry
    else:
        ratio = 1.0

    ratio1 = 1 / ratio
    for i, uv in enumerate(pts):
        pts[i] = (pts[i][0] * ratio1, pts[i][1])

    return pts


@dataclass
class ShapeTreeNode:
    """
    A node for the OpenCASCADE CAD data ShapeTree
    """

    parent: int
    index: int
    tag: int
    name: str
    children: list[int] = field(default_factory=list)
    local_transform: np.ndarray = field(default_factory=np.eye(4, dtype=np.float32))
    global_transform: np.ndarray = field(default_factory=np.eye(4, dtype=np.float32))
    shape: TopoDS_Shape = None

    def __init__(self, parent, index, tag, name):
        self.parent = parent
        self.index = index
        self.tag = tag
        self.name = name
        self.children = []
        self.local_transform = np.eye(4, dtype=np.float32)
        self.global_transform = np.eye(4, dtype=np.float32)
        self.shape = None

    def get_values(self):
        """
        parent, index, tag, name
        """
        return (
            self.parent,
            self.index,
            self.tag,
            self.name,
            self.shape,
            self.local_transform,
            self.global_transform,
        )

    def set_shape(self, shape):
        if shape:
            if not isinstance(shape, TopoDS_Shape):
                raise ValueError("Input shape is not OpenCASCADE TopoDS_Shape")
            self.shape = shape
        else:
            self.shape = None


class ShapeTree:
    """
    Intermediary data structure to partially abstract OpenCASCADE away from the rest of the program
    """

    def __init__(self):
        self.nodes = []

        # Root node has special values
        self.nodes.append(ShapeTreeNode(-1, 0, -1, "root"))

    def get_root_id(self):
        return 0

    def get_max_id(self):
        return len(self.nodes) - 1

    def add(self, parent, label) -> ShapeTreeNode:
        loc = len(self.nodes)
        node = ShapeTreeNode(parent, loc, label.Tag(), label.GetLabelName())
        self.nodes[parent].children.append(loc)
        self.nodes.append(node)
        return self.nodes[-1]

    def get_shapes(self):
        # return {i.shape: i.index for i in self.nodes if i.shape}
        return [(i.shape, i.index) for i in self.nodes]

    def print_transforms(self):
        for i in self.nodes:
            print(i.local_transform)


class ReadSTEP:
    def __init__(self, filename):
        self.read_file(filename)

    def query_color(self, label, overwrite=False):
        # default color = pink
        c = Quantity_Color(1.0, 0.0, 1.0, Quantity_TOC_RGB)
        colorset = False
        colortype = None

        c_gen = self.color_tool.GetColor(label, int(XCAFDoc_ColorGen), c)
        c_surf = self.color_tool.GetColor(label, int(XCAFDoc_ColorSurf), c)
        c_curv = self.color_tool.GetColor(label, int(XCAFDoc_ColorCurv), c)
        if c_gen or c_surf or c_curv:
            colorset = True
            colortype = c_gen * 1 + c_surf * 2 + c_curv * 3

        return c, colortype, colorset

    def print_all_colors(self):
        tcol = Quantity_Color(1.0, 0.0, 1.0, Quantity_TOC_RGB)
        clabs = TDF_LabelSequence()
        self.color_tool.GetColors(clabs)
        for i in range(clabs.Length()):
            res = self.color_tool.GetColor(clabs.Value(i + 1), tcol)
            if res:
                print(b_colorname(tcol))

    def label_matrix(self, lab):
        trsf = self.shape_tool.GetLocation(lab).Transformation()
        matrix = np.eye(4, dtype=np.float32)
        for row in range(1, 4):
            for col in range(1, 5):
                matrix[row - 1, col - 1] = trsf.Value(row, col)
        # print(matrix)
        return matrix

    def explore_partial(self, shp, te_type):
        c_set = set([])
        ex = TopExp_Explorer(shp, te_type)
        # Todo: use label->tag
        while ex.More():
            c = ex.Current()
            if c not in c_set:
                c_set.add(c)
            ex.Next()
        return len(c_set)

    def explore_shape(self, shp):
        return (
            self.explore_partial(shp, TopAbs_COMPOUND),
            self.explore_partial(shp, TopAbs_SOLID),
            self.explore_partial(shp, TopAbs_SHELL),
            self.explore_partial(shp, TopAbs_FACE),
            self.explore_partial(shp, TopAbs_WIRE),
            self.explore_partial(shp, TopAbs_EDGE),
            self.explore_partial(shp, TopAbs_VERTEX),
        )

    def shape_info(self, shp):
        st = self.shape_tool
        lab = self.shape_label[shp]
        vals = (
            st.IsAssembly(lab),
            st.IsFree(lab),
            st.IsShape(lab),
            st.IsCompound(lab),
            st.IsComponent(lab),
            st.IsSimpleShape(lab),
            shp.Locked(),
        )

        lookup = ["A", "F", "S", "C", "T", "s", "L"]
        res = "".join([lookup[i] for i, v in enumerate(vals) if v])

        # res += f", C:{shp.NbChildren()}"

        res += ", C:{} So:{} Sh:{} F:{} Wi:{} E:{} V:{}".format(*self.explore_shape(shp))

        return " " + res + " "

    def transfer_with_units(self, filename):
        print("Init transfer with units")

        # Init new doc and reader
        doc = TDocStd_Document("STEP")
        step_reader = STEPCAFControl_Reader()
        step_reader.SetColorMode(True)
        step_reader.SetNameMode(True)
        step_reader.SetMatMode(True)
        step_reader.SetLayerMode(True)

        # Read simple STEP file for correct units
        session = XSControl_WorkSession()
        step_simple_reader = STEPControl_Reader(session)

        print("DataExchange: Reading STEP")

        status = step_simple_reader.ReadFile(filename)
        if status != IFSelect_RetDone:
            raise AssertionError("Error: can't read file. File possibly damaged.")

        print("STEP read into memory")

        # Check the STEP data model for unresolved references
        has_data_failures = False
        try:
            step_model = step_simple_reader.StepModel()
            if step_model is not None:
                global_check = step_model.GlobalCheck(True)
                if global_check.HasFailed():
                    has_data_failures = True
                    nb_fails = global_check.NbFails()
                    print(f"WARNING: STEP file has {nb_fails} data model failure(s).")
                    print("This may indicate unresolved references.")
        except Exception as e:
            print(f"STEP model pre-check failed: {e}")

        # read units
        ulen_names = TColStd_SequenceOfAsciiString()
        uang_names = TColStd_SequenceOfAsciiString()
        usld_names = TColStd_SequenceOfAsciiString()
        step_simple_reader.FileUnits(ulen_names, uang_names, usld_names)

        # Info about unit conversions
        # https://dev.opencascade.org/content/step-unit-conversion-and-meshing

        # for i in range(ulen_names.Length()):
        #     ulen = ulen_names.Value(i + 1)
        #     uang = uang_names.Value(i + 1)
        #     usld = usld_names.Value(i + 1)
        #     print(ulen.ToCString(), uang.ToCString(), usld.ToCString())

        # default is MM
        scale = 0.001

        if ulen_names.Length() > 0:
            scaleval = ulen_names.Value(1).ToCString().lower()

            # INCH, MM, FT, MI, M, KM, MIL, CM
            # UM, UIN ??

            scales = {
                "millimeter": 0.001,
                "millimetre": 0.001,
                "centimeter": 0.01,
                "centimetre": 0.01,
                "kilometer": 1000.0,
                "kilometre": 1000.0,
                "meter": 1.0,
                "metre": 1.0,
                "inch": 0.0254,
                "foot": 0.3048,
                "mile": 1609.34,
                "mil": 0.0254 * 0.001,
            }

            if scaleval in scales:
                scale = scales[scaleval]
            else:
                print("ERROR: Undefined scale:", scaleval)

            print("Scale from file (meters per unit):", scaleval, scale)

        else:
            print("Using default scale (millimeters)")

        self.scale = scale

        status = step_reader.ReadFile(self.filename)
        assert status == IFSelect_RetDone

        print("DataExchange: Transferring")

        # Try transfer with default surface curve mode first (preserves more
        # geometry).  Only fall back to mode 0 (skip surface curves) when the
        # transfer crashes — this can happen on files with unresolved refs.
        # See: https://dev.opencascade.org/content/loading-step-file-crashes-edgeloop
        transfer_ok = False
        if has_data_failures:
            try:
                transfer_result = step_reader.Transfer(doc)
                transfer_ok = bool(transfer_result)
                if transfer_ok:
                    print("DataExchange: Transfer done")
                else:
                    print("DataExchange: Transfer returned failure, retrying in safe mode...")
            except Exception as e:
                print(f"DataExchange: Transfer failed ({e}), retrying in safe mode...")

            if not transfer_ok:
                # Retry with mode 0: discard surface curves from file
                from OCC.Core.Interface import Interface_Static
                Interface_Static.SetIVal("read.surfacecurve.mode", 0)
                doc = TDocStd_Document("STEP")
                step_reader = STEPCAFControl_Reader()
                step_reader.SetColorMode(True)
                step_reader.SetNameMode(True)
                step_reader.SetMatMode(True)
                step_reader.SetLayerMode(True)
                step_reader.ReadFile(self.filename)
                transfer_result = step_reader.Transfer(doc)
                if not transfer_result:
                    print("DataExchange: Safe mode transfer also FAILED.")
                else:
                    print("DataExchange: Transfer done (safe mode)")
                # Reset to default for any subsequent imports
                Interface_Static.SetIVal("read.surfacecurve.mode", 1)
        else:
            transfer_result = step_reader.Transfer(doc)
            if not transfer_result:
                print("Dataexchange transfer FAILED.")
            else:
                print("DataExchange: Transfer done")

        self.doc = doc
        self._xcaf_reader = step_reader

    def transfer_simple(self, fname):
        # see stepanalyzer.py for license details
        print("Init simple transfer")

        # Create the application, empty document and shape_tool
        doc = TDocStd_Document("STEP")
        app = XCAFApp_Application_GetApplication()
        app.NewDocument("MDTV-XCAF", doc)

        # Read file and return populated doc
        step_reader = STEPCAFControl_Reader()
        step_reader.SetColorMode(True)
        step_reader.SetLayerMode(True)
        step_reader.SetNameMode(True)
        step_reader.SetMatMode(True)
        status = step_reader.ReadFile(fname)
        if status == IFSelect_RetDone:
            step_reader.Transfer(doc)
        self.scale = 0.001

        self.doc = doc

    def init_reader(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError("%s not found." % filename)

        # self.filename = force_ascii(filename)
        self.filename = filename

        self.transfer_with_units(self.filename)
        # self.transfer_simple(self.filename)

        self.shape_tool = XCAFDoc_DocumentTool.ShapeTool(self.doc.Main())
        self.color_tool = XCAFDoc_DocumentTool.ColorTool(self.doc.Main())

        # material_tool = XCAFDoc_DocumentTool_MaterialTool(doc.Main())
        # layer_tool = XCAFDoc_DocumentTool_LayerTool(doc.Main())

        # use OrderedDict and make sure the order is maintained through the entire pipeline

        self.shape_label = {}
        self.sub_shapes = OrderedDict()

        self.face_colors = {}
        self.face_color_priority = {}
        self.tag_info = {}
        self.skipped_shapes = set([])
        self.import_problems = {"Triangulation": 0, "Undefined normals": 0, "Empty shape": 0}
        self.failed_parts = []  # List of names for parts that produced no geometry
        self.recovered_parts = []  # List of names for parts recovered via per-face transfer
        self._pre_tessellated = False
        self._recovery_compounds = None  # Lazy-built per-shape recovery compounds

    def read_file(self, filename):
        """Returns list of tuples (topods_shape, label, color)
        Use OCAF.
        """

        self.init_reader(filename)

        # output_shapes = {}
        # outliers = defaultdict(set)

        def _cprio(lab, shape):
            "Get label color"
            tc, ctype, ok = self.query_color(lab)
            self.face_colors[shape] = tc if ok else None
            if ok:
                return ctype
            else:
                return 0

        def _get_sub_shapes(lab, level, tree, leaf_id):

            # print(" " * (2 * level) + lab.GetLabelName())
            master_leaf = tree.nodes[leaf_id]
            # l_comps = TDF_LabelSequence()
            # self.shape_tool.GetComponents(lab, l_comps)
            if self.shape_tool.IsAssembly(lab):
                # Get transform for pure transform (empty)
                # Empty has eye transform, inherit global from parent

                # empty = tree.add(leaf.index, lab, empty=True)
                # output_shapes[shape] = empty

                # Read contained shapes
                l_c = TDF_LabelSequence()
                self.shape_tool.GetComponents(lab, l_c)
                for i in range(l_c.Length()):
                    label = l_c.Value(i + 1)
                    if self.shape_tool.IsReference(label):
                        label_reference = TDF_Label()
                        self.shape_tool.GetReferredShape(label, label_reference)

                        label_transform = self.label_matrix(label)
                        node = tree.add(master_leaf.index, label_reference)
                        new_leaf = tree.nodes[node.index]
                        new_leaf.local_transform = label_transform
                        new_leaf.global_transform = master_leaf.global_transform @ label_transform

                        _get_sub_shapes(label_reference, level + 1, tree, node.index)
                    else:
                        # TODO: process rest of the data
                        pass

            elif self.shape_tool.IsSimpleShape(lab):
                # TODO: self.shape_label stops being unique when shapes aren't transformed
                shape = self.shape_tool.GetShape(lab)
                master_leaf.set_shape(shape)
                if shape in self.shape_label:
                    # Shape already in
                    return

                self.shape_label[shape] = lab

                self.face_color_priority[shape] = _cprio(lab, shape)

                l_subss = TDF_LabelSequence()
                self.shape_tool.GetSubShapes(lab, l_subss)
                self.sub_shapes[shape] = []
                for i in range(l_subss.Length()):
                    lab_subs = l_subss.Value(i + 1)
                    shape_sub = self.shape_tool.GetShape(lab_subs)
                    self.shape_label[shape_sub] = lab_subs
                    self.sub_shapes[shape].append(shape_sub)
                    self.face_color_priority[shape_sub] = _cprio(lab_subs, shape_sub)
                # Color priority is the same as CAD assistant material tree display
            else:
                print("DataExchange error: Item is neither assembly or a simple shape")

        def _get_shapes():
            # self.shape_tool.UpdateAssemblies()

            labels = TDF_LabelSequence()
            self.shape_tool.GetFreeShapes(labels)

            tree = ShapeTree()
            for i in range(labels.Length()):
                print("DataExchange: Reading shape ({}/{})".format(i + 1, labels.Length()))

                root_item = labels.Value(i + 1)
                node = tree.add(tree.get_root_id(), root_item)
                _get_sub_shapes(root_item, 0, tree, node.index)

            return tree

        tree = _get_shapes()
        self.tree = tree

    def pre_tessellate_all(self, lin_def=0.8, ang_def=0.5):
        """Pre-tessellate all unique shapes using thread pool for parallelism.

        OpenCASCADE SWIG bindings release the GIL during C++ calls, so
        threading provides real speedup for CPU-bound tessellation.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        shapes = list(self.sub_shapes.keys())
        if not shapes:
            self._pre_tessellated = True
            return

        def _tessellate_group(shape):
            brt = breptools()
            iter_shapes = [shape] + self.sub_shapes.get(shape, [])
            for shp in iter_shapes:
                brt.Clean(shp)
                ex = TopExp_Explorer(shp, TopAbs_FACE)
                if ex.More():
                    brepmesh = BRepMesh_IncrementalMesh(shp, lin_def, False, ang_def, False)
                    brepmesh.Perform()

        n_workers = min(len(shapes), os.cpu_count() or 4)

        if n_workers > 1 and len(shapes) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_tessellate_group, s): s for s in shapes}
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        print(f"Warning: pre-tessellation failed for a shape: {e}")
        else:
            for s in shapes:
                _tessellate_group(s)

        self._pre_tessellated = True

    def triangulate_face(self, face, tform):
        bt = BRep_Tool()
        location = TopLoc_Location()
        facing = bt.Triangulation(face, location)
        if facing is None:
            # Mesh error, no triangulation found for part
            self.import_problems["Triangulation"] += 1
            return None

        normcalc = poly()
        normcalc.ComputeNormals(facing)

        # nsurf = bt.Surface(face)
        surface = BRepAdaptor_Surface(face)
        prop = BRepLProp_SLProps(surface, 2, gp.Resolution())
        # prop = BRepLProp_SLProps(surface, 2, 1e-4)
        # face_uv = facing.UVNode()

        # Calculate UV bounds
        Umin, Umax, Vmin, Vmax = 0.0, 0.0, 0.0, 0.0
        # for t in range(1, face_uv.Length()):
        for t in range(1, facing.NbNodes() + 1):
            # v = face_uv.Value(t)
            v = facing.UVNode(t)
            x, y = v.X(), v.Y()
            if t == 1:
                Umin, Umax, Vmin, Vmax = x, x, y, y
            if x < Umin:
                Umin = x
            if x > Umax:
                Umax = x
            if y < Vmin:
                Vmin = y
            if y > Vmax:
                Vmax = y

        Ucenter = (Umin + Umax) * 0.5
        Vcenter = (Vmin + Vmax) * 0.5

        # tab = facing.Nodes()
        tri = facing.Triangles()

        verts = []
        norms = []
        tris = []
        uvs = []

        undef_normals = False

        itform = tform.Inverted()

        # Build normals
        d_nbnodes = facing.NbNodes()
        for t in range(1, d_nbnodes + 1):
            # pt = tab.Value(t)
            pt = facing.Node(t)
            loc = b_XYZ(pt)

            # nvert = bm.verts.new(loc)
            # nvert.index = t - 1

            # assert len(loc) == 3
            # assert loc[0] is float
            # assert loc is tuple
            verts.append(loc)

            # Get triangulation normal

            # pt = gp_Pnt(loc[0], loc[1], loc[2])
            # pt_surf = GeomAPI_ProjectPointOnSurf(pt, nsurf)
            # fU, fV = pt_surf.Parameters(1)
            # prop = GeomLProp_SLProps(nsurf, fU, fV, 2, gp.Resolution())

            uv = facing.UVNode(t)
            u, v = uv.X(), uv.Y()
            uvs.append((u, v))

            # The edges of UV give invalid normals, hence this
            prop.SetParameters((u - Ucenter) * 0.999 + Ucenter, (v - Vcenter) * 0.999 + Vcenter)

            if prop.IsNormalDefined():
                normal = prop.Normal().Transformed(itform)
                # normal = prop.Normal()
                nn = np.array(b_XYZ(normal))
                if face.Orientation() == TopAbs_REVERSED:
                    nn = -nn
            else:
                nn = np.array((0.0, 0.0, 1.0))
                undef_normals = True

            # norms.append(tuple(float(nnn) for nnn in nn))
            norms.append(np.float32(nn))

        # Build triangulation
        d_nbtriangles = facing.NbTriangles()
        for t in range(1, d_nbtriangles + 1):
            T1, T2, T3 = tri.Value(t).Get()

            if face.Orientation() != TopAbs_FORWARD:
                T1, T2 = T2, T1

            # v_list = (verts[T1 - 1], verts[T2 - 1], verts[T3 - 1])
            # nf = bm.faces.new(v_list)
            # nf.smooth = True
            # nf.normal_update()
            tris.append((T1 - 1, T2 - 1, T3 - 1))

            # for v in (T1, T2, T3):
            #     if norms[v - 1] is None:
            #         added_norms.append(np.array(nf.normal))
            #     else:
            #         added_norms.append(norms[v - 1])

            # new_norms.append(norms[v - 1])

        if undef_normals:
            self.import_problems["Undefined normals"] += 1

        tri_data = []
        for ti, t in enumerate(tris):
            tri_data.append(trimesh.TriData(t, [norms[i] for i in t], [uvs[i] for i in t], None, None, None, None))

        return trimesh.TriMesh(verts=verts, tris=tri_data)

    def _build_recovery_compounds(self):
        """Build per-shape recovery compounds for parts that failed to transfer.

        For each empty shape, we need to recover exactly the faces belonging
        to that shape — not faces from other parts.  The approach:

        1. Index all AdvancedFace entities by hash for hierarchy matching.
        2. Build a ProductDefinition → Representation mapping by navigating:
           ShapeDefinitionRepresentation → ProductDefinitionShape → PD, and
           ShapeRepresentationRelationship to resolve generic → specific reprs.
        3. For each failed representation (not in the TransientProcess),
           traverse its hierarchy to collect its AdvancedFace entity indices.
        4. Transfer each repr's faces into a separate compound.
        5. Map each compound by ProductDefinition entity number so that
           _get_recovery_compound(shape) can match via EntityFromShapeResult.

        Results are cached in self._recovery_compounds (dict: PD# → compound).
        """
        if self._recovery_compounds is not None:
            return  # already built

        self._recovery_compounds = {}
        try:
            basic_reader = self._xcaf_reader.ChangeReader()
            tr = basic_reader.WS().TransferReader()
            tp = tr.TransientProcess()
            xcaf_model = basic_reader.StepModel()
            ne = xcaf_model.NbEntities()

            # Index all AdvancedFace entities by hash for hierarchy matching
            af_hash_to_idx = {}
            for i in range(1, ne + 1):
                ent = xcaf_model.Value(i)
                if ent is not None and ent.DynamicType().Name() == 'StepShape_AdvancedFace':
                    af_hash_to_idx[hash(ent)] = i

            if not af_hash_to_idx:
                return

            def _collect_faces_from_shell(shell, face_set):
                """Collect AdvancedFace model indices from a ConnectedFaceSet."""
                for m in range(1, shell.NbCfsFaces() + 1):
                    idx = af_hash_to_idx.get(hash(shell.CfsFacesValue(m)))
                    if idx is not None:
                        face_set.add(idx)

            # --- Build ProductDefinition → Representation mapping ---

            # Resolve generic ShapeRepresentation → specific MSSR/ABSR
            # via ShapeRepresentationRelationship
            generic_to_specific = {}
            for i in range(1, ne + 1):
                ent = xcaf_model.Value(i)
                if ent is None or ent.DynamicType().Name() != 'StepRepr_ShapeRepresentationRelationship':
                    continue
                srr = StepRepr_ShapeRepresentationRelationship.DownCast(ent)
                rep1, rep2 = srr.Rep1(), srr.Rep2()
                t1 = rep1.DynamicType().Name() if rep1 else ""
                t2 = rep2.DynamicType().Name() if rep2 else ""
                if 'ManifoldSurface' in t2 or 'AdvancedBrep' in t2:
                    generic_to_specific[xcaf_model.Number(rep1)] = rep2
                elif 'ManifoldSurface' in t1 or 'AdvancedBrep' in t1:
                    generic_to_specific[xcaf_model.Number(rep2)] = rep1

            # SDR → PDS → PD, giving us PD entity number → repr entity
            pd_num_to_repr_ent = {}
            for i in range(1, ne + 1):
                ent = xcaf_model.Value(i)
                if ent is None or ent.DynamicType().Name() != 'StepShape_ShapeDefinitionRepresentation':
                    continue
                sdr = StepShape_ShapeDefinitionRepresentation.DownCast(ent)
                used_repr = sdr.UsedRepresentation()
                if used_repr is None:
                    continue
                actual_repr = generic_to_specific.get(
                    xcaf_model.Number(used_repr), used_repr)
                defn = sdr.Definition()
                if defn is None:
                    continue
                try:
                    pds = StepRepr_ProductDefinitionShape.DownCast(defn.Value())
                    pd = StepBasic_ProductDefinition.DownCast(
                        pds.Definition().Value())
                    pd_num_to_repr_ent[xcaf_model.Number(pd)] = actual_repr
                except Exception:
                    pass

            # --- Collect faces per failed representation ---
            repr_num_to_faces = {}  # repr entity number → set of face indices

            for i in range(1, ne + 1):
                ent = xcaf_model.Value(i)
                if ent is None or tp.MapIndex(ent) > 0:
                    continue
                tname = ent.DynamicType().Name()
                faces = set()

                if tname == 'StepShape_ManifoldSurfaceShapeRepresentation':
                    mssr = StepShape_ManifoldSurfaceShapeRepresentation.DownCast(ent)
                    for j in range(1, mssr.NbItems() + 1):
                        item = mssr.ItemsValue(j)
                        if item.DynamicType().Name() != 'StepShape_ShellBasedSurfaceModel':
                            continue
                        sbsm = StepShape_ShellBasedSurfaceModel.DownCast(item)
                        for k in range(1, sbsm.NbSbsmBoundary() + 1):
                            shell_select = sbsm.SbsmBoundaryValue(k)
                            shell = None
                            try:
                                shell = shell_select.ClosedShell()
                            except Exception:
                                pass
                            if shell is None:
                                try:
                                    shell = shell_select.OpenShell()
                                except Exception:
                                    pass
                            if shell is not None:
                                _collect_faces_from_shell(shell, faces)

                elif tname == 'StepShape_AdvancedBrepShapeRepresentation':
                    absr = StepShape_AdvancedBrepShapeRepresentation.DownCast(ent)
                    for j in range(1, absr.NbItems() + 1):
                        item = absr.ItemsValue(j)
                        if item.DynamicType().Name() == 'StepShape_ManifoldSolidBrep':
                            msb = StepShape_ManifoldSolidBrep.DownCast(item)
                            outer = msb.Outer()
                            if outer is not None:
                                _collect_faces_from_shell(outer, faces)

                if faces:
                    repr_num_to_faces[i] = faces

            if not repr_num_to_faces:
                return

            # --- Build PD# → repr# mapping (only for failed reprs) ---
            pd_num_to_repr_num = {}
            for pd_num, repr_ent in pd_num_to_repr_ent.items():
                repr_num = xcaf_model.Number(repr_ent)
                if repr_num in repr_num_to_faces:
                    pd_num_to_repr_num[pd_num] = repr_num

            # --- Transfer faces per repr into compounds ---
            reader = STEPControl_Reader()
            status = reader.ReadFile(self.filename)
            if status != IFSelect_RetDone:
                return

            recovery_model = reader.StepModel()
            repr_num_to_compound = {}

            for repr_num, face_indices in repr_num_to_faces.items():
                print(f"[recovering {len(face_indices)} faces]", end="", flush=True)
                builder = BRep_Builder()
                compound = TopoDS_Compound()
                builder.MakeCompound(compound)
                ok = 0
                for idx in face_indices:
                    try:
                        if reader.TransferEntity(recovery_model.Value(idx)):
                            ns = reader.NbShapes()
                            if ns > 0:
                                builder.Add(compound, reader.Shape(ns))
                                ok += 1
                    except Exception:
                        pass
                print(f"[{ok}/{len(face_indices)} OK]", end="", flush=True)
                if ok > 0:
                    repr_num_to_compound[repr_num] = compound

            # --- Build final PD# → compound map ---
            for pd_num, repr_num in pd_num_to_repr_num.items():
                compound = repr_num_to_compound.get(repr_num)
                if compound is not None:
                    self._recovery_compounds[pd_num] = compound

            # Store TransferReader reference for shape→entity lookup
            self._transfer_reader = tr

        except Exception as e:
            print(f"[recovery failed: {e}]", end="", flush=True)

    def _get_recovery_compound(self, shape):
        """Return the recovery compound for a specific empty shape, or None.

        Uses EntityFromShapeResult to find the STEP ProductDefinition entity
        for the shape, then looks up the pre-built compound for that PD.
        """
        self._build_recovery_compounds()
        if not self._recovery_compounds:
            return None

        # Find the ProductDefinition entity for this shape
        tr = getattr(self, '_transfer_reader', None)
        if tr is None:
            return None

        model = self._xcaf_reader.ChangeReader().StepModel()
        for mode in [-1, 1]:
            try:
                ent = tr.EntityFromShapeResult(shape, mode)
                if ent is not None:
                    pd_num = model.Number(ent)
                    compound = self._recovery_compounds.get(pd_num)
                    if compound is not None:
                        return compound
            except Exception:
                pass

        return None

    def _heal_shape(self, shp):
        """Attempt to repair a shape using ShapeFix.

        Returns the healed shape, or the original if healing fails or makes
        things worse.
        """
        try:
            fixer = ShapeFix_Shape(shp)
            fixer.Perform()
            healed = fixer.Shape()
            # Only use healed shape if it actually has faces
            ex = TopExp_Explorer(healed, TopAbs_FACE)
            if ex.More():
                return healed
        except Exception:
            pass
        return shp

    def _tessellate_shape(self, shp, lin_def, ang_def):
        """Tessellate a shape, retrying with relaxed tolerances and shape
        healing if the first attempt produces no triangulation."""
        breptools().Clean(shp)
        brepmesh = BRepMesh_IncrementalMesh(shp, lin_def, False, ang_def, False)
        brepmesh.Perform()

        # Check if tessellation produced any triangles
        test_loc = TopLoc_Location()
        ex = TopExp_Explorer(shp, TopAbs_FACE)
        has_tris = False
        while ex.More():
            face = topods.Face(ex.Current())
            if BRep_Tool().Triangulation(face, test_loc) is not None:
                has_tris = True
                break
            ex.Next()

        if has_tris:
            return shp

        # Retry 1: heal shape then tessellate
        healed = self._heal_shape(shp)
        if healed is not shp:
            breptools().Clean(healed)
            brepmesh = BRepMesh_IncrementalMesh(healed, lin_def, False, ang_def, False)
            brepmesh.Perform()
            print("[healed]", end="", flush=True)
            return healed

        # Retry 2: relax tolerances significantly
        breptools().Clean(shp)
        brepmesh = BRepMesh_IncrementalMesh(shp, lin_def * 4.0, False, ang_def * 2.0, False)
        brepmesh.Perform()
        print("[relaxed-tess]", end="", flush=True)
        return shp

    def build_trimesh(self, shape, lin_def=0.8, ang_def=0.5, hacks=set([])):
        out_mesh = trimesh.TriMesh()
        out_mesh.matrix = np.eye(4, dtype=np.float32)

        # TODO: this is hack
        if "skip_solids" in hacks and self.explore_partial(shape, TopAbs_SOLID) == 0:
            self.skipped_shapes.add(self.shape_label[shape].GetLabelName())
            return out_mesh

        # Check if the main shape has any faces at all; if not, try
        # per-face entity recovery from the STEP file before iterating.
        all_faces = TopExp_Explorer(shape, TopAbs_FACE)
        all_subs_empty = not all_faces.More()
        if all_subs_empty:
            for ss in self.sub_shapes.get(shape, []):
                ex_ss = TopExp_Explorer(ss, TopAbs_FACE)
                if ex_ss.More():
                    all_subs_empty = False
                    break

        recovered_shape = None
        if all_subs_empty:
            recovered_shape = self._get_recovery_compound(shape)
            if recovered_shape is not None:
                # Use recovered compound as the sole shape; discard sub_shapes
                # since we no longer have XCAF label associations.
                self.face_colors[recovered_shape] = self.face_colors.get(shape)
                self.face_color_priority[recovered_shape] = self.face_color_priority.get(shape, 0)
                label = self.shape_label.get(shape)
                if label is not None:
                    self.recovered_parts.append(label.GetLabelName())

        iter_shapes = [shape] + self.sub_shapes[shape]
        if recovered_shape is not None:
            iter_shapes = [recovered_shape]
        iter_shapes.sort(key=lambda x: x.Checked())

        face_data = OrderedDict()
        batch = 0

        # Iterate over the main shape and its sub shapes
        for shp_i, shp in enumerate(iter_shapes):
            col = self.face_colors.get(shp)
            if col is not None:
                col_rgb = b_RGB(col)
                col_name = b_colorname(col)
            else:
                col_name = ""

            # Subshape transforms can be different from the mainshape transform
            ex = TopExp_Explorer(shp, TopAbs_FACE)
            if not ex.More():
                # Shape has no faces — try healing before giving up
                healed = self._heal_shape(shp)
                if healed is not shp:
                    shp = healed
                    ex = TopExp_Explorer(shp, TopAbs_FACE)
                if not ex.More():
                    self.import_problems["Empty shape"] += 1
                    continue

            # Skip meshing if already done by pre_tessellate_all
            if not self._pre_tessellated:
                shp = self._tessellate_shape(shp, lin_def, ang_def)
                # Re-create explorer after potential shape replacement from healing
                ex = TopExp_Explorer(shp, TopAbs_FACE)
            else:
                # Verify pre-tessellation succeeded; if not, re-tessellate.
                # Threading race conditions in pre_tessellate_all can cause
                # BRepMesh to silently produce no output for some shapes
                # (especially COMPOUND shapes that share internal topology).
                test_loc = TopLoc_Location()
                test_face = topods.Face(ex.Current())
                if BRep_Tool().Triangulation(test_face, test_loc) is None:
                    shp = self._tessellate_shape(shp, lin_def, ang_def)
                    ex = TopExp_Explorer(shp, TopAbs_FACE)
                    print("[re-tess]", end="", flush=True)
            trf = shp.Location().Transformation()
            # Iterate through faces with TopExp_Explorer
            while ex.More():
                exc = ex.Current()
                face = topods.Face(exc)

                try:
                    mesh = self.triangulate_face(face, trf)
                except Exception:
                    # Individual face tessellation/triangulation error —
                    # skip this face but keep processing the rest
                    mesh = None

                if mesh:
                    # If shape or sub-shape has defined color, set it so
                    mesh.set_batch(batch)
                    if col is not None:
                        mesh.colorize(col_rgb)
                        mesh.set_material_name(col_name)

                    # First filter in overwriting a face/color
                    face_data[face] = (0, mesh, "EMPTY")

                ex.Next()
                batch += 1

        for fc, b in face_data.items():
            prio, mesh, col_name = b
            if len(mesh.verts) > 0:
                out_mesh.add_mesh(mesh)

        print("[l]", end="", flush=True)

        return out_mesh

    def build_nurbs(self, shape):
        iter_shapes = [shape]
        nbs = []
        for shp_i, shp in enumerate(iter_shapes):
            ex = TopExp_Explorer(shp, TopAbs_FACE)
            if not ex.More():
                self.import_problems["Empty shape"] += 1
                return []

            while ex.More():
                pt = nurbs_parse(topods.Face(ex.Current()))
                nbs.append(pt)
                ex.Next()

        return nbs
