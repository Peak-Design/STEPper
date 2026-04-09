/*
 * stepper_native.cpp — Fast mesh extraction from OCC triangulations.
 *
 * Python extension module that accepts SWIG pointer addresses for
 * TopoDS_Face and gp_Trsf objects, extracts triangulation data
 * (vertices, normals, UVs, faces) from each face, and returns
 * concatenated numpy arrays ready for Blender import.
 *
 * Copyright 2026 Peak-Design
 * License: GPL v3
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* numpy C-API */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* OpenCascade headers */
#include <BRep_Tool.hxx>
#include <Poly_Triangulation.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopAbs_Orientation.hxx>
#include <TopLoc_Location.hxx>
#include <gp_Trsf.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>

#include <vector>
#include <cstdint>
#include <cmath>

/* ------------------------------------------------------------------ */
/* Helper: extract triangulation data from a single face              */
/* ------------------------------------------------------------------ */
struct FaceMesh {
    std::vector<float> verts;   /* Nx3 flattened */
    std::vector<float> norms;   /* Nx3 flattened (placeholder) */
    std::vector<float> uvs;     /* Nx2 flattened */
    std::vector<int32_t> faces; /* Tx3 flattened (global indices) */
    int n_verts = 0;
    int n_tris  = 0;
    bool failed = false;
    bool undef_norms = false;
};

static FaceMesh extract_one_face(const TopoDS_Face& face,
                                 const gp_Trsf& trsf,
                                 int global_vert_offset)
{
    FaceMesh fm;
    TopLoc_Location loc;
    Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(face, loc);

    if (tri.IsNull() || tri->NbTriangles() == 0 || tri->NbNodes() == 0) {
        fm.failed = true;
        return fm;
    }

    bool not_forward = (face.Orientation() != TopAbs_FORWARD);
    int nb_nodes = tri->NbNodes();
    int nb_tris  = tri->NbTriangles();
    bool has_uvs = tri->HasUVNodes();

    fm.n_verts = nb_nodes;
    fm.n_tris  = nb_tris;
    fm.verts.resize(nb_nodes * 3);
    fm.norms.resize(nb_nodes * 3);
    fm.uvs.resize(nb_nodes * 2);
    fm.faces.resize(nb_tris * 3);

    /* Extract vertices and UVs */
    for (int i = 1; i <= nb_nodes; ++i) {
        gp_Pnt pt = tri->Node(i);
        /* Node positions are already in the triangulation's local frame.
         * The location embedded in the triangulation (loc) is the mesh
         * placement; we must apply it so coordinates are in shape space. */
        if (!loc.IsIdentity()) {
            pt.Transform(loc);
        }
        int idx = (i - 1) * 3;
        fm.verts[idx + 0] = static_cast<float>(pt.X());
        fm.verts[idx + 1] = static_cast<float>(pt.Y());
        fm.verts[idx + 2] = static_cast<float>(pt.Z());

        /* UVs */
        int ui = (i - 1) * 2;
        if (has_uvs) {
            gp_Pnt2d uv = tri->UVNode(i);
            fm.uvs[ui + 0] = static_cast<float>(uv.X());
            fm.uvs[ui + 1] = static_cast<float>(uv.Y());
        } else {
            fm.uvs[ui + 0] = 0.0f;
            fm.uvs[ui + 1] = 0.0f;
        }

        /* Placeholder normals — Python recomputes from analytic surface */
        int ni = (i - 1) * 3;
        fm.norms[ni + 0] = 0.0f;
        fm.norms[ni + 1] = 0.0f;
        fm.norms[ni + 2] = 1.0f;
    }
    fm.undef_norms = true;  /* normals are placeholders */

    /* Extract triangles with global vertex offset */
    for (int i = 1; i <= nb_tris; ++i) {
        int n1, n2, n3;
        tri->Triangle(i).Get(n1, n2, n3);
        if (not_forward) {
            std::swap(n1, n2);
        }
        int idx = (i - 1) * 3;
        fm.faces[idx + 0] = (n1 - 1) + global_vert_offset;
        fm.faces[idx + 1] = (n2 - 1) + global_vert_offset;
        fm.faces[idx + 2] = (n3 - 1) + global_vert_offset;
    }

    return fm;
}

/* ------------------------------------------------------------------ */
/* Python-callable function                                           */
/* ------------------------------------------------------------------ */
static PyObject* py_extract_face_meshes(PyObject* /*self*/, PyObject* args)
{
    PyObject* face_ptrs_list;
    PyObject* tform_ptrs_list;

    if (!PyArg_ParseTuple(args, "OO", &face_ptrs_list, &tform_ptrs_list))
        return NULL;

    Py_ssize_t n = PyList_Size(face_ptrs_list);
    if (n != PyList_Size(tform_ptrs_list)) {
        PyErr_SetString(PyExc_ValueError,
                        "face_ptrs and tform_ptrs must have the same length");
        return NULL;
    }

    /* Extract all face meshes */
    std::vector<FaceMesh> meshes(n);
    int total_verts = 0;
    int total_tris  = 0;

    for (Py_ssize_t i = 0; i < n; ++i) {
        /* Get SWIG pointer addresses */
        uintptr_t face_addr = static_cast<uintptr_t>(
            PyLong_AsUnsignedLongLong(PyList_GetItem(face_ptrs_list, i)));
        uintptr_t trsf_addr = static_cast<uintptr_t>(
            PyLong_AsUnsignedLongLong(PyList_GetItem(tform_ptrs_list, i)));

        if (PyErr_Occurred()) return NULL;

        const TopoDS_Face& face = *reinterpret_cast<TopoDS_Face*>(face_addr);
        const gp_Trsf& trsf     = *reinterpret_cast<gp_Trsf*>(trsf_addr);

        meshes[i] = extract_one_face(face, trsf, total_verts);

        if (!meshes[i].failed) {
            total_verts += meshes[i].n_verts;
            total_tris  += meshes[i].n_tris;
        }
    }

    /* Allocate output numpy arrays */
    npy_intp verts_dims[2] = {total_verts, 3};
    npy_intp norms_dims[2] = {total_verts, 3};
    npy_intp uvs_dims[2]   = {total_verts, 2};
    npy_intp faces_dims[2] = {total_tris,  3};
    npy_intp n_faces_dim   = {n};

    PyObject* all_verts = PyArray_ZEROS(2, verts_dims, NPY_FLOAT32, 0);
    PyObject* all_norms = PyArray_ZEROS(2, norms_dims, NPY_FLOAT32, 0);
    PyObject* all_uvs   = PyArray_ZEROS(2, uvs_dims,   NPY_FLOAT32, 0);
    PyObject* all_faces = PyArray_ZEROS(2, faces_dims, NPY_INT32, 0);

    PyObject* face_starts  = PyArray_ZEROS(1, &n_faces_dim, NPY_INT32, 0);
    PyObject* face_counts  = PyArray_ZEROS(1, &n_faces_dim, NPY_INT32, 0);
    PyObject* vert_starts  = PyArray_ZEROS(1, &n_faces_dim, NPY_INT32, 0);
    PyObject* vert_counts  = PyArray_ZEROS(1, &n_faces_dim, NPY_INT32, 0);
    PyObject* failed_mask  = PyArray_ZEROS(1, &n_faces_dim, NPY_BOOL, 0);
    PyObject* undef_mask   = PyArray_ZEROS(1, &n_faces_dim, NPY_BOOL, 0);

    if (!all_verts || !all_norms || !all_uvs || !all_faces ||
        !face_starts || !face_counts || !vert_starts || !vert_counts ||
        !failed_mask || !undef_mask) {
        Py_XDECREF(all_verts); Py_XDECREF(all_norms);
        Py_XDECREF(all_uvs);   Py_XDECREF(all_faces);
        Py_XDECREF(face_starts); Py_XDECREF(face_counts);
        Py_XDECREF(vert_starts); Py_XDECREF(vert_counts);
        Py_XDECREF(failed_mask); Py_XDECREF(undef_mask);
        return PyErr_NoMemory();
    }

    /* Get raw data pointers */
    float*   v_ptr = static_cast<float*>(PyArray_DATA((PyArrayObject*)all_verts));
    float*   n_ptr = static_cast<float*>(PyArray_DATA((PyArrayObject*)all_norms));
    float*   u_ptr = static_cast<float*>(PyArray_DATA((PyArrayObject*)all_uvs));
    int32_t* f_ptr = static_cast<int32_t*>(PyArray_DATA((PyArrayObject*)all_faces));

    int32_t* fs_ptr = static_cast<int32_t*>(PyArray_DATA((PyArrayObject*)face_starts));
    int32_t* fc_ptr = static_cast<int32_t*>(PyArray_DATA((PyArrayObject*)face_counts));
    int32_t* vs_ptr = static_cast<int32_t*>(PyArray_DATA((PyArrayObject*)vert_starts));
    int32_t* vc_ptr = static_cast<int32_t*>(PyArray_DATA((PyArrayObject*)vert_counts));
    npy_bool* fl_ptr = static_cast<npy_bool*>(PyArray_DATA((PyArrayObject*)failed_mask));
    npy_bool* un_ptr = static_cast<npy_bool*>(PyArray_DATA((PyArrayObject*)undef_mask));

    /* Copy data from per-face meshes into concatenated arrays */
    int v_offset = 0;
    int f_offset = 0;

    for (Py_ssize_t i = 0; i < n; ++i) {
        const FaceMesh& fm = meshes[i];

        fl_ptr[i] = fm.failed ? NPY_TRUE : NPY_FALSE;
        un_ptr[i] = fm.undef_norms ? NPY_TRUE : NPY_FALSE;

        if (fm.failed) {
            fs_ptr[i] = f_offset;
            fc_ptr[i] = 0;
            vs_ptr[i] = v_offset;
            vc_ptr[i] = 0;
            continue;
        }

        vs_ptr[i] = v_offset;
        vc_ptr[i] = fm.n_verts;
        fs_ptr[i] = f_offset;
        fc_ptr[i] = fm.n_tris;

        /* Copy vertices: n_verts * 3 floats */
        std::memcpy(v_ptr + v_offset * 3, fm.verts.data(),
                    fm.n_verts * 3 * sizeof(float));
        /* Copy normals */
        std::memcpy(n_ptr + v_offset * 3, fm.norms.data(),
                    fm.n_verts * 3 * sizeof(float));
        /* Copy UVs: n_verts * 2 floats */
        std::memcpy(u_ptr + v_offset * 2, fm.uvs.data(),
                    fm.n_verts * 2 * sizeof(float));
        /* Copy faces: n_tris * 3 int32s */
        std::memcpy(f_ptr + f_offset * 3, fm.faces.data(),
                    fm.n_tris * 3 * sizeof(int32_t));

        v_offset += fm.n_verts;
        f_offset += fm.n_tris;
    }

    /* Return tuple of 10 arrays */
    PyObject* result = PyTuple_Pack(10,
        all_verts, all_norms, all_uvs, all_faces,
        face_starts, face_counts, vert_starts, vert_counts,
        failed_mask, undef_mask);

    /* PyTuple_Pack increments refcounts; release our references */
    Py_DECREF(all_verts); Py_DECREF(all_norms);
    Py_DECREF(all_uvs);   Py_DECREF(all_faces);
    Py_DECREF(face_starts); Py_DECREF(face_counts);
    Py_DECREF(vert_starts); Py_DECREF(vert_counts);
    Py_DECREF(failed_mask); Py_DECREF(undef_mask);

    return result;
}

/* ------------------------------------------------------------------ */
/* Module definition                                                  */
/* ------------------------------------------------------------------ */
static PyMethodDef methods[] = {
    {"extract_face_meshes", py_extract_face_meshes, METH_VARARGS,
     "Extract triangulation data from OCC faces.\n\n"
     "Args:\n"
     "    face_ptrs: list of int — SWIG pointer addresses for TopoDS_Face objects\n"
     "    tform_ptrs: list of int — SWIG pointer addresses for gp_Trsf objects\n\n"
     "Returns:\n"
     "    Tuple of 10 numpy arrays:\n"
     "    (all_verts, all_norms, all_uvs, all_faces,\n"
     "     face_starts, face_counts, vert_starts, vert_counts,\n"
     "     failed_mask, undef_mask)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "stepper_native",
    "Native C++ mesh extraction for STEPper NEXT Blender addon.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_stepper_native(void)
{
    import_array();  /* Initialize numpy C-API */
    return PyModule_Create(&module_def);
}
