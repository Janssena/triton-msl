/* Narrow flat i32 binder. Admission performs no Python callbacks; rejection
 * returns NotImplemented before traversing an observable Python prefix. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    PyObject *dict, *proxy, *packing;
} Recipe;
typedef struct { PyTypeObject *type; unsigned int version; } TypeProof;
typedef struct {
    PyObject *globals, *builtins, *bind, *bind_code, *plan_type;
    PyObject *fallback, *fallback_code;
    PyObject *module, *bind_key, *plan_keyword;
    PyObject *keys[16], *expected[16], *i32, *pointer, *class_name;
    PyTypeObject *struct_type;
    unsigned int plan_version;
    unsigned long long hits, misses, errors;
    TypeProof types[16];
} State;
static const char *binding_names[] = {
    "type", "len", "isinstance", "getattr", "callable", "enumerate", "zip",
    "int", "bool", "float", "str", "tuple", "list", "dict", "MappingProxyType", "_BindingPlan"
};

static int exact_keys(PyObject *dict) {
    if (!dict || !PyDict_CheckExact(dict)) return 0;
    Py_ssize_t pos=0; PyObject *key, *value;
    while (PyDict_Next(dict,&pos,&key,&value)) if (!PyUnicode_CheckExact(key)) return 0;
    return 1;
}
static PyObject *lookup(State *s, PyObject *key) {
    PyObject *v=PyDict_GetItemWithError(s->globals,key);
    return v ? v : PyDict_GetItemWithError(s->builtins,key);
}
static int ordinary(State *s) {
    if (!s->globals || !exact_keys(s->globals) || !exact_keys(s->builtins)) return 0;
    if (PyDict_GetItemString(s->globals,"_binder_native")!=s->module) return 0;
    if (!PyFunction_Check(s->bind) || PyFunction_GET_CODE(s->bind)!=s->bind_code) return 0;
    if (PyFunction_GET_CODE(s->fallback)!=s->fallback_code) return 0;
    if (PyDict_GetItemString(s->globals,"bind_arguments")!=s->bind) return 0;
    if (!s->plan_version || ((PyTypeObject *)s->plan_type)->tp_version_tag!=s->plan_version) return 0;
    for (int i=0;i<16;i++) if (lookup(s,s->keys[i])!=s->expected[i]) return 0;
    return 1;
}
static void recipe_free(PyObject *capsule) {
    Recipe *r=PyCapsule_GetPointer(capsule,"triton_msl.flat_i32_binder");
    if (!r) { PyErr_Clear(); return; }
    Py_DECREF(r->dict); Py_DECREF(r->proxy); Py_DECREF(r->packing); PyMem_Free(r);
}

static PyObject *configure(PyObject *module, PyObject *args) {
    State *s=PyModule_GetState(module); PyObject *fn,*plan,*fallback;
    if (!PyArg_ParseTuple(args,"OOO",&fn,&plan,&fallback)) return NULL;
    if (s->globals || !PyFunction_Check(fn) || !PyFunction_Check(fallback) || !PyType_Check(plan) || !PyType_IsSubtype((PyTypeObject *)plan,&PyTuple_Type)) Py_RETURN_FALSE;
    PyObject *g=PyFunction_GET_GLOBALS(fn), *b=((PyFunctionObject *)fn)->func_builtins;
    if (!exact_keys(g) || !exact_keys(b) || PyFunction_GET_GLOBALS(fallback)!=g ||
        ((PyFunctionObject *)fallback)->func_builtins!=b) Py_RETURN_FALSE;
    s->globals=Py_NewRef(g); s->builtins=Py_NewRef(b);
    s->bind=Py_NewRef(fn); s->bind_code=Py_NewRef(PyFunction_GET_CODE(fn)); s->plan_type=Py_NewRef(plan);
    s->fallback=Py_NewRef(fallback); s->fallback_code=Py_NewRef(PyFunction_GET_CODE(fallback));
    PyCodeObject *code=(PyCodeObject *)s->fallback_code;
    if (PyTuple_GET_SIZE(code->co_names)!=1 ||
        PyUnicode_CompareWithASCIIString(PyTuple_GET_ITEM(code->co_names,0),"bind_arguments")) Py_RETURN_FALSE;
    s->module=Py_NewRef(module);
    /* Preserve the original LOAD_GLOBAL name object's identity on fallback. */
    s->bind_key=Py_NewRef(PyTuple_GET_ITEM(code->co_names,0));
    for (Py_ssize_t i=0;i<PyTuple_GET_SIZE(code->co_consts);i++) {
        PyObject *item=PyTuple_GET_ITEM(code->co_consts,i);
        if (PyTuple_CheckExact(item) && PyTuple_GET_SIZE(item)==1 &&
            PyUnicode_CheckExact(PyTuple_GET_ITEM(item,0)) &&
            PyUnicode_CompareWithASCIIString(PyTuple_GET_ITEM(item,0),"_plan")==0) {
            s->plan_keyword=Py_NewRef(item); break;
        }
    }
    if (!s->plan_keyword) Py_RETURN_FALSE;
    if (!PyUnstable_Type_AssignVersionTag((PyTypeObject *)plan)) Py_RETURN_FALSE;
    s->plan_version=((PyTypeObject *)plan)->tp_version_tag;
    PyObject *types[]={ (PyObject *)&PyType_Type,NULL,NULL,NULL,NULL,(PyObject *)&PyEnum_Type,(PyObject *)&PyZip_Type,
        (PyObject *)&PyLong_Type,(PyObject *)&PyBool_Type,(PyObject *)&PyFloat_Type,
        (PyObject *)&PyUnicode_Type,(PyObject *)&PyTuple_Type,(PyObject *)&PyList_Type,
        (PyObject *)&PyDict_Type,(PyObject *)&PyDictProxy_Type,plan };
    PyObject *builtin_module=PyImport_AddModule("builtins");
    for (int i=0;i<16;i++) {
        s->keys[i]=PyUnicode_FromString(binding_names[i]); if (!s->keys[i]) return NULL;
        PyObject *v=lookup(s,s->keys[i]);
        if (!v) Py_RETURN_FALSE;
        if (types[i]) { if (v!=types[i]) Py_RETURN_FALSE; }
        else if (!PyCFunction_Check(v) || PyCFunction_GET_SELF(v)!=builtin_module ||
                 strcmp(((PyCFunctionObject *)v)->m_ml->ml_name,binding_names[i])) Py_RETURN_FALSE;
        s->expected[i]=Py_NewRef(v);
    }
    s->i32=PyUnicode_FromString("i32"); s->pointer=PyUnicode_FromString("data_ptr"); s->class_name=PyUnicode_FromString("__class__");
    PyObject *st=PyImport_ImportModule("_struct"); if (!st) return NULL;
    PyObject *type=PyObject_GetAttrString(st,"Struct"); Py_DECREF(st);
    if (!type) return NULL;
    if (!PyType_Check(type) || !( ((PyTypeObject *)type)->tp_flags & Py_TPFLAGS_IMMUTABLETYPE) ||
        strcmp(((PyTypeObject *)type)->tp_name,"_struct.Struct")) { Py_DECREF(type); Py_RETURN_FALSE; }
    s->struct_type=(PyTypeObject *)type;
    if (!s->i32 || !s->pointer || !s->class_name) return NULL;
    Py_RETURN_TRUE;
}

static PyObject *make_parts(PyObject *module, PyObject *dict) {
    State *s=PyModule_GetState(module);
    if (!ordinary(s) || !s->struct_type || !exact_keys(dict)) Py_RETURN_NONE;
    /* A mixed/non-i32 plan stays on the original Python implementation even
     * if a later live declaration changes. This changes eligibility only. */
    if (PyDict_GET_SIZE(dict)!=1) Py_RETURN_NONE;
    PyObject *packing=PyDict_GetItemWithError(dict,s->i32);
    if (!packing || !PyTuple_CheckExact(packing) || PyTuple_GET_SIZE(packing)!=2 || PyTuple_GET_ITEM(packing,0)!=Py_False) Py_RETURN_NONE;
    PyObject *pack=PyTuple_GET_ITEM(packing,1);
    if (!PyCFunction_Check(pack)) Py_RETURN_NONE;
    PyObject *self=PyCFunction_GET_SELF(pack);
    if (!self || Py_TYPE(self)!=s->struct_type) Py_RETURN_NONE;
    PyObject *struct_dict=PyType_GetDict(s->struct_type); if (!struct_dict) return NULL;
    PyObject *descriptor=PyDict_GetItemString(struct_dict,"pack"); Py_DECREF(struct_dict);
    if (!descriptor || Py_TYPE(descriptor)!=&PyMethodDescr_Type ||
        PyCFunction_GET_FUNCTION(pack)!=((PyMethodDescrObject *)descriptor)->d_method->ml_meth) Py_RETURN_NONE;
    PyObject *format=PyObject_GetAttrString(self,"format"); if (!format) return NULL;
    int good=PyUnicode_CheckExact(format) && PyUnicode_CompareWithASCIIString(format,"<i")==0;
    Py_DECREF(format); if (!good) Py_RETURN_NONE;
    Recipe *r=PyMem_Calloc(1,sizeof(*r)); if (!r) return PyErr_NoMemory();
    r->dict=Py_NewRef(dict); r->packing=Py_NewRef(packing); r->proxy=PyDictProxy_New(dict);
    if (!r->proxy) { Py_DECREF(r->dict); Py_DECREF(r->packing); PyMem_Free(r); return NULL; }
    PyObject *cap=PyCapsule_New(r,"triton_msl.flat_i32_binder",recipe_free);
    if (!cap) { Py_DECREF(r->dict); Py_DECREF(r->proxy); Py_DECREF(r->packing); PyMem_Free(r); return NULL; }
    PyObject *result=PyTuple_Pack(2,r->proxy,cap); Py_DECREF(cap); return result;
}

/* A generic attribute lookup of a standard C method descriptor produces a
 * callable bound method without running the pointer's data_ptr function. */
static int certify_type(State *s, PyTypeObject *type) {
    if (type->tp_getattro!=PyObject_GenericGetAttr) return 0;
    for (int i=0;i<16;i++) if (s->types[i].type==type && s->types[i].version && s->types[i].version==type->tp_version_tag) return 1;
    PyObject *mro=type->tp_mro,*method=NULL,*cls=NULL;
    if (!mro || !PyTuple_CheckExact(mro)) return 0;
    for (Py_ssize_t i=0;i<PyTuple_GET_SIZE(mro);i++) {
        PyObject *entry=PyTuple_GET_ITEM(mro,i); if (!PyType_Check(entry)) return 0;
        PyObject *dict=PyType_GetDict((PyTypeObject *)entry);
        if (!exact_keys(dict)) { Py_XDECREF(dict); return 0; }
        if (!method) method=PyDict_GetItemWithError(dict,s->pointer);
        if (!cls) cls=PyDict_GetItemWithError(dict,s->class_name);
        Py_DECREF(dict);
    }
    if (!method || Py_TYPE(method)!=&PyMethodDescr_Type || !PyType_IsSubtype(type,PyDescr_TYPE(method))) return 0;
    PyObject *base_dict=PyType_GetDict(&PyBaseObject_Type); if (!base_dict) return 0;
    PyObject *ordinary_class=PyDict_GetItemWithError(base_dict,s->class_name); Py_DECREF(base_dict);
    if (cls!=ordinary_class) return 0;
    if (!PyUnstable_Type_AssignVersionTag(type)) return 0;
    for (int i=0;i<16;i++) if (!s->types[i].type || s->types[i].type==type) {
        if (!s->types[i].type) s->types[i].type=(PyTypeObject *)Py_NewRef((PyObject *)type);
        s->types[i].version=type->tp_version_tag; return 1;
    }
    return 0;
}
typedef struct { State *s; int rejected; } Visit;
static int visit_dict(PyObject *value, void *raw) {
    Visit *v=raw;
    /* Managed inline attribute values lack key names through this API: decline
     * those conservatively. Materialized exact dictionaries can be inspected. */
    if (!exact_keys(value)) { v->rejected=1; return 1; }
    PyObject *override=PyDict_GetItemWithError(value,v->s->pointer);
    if (override) { v->rejected=1; return 1; }
    return 0;
}
static int pointer_ok(State *s, PyObject *value) {
    if (value==Py_None) return 1;
    /* The Python binder refuses tuple values before pointer lookup, including
     * tuple subclasses exposing an otherwise ordinary C method descriptor. */
    if (PyTuple_Check(value)) return 0;
    PyTypeObject *type=Py_TYPE(value);
    if (!certify_type(s,type)) return 0;
    if (type->tp_flags & Py_TPFLAGS_MANAGED_DICT) {
        Visit visit={s,0}; (void)PyObject_VisitManagedDict(value,visit_dict,&visit);
        return !visit.rejected;
    }
    /* No raw offsets or guessed third-party object layouts. */
    return type->tp_dictoffset==0;
}

typedef struct {
    PyObject *types[128]; int kinds[128]; int32_t integers[128];
    Py_ssize_t n, count;
} Flat;
static int admit(State *s, PyObject *const *args, Flat *f) {
    PyObject *values=args[0],*names=args[1],*signature=args[2],*plan=args[3];
    if (Py_TYPE(plan)!=(PyTypeObject *)s->plan_type || PyTuple_GET_SIZE(plan)!=2 ||
        !PyTuple_CheckExact(values) || !PyList_CheckExact(names) || !PyDict_CheckExact(signature)) return 0;
    PyObject *cap=PyTuple_GET_ITEM(plan,1);
    if (!PyCapsule_IsValid(cap,"triton_msl.flat_i32_binder")) return 0;
    Recipe *r=PyCapsule_GetPointer(cap,"triton_msl.flat_i32_binder");
    if (PyTuple_GET_ITEM(plan,0)!=r->proxy || !exact_keys(r->dict) || PyDict_GetItemWithError(r->dict,s->i32)!=r->packing) return 0;
    if (!exact_keys(signature)) return 0;
    Py_ssize_t n=PyList_GET_SIZE(names), count=0;
    if (n>128 || PyTuple_GET_SIZE(values)!=n) return 0;
    PyObject **types=f->types; int *kinds=f->kinds; int32_t *integers=f->integers;
    /* All admission precedes output construction. No descriptor, scalar
     * conversion hook, Python binder, or callbackful dictionary lookup occurs. */
    for (Py_ssize_t i=0;i<n;i++) {
        PyObject *name=PyList_GET_ITEM(names,i); if (!PyUnicode_CheckExact(name)) return 0;
        PyObject *ty=PyDict_GetItemWithError(signature,name); if (!ty || !PyUnicode_CheckExact(ty)) return 0;
        types[i]=ty;
        if (PyUnicode_CompareWithASCIIString(ty,"constexpr")==0) { kinds[i]=0; continue; }
        PyObject *value=PyTuple_GET_ITEM(values,i);
        if (PyUnicode_GET_LENGTH(ty)>0 && PyUnicode_ReadChar(ty,0)=='*') {
            if (!pointer_ok(s,value)) return 0;
            kinds[i]=1;
        } else if (PyUnicode_CompareWithASCIIString(ty,"i32")==0 && (PyLong_CheckExact(value) || PyBool_Check(value))) {
            int overflow=0; long long number=PyLong_AsLongLongAndOverflow(value,&overflow);
            if (overflow || PyErr_Occurred()) { PyErr_Clear(); return 0; }
            if (number<INT32_MIN || number>INT32_MAX) return 0;
            integers[i]=(int32_t)number; kinds[i]=2;
        } else return 0;
        count++;
    }
    /* Structural declines above are callback-free. Only potentially admitted
     * calls pay the live global/builtin census and packer-state proof. */
    if (!ordinary(s)) return 0;
    /* Struct's type is immutable; its INSTANCE can still be reinitialized. */
    PyObject *packer=PyTuple_GET_ITEM(r->packing,1);
    PyObject *format=PyObject_GetAttrString(PyCFunction_GET_SELF(packer),"format");
    if (!format) return -1;
    int format_ok=PyUnicode_CheckExact(format) && PyUnicode_CompareWithASCIIString(format,"<i")==0;
    Py_DECREF(format); if (!format_ok) return 0;
    f->n=n; f->count=count;
    return 1;
}
static void release_types(Flat *f) {
    for (Py_ssize_t i=0;i<f->n;i++) Py_DECREF(f->types[i]);
}
static PyObject *bind_flat_impl(PyObject *module, PyObject *const *args, Py_ssize_t nargs) {
    State *s=PyModule_GetState(module);
    if (nargs!=4) { PyErr_SetString(PyExc_TypeError,"bind_flat requires args, names, signature and plan"); return NULL; }
    Flat f={0}; int ok=admit(s,args,&f);
    if (ok<0) return NULL;
    if (!ok) Py_RETURN_NOTIMPLEMENTED;
    /* Hold each exact type string through output construction. On the gated
     * GIL-enabled CPython 3.14 runtime, container allocation schedules cyclic GC
     * for an evaluator safe point; it does not execute callbacks inside this C
     * window. No API below executes Python code or releases the GIL. */
    for (Py_ssize_t i=0;i<f.n;i++) Py_INCREF(f.types[i]);
    PyObject *result=PyTuple_New(4);
    if (!result) { release_types(&f); return NULL; }
    for (int i=0;i<4;i++) {
        PyObject *list=PyList_New(f.count);
        if (!list) { Py_DECREF(result); release_types(&f); return NULL; }
        PyTuple_SET_ITEM(result,i,list);
    }
    PyObject *values=args[0]; Py_ssize_t at=0;
    /* Exact longs and bytes are untracked; ref copies retain original leaves.
     * There is no Python call, descriptor lookup, or GIL release in this loop. */
    for (Py_ssize_t i=0;i<f.n;i++) {
        if (!f.kinds[i]) continue;
        PyObject *origin=PyLong_FromSsize_t(i),*payload=NULL;
        if (f.kinds[i]==1) payload=Py_NewRef(Py_None);
        else { uint32_t bits=(uint32_t)f.integers[i]; char bytes[4]; for (int k=0;k<4;k++) bytes[k]=(char)(bits>>(8*k)); payload=PyBytes_FromStringAndSize(bytes,4); }
        if (!origin || !payload) { Py_XDECREF(origin); Py_XDECREF(payload); Py_DECREF(result); release_types(&f); return NULL; }
        PyList_SET_ITEM(PyTuple_GET_ITEM(result,0),at,Py_NewRef(PyTuple_GET_ITEM(values,i)));
        PyList_SET_ITEM(PyTuple_GET_ITEM(result,1),at,Py_NewRef(f.types[i]));
        PyList_SET_ITEM(PyTuple_GET_ITEM(result,2),at,origin);
        PyList_SET_ITEM(PyTuple_GET_ITEM(result,3),at,payload); at++;
    }
    release_types(&f);
    return result;
}
static PyObject *bind_flat(PyObject *module, PyObject *const *args, Py_ssize_t nargs) {
    State *s=PyModule_GetState(module);
    PyObject *result=bind_flat_impl(module,args,nargs);
    if (!result) s->errors++;
    else if (result==Py_NotImplemented) s->misses++;
    else s->hits++;
    return result;
}
static PyObject *statistics(PyObject *module, PyObject *unused) {
    State *s=PyModule_GetState(module);
    return Py_BuildValue("{s:K,s:K,s:K}","hits",s->hits,"misses",s->misses,"errors",s->errors);
}
static PyObject *bind_with_plan(PyObject *module, PyObject *const *args,
                                Py_ssize_t nargs, PyObject *kwnames) {
    State *s=PyModule_GetState(module);
    if (!s->fallback) { PyErr_SetString(PyExc_RuntimeError,"native binder is not configured"); return NULL; }
    if (nargs==4 && (!kwnames || PyTuple_GET_SIZE(kwnames)==0)) {
        PyObject *result=bind_flat(module,args,nargs);
        if (result!=Py_NotImplemented) return result;
        Py_DECREF(result);
    }
    if (nargs==4 && (!kwnames || PyTuple_GET_SIZE(kwnames)==0) &&
        PyFunction_GET_CODE(s->fallback)==s->fallback_code) {
        /* Exact four-position shim: LOAD_GLOBAL using its actual co_names key,
         * then CALL with three positions and _plan. Do not capture the callee.
         * Foreign-key callbacks are observed once; never retry via the shim
         * after this lookup has performed an observable operation. */
        PyObject *fn=NULL;
        int found=PyDict_GetItemRef(s->globals,s->bind_key,&fn);
        if (found==0) found=PyDict_GetItemRef(s->builtins,s->bind_key,&fn);
        if (found<0) return NULL;
        if (!found) {
            PyErr_Format(PyExc_NameError,"name '%U' is not defined",s->bind_key);
            PyObject *error=PyErr_GetRaisedException();
            if (PyObject_SetAttrString(error,"name",s->bind_key)<0) { Py_DECREF(error); return NULL; }
            PyErr_SetRaisedException(error); return NULL;
        }
        PyObject *result=PyObject_Vectorcall(fn,args,3,s->plan_keyword);
        Py_DECREF(fn); return result;
    }
    /* Changed code, keyword calls, and arity handling remain in Python. */
    return PyObject_Vectorcall(s->fallback,args,(size_t)nargs,kwnames);
}
static PyMethodDef methods[]={
    {"configure",configure,METH_VARARGS,NULL}, {"make_parts",make_parts,METH_O,NULL},
    {"statistics",statistics,METH_NOARGS,NULL},
    {"bind_with_plan",(PyCFunction)(void(*)(void))bind_with_plan,METH_FASTCALL|METH_KEYWORDS,NULL},
    {"bind_flat",(PyCFunction)(void(*)(void))bind_flat,METH_FASTCALL,NULL}, {NULL,NULL,0,NULL}
};
static int traverse(PyObject *module,visitproc visit,void *arg) {
    State *s=PyModule_GetState(module); Py_VISIT(s->globals);Py_VISIT(s->builtins);Py_VISIT(s->bind);Py_VISIT(s->bind_code);Py_VISIT(s->plan_type);
    Py_VISIT(s->fallback);Py_VISIT(s->fallback_code);
    Py_VISIT(s->module);Py_VISIT(s->bind_key);Py_VISIT(s->plan_keyword);
    Py_VISIT(s->struct_type);Py_VISIT(s->i32);Py_VISIT(s->pointer);Py_VISIT(s->class_name);
    for(int i=0;i<16;i++){Py_VISIT(s->keys[i]);Py_VISIT(s->expected[i]);Py_VISIT(s->types[i].type);}return 0;
}
static int clear(PyObject *module) {
    State *s=PyModule_GetState(module);Py_CLEAR(s->globals);Py_CLEAR(s->builtins);Py_CLEAR(s->bind);Py_CLEAR(s->bind_code);Py_CLEAR(s->plan_type);
    Py_CLEAR(s->fallback);Py_CLEAR(s->fallback_code);
    Py_CLEAR(s->module);Py_CLEAR(s->bind_key);Py_CLEAR(s->plan_keyword);
    Py_CLEAR(s->struct_type);Py_CLEAR(s->i32);Py_CLEAR(s->pointer);Py_CLEAR(s->class_name);
    for(int i=0;i<16;i++){Py_CLEAR(s->keys[i]);Py_CLEAR(s->expected[i]);Py_CLEAR(s->types[i].type);}return 0;
}
static struct PyModuleDef definition={PyModuleDef_HEAD_INIT,"_binder_native",NULL,sizeof(State),methods,NULL,traverse,clear,NULL};
PyMODINIT_FUNC PyInit__binder_native(void){return PyModule_Create(&definition);}
