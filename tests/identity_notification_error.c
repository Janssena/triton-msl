#define PY_SSIZE_T_CLEAN
#include <Python.h>
/* A legitimate failing watcher. It executes no Python and changes no dict. */
static PyObject *target;
static int watcher = -1;
static unsigned long events;
static int notify(PyDict_WatchEvent event, PyObject *dict, PyObject *key, PyObject *value) {
    (void)key; (void)value;
    if (dict == target && event != PyDict_EVENT_DEALLOCATED) {
        events++;
        PyErr_SetString(PyExc_RuntimeError, "notification control error");
        return -1;
    }
    return 0;
}
static PyObject *install(PyObject *self, PyObject *dict) {
    (void)self;
    if (!PyDict_CheckExact(dict) || watcher >= 0)
        return PyErr_Format(PyExc_RuntimeError, "expected one fresh exact-dict watch");
    watcher = PyDict_AddWatcher(notify);
    if (watcher < 0) return NULL;
    target = Py_NewRef(dict);
    events = 0;
    if (PyDict_Watch(watcher, dict) < 0) return NULL;
    return PyLong_FromLong(watcher);
}
static PyObject *remove_watch(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    if (watcher >= 0) {
        if (target && PyDict_Unwatch(watcher, target) < 0) return NULL;
        if (PyDict_ClearWatcher(watcher) < 0) return NULL;
        watcher = -1;
    }
    Py_CLEAR(target);
    return PyLong_FromUnsignedLong(events);
}
static PyMethodDef methods[] = {
    {"install", install, METH_O, NULL}, {"remove", remove_watch, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "notification_error", NULL, -1, methods};
PyMODINIT_FUNC PyInit_notification_error(void) { return PyModule_Create(&module); }
