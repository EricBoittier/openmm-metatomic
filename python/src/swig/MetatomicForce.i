%pythonbegin %{
import os
from pathlib import Path

def _load_openmmmetatomic_plugin():
    from openmm import Platform
    dirs = []
    env = os.environ.get("OPENMM_PLUGIN_DIR")
    if env:
        dirs.append(env)
    here = Path(__file__).resolve().parent
    for cand in (
        here,
        here.parent,
        here.parent.parent / "build",
        Path(os.environ.get("OPENMM_METATOMIC_ROOT", "")) / "build",
    ):
        if cand and (cand / "libOpenMMMetatomic.so").is_file():
            dirs.append(str(cand))
    seen = set()
    for directory in dirs:
        if not directory or directory in seen:
            continue
        seen.add(directory)
        try:
            Platform.loadPluginsFromDirectory(directory)
        except Exception:
            pass

_load_openmmmetatomic_plugin()
%}

%module(moduleimport="from . import $module") openmmmetatomic

%include "factory.i"
%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_vector.i>

%{
#include "openmmmetatomic/MetatomicForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace std {
    %template(ivector) vector<int>;
}

namespace OpenMMMetatomic {

class MetatomicForce : public OpenMM::Force {
public:
    explicit MetatomicForce(const std::string& modelPath);
    void setDevice(const std::string& device);
    void setExtensionsDirectory(const std::string& path);
    void setCheckConsistency(bool enabled);
    void setAtomicTypes(const std::vector<int>& types);
    void setUsesPeriodicBoundaryConditions(bool periodic);
    void setBackend(const std::string& backend);

    const std::string& getModelPath() const;
    const std::string& getDevice() const;
    const std::string& getExtensionsDirectory() const;
    bool getCheckConsistency() const;
    const std::vector<int>& getAtomicTypes() const;
    bool usesPeriodicBoundaryConditions() const;
    const std::string& getBackend() const;

    %extend {
        static OpenMMMetatomic::MetatomicForce& cast(OpenMM::Force& force) {
            return dynamic_cast<OpenMMMetatomic::MetatomicForce&>(force);
        }
        static bool isinstance(OpenMM::Force& force) {
            return dynamic_cast<OpenMMMetatomic::MetatomicForce*>(&force) != NULL;
        }
    }
};

}

%factory(OpenMM::Force& OpenMM::System::getForce, OpenMMMetatomic::MetatomicForce);
