%pythonbegin %{
import os
import warnings
from pathlib import Path

_PLUGIN_LIBRARY = "libOpenMMMetatomic.so"


def _openmmmetatomic_plugin_candidates():
    """Directories to look for the plugin library in, most specific first.

    Every path here is absolute. An earlier version appended
    ``Path(os.environ.get("OPENMM_METATOMIC_ROOT", "")) / "build"``, which
    collapses to the relative ``build`` when the variable is unset, so running
    from a directory that happened to contain a ``build/`` subdirectory loaded
    that copy of the plugin instead of the installed one.
    """
    env = os.environ.get("OPENMM_PLUGIN_DIR")
    if env:
        yield Path(env).resolve()
    root = os.environ.get("OPENMM_METATOMIC_ROOT")
    if root:
        yield (Path(root) / "build").resolve()
    here = Path(__file__).resolve().parent
    yield here                       # installed beside the package
    yield here.parent                # python/ in a build tree
    yield here.parent.parent         # the CMake build root, where the
                                     # library is written before install
    yield here.parent.parent / "build"


def _load_openmmmetatomic_plugin():
    from openmm import Platform

    failures = []
    seen = set()
    for directory in _openmmmetatomic_plugin_candidates():
        if directory in seen or not (directory / _PLUGIN_LIBRARY).is_file():
            continue
        seen.add(directory)
        try:
            Platform.loadPluginsFromDirectory(str(directory))
        except Exception as error:
            failures.append((directory, error))
            continue
        # Stop at the first one that worked: loading a second copy registers
        # the plugin twice, and which registration wins is not defined.
        return directory

    warnings.warn(
        "could not load {}, so its registerPlatforms/registerKernelFactories "
        "entry points were never called. Looked in: {}. {}".format(
            _PLUGIN_LIBRARY,
            ", ".join(str(d) for d in _openmmmetatomic_plugin_candidates()),
            "; ".join("{}: {}".format(d, e) for d, e in failures) or "no directory had it",
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    return None


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
    %template(svector) vector<string>;
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
    void setParticles(const std::vector<int>& particles);
    void setPeriodicDirections(bool a, bool b, bool c);
    void setNonConservative(const std::string& mode);
    void setVariant(const std::string& output, const std::string& variant);
    void setUncertaintyThreshold(double eVPerAtom);
    void setCharge(double charge);
    void setSpinMultiplicity(double multiplicity);

    const std::string& getModelPath() const;
    const std::string& getDevice() const;
    const std::string& getExtensionsDirectory() const;
    bool getCheckConsistency() const;
    const std::vector<int>& getAtomicTypes() const;
    bool usesPeriodicBoundaryConditions() const;
    const std::string& getBackend() const;
    const std::vector<int>& getParticles() const;
    bool getPeriodicDirection(int axis) const;
    const std::string& getNonConservative() const;
    std::string getVariant(const std::string& output) const;
    std::vector<std::string> getVariantOutputs() const;
    double getUncertaintyThreshold() const;
    double getCharge() const;
    double getSpinMultiplicity() const;

    // cast() hands back a reference it does not own, so a caller writing
    // cast(XmlSerializer.deserialize(xml)) would be left with a dangling
    // pointer once the temporary is collected.
    %pythonappend cast %{
        val._cast_source = force
    %}
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
