/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
#include <windows.h>
#else
#endif

#include "openmm/serialization/SerializationProxy.h"
#include "openmmmetatomic/MetatomicForce.h"
#include "openmmmetatomic/serialization/MetatomicForceProxy.h"
#include "openmmmetatomic/internal/windowsExportMetatomic.h"

#if defined(WIN32)
    extern "C" OPENMM_EXPORT_METATOMIC void registerMetatomicSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerMetatomicSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerMetatomicSerializationProxies();
#endif

using namespace OpenMM;
using namespace OpenMMMetatomic;

extern "C" OPENMM_EXPORT_METATOMIC void registerMetatomicSerializationProxies() {
    SerializationProxy::registerProxy(typeid(MetatomicForce), new MetatomicForceProxy());
}

extern "C" OPENMM_EXPORT_METATOMIC void registerPlatforms() {
}

extern "C" OPENMM_EXPORT_METATOMIC void registerKernelFactories() {
}
