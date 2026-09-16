#ifndef OPENMM_WINDOWSEXPORTMETATOMIC_H_
#define OPENMM_WINDOWSEXPORTMETATOMIC_H_

#ifdef _MSC_VER
    #pragma warning(disable:4996)
    #pragma warning(disable:4251)
    #if defined(OPENMM_METATOMIC_BUILDING_SHARED_LIBRARY)
        #define OPENMM_EXPORT_METATOMIC __declspec(dllexport)
    #elif defined(OPENMM_METATOMIC_BUILDING_STATIC_LIBRARY) || defined(OPENMM_METATOMIC_USE_STATIC_LIBRARIES)
        #define OPENMM_EXPORT_METATOMIC
    #else
        #define OPENMM_EXPORT_METATOMIC __declspec(dllimport)
    #endif
#else
    #define OPENMM_EXPORT_METATOMIC
#endif

#endif
