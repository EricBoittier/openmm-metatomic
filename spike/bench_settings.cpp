/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- *
 * Settings matrix for MetatomicForce: backend, device, consistency, neighbor
 * list, platform, model, and eval vs NVE vs NVT. Each case constructs its
 * own Context; eval timing is repeats after one untimed getState warmup.
 * -------------------------------------------------------------------------- */

#include "SimHelpers.h"

#include "OpenMM.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace OpenMMMetatomic;
using namespace OpenMMMetatomic::sim;
using namespace OpenMM;

namespace {

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty())
            out.push_back(tok);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> models = {"harmonic", "harmonic-nl"};
        std::vector<std::string> backends = {"core", "torch"};
        std::vector<std::string> devices = {"cpu"};
        std::vector<std::string> platforms;
        for (int i = 0; i < Platform::getNumPlatforms(); i++) {
            const auto name = Platform::getPlatform(i).getName();
            if (name == "Reference" || name == "CPU")
                platforms.push_back(name);
        }
        if (platforms.empty())
            platforms = {"Reference"};
        std::vector<std::string> nls = {"vesin", "naive"};
        std::vector<int> consistency = {0, 1};
        std::string torchHarmonic;
        std::string torchNl;
        std::string soapPt;
        std::string petmadPt;
        std::string toluenePdb;
        int nveSteps = 20;
        int nvtSteps = 20;
        int evalRepeats = 15;
        int nMol = 8;
        bool includeMd = true;
        std::string pluginsDir;

        auto need = [&](int i) {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + std::string(argv[i]));
            return std::string(argv[i + 1]);
        };
        for (int i = 1; i < argc; i++) {
            const std::string arg = argv[i];
            if (arg == "--models") { models = split(need(i)); i++; }
            else if (arg == "--backends") { backends = split(need(i)); i++; }
            else if (arg == "--devices") { devices = split(need(i)); i++; }
            else if (arg == "--platforms") { platforms = split(need(i)); i++; }
            else if (arg == "--nl") { nls = split(need(i)); i++; }
            else if (arg == "--torch-harmonic") { torchHarmonic = need(i); i++; }
            else if (arg == "--torch-nl") { torchNl = need(i); i++; }
            else if (arg == "--soap") { soapPt = need(i); i++; }
            else if (arg == "--petmad") { petmadPt = need(i); i++; }
            else if (arg == "--toluene") { toluenePdb = need(i); i++; }
            else if (arg == "--nve") { nveSteps = std::stoi(need(i)); i++; }
            else if (arg == "--nvt") { nvtSteps = std::stoi(need(i)); i++; }
            else if (arg == "--eval") { evalRepeats = std::stoi(need(i)); i++; }
            else if (arg == "--n-mol") { nMol = std::stoi(need(i)); i++; }
            else if (arg == "--no-md") { includeMd = false; }
            else if (arg == "--plugins-dir") { pluginsDir = need(i); i++; }
            else if (arg == "--help") {
                std::cout << "openmm-metatomic-bench-settings [options]\n";
                return 0;
            }
        }
        if (!pluginsDir.empty())
            Platform::loadPluginsFromDirectory(pluginsDir);

        std::cout << std::left << std::setw(52) << "case"
                  << std::right << std::setw(8) << "N"
                  << std::setw(12) << "eval/ms"
                  << std::setw(12) << "nve/ms"
                  << std::setw(12) << "nvt/ms"
                  << std::setw(14) << "nve_drift"
                  << "\n";

        auto emit = [&](const std::string& label, const Geometry& geom, const ForceConfig& cfg,
                        const std::string& platform, const std::string& nlEnv) {
            const char* prev = std::getenv("OPENMM_METATOMIC_NEIGHBOR_LIST");
            if (nlEnv == "naive")
                setenv("OPENMM_METATOMIC_NEIGHBOR_LIST", "naive", 1);
            else
                unsetenv("OPENMM_METATOMIC_NEIGHBOR_LIST");
            try {
                const auto result = runMd(
                    geom, cfg, platform,
                    includeMd ? nveSteps : 0,
                    includeMd ? nvtSteps : 0,
                    0.0005, 300.0, 1.0, evalRepeats
                );
                std::cout << std::left << std::setw(52) << label
                          << std::right << std::setw(8) << geom.types.size()
                          << std::setw(12) << std::fixed << std::setprecision(3) << result.evalMs
                          << std::setw(12) << result.nveMsPerStep
                          << std::setw(12) << result.nvtMsPerStep
                          << std::setw(14) << std::setprecision(4) << result.nveDrift
                          << "\n";
            }
            catch (const std::exception& e) {
                std::string msg = e.what();
                const auto nl = msg.find('\n');
                if (nl != std::string::npos)
                    msg.resize(nl);
                if (msg.size() > 120)
                    msg.resize(117), msg += "...";
                std::cout << std::left << std::setw(52) << label
                          << "  skipped (" << msg << ")\n";
            }
            if (prev)
                setenv("OPENMM_METATOMIC_NEIGHBOR_LIST", prev, 1);
            else
                unsetenv("OPENMM_METATOMIC_NEIGHBOR_LIST");
        };

        const Geometry water = waterVacuum();
        const Geometry box = waterBox(nMol);

        for (const auto& model : models) {
            for (const auto& backend : backends) {
                std::string path = model;
                Geometry geom = water;
                bool periodic = false;
                if (model == "harmonic") {
                    path = (backend == "torch") ? torchHarmonic : "harmonic";
                    if (backend == "torch" && path.empty())
                        continue;
                }
                else if (model == "harmonic-nl") {
                    path = (backend == "torch") ? torchNl : "harmonic-nl";
                    geom = box;
                    periodic = true;
                    if (backend == "torch" && path.empty())
                        continue;
                }
                else if (model == "soap-bpnn") {
                    if (backend != "torch" || soapPt.empty())
                        continue;
                    path = soapPt;
                }
                else if (model == "pet-mad") {
                    if (backend != "torch" || petmadPt.empty())
                        continue;
                    path = petmadPt;
                    geom = water;
                }
                else if (model == "pet-mad-box") {
                    if (backend != "torch" || petmadPt.empty())
                        continue;
                    path = petmadPt;
                    geom = box;
                    periodic = true;
                }
                else if (model == "toluene") {
                    if (backend != "torch" || petmadPt.empty() || toluenePdb.empty())
                        continue;
                    path = petmadPt;
                    geom = loadPdb(toluenePdb);
                }
                else {
                    continue;
                }

                                const auto nlsForModel = (model == "harmonic")
                    ? std::vector<std::string>{"none"}
                    : nls;
                for (const auto& device : devices) {
                    if (device == "cuda" && backend != "torch")
                        continue;
                    for (const auto& platform : platforms) {
                        for (int cons : consistency) {
                            if (cons && device == "cuda")
                                continue;
                            for (const auto& nl : nlsForModel) {
                                if (nl == "naive" && model == "harmonic")
                                    continue;
                                ForceConfig cfg;
                                cfg.modelPath = path;
                                cfg.backend = backend;
                                cfg.device = device;
                                cfg.checkConsistency = cons != 0;
                                cfg.periodic = periodic;
                                std::ostringstream label;
                                label << model << "/" << backend << "/" << device
                                      << "/" << platform
                                      << (cons ? "/cons" : "/nocon")
                                      << "/" << nl;
                                emit(label.str(), geom, cfg, platform, nl);
                            }
                        }
                    }
                }
            }
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "bench_settings failed: " << e.what() << "\n";
        return 1;
    }
}
