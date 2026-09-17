/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- *
 * Settings matrix for MetatomicForce. Each case uses one Context: warmup
 * evals + warmup MD, then median getState, then Integrator::step(N) with no
 * per-step getState. Default atom counts are large enough that load/JIT is
 * not the number being measured.
 * -------------------------------------------------------------------------- */

#include "SimHelpers.h"

#include "OpenMM.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
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

std::vector<int> splitInts(const std::string& s) {
    std::vector<int> out;
    for (const auto& tok : split(s))
        out.push_back(std::stoi(tok));
    return out;
}

std::string torchHarmonicFor(int n, const std::string& single, const std::string& dir) {
    if (!dir.empty()) {
        const std::string path = dir + "/harmonic-" + std::to_string(n) + ".pt";
        std::ifstream in(path);
        if (in)
            return path;
    }
    return single;
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
        std::vector<int> consistency = {0};
        std::vector<int> harmonicAtoms = {3000, 15000, 60000};
        std::vector<int> boxMol = {256, 1000, 5000};
        std::vector<int> petMol = {32, 96};
        std::string torchHarmonic;
        std::string torchHarmonicDir;
        std::string torchNl;
        std::string soapPt;
        std::string petmadPt;
        std::string toluenePdb;
        int nveSteps = 20;
        int nvtSteps = 20;
        int evalRepeats = 11;
        int warmupEvals = 8;
        int warmupSteps = 10;
        int naiveMaxAtoms = 800;
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
            else if (arg == "--consistency") { consistency = splitInts(need(i)); i++; }
            else if (arg == "--harmonic-atoms") { harmonicAtoms = splitInts(need(i)); i++; }
            else if (arg == "--box-mol") { boxMol = splitInts(need(i)); i++; }
            else if (arg == "--pet-mol") { petMol = splitInts(need(i)); i++; }
            else if (arg == "--torch-harmonic") { torchHarmonic = need(i); i++; }
            else if (arg == "--torch-harmonic-dir") { torchHarmonicDir = need(i); i++; }
            else if (arg == "--torch-nl") { torchNl = need(i); i++; }
            else if (arg == "--soap") { soapPt = need(i); i++; }
            else if (arg == "--petmad") { petmadPt = need(i); i++; }
            else if (arg == "--toluene") { toluenePdb = need(i); i++; }
            else if (arg == "--nve") { nveSteps = std::stoi(need(i)); i++; }
            else if (arg == "--nvt") { nvtSteps = std::stoi(need(i)); i++; }
            else if (arg == "--eval") { evalRepeats = std::stoi(need(i)); i++; }
            else if (arg == "--warmup-eval") { warmupEvals = std::stoi(need(i)); i++; }
            else if (arg == "--warmup-step") { warmupSteps = std::stoi(need(i)); i++; }
            else if (arg == "--naive-max-n") { naiveMaxAtoms = std::stoi(need(i)); i++; }
            else if (arg == "--n-mol") { boxMol = {std::stoi(need(i))}; i++; }
            else if (arg == "--no-md") { includeMd = false; }
            else if (arg == "--plugins-dir") { pluginsDir = need(i); i++; }
            else if (arg == "--help") {
                std::cout <<
                    "openmm-metatomic-bench-settings [options]\n"
                    "  --harmonic-atoms 3000,15000,60000  --box-mol 256,1000,5000  --pet-mol 32,96\n"
                    "  --eval 11 --warmup-eval 8 --warmup-step 10 --nve 20 --nvt 20\n"
                    "  --consistency 0  --naive-max-n 800\n";
                return 0;
            }
        }
        if (!pluginsDir.empty())
            Platform::loadPluginsFromDirectory(pluginsDir);

        std::cout << std::left << std::setw(56) << "case"
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
                    0.0005, 300.0, 1.0,
                    evalRepeats, warmupEvals, warmupSteps
                );
                std::cout << std::left << std::setw(56) << label
                          << std::right << std::setw(8) << geom.types.size()
                          << std::setw(12) << std::fixed << std::setprecision(3) << result.evalMs
                          << std::setw(12) << result.nveMsPerStep
                          << std::setw(12) << result.nvtMsPerStep
                          << std::setw(14) << std::setprecision(4) << result.nveDrift
                          << "\n" << std::flush;
            }
            catch (const std::exception& e) {
                std::string msg = e.what();
                const auto nl = msg.find('\n');
                if (nl != std::string::npos)
                    msg.resize(nl);
                if (msg.size() > 120)
                    msg.resize(117), msg += "...";
                std::cout << std::left << std::setw(56) << label
                          << "  skipped (" << msg << ")\n" << std::flush;
            }
            if (prev)
                setenv("OPENMM_METATOMIC_NEIGHBOR_LIST", prev, 1);
            else
                unsetenv("OPENMM_METATOMIC_NEIGHBOR_LIST");
        };

        auto sweep = [&](const std::string& model, const Geometry& geom, const std::string& path,
                         const std::string& backend, bool periodic, const std::vector<std::string>& nlsFor) {
            for (const auto& device : devices) {
                        if (device == "cuda" && backend != "torch")
                            continue;
                        if (device == "cuda" && (model == "harmonic" || model == "harmonic-nl"
                                                || model == "soap-bpnn"))
                            continue;
                for (const auto& platform : platforms) {
                    for (int cons : consistency) {
                        if (cons && device == "cuda")
                            continue;
                        for (const auto& nl : nlsFor) {
                            if (nl == "naive" && static_cast<int>(geom.types.size()) > naiveMaxAtoms)
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
                                  << "/" << nl
                                  << "/N" << geom.types.size();
                            emit(label.str(), geom, cfg, platform, nl);
                        }
                    }
                }
            }
        };

        auto has = [&](const std::string& name) {
            return std::find(models.begin(), models.end(), name) != models.end();
        };

        if (has("harmonic")) {
            for (int n : harmonicAtoms) {
                const auto geom = harmonicCloud(n);
                for (const auto& backend : backends) {
                    std::string path = "harmonic";
                    if (backend == "torch") {
                        path = torchHarmonicFor(n, torchHarmonic, torchHarmonicDir);
                        if (path.empty())
                            continue;
                    }
                    sweep("harmonic", geom, path, backend, false, {"none"});
                }
            }
        }

        if (has("harmonic-nl")) {
            for (int nMol : boxMol) {
                const auto geom = waterBox(nMol);
                for (const auto& backend : backends) {
                    std::string path = "harmonic-nl";
                    if (backend == "torch") {
                        path = torchNl;
                        if (path.empty())
                            continue;
                    }
                    sweep("harmonic-nl", geom, path, backend, true, nls);
                }
            }
        }

        if (has("soap-bpnn") && soapPt.size()) {
            const auto geom = waterBox(petMol.empty() ? 32 : petMol.front());
            if (std::find(backends.begin(), backends.end(), "torch") != backends.end())
                sweep("soap-bpnn", geom, soapPt, "torch", true, {"vesin"});
        }

        if ((has("pet-mad") || has("pet-mad-box")) && petmadPt.size()) {
            if (std::find(backends.begin(), backends.end(), "torch") == backends.end())
                ;
            else {
                for (int nMol : petMol) {
                    const auto geom = waterBox(nMol);
                    sweep("pet-mad-box", geom, petmadPt, "torch", true, {"vesin"});
                }
            }
        }

        if (has("pet-mad-vacuum") && petmadPt.size()) {
            const auto geom = waterVacuum();
            if (std::find(backends.begin(), backends.end(), "torch") != backends.end())
                sweep("pet-mad", geom, petmadPt, "torch", false, {"vesin"});
        }

        if (has("toluene") && petmadPt.size() && toluenePdb.size()) {
            const auto geom = loadPdb(toluenePdb);
            if (std::find(backends.begin(), backends.end(), "torch") != backends.end())
                sweep("toluene", geom, petmadPt, "torch", false, {"vesin"});
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "bench_settings failed: " << e.what() << "\n";
        return 1;
    }
}
