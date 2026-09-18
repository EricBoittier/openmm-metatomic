/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- *
 * NVE / NVT through a real OpenMM::Context and MetatomicForce.
 * -------------------------------------------------------------------------- */

#include "SimHelpers.h"

#include <iostream>
#include <string>
#include <vector>

using namespace OpenMMMetatomic::sim;

int main(int argc, char** argv) {
    try {
        std::string model = "harmonic";
        std::string backend = "auto";
        std::string device;
        std::string platform;
        std::string systemName = "water";
        std::string pdb;
        std::string pluginsDir;
        int nMol = 8;
        int nAtoms = 0;
        int nveSteps = 200;
        int nvtSteps = 100;
        int evalRepeats = 11;
        int warmupEvals = 8;
        int warmupSteps = 10;
        double dt = 0.0005;
        double temperature = 300.0;
        double friction = 1.0;
        bool checkConsistency = false;
        bool periodic = false;

        auto need = [&](int i) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + std::string(argv[i]));
            return argv[i + 1];
        };
        for (int i = 1; i < argc; i++) {
            const std::string arg = argv[i];
            if (arg == "--model") { model = need(i); i++; }
            else if (arg == "--backend") { backend = need(i); i++; }
            else if (arg == "--device") { device = need(i); i++; }
            else if (arg == "--platform") { platform = need(i); i++; }
            else if (arg == "--system") { systemName = need(i); i++; }
            else if (arg == "--pdb") { pdb = need(i); i++; }
            else if (arg == "--n-mol") { nMol = std::stoi(need(i)); i++; }
            else if (arg == "--n-atoms") { nAtoms = std::stoi(need(i)); i++; }
            else if (arg == "--nve") { nveSteps = std::stoi(need(i)); i++; }
            else if (arg == "--nvt") { nvtSteps = std::stoi(need(i)); i++; }
            else if (arg == "--eval") { evalRepeats = std::stoi(need(i)); i++; }
            else if (arg == "--warmup-eval") { warmupEvals = std::stoi(need(i)); i++; }
            else if (arg == "--warmup-step") { warmupSteps = std::stoi(need(i)); i++; }
            else if (arg == "--dt") { dt = std::stod(need(i)); i++; }
            else if (arg == "--temperature") { temperature = std::stod(need(i)); i++; }
            else if (arg == "--friction") { friction = std::stod(need(i)); i++; }
            else if (arg == "--check-consistency") { checkConsistency = true; }
            else if (arg == "--periodic") { periodic = true; }
            else if (arg == "--plugins-dir") { pluginsDir = need(i); i++; }
            else if (arg == "--help" || arg == "-h") {
                std::cout <<
                    "openmm-metatomic-run-md --model harmonic|harmonic-nl|<file.pt>\n"
                    "  --system water|water-box|cloud|pdb   --pdb file.pdb   --n-mol 8   --n-atoms 3000\n"
                    "  --backend auto|core|torch  --device cpu|cuda\n"
                    "  --platform Reference|CPU|CUDA  --plugins-dir <OpenMM lib/plugins>\n"
                    "  --nve 200 --nvt 100 --dt 0.0005 --eval 11 --warmup-eval 8 --warmup-step 10\n"
                    "  [--check-consistency] [--periodic]\n";
                return 0;
            }
            else {
                throw std::runtime_error("unknown argument " + arg);
            }
        }

        loadPlatformPlugins(pluginsDir);

        Geometry geom;
        if (!pdb.empty() || systemName == "pdb") {
            if (pdb.empty())
                throw std::runtime_error("--system pdb requires --pdb");
            geom = loadPdb(pdb);
        }
        else if (systemName == "water-box") {
            geom = waterBox(nMol);
        }
        else if (systemName == "water") {
            geom = waterVacuum();
            if (periodic) {
                geom.periodic = true;
                geom.a = OpenMM::Vec3(1.5, 0, 0);
                geom.b = OpenMM::Vec3(0, 1.5, 0);
                geom.c = OpenMM::Vec3(0, 0, 1.5);
            }
        }
        else if (systemName == "cloud") {
            geom = harmonicCloud(nAtoms > 0 ? nAtoms : 3000);
        }
        else {
            throw std::runtime_error("unknown --system " + systemName);
        }

        ForceConfig config;
        config.modelPath = model;
        config.backend = backend;
        config.device = device;
        config.checkConsistency = checkConsistency;
        config.periodic = geom.periodic || periodic;

        std::cout << "model=" << model
                  << " backend=" << backend
                  << " device=" << (device.empty() ? "default" : device)
                  << " platform=" << (platform.empty() ? "auto" : platform)
                  << " N=" << geom.types.size()
                  << " periodic=" << (config.periodic ? "true" : "false")
                  << " available=" << availablePlatforms()
                  << "\n";

        const auto result = runMd(
            geom, config, platform, nveSteps, nvtSteps, dt, temperature, friction,
            evalRepeats, warmupEvals, warmupSteps
        );
        std::cout << "eval_ms=" << result.evalMs << " (median, hot Context)\n";
        if (!result.nve.empty()) {
            printTrace("NVE", result.nve, dt);
            std::cout << "nve_ms_per_step=" << result.nveMsPerStep
                      << "  steps=" << nveSteps
                      << "  drift=" << result.nveDrift << " kJ/mol\n";
        }
        if (!result.nvt.empty()) {
            printTrace("NVT", result.nvt, dt);
            std::cout << "nvt_ms_per_step=" << result.nvtMsPerStep
                      << "  steps=" << nvtSteps << "\n";
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "run_md failed: " << e.what() << "\n";
        return 1;
    }
}
