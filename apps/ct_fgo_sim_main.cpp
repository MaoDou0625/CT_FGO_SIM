#include "ct_fgo_sim/core/system.h"

#include <glog/logging.h>

#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    if (argc < 2) {
        std::cerr << "Usage: ct_fgo_sim_main <config.yaml>\n";
        return 1;
    }

    ct_fgo_sim::System system;
    if (!system.LoadConfig(std::filesystem::path(argv[1]))) {
        return 2;
    }

    system.Describe();
    LOG(INFO) << "Next step: wire IMU/GNSS loaders and optimization problem construction.";
    return 0;
}
