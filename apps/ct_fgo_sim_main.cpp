#include "ct_fgo_sim/core/system.h"

#include <glog/logging.h>

#include <cstdio>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    std::fprintf(stderr, "[probe] main enter\n");
    google::InitGoogleLogging(argv[0]);

    if (argc < 2) {
        std::cerr << "Usage: ct_fgo_sim_main <config.yaml>\n";
        return 1;
    }

    std::fprintf(stderr, "[probe] loading config %s\n", argv[1]);
    ct_fgo_sim::System system;
    if (!system.LoadConfig(std::filesystem::path(argv[1]))) {
        std::fprintf(stderr, "[probe] LoadConfig failed\n");
        return 2;
    }
    std::fprintf(stderr, "[probe] Run enter\n");
    const bool ok = system.Run();
    std::fprintf(stderr, "[probe] Run exit ok=%d\n", ok ? 1 : 0);
    return ok ? 0 : 3;
}
