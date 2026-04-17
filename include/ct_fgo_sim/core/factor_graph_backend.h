#pragma once

#include "ct_fgo_sim/core/factor_graph_session.h"

#include <string>

namespace ct_fgo_sim {

enum class GraphBackend {
    Ceres,
    Gtsam,
};

GraphBackend ParseGraphBackend(const std::string& value);
const char* GraphBackendName(GraphBackend backend);

/// Backend-neutral graph solve entry used by System.
bool BuildAndSolveFactorGraphWithBackend(FactorGraphSession& session, GraphBackend backend);
const char* ActiveGraphBackendImpl(GraphBackend backend);
const char* LastBackendImpl();
const char* LastBackendFallbackReason();

/// Runtime capability probe (true only when GTSAM backend is built in).
bool IsGtsamBackendAvailable();

}  // namespace ct_fgo_sim

