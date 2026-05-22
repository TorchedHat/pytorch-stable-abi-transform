#pragma once

#include "PreprocessorCallbacks.h"
#include "Reporter.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace stable_abi {

struct MigrationGroup {
    size_t id = 0;
    std::vector<std::string> files;
    std::map<std::string, size_t> api_counts;
    size_t total_findings = 0;
};

struct MigrationPlan {
    std::vector<MigrationGroup> groups;
    std::vector<std::vector<size_t>> rounds;
    size_t critical_path_length = 0;
};

class DepGraph {
public:
    void build(const IncludeGraph &includes,
               std::map<std::string, std::vector<Finding>> fileFindings);

    MigrationPlan computePlan() const;

    void printPlan(const MigrationPlan &plan, bool json) const;

private:
    std::set<std::string> nodes_;
    std::map<std::string, std::set<std::string>> edges_;
    std::map<std::string, std::vector<Finding>> findings_;

    std::vector<std::vector<std::string>> partitions() const;
    std::map<size_t, std::set<size_t>> condensedDag(
        const std::vector<std::vector<std::string>> &parts) const;
    std::vector<std::vector<size_t>> topoRounds(
        const std::map<size_t, std::set<size_t>> &dag, size_t n) const;
};

} // namespace stable_abi
