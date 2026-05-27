#include "DepGraph.h"
#include "Helpers.h"
#include <algorithm>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <queue>

namespace stable_abi {

static std::string canonicalize(llvm::StringRef path) {
    llvm::SmallString<256> result(path);
    llvm::sys::path::remove_dots(result, true);
    return std::string(result);
}

void DepGraph::build(const IncludeGraph &includes,
                     std::map<std::string, std::vector<Finding>> fileFindings) {
    for (auto &[file, findings] : fileFindings)
        findings_[canonicalize(file)] = std::move(findings);

    for (const auto &[file, _] : findings_)
        nodes_.insert(file);

    IncludeGraph canonIncludes;
    for (const auto &[from, tos] : includes) {
        auto cFrom = canonicalize(from);
        for (const auto &to : tos)
            canonIncludes[cFrom].insert(canonicalize(to));
    }

    for (const auto &start : nodes_) {
        std::set<std::string> visited;
        std::queue<std::string> q;
        q.push(start);
        visited.insert(start);
        while (!q.empty()) {
            auto cur = q.front(); q.pop();
            auto it = canonIncludes.find(cur);
            if (it == canonIncludes.end()) continue;
            for (const auto &next : it->second) {
                if (visited.count(next)) continue;
                visited.insert(next);
                if (nodes_.count(next))
                    edges_[start].insert(next);
                q.push(next);
            }
        }
    }
}

std::vector<std::vector<std::string>> DepGraph::partitions() const {
    std::map<std::string, std::set<std::string>> undirected;
    for (const auto &[from, tos] : edges_) {
        if (!nodes_.count(from)) continue;
        for (const auto &to : tos) {
            if (!nodes_.count(to)) continue;
            undirected[from].insert(to);
            undirected[to].insert(from);
        }
    }

    std::set<std::string> visited;
    std::vector<std::vector<std::string>> components;

    for (const auto &node : nodes_) {
        if (visited.count(node)) continue;
        std::vector<std::string> component;
        std::queue<std::string> q;
        q.push(node);
        visited.insert(node);
        while (!q.empty()) {
            auto cur = q.front(); q.pop();
            component.push_back(cur);
            for (const auto &neighbor : undirected[cur]) {
                if (!visited.count(neighbor) && nodes_.count(neighbor)) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }
        std::sort(component.begin(), component.end());
        components.push_back(std::move(component));
    }

    std::sort(components.begin(), components.end());
    return components;
}

std::map<size_t, std::set<size_t>> DepGraph::condensedDag(
    const std::vector<std::vector<std::string>> &parts) const {
    std::map<std::string, size_t> fileToGroup;
    for (size_t i = 0; i < parts.size(); ++i)
        for (const auto &f : parts[i])
            fileToGroup[f] = i;

    std::map<size_t, std::set<size_t>> dag;
    for (size_t i = 0; i < parts.size(); ++i)
        dag[i];

    for (const auto &[from, tos] : edges_) {
        auto itFrom = fileToGroup.find(from);
        if (itFrom == fileToGroup.end()) continue;
        for (const auto &to : tos) {
            auto itTo = fileToGroup.find(to);
            if (itTo == fileToGroup.end()) continue;
            if (itFrom->second != itTo->second)
                dag[itFrom->second].insert(itTo->second);
        }
    }
    return dag;
}

std::vector<std::vector<size_t>> DepGraph::topoRounds(
    const std::map<size_t, std::set<size_t>> &dag, size_t n) const {
    std::map<size_t, size_t> inDegree;
    for (size_t i = 0; i < n; ++i)
        inDegree[i] = 0;
    for (const auto &[from, tos] : dag)
        for (auto to : tos)
            ++inDegree[to];

    std::vector<std::vector<size_t>> rounds;
    std::set<size_t> remaining;
    for (size_t i = 0; i < n; ++i)
        remaining.insert(i);

    while (!remaining.empty()) {
        std::vector<size_t> round;
        for (auto id : remaining) {
            if (inDegree[id] == 0)
                round.push_back(id);
        }
        if (round.empty())
            llvm::report_fatal_error(
                "cycle in condensed DAG — this can't happen");

        for (auto id : round) {
            remaining.erase(id);
            auto it = dag.find(id);
            if (it != dag.end())
                for (auto dep : it->second)
                    --inDegree[dep];
        }
        std::sort(round.begin(), round.end());
        rounds.push_back(std::move(round));
    }
    return rounds;
}

MigrationPlan DepGraph::computePlan() const {
    MigrationPlan plan;

    auto parts = partitions();
    auto dag = condensedDag(parts);
    auto rounds = topoRounds(dag, parts.size());

    for (size_t i = 0; i < parts.size(); ++i) {
        MigrationGroup group;
        group.id = i;
        for (const auto &file : parts[i]) {
            bool isHeader = llvm::StringRef(file).ends_with(".h") ||
                            llvm::StringRef(file).ends_with(".cuh") ||
                            llvm::StringRef(file).ends_with(".hpp");
            if (isHeader)
                group.headers.push_back(file);
            else
                group.sources.push_back(file);
            auto it = findings_.find(file);
            if (it == findings_.end()) continue;
            for (const auto &f : it->second) {
                ++group.api_counts[f.old_text];
                ++group.total_findings;
            }
        }
        plan.groups.push_back(std::move(group));
    }

    plan.rounds = std::move(rounds);
    plan.critical_path_length = plan.rounds.size();
    return plan;
}

void printMigrationPlan(const MigrationPlan &plan, bool json) {
    bool fully_parallel = plan.rounds.size() <= 1;

    if (json) {
        llvm::json::OStream J(llvm::outs(), 2);
        J.objectBegin();

        J.attributeBegin("groups");
        J.arrayBegin();
        for (const auto &g : plan.groups) {
            J.objectBegin();
            J.attribute("id", static_cast<int64_t>(g.id));
            J.attributeBegin("sources");
            J.arrayBegin();
            for (const auto &f : g.sources)
                J.value(f);
            J.arrayEnd();
            J.attributeEnd();
            J.attributeBegin("headers");
            J.arrayBegin();
            for (const auto &f : g.headers)
                J.value(f);
            J.arrayEnd();
            J.attributeEnd();
            J.attribute("findings", static_cast<int64_t>(g.total_findings));
            J.attributeBegin("apis");
            J.objectBegin();
            for (const auto &[api, count] : g.api_counts)
                J.attribute(api, static_cast<int64_t>(count));
            J.objectEnd();
            J.attributeEnd();
            J.objectEnd();
        }
        J.arrayEnd();
        J.attributeEnd();

        J.attributeBegin("rounds");
        J.arrayBegin();
        for (const auto &round : plan.rounds) {
            J.arrayBegin();
            for (auto id : round)
                J.value(static_cast<int64_t>(id));
            J.arrayEnd();
        }
        J.arrayEnd();
        J.attributeEnd();

        J.attribute("critical_path_length",
                    static_cast<int64_t>(plan.critical_path_length));
        J.attribute("fully_parallel", fully_parallel);
        J.objectEnd();
        llvm::outs() << "\n";
        return;
    }

    llvm::outs() << "Migration plan: " << plan.groups.size() << " group"
                 << (plan.groups.size() != 1 ? "s" : "");
    if (fully_parallel && plan.groups.size() > 1)
        llvm::outs() << " (all independent — transform in parallel)";
    llvm::outs() << "\n\n";

    for (size_t ri = 0; ri < plan.rounds.size(); ++ri) {
        llvm::outs() << "Round " << (ri + 1) << ":\n";

        for (auto gid : plan.rounds[ri]) {
            const auto &g = plan.groups[gid];
            llvm::outs() << "  Group ";
            if (gid < 26)
                llvm::outs() << static_cast<char>('A' + gid);
            else
                llvm::outs() << gid;
            llvm::outs() << " ("
                         << g.file_count() << " file"
                         << (g.file_count() != 1 ? "s" : "")
                         << ", " << g.total_findings << " finding"
                         << (g.total_findings != 1 ? "s" : "") << "):\n";
            for (const auto &f : g.sources)
                llvm::outs() << "    " << f << "\n";
            if (!g.headers.empty()) {
                llvm::outs() << "    Headers:\n";
                for (const auto &f : g.headers)
                    llvm::outs() << "      " << f << "\n";
            }
            if (!g.api_counts.empty()) {
                llvm::outs() << "    APIs: ";
                size_t k = 0;
                for (const auto &[api, count] : g.api_counts) {
                    llvm::outs() << api << " (" << count << ")";
                    if (++k < g.api_counts.size()) llvm::outs() << ", ";
                }
                llvm::outs() << "\n";
            }
            llvm::outs() << "\n";
        }
    }
}

} // namespace stable_abi
