#pragma once

#include <string>
#include <vector>

namespace stable_abi {

struct Violation {
    std::string file;
    unsigned line;
    unsigned col;
    std::string text;
    std::string reason;
};

struct VerifyOptions {
    std::string pytorch_root;
    std::string resource_dir;
    std::vector<std::string> extra_includes;
    std::string cuda_include;
};

// Compile-based verification: parse the file with only stable-ABI headers
// reachable via a shadow include tree. Compiler errors = unstable API usage.
[[nodiscard]] std::vector<Violation> verifyStableAbi(const std::string &filepath,
                                                      const VerifyOptions &opts);

// Regex-based verification (legacy fallback): scan for forbidden patterns.
[[nodiscard]] std::vector<Violation> verifyStableAbiRegex(const std::string &filepath);

void printViolations(const std::vector<Violation> &violations);
void printViolationsJson(const std::vector<Violation> &violations);

} // namespace stable_abi
