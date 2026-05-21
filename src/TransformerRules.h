#pragma once

#include "Helpers.h"
#include "Reporter.h"
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Transformer/RewriteRule.h>

namespace stable_abi {

[[nodiscard]] clang::transformer::RewriteRule buildTransformerRules(
    Reporter &reporter, bool rewrite_mode,
    const std::string &projectRoot = "");

class DeviceGuardCallback
    : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
    DeviceGuardCallback(FileReplacements &fileRepls, Reporter &rep,
                        bool rewrite, const std::string &projectRoot = "")
        : file_repls_(fileRepls), reporter_(rep), rewrite_mode_(rewrite),
          project_root_(projectRoot) {}

    void run(const clang::ast_matchers::MatchFinder::MatchResult &Result)
        override;

    const std::string &getLastDeviceExpr() const { return last_device_expr_; }

private:
    FileReplacements &file_repls_;
    Reporter &reporter_;
    bool rewrite_mode_;
    std::string project_root_;
    std::string last_device_expr_;
};

class CudaStreamCallback
    : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
    CudaStreamCallback(FileReplacements &fileRepls, Reporter &rep,
                        bool rewrite, const DeviceGuardCallback &guard_cb,
                        const std::string &projectRoot = "")
        : file_repls_(fileRepls), reporter_(rep), rewrite_mode_(rewrite),
          guard_cb_(guard_cb), project_root_(projectRoot) {}

    void run(const clang::ast_matchers::MatchFinder::MatchResult &Result)
        override;

private:
    FileReplacements &file_repls_;
    Reporter &reporter_;
    bool rewrite_mode_;
    const DeviceGuardCallback &guard_cb_;
    std::string project_root_;
};

void registerManualMatchers(clang::ast_matchers::MatchFinder &finder,
                            CudaStreamCallback &streamCallback,
                            DeviceGuardCallback &guardCallback);

} // namespace stable_abi
