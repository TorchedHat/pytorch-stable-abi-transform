#include "Config.h"
#include "StableAbiAction.h"
#include "Verifier.h"
#include <array>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <optional>
#include <string_view>

using stable_abi::Mode;
using stable_abi::OutputFormat;
using stable_abi::VerifyMethod;

static void printVersion(llvm::raw_ostream &OS) {
    OS << "stable-abi-transform " << TOOL_VERSION << "\n";
}

static llvm::cl::OptionCategory
    ToolCategory("stable-abi-transform options");

static llvm::cl::opt<std::string>
    ModeOpt("mode",
            llvm::cl::desc("Operating mode: audit (default), rewrite, or verify"),
            llvm::cl::init("audit"), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string>
    FormatOpt("format",
              llvm::cl::desc("Output format: text (default) or json"),
              llvm::cl::init("text"), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string>
    PytorchRoot("pytorch-root",
                llvm::cl::desc("Path to PyTorch source/install root (required for compile-based verification)"),
                llvm::cl::init(""), llvm::cl::cat(ToolCategory));

static llvm::cl::list<std::string>
    ExtraIncludes("extra-includes",
                  llvm::cl::desc("Additional include paths for verification (project-specific headers)"),
                  llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string>
    CudaInclude("cuda-include",
                llvm::cl::desc("Path to CUDA include directory (default: auto-detect)"),
                llvm::cl::init(""), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string>
    VerifyMethodOpt("verify-method",
                    llvm::cl::desc("Verification method: compile (default) or regex"),
                    llvm::cl::init("compile"), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string>
    ProjectRoot("project-root",
                llvm::cl::desc("Project root directory — rewrites files under this path (not just main file). "
                               "Also auto-discovers .cpp/.cu source files when no sources given."),
                llvm::cl::init(""), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string>
    ConfigFile("config",
               llvm::cl::desc("Path to YAML config file (.stable-abi.yaml)"),
               llvm::cl::init(""), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<bool>
    InitConfig("init-config",
               llvm::cl::desc("Print an example .stable-abi.yaml config to stdout and exit"),
               llvm::cl::init(false), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<bool>
    DryRun("dry-run",
           llvm::cl::desc("Show unified diff of what --mode=rewrite would change, without modifying files"),
           llvm::cl::init(false), llvm::cl::cat(ToolCategory));

static std::string detectResourceDir() {
    llvm::SmallString<256> path(LLVM_INSTALL_PREFIX);
    llvm::sys::path::append(path, "lib", "clang", CLANG_VERSION_MAJOR_STR);
    if (llvm::sys::fs::is_directory(path))
        return std::string(path);
    return "";
}

static std::string detectCudaInclude() {
    static constexpr std::array candidates = {
        std::string_view{"/usr/local/cuda/include"},
        std::string_view{"/usr/include/cuda"},
    };
    for (const auto &path : candidates) {
        if (llvm::sys::fs::is_directory(path))
            return std::string(path);
    }
    return "";
}

static std::optional<Mode> parseMode(llvm::StringRef s) {
    if (s == "audit") return Mode::Audit;
    if (s == "rewrite") return Mode::Rewrite;
    if (s == "verify") return Mode::Verify;
    return std::nullopt;
}

static std::optional<OutputFormat> parseFormat(llvm::StringRef s) {
    if (s == "text") return OutputFormat::Text;
    if (s == "json") return OutputFormat::Json;
    return std::nullopt;
}

static std::optional<VerifyMethod> parseVerifyMethod(llvm::StringRef s) {
    if (s == "compile") return VerifyMethod::Compile;
    if (s == "regex") return VerifyMethod::Regex;
    return std::nullopt;
}

static stable_abi::VerifyOptions buildVerifyOptions(
    const std::string &resourceDir,
    const std::string &pytorchRoot,
    const std::vector<std::string> &extraIncludes,
    const std::string &cudaInclude) {
    stable_abi::VerifyOptions opts;
    opts.pytorch_root = pytorchRoot;
    opts.resource_dir = resourceDir;
    opts.extra_includes = extraIncludes;
    opts.cuda_include = cudaInclude;
    if (opts.cuda_include.empty())
        opts.cuda_include = detectCudaInclude();
    return opts;
}

static int runVerify(const std::vector<std::string> &sources,
                     const std::string &resourceDir,
                     const std::string &pytorchRoot,
                     const std::vector<std::string> &extraIncludes,
                     const std::string &cudaInclude,
                     VerifyMethod method,
                     OutputFormat format,
                     bool allow_fallback = false) {
    bool use_compile = (method == VerifyMethod::Compile);
    bool json = (format == OutputFormat::Json);
    auto opts = buildVerifyOptions(resourceDir, pytorchRoot,
                                   extraIncludes, cudaInclude);

    if (use_compile && opts.pytorch_root.empty()) {
        if (allow_fallback) {
            use_compile = false;
            llvm::errs() << "note: --pytorch-root not set, using regex-based "
                            "verification (less precise).\n"
                         << "      Pass --pytorch-root for compile-based "
                            "verification.\n";
        } else {
            llvm::errs() << "error: --pytorch-root required for compile-based verification\n"
                         << "       use --verify-method=regex for regex-based fallback\n";
            return 1;
        }
    }

    size_t total_violations = 0;
    for (const auto &src : sources) {
        auto violations = use_compile
            ? stable_abi::verifyStableAbi(src, opts)
            : stable_abi::verifyStableAbiRegex(src);
        if (json)
            stable_abi::printViolationsJson(violations);
        else
            stable_abi::printViolations(violations);
        total_violations += violations.size();
    }
    return total_violations > 0 ? 1 : 0;
}

static std::vector<std::string> discoverSources(llvm::StringRef root) {
    std::vector<std::string> result;
    std::error_code ec;
    for (llvm::sys::fs::recursive_directory_iterator it(root, ec, /*follow_symlinks=*/false), end;
         it != end && !ec; it.increment(ec)) {
        auto path = it->path();
        auto ext = llvm::sys::path::extension(path);
        if (ext == ".cpp" || ext == ".cu" || ext == ".cuh")
            result.push_back(std::string(path));
    }
    std::sort(result.begin(), result.end());
    return result;
}

enum class ConfigResult { NotFound, Loaded, Error };

static ConfigResult tryLoadConfig(stable_abi::Config &cfg, std::string &error) {
    std::string configPath = ConfigFile.getValue();
    if (configPath.empty()) {
        if (llvm::sys::fs::exists(".stable-abi.yaml"))
            configPath = ".stable-abi.yaml";
        else
            return ConfigResult::NotFound;
    }

    if (!stable_abi::loadConfig(configPath, cfg, error))
        return ConfigResult::Error;
    llvm::errs() << "note: loaded config from " << configPath << "\n";
    return ConfigResult::Loaded;
}

static bool applyCliOverrides(stable_abi::Config &cfg) {
    if (ModeOpt.getNumOccurrences() > 0) {
        auto m = parseMode(ModeOpt.getValue());
        if (!m) {
            llvm::errs() << "error: invalid mode '" << ModeOpt.getValue()
                         << "'. Must be one of: audit, rewrite, verify\n";
            return false;
        }
        cfg.mode = *m;
    }
    if (FormatOpt.getNumOccurrences() > 0) {
        auto f = parseFormat(FormatOpt.getValue());
        if (!f) {
            llvm::errs() << "error: invalid format '" << FormatOpt.getValue()
                         << "'. Must be one of: text, json\n";
            return false;
        }
        cfg.format = *f;
    }
    if (PytorchRoot.getNumOccurrences() > 0)
        cfg.pytorch_root = PytorchRoot.getValue();
    if (ProjectRoot.getNumOccurrences() > 0)
        cfg.project_root = ProjectRoot.getValue();
    if (VerifyMethodOpt.getNumOccurrences() > 0) {
        auto v = parseVerifyMethod(VerifyMethodOpt.getValue());
        if (!v) {
            llvm::errs() << "error: invalid verify-method '"
                         << VerifyMethodOpt.getValue()
                         << "'. Must be one of: compile, regex\n";
            return false;
        }
        cfg.verify_method = *v;
    }
    if (CudaInclude.getNumOccurrences() > 0)
        cfg.cuda_include = CudaInclude.getValue();
    if (ExtraIncludes.getNumOccurrences() > 0)
        cfg.extra_includes = std::vector<std::string>(
            ExtraIncludes.begin(), ExtraIncludes.end());
    return true;
}

static bool validateSources(const std::vector<std::string> &sources) {
    bool missing = false;
    for (const auto &src : sources) {
        if (!llvm::sys::fs::exists(src)) {
            llvm::errs() << "error: source file not found: " << src << "\n";
            missing = true;
        }
    }
    return !missing;
}

static int runWithConfig(stable_abi::Config &cfg,
                         clang::tooling::CompilationDatabase *externalDB = nullptr) {
    auto resourceDir = detectResourceDir();

    bool json = (cfg.format == OutputFormat::Json);

    std::string projectRoot = cfg.project_root;
    if (!projectRoot.empty()) {
        llvm::SmallString<256> abs(projectRoot);
        llvm::sys::fs::make_absolute(abs);
        projectRoot = std::string(abs);
    }

    std::vector<std::string> sources = cfg.sources;
    if (sources.empty() && !projectRoot.empty()) {
        if (!llvm::sys::fs::is_directory(projectRoot)) {
            llvm::errs() << "error: project root is not a directory: "
                         << projectRoot << "\n";
            return 1;
        }
        sources = discoverSources(projectRoot);
        if (sources.empty()) {
            llvm::errs() << "error: no .cpp/.cu/.cuh files found under "
                         << projectRoot << "\n";
            return 1;
        }
        llvm::errs() << "note: auto-discovered " << sources.size()
                     << " source files under " << projectRoot << "\n";
    }

    if (sources.empty()) {
        llvm::errs() << "error: no source files specified\n";
        return 1;
    }

    if (!validateSources(sources))
        return 1;

    if (cfg.mode == Mode::Verify) {
        return runVerify(sources, resourceDir, cfg.pytorch_root,
                         cfg.extra_includes, cfg.cuda_include,
                         cfg.verify_method, cfg.format);
    }

    auto writeMode = (cfg.mode == Mode::Rewrite)
        ? (DryRun.getValue() ? stable_abi::WriteMode::DryRun
                             : stable_abi::WriteMode::Rewrite)
        : stable_abi::WriteMode::Audit;

    std::unique_ptr<clang::tooling::FixedCompilationDatabase> ownedDB;
    clang::tooling::CompilationDatabase *db = externalDB;
    if (!db) {
        std::vector<std::string> clangArgs;
        for (const auto &flag : cfg.compiler_flags)
            clangArgs.push_back(flag);
        for (const auto &inc : cfg.include_paths)
            clangArgs.push_back("-I" + inc);
        ownedDB = std::make_unique<clang::tooling::FixedCompilationDatabase>(
            ".", clangArgs);
        db = ownedDB.get();
    }
    clang::tooling::ClangTool Tool(*db, sources);

    Tool.appendArgumentsAdjuster(
        [&resourceDir](const clang::tooling::CommandLineArguments &Args,
                        llvm::StringRef Filename) {
            clang::tooling::CommandLineArguments AdjustedArgs = Args;
            if (Filename.ends_with(".cu") || Filename.ends_with(".cuh")) {
                AdjustedArgs.push_back("--cuda-host-only");
            }
            if (!resourceDir.empty()) {
                AdjustedArgs.push_back("-resource-dir");
                AdjustedArgs.push_back(resourceDir);
            }
            return AdjustedArgs;
        });

    stable_abi::ActionOptions actionOpts{
        .write_mode = writeMode,
        .project_root = projectRoot,
    };
    auto Factory = std::make_unique<stable_abi::StableAbiActionFactory>(
        actionOpts);

    stable_abi::ParseDiagConsumer diagConsumer(Factory->getReporter());
    Tool.setDiagnosticConsumer(&diagConsumer);

    int result = Tool.run(Factory.get());

    auto &reporter = Factory->getReporter();
    reporter.suppressRedundantFlags();
    if (json) {
        reporter.printJson();
    } else {
        reporter.printReport();
        reporter.printSummary();
    }
    reporter.printParseWarnings();

    if (writeMode == stable_abi::WriteMode::Rewrite && result == 0) {
        if (!json)
            llvm::outs() << "\n--- Post-rewrite ABI verification ---\n";
        int verify_result = runVerify(sources, resourceDir, cfg.pytorch_root,
                                      cfg.extra_includes, cfg.cuda_include,
                                      cfg.verify_method, cfg.format, true);
        if (verify_result > 0) {
            if (!json)
                llvm::outs() << "\nUnstable API references remain. "
                                "Manual review needed for flagged items.\n";
            result = 1;
        }
    }

    if (result == 0) {
        if (writeMode == stable_abi::WriteMode::Rewrite) {
            if (reporter.flagCount() > 0)
                result = 1;
        } else {
            if (reporter.rewriteCount() > 0 || reporter.flagCount() > 0)
                result = 1;
        }
    }

    return result;
}

int main(int argc, const char **argv) {
    llvm::cl::SetVersionPrinter(printVersion);

    auto ExpectedParser =
        clang::tooling::CommonOptionsParser::create(
            argc, argv, ToolCategory, llvm::cl::ZeroOrMore);

    if (!ExpectedParser) {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }

    if (InitConfig) {
        stable_abi::printExampleConfig();
        return 0;
    }

    stable_abi::Config cfg;
    std::string configError;
    auto configResult = tryLoadConfig(cfg, configError);
    if (configResult == ConfigResult::Error) {
        llvm::errs() << "error: " << configError << "\n";
        return 1;
    }

    clang::tooling::CommonOptionsParser &OptionsParser = ExpectedParser.get();

    if (configResult == ConfigResult::Loaded) {
        if (!applyCliOverrides(cfg))
            return 1;
        auto &cliSources = OptionsParser.getSourcePathList();
        if (!cliSources.empty())
            cfg.sources = cliSources;
        return runWithConfig(cfg);
    }

    // Legacy CLI path — no config file
    auto &cliSources = OptionsParser.getSourcePathList();

    if (cliSources.empty() && ProjectRoot.getValue().empty()) {
        llvm::errs() << "stable-abi-transform: no input files\n\n"
                     << "Usage:\n"
                     << "  stable-abi-transform [options] <source files> -- "
                        "[clang options]\n\n"
                     << "Quick start:\n"
                     << "  # Audit a file (report what needs to change)\n"
                     << "  stable-abi-transform file.cu -- -std=c++20 "
                        "-I/path/to/pytorch\n\n"
                     << "  # Rewrite in-place\n"
                     << "  stable-abi-transform --mode=rewrite file.cu -- "
                        "-std=c++20 -I/path/to/pytorch\n\n"
                     << "  # Generate config file\n"
                     << "  stable-abi-transform --init-config\n\n"
                     << "Run with --help for all options.\n";
        return 1;
    }

    if (!applyCliOverrides(cfg))
        return 1;
    cfg.sources = cliSources;
    return runWithConfig(cfg, &OptionsParser.getCompilations());
}
