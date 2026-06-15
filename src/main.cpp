#include "Config.h"
#include "DepGraph.h"
#include "StableAbiAction.h"
#include "Verifier.h"
#include <array>
#include <atomic>
#include <chrono>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/ThreadPool.h>
#include <llvm/Support/Threading.h>
#include <mutex>
#include <optional>
#include <string_view>

using stable_abi::Mode;
using stable_abi::OutputFormat;
using stable_abi::VerifyMethod;

static void printVersion(llvm::raw_ostream &OS) {
    OS << "stable-abi-transform " << TOOL_VERSION << "\n";
}

static llvm::cl::OptionCategory ToolCategory("stable-abi-transform options");

static llvm::cl::opt<std::string> ModeOpt(
    "mode",
    llvm::cl::desc("Operating mode: audit (default), rewrite, verify, or plan"),
    llvm::cl::init("audit"), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string>
    FormatOpt("format", llvm::cl::desc("Output format: text (default) or json"),
              llvm::cl::init("text"), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string> PytorchRoot(
    "pytorch-root",
    llvm::cl::desc(
        "Path to PyTorch root, or \"auto\" to detect from pip-installed torch"),
    llvm::cl::init(""), llvm::cl::cat(ToolCategory));

static llvm::cl::list<std::string> ExtraIncludes(
    "extra-includes",
    llvm::cl::desc(
        "Additional include paths for verification (project-specific headers)"),
    llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string> CudaInclude(
    "cuda-include",
    llvm::cl::desc("Path to CUDA include directory (default: auto-detect)"),
    llvm::cl::init(""), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string> VerifyMethodOpt(
    "verify-method",
    llvm::cl::desc("Verification method: compile (default) or regex"),
    llvm::cl::init("compile"), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string>
    ProjectRoot("project-root",
                llvm::cl::desc("Project root directory — rewrites files under "
                               "this path (not just main file). "
                               "Also auto-discovers .cpp/.cu source files when "
                               "no transform targets given."),
                llvm::cl::init(""), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string> OutputDir(
    "output-dir",
    llvm::cl::desc(
        "Write transformed files to this directory instead of in-place"),
    llvm::cl::init(""), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string>
    ConfigFile("config",
               llvm::cl::desc("Path to YAML config file (.stable-abi.yaml)"),
               llvm::cl::init(""), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<bool> InitConfig(
    "init-config",
    llvm::cl::desc(
        "Print an example .stable-abi.yaml config to stdout and exit"),
    llvm::cl::init(false), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<bool>
    DryRun("dry-run",
           llvm::cl::desc("Show unified diff of what --mode=rewrite would "
                          "change, without modifying files"),
           llvm::cl::init(false), llvm::cl::cat(ToolCategory));

static llvm::cl::opt<unsigned> Jobs(
    "jobs",
    llvm::cl::desc(
        "Number of parallel TU processing threads (0 = auto, 1 = sequential)"),
    llvm::cl::init(0), llvm::cl::cat(ToolCategory));

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
    if (s == "audit")
        return Mode::Audit;
    if (s == "rewrite")
        return Mode::Rewrite;
    if (s == "verify")
        return Mode::Verify;
    if (s == "plan")
        return Mode::Plan;
    return std::nullopt;
}

static std::optional<OutputFormat> parseFormat(llvm::StringRef s) {
    if (s == "text")
        return OutputFormat::Text;
    if (s == "json")
        return OutputFormat::Json;
    return std::nullopt;
}

static std::optional<VerifyMethod> parseVerifyMethod(llvm::StringRef s) {
    if (s == "compile")
        return VerifyMethod::Compile;
    if (s == "regex")
        return VerifyMethod::Regex;
    return std::nullopt;
}

static stable_abi::VerifyOptions
buildVerifyOptions(const std::string &resourceDir,
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
                     const std::string &cudaInclude, VerifyMethod method,
                     OutputFormat format, bool allow_fallback = false) {
    bool use_compile = (method == VerifyMethod::Compile);
    bool json = (format == OutputFormat::Json);
    auto opts = buildVerifyOptions(resourceDir, pytorchRoot, extraIncludes,
                                   cudaInclude);

    if (use_compile && opts.pytorch_root.empty()) {
        if (allow_fallback) {
            use_compile = false;
            llvm::errs() << "note: --pytorch-root not set, using regex-based "
                            "verification (less precise).\n"
                         << "      Pass --pytorch-root for compile-based "
                            "verification.\n";
        } else {
            llvm::errs() << "error: --pytorch-root required for compile-based "
                            "verification\n"
                         << "       use --verify-method=regex for regex-based "
                            "fallback\n";
            return 1;
        }
    }

    size_t total_violations = 0;
    for (const auto &src : sources) {
        auto violations = use_compile ? stable_abi::verifyStableAbi(src, opts)
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
    for (llvm::sys::fs::recursive_directory_iterator
             it(root, ec, /*follow_symlinks=*/false),
         end;
         it != end && !ec; it.increment(ec)) {
        auto path = it->path();
        auto ext = llvm::sys::path::extension(path);
        if (ext == ".cpp" || ext == ".cu" || ext == ".cuh")
            result.push_back(std::string(path));
    }
    std::sort(result.begin(), result.end());
    return result;
}

static std::vector<std::string>
expandSources(const std::vector<std::string> &entries) {
    std::set<std::string> seen;
    std::vector<std::string> result;
    for (const auto &entry : entries) {
        if (llvm::sys::fs::is_directory(entry)) {
            for (auto &f : discoverSources(entry)) {
                if (seen.insert(f).second)
                    result.push_back(std::move(f));
            }
        } else {
            if (seen.insert(entry).second)
                result.push_back(entry);
        }
    }
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
                         << "'. Must be one of: audit, rewrite, verify, plan\n";
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
        cfg.extra_includes = std::vector<std::string>(ExtraIncludes.begin(),
                                                      ExtraIncludes.end());
    if (OutputDir.getNumOccurrences() > 0)
        cfg.output_dir = OutputDir.getValue();
    if (Jobs.getNumOccurrences() > 0)
        cfg.jobs = Jobs.getValue();
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

static int
runWithConfig(stable_abi::Config &cfg,
              clang::tooling::CompilationDatabase *externalDB = nullptr) {
    auto resourceDir = detectResourceDir();

    bool json = (cfg.format == OutputFormat::Json);

    std::string projectRoot = cfg.project_root;
    if (!projectRoot.empty()) {
        llvm::SmallString<256> abs(projectRoot);
        llvm::sys::fs::make_absolute(abs);
        projectRoot = std::string(abs);
    }

    std::vector<std::string> sources = expandSources(cfg.transform);
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

    if (cfg.mode == Mode::Plan && projectRoot.empty()) {
        llvm::errs() << "error: --project-root required for plan mode\n";
        return 1;
    }

    auto writeMode = (cfg.mode == Mode::Rewrite)
                         ? (DryRun.getValue() ? stable_abi::WriteMode::DryRun
                                              : stable_abi::WriteMode::Rewrite)
                         : stable_abi::WriteMode::Audit;

    std::unique_ptr<clang::tooling::CompilationDatabase> ownedDB;
    clang::tooling::CompilationDatabase *db = externalDB;
    if (!db) {
        if (!cfg.compile_commands_dir.empty()) {
            std::string dbError;
            ownedDB = clang::tooling::CompilationDatabase::loadFromDirectory(
                cfg.compile_commands_dir, dbError);
            if (!ownedDB) {
                llvm::errs()
                    << "error: failed to load compile_commands.json "
                       "from "
                    << cfg.compile_commands_dir << ": " << dbError << "\n";
                return 1;
            }
            llvm::errs() << "note: loaded " << ownedDB->getAllFiles().size()
                         << " entries from " << cfg.compile_commands_dir
                         << "/compile_commands.json\n";
        } else if (!projectRoot.empty()) {
            std::string dbError;
            ownedDB =
                clang::tooling::CompilationDatabase::autoDetectFromDirectory(
                    projectRoot, dbError);
        }
        if (!ownedDB) {
            std::vector<std::string> clangArgs;
            for (const auto &flag : cfg.compiler_flags)
                clangArgs.push_back(flag);
            for (const auto &inc : cfg.include_paths)
                clangArgs.push_back("-I" + inc);
            ownedDB =
                std::make_unique<clang::tooling::FixedCompilationDatabase>(
                    ".", clangArgs);
        }
        db = ownedDB.get();
    }

    if (!cfg.compile_commands_dir.empty()) {
        auto dbFiles = db->getAllFiles();
        std::set<std::string> known(dbFiles.begin(), dbFiles.end());
        std::erase_if(sources, [&](const std::string &src) {
            llvm::SmallString<256> real;
            if (llvm::sys::fs::real_path(src, real))
                return true;
            return !known.count(std::string(real));
        });
        llvm::errs() << "note: processing " << sources.size()
                     << " files with database entries\n";
    }

    std::string outputDir = cfg.output_dir;
    if (!outputDir.empty()) {
        if (projectRoot.empty()) {
            llvm::errs() << "error: --output-dir requires --project-root "
                            "(needed to compute relative paths)\n";
            return 1;
        }
        llvm::SmallString<256> abs(outputDir);
        llvm::sys::fs::make_absolute(abs);
        outputDir = std::string(abs);
    }

    auto pytorchIncs = stable_abi::pytorchIncludePaths(cfg.pytorch_root);
    std::string cudaPath;
    if (!cfg.cuda_include.empty()) {
        llvm::SmallString<256> cp(cfg.cuda_include);
        llvm::sys::path::remove_filename(cp);
        cudaPath = std::string(cp);
    }
    bool hasArchFlag = false;
    for (const auto &f : cfg.compiler_flags) {
        if (llvm::StringRef(f).starts_with("-march") ||
            llvm::StringRef(f).starts_with("-mavx") ||
            llvm::StringRef(f).starts_with("-msse"))
            hasArchFlag = true;
    }

    auto stripAdjuster = stable_abi::makeStripNonClangAdjuster();
    auto toolAdjuster =
        stable_abi::makeToolAdjuster(resourceDir, cudaPath, hasArchFlag);
    clang::tooling::CommandLineArguments pytorchFlags;
    for (const auto &inc : pytorchIncs)
        pytorchFlags.push_back("-I" + inc);
    auto pytorchAdjuster = clang::tooling::getInsertArgumentAdjuster(
        pytorchFlags, clang::tooling::ArgumentInsertPosition::BEGIN);

    stable_abi::IncludeGraph includeGraph;
    std::mutex includeGraphMutex;
    std::mutex writeMutex;

    unsigned jobs = cfg.jobs;
    if (jobs == 0)
        jobs = llvm::hardware_concurrency().compute_thread_count();
    bool generates_edits = (writeMode != stable_abi::WriteMode::Audit);
    bool parallel = jobs > 1 && sources.size() > 1 && !generates_edits;
    if (jobs > 1 && generates_edits) {
        llvm::errs() << "note: --jobs ignored in rewrite mode "
                        "(parallel rewrite is not yet safe)\n";
    }

    std::set<std::string> incompleteFiles;
    std::mutex incompleteFilesMutex;

    stable_abi::ActionOptions actionOpts{
        .write_mode = writeMode,
        .project_root = projectRoot,
        .output_dir = outputDir,
        .include_graph = (cfg.mode == Mode::Plan) ? &includeGraph : nullptr,
        .include_graph_mutex = parallel ? &includeGraphMutex : nullptr,
        .write_mutex = parallel ? &writeMutex : nullptr,
        .incomplete_files = &incompleteFiles,
        .incomplete_files_mutex = &incompleteFilesMutex,
    };
    auto Factory =
        std::make_unique<stable_abi::StableAbiActionFactory>(actionOpts);

    int result;
    if (parallel) {
        llvm::DefaultThreadPool pool(llvm::hardware_concurrency(jobs));
        std::atomic<int> toolResult{0};

        llvm::errs() << "note: processing " << sources.size() << " files with "
                     << jobs << " threads\n";
        auto t0 = std::chrono::steady_clock::now();

        for (const auto &source : sources) {
            pool.async([&, source]() {
                clang::tooling::ClangTool tool(*db, {source});
                tool.appendArgumentsAdjuster(stripAdjuster);
                tool.appendArgumentsAdjuster(toolAdjuster);
                tool.appendArgumentsAdjuster(pytorchAdjuster);
                stable_abi::ParseDiagConsumer diag(Factory->getReporter(),
                                                   &incompleteFiles,
                                                   &incompleteFilesMutex);
                diag.setMainFile(source);
                tool.setDiagnosticConsumer(&diag);
                int r = tool.run(Factory.get());
                if (r != 0)
                    toolResult.store(1, std::memory_order_relaxed);
            });
        }
        pool.wait();
        auto elapsed = std::chrono::steady_clock::now() - t0;
        auto secs = std::chrono::duration<double>(elapsed).count();
        llvm::errs() << llvm::format("note: processed %zu files in %.1fs\n",
                                     sources.size(), secs);
        result = toolResult.load();
    } else {
        clang::tooling::ClangTool Tool(*db, sources);
        Tool.appendArgumentsAdjuster(stripAdjuster);
        Tool.appendArgumentsAdjuster(toolAdjuster);
        Tool.appendArgumentsAdjuster(pytorchAdjuster);
        stable_abi::ParseDiagConsumer diagConsumer(
            Factory->getReporter(), &incompleteFiles, &incompleteFilesMutex);
        if (sources.size() == 1)
            diagConsumer.setMainFile(sources[0]);
        Tool.setDiagnosticConsumer(&diagConsumer);
        result = Tool.run(Factory.get());
    }

    auto &reporter = Factory->getReporter();

    if (cfg.mode == Mode::Plan) {
        stable_abi::DepGraph graph;
        graph.build(includeGraph, reporter.findingsByFile());
        auto plan = graph.computePlan();
        stable_abi::printMigrationPlan(plan, json);
        return 0;
    }

    if (writeMode == stable_abi::WriteMode::Audit)
        reporter.runTextScanComplement(sources);

    reporter.sortFindings();
    reporter.suppressRedundantFlags();
    if (json) {
        reporter.printJson();
    } else {
        reporter.printReport(projectRoot);
        reporter.printSummary();
        reporter.printFileReport(projectRoot, incompleteFiles);
    }
    reporter.printParseWarnings();

    if (writeMode == stable_abi::WriteMode::Rewrite && result == 0) {
        if (!json)
            llvm::outs() << "\n--- Post-rewrite ABI verification ---\n";
        std::vector<std::string> verifySources = sources;
        if (!outputDir.empty()) {
            verifySources.clear();
            for (const auto &src : sources) {
                llvm::SmallString<256> rel(src);
                llvm::sys::path::replace_path_prefix(rel, projectRoot, "");
                llvm::SmallString<256> out(outputDir);
                llvm::sys::path::append(out, rel);
                if (llvm::sys::fs::exists(out))
                    verifySources.push_back(std::string(out));
            }
        }
        int verify_result = runVerify(
            verifySources, resourceDir, cfg.pytorch_root, cfg.extra_includes,
            cfg.cuda_include, cfg.verify_method, cfg.format, true);
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

    auto ExpectedParser = clang::tooling::CommonOptionsParser::create(
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
        std::string resolveErr;
        if (!stable_abi::resolvePytorchRoot(cfg, resolveErr)) {
            llvm::errs() << "error: " << resolveErr << "\n";
            return 1;
        }
        auto &cliSources = OptionsParser.getSourcePathList();
        if (!cliSources.empty())
            cfg.transform = cliSources;
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
    std::string resolveErr;
    if (!stable_abi::resolvePytorchRoot(cfg, resolveErr)) {
        llvm::errs() << "error: " << resolveErr << "\n";
        return 1;
    }
    cfg.transform = cliSources;
    return runWithConfig(cfg, &OptionsParser.getCompilations());
}
