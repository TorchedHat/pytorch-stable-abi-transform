#pragma once

#include <llvm/Support/YAMLTraits.h>
#include <string>
#include <vector>

namespace stable_abi {

enum class Mode { Audit, Rewrite, Verify };
enum class OutputFormat { Text, Json };
enum class VerifyMethod { Compile, Regex };

struct Config {
    Mode mode = Mode::Audit;
    OutputFormat format = OutputFormat::Text;
    std::string pytorch_root;
    std::string project_root;
    std::vector<std::string> compiler_flags;
    std::vector<std::string> include_paths;
    std::vector<std::string> extra_includes;
    std::vector<std::string> sources;
    VerifyMethod verify_method = VerifyMethod::Compile;
    std::string cuda_include;
};

[[nodiscard]] bool loadConfig(const std::string &path, Config &out, std::string &error);
void printExampleConfig();

} // namespace stable_abi

namespace llvm {
namespace yaml {

template <> struct ScalarEnumerationTraits<stable_abi::Mode> {
    static void enumeration(IO &io, stable_abi::Mode &val) {
        io.enumCase(val, "audit", stable_abi::Mode::Audit);
        io.enumCase(val, "rewrite", stable_abi::Mode::Rewrite);
        io.enumCase(val, "verify", stable_abi::Mode::Verify);
    }
};

template <> struct ScalarEnumerationTraits<stable_abi::OutputFormat> {
    static void enumeration(IO &io, stable_abi::OutputFormat &val) {
        io.enumCase(val, "text", stable_abi::OutputFormat::Text);
        io.enumCase(val, "json", stable_abi::OutputFormat::Json);
    }
};

template <> struct ScalarEnumerationTraits<stable_abi::VerifyMethod> {
    static void enumeration(IO &io, stable_abi::VerifyMethod &val) {
        io.enumCase(val, "compile", stable_abi::VerifyMethod::Compile);
        io.enumCase(val, "regex", stable_abi::VerifyMethod::Regex);
    }
};

template <> struct MappingTraits<stable_abi::Config> {
    static void mapping(IO &io, stable_abi::Config &c) {
        io.mapOptional("mode", c.mode);
        io.mapOptional("format", c.format);
        io.mapOptional("pytorch_root", c.pytorch_root);
        io.mapOptional("project_root", c.project_root);
        io.mapOptional("compiler_flags", c.compiler_flags);
        io.mapOptional("include_paths", c.include_paths);
        io.mapOptional("extra_includes", c.extra_includes);
        io.mapOptional("sources", c.sources);
        io.mapOptional("verify_method", c.verify_method);
        io.mapOptional("cuda_include", c.cuda_include);
    }
};

} // namespace yaml
} // namespace llvm
