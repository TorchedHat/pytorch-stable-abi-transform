#include "TransformerRules.h"
#include "Helpers.h"
#include "Rules.h"
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Lex/Lexer.h>
#include <clang/Tooling/Transformer/RangeSelector.h>
#include <clang/Tooling/Transformer/RewriteRule.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Regex.h>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::transformer;

namespace stable_abi {

struct LocFilter {
    ast_matchers::internal::Matcher<Stmt> stmt;
    ast_matchers::internal::Matcher<TypeLoc> typeLoc;
    ast_matchers::internal::Matcher<Decl> decl;
    std::string projectRoot;
};

static LocFilter buildLocFilter(const std::string &projectRoot) {
    if (projectRoot.empty())
        return {isExpansionInMainFile(), isExpansionInMainFile(),
                isExpansionInMainFile(), ""};
    auto regex = "^" + llvm::Regex::escape(projectRoot);
    return {isExpansionInFileMatching(regex),
            isExpansionInFileMatching(regex),
            isExpansionInFileMatching(regex),
            projectRoot};
}

static bool shouldUseConstDataPtr(const CXXMemberCallExpr *CE,
                                  ASTContext &Ctx) {
    auto parents = Ctx.getParents(*CE);
    while (!parents.empty()) {
        const auto &parent = parents[0];

        if (const auto *cast = parent.get<CastExpr>()) {
            const auto resultType = cast->getType();
            if (resultType->isPointerType())
                return resultType->getPointeeType().isConstQualified();
        }

        if (const auto *call = parent.get<CallExpr>()) {
            if (const auto *FD = call->getDirectCallee()) {
                for (unsigned i = 0; i < call->getNumArgs(); ++i) {
                    if (call->getArg(i)->IgnoreParenImpCasts() == CE) {
                        if (i < FD->getNumParams()) {
                            auto paramType = FD->getParamDecl(i)->getType();
                            if (paramType->isPointerType())
                                return paramType->getPointeeType()
                                    .isConstQualified();
                        }
                        break;
                    }
                }
            }
            break;
        }

        if (const auto *VD = parent.get<VarDecl>()) {
            const auto varType = VD->getType();
            if (varType->isPointerType())
                return varType->getPointeeType().isConstQualified();
            if (varType->getContainedAutoType())
                break;
            return false;
        }

        if (const auto *RCE = parent.get<CXXReinterpretCastExpr>()) {
            const auto resultType = RCE->getType();
            if (resultType->isPointerType())
                return resultType->getPointeeType().isConstQualified();
            break;
        }
        if (const auto *SCE = parent.get<CXXStaticCastExpr>()) {
            const auto resultType = SCE->getType();
            if (resultType->isPointerType())
                return resultType->getPointeeType().isConstQualified();
            break;
        }

        parents = Ctx.getParents(parent);
    }

    const auto *objArg = CE->getImplicitObjectArgument();
    if (objArg) {
        const auto objType = objArg->getType();
        if (objType.isConstQualified() ||
            (objType->isReferenceType() &&
             objType.getNonReferenceType().isConstQualified()))
            return true;
    }
    return false;
}

static llvm::Expected<SmallVector<Edit, 1>> noEdits() {
    return SmallVector<Edit, 1>{};
}

static ast_matchers::internal::Matcher<NamedDecl>
hasAnyNameFromVec(std::vector<std::string> names) {
    return ast_matchers::internal::Matcher<NamedDecl>(
        new ast_matchers::internal::HasNameMatcher(std::move(names)));
}

enum class MacroPolicy { RejectAll, AllowArgs };

static std::optional<SourceLocation> getRewritableLoc(
    SourceLocation loc, const SourceManager &SM,
    llvm::StringRef projectRoot = "",
    MacroPolicy policy = MacroPolicy::RejectAll) {
    auto spelling = SM.getSpellingLoc(loc);
    if (!isInProjectScope(SM, spelling, projectRoot))
        return std::nullopt;
    if (SM.isMacroBodyExpansion(loc))
        return std::nullopt;
    if (policy == MacroPolicy::RejectAll && SM.isMacroArgExpansion(loc))
        return std::nullopt;
    return spelling;
}

static llvm::Expected<SmallVector<Edit, 1>>
singleEdit(SourceLocation start, unsigned len, std::string replacement) {
    Edit E;
    E.Kind = EditKind::Range;
    E.Range = CharSourceRange::getCharRange(
        start, start.getLocWithOffset(static_cast<int>(len)));
    E.Replacement = std::move(replacement);
    return SmallVector<Edit, 1>{std::move(E)};
}

static llvm::Expected<SmallVector<Edit, 1>>
singleEdit(SourceRange range, std::string replacement) {
    Edit E;
    E.Kind = EditKind::Range;
    E.Range = CharSourceRange::getTokenRange(range);
    E.Replacement = std::move(replacement);
    return SmallVector<Edit, 1>{std::move(E)};
}

// ---------------------------------------------------------------------------
// Type rewrite rules (from kTypeRules)
// ---------------------------------------------------------------------------

static bool isEnumQualifierType(std::string_view typeName) {
    return typeName == "at::ScalarType" || typeName == "c10::ScalarType" ||
           typeName == "at::DeviceType" || typeName == "c10::DeviceType";
}

// Returns the bracket depth at `pos` in `text`, counting < and >.
// A match inside template arguments (depth > 0) belongs to an inner
// TypeLoc and should not be rewritten on the outer one.
static unsigned templateDepthAt(std::string_view text, size_t pos) {
    unsigned depth = 0;
    for (size_t i = 0; i < pos; ++i) {
        if (text[i] == '<') ++depth;
        if (text[i] == '>' && depth > 0) --depth;
    }
    return depth;
}

// Returns the match position if `rule` matches `text` at template depth 0
// with valid word boundaries, or std::string::npos otherwise.
static size_t findTypeRuleMatch(std::string_view text, const TypeRule &rule) {
    auto pos = text.find(rule.from);
    if (pos == std::string::npos)
        return std::string::npos;
    if (templateDepthAt(text, pos) > 0)
        return std::string::npos;
    auto endPos = pos + rule.from.size();
    if (endPos < text.size()) {
        char next = text[endPos];
        if (next == ':' && isEnumQualifierType(rule.from))
            return std::string::npos;
        if (std::isalnum(static_cast<unsigned char>(next)) || next == '_')
            return std::string::npos;
    }
    return pos;
}

static void addTypeRules(std::vector<RewriteRule> &rules, Reporter &rep,
                         bool rewrite, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *TL = R.Nodes.getNodeAs<TypeLoc>("tl");
        if (!TL)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(TL->getBeginLoc(), *R.SourceManager,
                                          projectRoot);
        if (!rewriteLoc)
            return noEdits();

        const auto &SM = *R.SourceManager;
        auto text = getSourceText(TL->getSourceRange(), SM,
                                  R.Context->getLangOpts());

        for (const auto &rule : kTypeRules) {
            auto pos = findTypeRuleMatch(text, rule);
            if (pos == std::string::npos)
                continue;

            const bool flag_only = rule.to.empty();
            std::string_view suggestion = flag_only
                ? "decompose into explicit scalar_type, layout, device args"
                : rule.to;

            rep.addFinding(FindingKind::Type, SM, TL->getBeginLoc(),
                           rule.from, suggestion,
                           flag_only ? FindingAction::Flag
                                     : FindingAction::Rewrite);
            if (!rewrite || flag_only)
                return noEdits();

            auto replaceStart = rewriteLoc->getLocWithOffset(
                static_cast<int>(pos));
            return singleEdit(replaceStart,
                              static_cast<unsigned>(rule.from.size()),
                              std::string(suggestion));
        }

        return noEdits();
    };

    rules.push_back(makeRule(
        typeLoc(loc.typeLoc,
                ast_matchers::loc(qualType(hasDeclaration(namedDecl(hasAnyName(
            "at::Tensor", "torch::Tensor",
            "c10::Device", "at::Device", "torch::Device",
            "at::ScalarType", "c10::ScalarType", "torch::Dtype",
            "at::DeviceType", "c10::DeviceType",
            "at::Layout", "c10::Layout",
            "at::MemoryFormat", "c10::MemoryFormat",
            "at::Half", "c10::Half",
            "at::BFloat16", "c10::BFloat16",
            "c10::Float8_e4m3fn", "c10::Float8_e4m3fnuz",
            "c10::Float8_e5m2", "c10::Float8_e5m2fnuz",
            "c10::CppTypeToScalarType",
            "c10::optional", "std::optional",
            "c10::ArrayRef", "c10::IntArrayRef", "at::IntArrayRef",
            "c10::string_view", "std::string_view",
            "torch::TensorOptions", "at::TensorOptions", "c10::TensorOptions"
        ))))))
            .bind("tl"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// Enum shorthand rules (shared by ScalarType and DeviceType)
// ---------------------------------------------------------------------------

// Macro-expanded arguments can produce multiple AST matches at the same source
// offset. shared_ptr is needed because std::function (inside EditGenerator)
// requires a copyable lambda.
template <MappingRuleRange ShorthandTable>
static llvm::Expected<SmallVector<Edit, 1>> rewriteEnumRef(
    const MatchFinder::MatchResult &R,
    Reporter &rep, bool rewrite,
    llvm::DenseSet<unsigned> &dedup,
    llvm::StringRef bindName,
    const ShorthandTable &shorthands,
    std::string_view typePrefix1, std::string_view typePrefix2,
    std::string_view macroBodyDesc,
    llvm::StringRef projectRoot = "") {

    const auto *DRE = R.Nodes.getNodeAs<DeclRefExpr>(bindName);
    if (!DRE)
        return noEdits();

    SourceLocation loc = DRE->getBeginLoc();
    const auto &SM = *R.SourceManager;
    const auto &LO = R.Context->getLangOpts();
    SourceLocation spellingLoc = SM.getSpellingLoc(loc);

    if (!isInProjectScope(SM, spellingLoc, projectRoot))
        return noEdits();
    if (!dedup.insert(SM.getFileOffset(spellingLoc)).second)
        return noEdits();
    if (SM.isMacroBodyExpansion(loc)) {
        rep.addFinding(FindingKind::ScalarType, SM, spellingLoc,
                       macroBodyDesc, "(inside macro body)",
                       FindingAction::Flag);
        return noEdits();
    }

    const bool in_macro_arg = SM.isMacroArgExpansion(loc);
    SourceRange textRange = in_macro_arg
        ? SourceRange(SM.getSpellingLoc(DRE->getBeginLoc()),
                       SM.getSpellingLoc(DRE->getEndLoc()))
        : DRE->getSourceRange();
    std::string text = getSourceText(textRange, SM, LO);
    SourceLocation rewriteLoc = in_macro_arg ? spellingLoc : loc;

    for (const auto &rule : shorthands) {
        if (text == rule.from) {
            rep.addFinding(FindingKind::ScalarType, SM, spellingLoc,
                           rule.from, rule.to);
            if (!rewrite)
                return noEdits();
            return singleEdit(rewriteLoc,
                              static_cast<unsigned>(rule.from.size()),
                              std::string(rule.to));
        }
    }

    for (const auto &rule : kTypeRules) {
        if (rule.from != typePrefix1 && rule.from != typePrefix2)
            continue;
        if (text.starts_with(rule.from)) {
            std::string newText = std::string(rule.to) +
                                  text.substr(rule.from.size());
            rep.addFinding(FindingKind::ScalarType, SM, spellingLoc,
                           text, newText);
            if (!rewrite)
                return noEdits();
            return singleEdit(rewriteLoc,
                              static_cast<unsigned>(text.size()),
                              newText);
        }
    }

    return noEdits();
}

template <MappingRuleRange ShorthandArray>
static void addEnumShorthandRules(
    std::vector<RewriteRule> &rules, Reporter &rep, bool rewrite,
    const LocFilter &loc,
    StatementMatcher shorthandMatcher, StatementMatcher enumMatcher,
    llvm::StringRef bindName, const ShorthandArray &shorthands,
    std::string_view typePrefix1, std::string_view typePrefix2,
    std::string_view macroBodyDesc) {

    auto dedup = std::make_shared<llvm::DenseSet<unsigned>>();
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, dedup, projectRoot, bindName,
                    &shorthands, typePrefix1, typePrefix2,
                    macroBodyDesc](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        return rewriteEnumRef(R, rep, rewrite, *dedup, bindName,
                              shorthands, typePrefix1, typePrefix2,
                              macroBodyDesc, projectRoot);
    };

    rules.push_back(makeRule(shorthandMatcher, EditGenerator(editGen)));
    rules.push_back(makeRule(enumMatcher, EditGenerator(editGen)));
}

static void addScalarTypeRules(std::vector<RewriteRule> &rules, Reporter &rep,
                               bool rewrite, const LocFilter &loc) {
    addEnumShorthandRules(rules, rep, rewrite, loc,
        declRefExpr(loc.stmt,
            to(namedDecl(hasAnyName(
            "kFloat", "kFloat16", "kFloat32", "kFloat64",
            "kDouble", "kHalf", "kBFloat16", "kBool",
            "kByte", "kChar", "kShort", "kInt", "kInt8",
            "kInt16", "kInt32", "kInt64", "kLong",
            "kFloat8_e4m3fn", "kFloat8_e5m2", "kFloat8_e4m3fnuz",
            "kFloat8_e5m2fnuz", "kFloat8_e8m0fnu",
            "kFloat4_e2m1fn_x2",
            "kComplexHalf", "kComplexFloat",
            "kComplexDouble", "kUInt8"))))
            .bind("scalarRef"),
        declRefExpr(loc.stmt,
            to(enumConstantDecl(hasDeclContext(enumDecl(hasName("ScalarType"))))))
            .bind("scalarRef"),
        "scalarRef", kScalarTypeShorthands,
        "at::ScalarType", "c10::ScalarType", "scalar type shorthand");
}

static void addDeviceTypeRules(std::vector<RewriteRule> &rules, Reporter &rep,
                               bool rewrite, const LocFilter &loc) {
    addEnumShorthandRules(rules, rep, rewrite, loc,
        declRefExpr(loc.stmt,
            to(namedDecl(hasAnyName(
            "kCPU", "kCUDA", "kMKLDNN", "kOPENGL", "kOPENCL",
            "kIDEEP", "kHIP", "kFPGA", "kMAIA", "kXLA",
            "kVulkan", "kMetal", "kXPU", "kMPS", "kMeta",
            "kHPU", "kVE", "kLazy", "kIPU", "kMTIA",
            "kPrivateUse1"))))
            .bind("deviceRef"),
        declRefExpr(loc.stmt,
            to(enumConstantDecl(hasDeclContext(enumDecl(hasName("DeviceType"))))))
            .bind("deviceRef"),
        "deviceRef", kDeviceTypeShorthands,
        "at::DeviceType", "c10::DeviceType", "device type");
}

// ---------------------------------------------------------------------------
// data_ptr → mutable_data_ptr / const_data_ptr
// ---------------------------------------------------------------------------

static void addDataPtrRule(std::vector<RewriteRule> &rules, Reporter &rep,
                           bool rewrite, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CXXMemberCallExpr>("dataPtrCall");
        if (!CE || !isTensorType(CE->getImplicitObjectArgument()))
            return noEdits();

        const auto &SM = *R.SourceManager;
        auto spellingLoc = getRewritableLoc(CE->getBeginLoc(), SM,
                                            projectRoot, MacroPolicy::AllowArgs);
        if (!spellingLoc)
            return noEdits();

        const auto *ME = R.Nodes.getNodeAs<MemberExpr>("dataPtrMember");
        if (!ME)
            return noEdits();

        if (ME->hasExplicitTemplateArgs()) {
            const auto tArgs = ME->template_arguments();
            if (!tArgs.empty()) {
                const auto argType = tArgs[0].getArgument().getAsType();
                if (argType->isDependentType()) {
                    rep.addFinding(FindingKind::DataPtr, SM, *spellingLoc,
                                   "data_ptr<dependent>",
                                   "mutable_data_ptr<T> (dependent type)",
                                   FindingAction::Flag);
                    return noEdits();
                }
            }
        }

        const bool use_const = shouldUseConstDataPtr(CE, *R.Context);
        const std::string replacement =
            use_const ? "const_data_ptr" : "mutable_data_ptr";

        auto loc = CE->getBeginLoc();
        const bool in_macro = SM.isMacroArgExpansion(loc);
        auto memberLoc = in_macro ? SM.getSpellingLoc(ME->getMemberLoc())
                                  : ME->getMemberLoc();

        rep.addFinding(FindingKind::DataPtr, SM, *spellingLoc, "data_ptr",
                       replacement);
        if (!rewrite)
            return noEdits();

        return singleEdit(memberLoc,
                          static_cast<unsigned>(std::string_view("data_ptr").size()),
                          replacement);
    };

    rules.push_back(makeRule(
        cxxMemberCallExpr(loc.stmt,
            callee(memberExpr(member(hasName("data_ptr"))).bind("dataPtrMember")))
            .bind("dataPtrCall"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// Method-to-free-function rules (from kMethodToFreeFuncRules)
// ---------------------------------------------------------------------------

static void addMethodToFuncRules(std::vector<RewriteRule> &rules,
                                 Reporter &rep, bool rewrite,
                                 const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CXXMemberCallExpr>("methodCall");
        if (!CE)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(CE->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        const auto &SM = *R.SourceManager;
        const auto &LO = R.Context->getLangOpts();

        const auto *ME = R.Nodes.getNodeAs<MemberExpr>("methodMember");
        if (!ME)
            return noEdits();
        auto methodName = ME->getMemberDecl()->getNameAsString();

        const auto *obj = CE->getImplicitObjectArgument();
        if (!isTensorType(obj))
            return noEdits();

        for (const auto &rule : kMethodToFreeFuncRules) {
            if (methodName != rule.from)
                continue;

            auto objText = getSourceText(obj->getSourceRange(), SM, LO);
            std::string argsText;
            auto argsRange = callArgs("methodCall")(R);
            if (argsRange)
                argsText = Lexer::getSourceText(*argsRange, SM, LO).str();
            else
                llvm::consumeError(argsRange.takeError());

            std::string replacement =
                std::string(rule.to) + "(" + objText;
            if (!argsText.empty())
                replacement += ", " + argsText;
            replacement += ")";

            auto fullText = getSourceText(CE->getSourceRange(), SM, LO);
            rep.addFinding(FindingKind::MethodToFunc, SM, CE->getBeginLoc(),
                           fullText, replacement);
            if (!rewrite)
                return noEdits();

            return singleEdit(CE->getSourceRange(), replacement);
        }
        return noEdits();
    };

    std::vector<std::string> methodNames;
    for (const auto &rule : kMethodToFreeFuncRules)
        methodNames.push_back(std::string(rule.from));

    rules.push_back(makeRule(
        cxxMemberCallExpr(loc.stmt,
            callee(memberExpr(member(cxxMethodDecl(
                hasAnyNameFromVec(std::move(methodNames)))))
                .bind("methodMember")))
            .bind("methodCall"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// Method rename rules (dtype→scalar_type, sizes()[i]→size(i))
// ---------------------------------------------------------------------------

static void addMethodRenameRules(std::vector<RewriteRule> &rules,
                                 Reporter &rep, bool rewrite,
                                 const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;

    // TensorOptions constructor — flag only
    auto tensorOptsGen = [&rep, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *Ctor =
            R.Nodes.getNodeAs<CXXConstructExpr>("tensorOptionsConstruct");
        if (!Ctor)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(Ctor->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        auto text = getSourceText(Ctor->getSourceRange(), *R.SourceManager,
                                  R.Context->getLangOpts());
        rep.addFinding(FindingKind::Type, *R.SourceManager, Ctor->getBeginLoc(),
                       text,
                       "decompose into explicit scalar_type, layout, device args",
                       FindingAction::Flag);
        return noEdits();
    };

    rules.push_back(makeRule(
        cxxConstructExpr(loc.stmt,
            hasType(hasUnqualifiedDesugaredType(recordType(
            hasDeclaration(cxxRecordDecl(hasName("TensorOptions")))))))
            .bind("tensorOptionsConstruct"),
        EditGenerator(tensorOptsGen)));

    // .sizes()[i] → .size(i), .strides()[i] → .stride(i)
    auto rewriteSizes = [&rep, rewrite](
            const MatchFinder::MatchResult &R,
            const CXXMemberCallExpr *sizesCall,
            const Expr *idx, SourceRange fullRange)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto &SM = *R.SourceManager;
        const auto &LO = R.Context->getLangOpts();

        const auto *calleeME = dyn_cast<MemberExpr>(sizesCall->getCallee());
        if (!calleeME)
            return noEdits();

        auto methodName = calleeME->getMemberDecl()->getName();
        const char *newMethod;
        if (methodName == "sizes")
            newMethod = "size";
        else if (methodName == "strides")
            newMethod = "stride";
        else
            return noEdits();

        const auto *obj = sizesCall->getImplicitObjectArgument();
        if (!obj)
            return noEdits();

        auto objText = getSourceText(obj->getSourceRange(), SM, LO);
        auto idxText = getSourceText(idx->getSourceRange(), SM, LO);
        auto fullText = getSourceText(fullRange, SM, LO);

        std::string replacement =
            objText + "." + newMethod + "(" + idxText + ")";
        rep.addFinding(FindingKind::MethodToFunc, SM, fullRange.getBegin(),
                       fullText, replacement);
        if (!rewrite)
            return noEdits();

        return singleEdit(fullRange, replacement);
    };

    auto sizesGen = [rewriteSizes, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *ASE =
            R.Nodes.getNodeAs<ArraySubscriptExpr>("sizesSubscript");
        if (!ASE)
            return noEdits();
        if (!getRewritableLoc(ASE->getBeginLoc(), *R.SourceManager, projectRoot))
            return noEdits();
        const auto *sizesCall =
            R.Nodes.getNodeAs<CXXMemberCallExpr>("sizesCall");
        if (!sizesCall)
            return noEdits();
        return rewriteSizes(R, sizesCall, ASE->getIdx(),
                            ASE->getSourceRange());
    };

    rules.push_back(makeRule(
        arraySubscriptExpr(loc.stmt,
            hasBase(hasDescendant(
            cxxMemberCallExpr(callee(cxxMethodDecl(
                hasAnyName("sizes", "strides"))))
                .bind("sizesCall"))))
            .bind("sizesSubscript"),
        EditGenerator(sizesGen)));

    // CXXOperatorCallExpr version (overloaded operator[])
    auto sizesOpGen = [rewriteSizes, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *Sub =
            R.Nodes.getNodeAs<CXXOperatorCallExpr>("sizesSubscriptOp");
        if (!Sub || Sub->getNumArgs() < 2)
            return noEdits();
        if (!getRewritableLoc(Sub->getBeginLoc(), *R.SourceManager, projectRoot))
            return noEdits();
        const auto *sizesCall = dyn_cast<CXXMemberCallExpr>(
            Sub->getArg(0)->IgnoreImplicit());
        if (!sizesCall)
            return noEdits();
        return rewriteSizes(R, sizesCall, Sub->getArg(1),
                            Sub->getSourceRange());
    };

    rules.push_back(makeRule(
        cxxOperatorCallExpr(loc.stmt,
            hasOverloadedOperatorName("[]"),
            hasArgument(0, ignoringImplicit(cxxMemberCallExpr(
                callee(cxxMethodDecl(hasAnyName("sizes", "strides")))))))
            .bind("sizesSubscriptOp"),
        EditGenerator(sizesOpGen)));

    // .dtype() → .scalar_type()
    auto dtypeGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CXXMemberCallExpr>("methodRename");
        if (!CE || !isTensorType(CE->getImplicitObjectArgument()))
            return noEdits();

        auto exprLoc = CE->getBeginLoc();
        const auto &SM = *R.SourceManager;
        auto spellingLoc = SM.getSpellingLoc(exprLoc);
        if (!isInProjectScope(SM, spellingLoc, projectRoot))
            return noEdits();
        if (SM.isMacroBodyExpansion(exprLoc)) {
            rep.addFinding(FindingKind::MethodToFunc, SM, spellingLoc,
                           "dtype", "scalar_type (inside macro body)",
                           FindingAction::Flag);
            return noEdits();
        }
        const bool in_macro_arg = SM.isMacroArgExpansion(exprLoc);

        const auto *ME = R.Nodes.getNodeAs<MemberExpr>("dtypeMember");
        if (!ME)
            return noEdits();

        auto methodName = ME->getMemberDecl()->getNameAsString();
        for (const auto &rule : kMethodRenameRules) {
            if (methodName != rule.from)
                continue;

            auto memberLoc = in_macro_arg
                ? SM.getSpellingLoc(ME->getMemberLoc())
                : ME->getMemberLoc();
            rep.addFinding(FindingKind::MethodToFunc, SM, memberLoc,
                           rule.from, rule.to);
            if (!rewrite)
                return noEdits();

            return singleEdit(memberLoc,
                              static_cast<unsigned>(rule.from.size()),
                              std::string(rule.to));
        }
        return noEdits();
    };

    std::vector<std::string> renameNames;
    for (const auto &rule : kMethodRenameRules)
        renameNames.push_back(std::string(rule.from));

    rules.push_back(makeRule(
        cxxMemberCallExpr(loc.stmt,
            callee(memberExpr(member(hasAnyNameFromVec(std::move(renameNames))))
                .bind("dtypeMember")))
            .bind("methodRename"),
        EditGenerator(dtypeGen)));
}

// ---------------------------------------------------------------------------
// Free function rules (from kFreeFuncRules)
// ---------------------------------------------------------------------------

static void addElementSizeRules(std::vector<RewriteRule> &rules, Reporter &rep,
                                bool rewrite, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    // at::elementSize(tensor.scalar_type/dtype()) → tensor.element_size()
    auto exactGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CallExpr>("elemSizeExact");
        if (!CE)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(CE->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        const auto &SM = *R.SourceManager;
        const auto &LO = R.Context->getLangOpts();
        const auto *obj = R.Nodes.getNodeAs<Expr>("elemObj");
        if (!obj)
            return noEdits();

        auto objText = getSourceText(obj->getSourceRange(), SM, LO);
        std::string replacement = objText + ".element_size()";
        auto fullText = getSourceText(CE->getSourceRange(), SM, LO);

        rep.addFinding(FindingKind::FreeFunc, SM, CE->getBeginLoc(),
                       fullText, replacement);
        if (!rewrite)
            return noEdits();

        return singleEdit(CE->getSourceRange(), replacement);
    };

    rules.push_back(makeRule(
        callExpr(loc.stmt,
            callee(functionDecl(hasAnyName("at::elementSize", "c10::elementSize"))),
            hasArgument(0, ignoringImplicit(
                cxxMemberCallExpr(
                    callee(cxxMethodDecl(hasAnyName("scalar_type", "dtype"))),
                    on(expr().bind("elemObj"))))))
            .bind("elemSizeExact"),
        EditGenerator(exactGen)));

    // at::elementSize(...) — flag only (can't auto-rewrite)
    auto fallbackGen = [&rep, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CallExpr>("elemSizeFallback");
        if (!CE)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(CE->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        auto text = getSourceText(CE->getSourceRange(), *R.SourceManager,
                                  R.Context->getLangOpts());
        rep.addFinding(FindingKind::FreeFunc, *R.SourceManager,
                       CE->getBeginLoc(), text,
                       "tensor.element_size()", FindingAction::Flag);
        return noEdits();
    };

    rules.push_back(makeRule(
        callExpr(loc.stmt,
            callee(functionDecl(hasAnyName("at::elementSize", "c10::elementSize"))))
            .bind("elemSizeFallback"),
        EditGenerator(fallbackGen)));
}

static void addFreeFuncRules(std::vector<RewriteRule> &rules, Reporter &rep,
                             bool rewrite, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CallExpr>("freeFuncCall");
        if (!CE)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(CE->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        const auto &SM = *R.SourceManager;
        const auto &LO = R.Context->getLangOpts();

        const auto *calleeExpr = CE->getCallee()->IgnoreParenImpCasts();
        const auto *DRE = dyn_cast<DeclRefExpr>(calleeExpr);
        if (!DRE)
            return noEdits();
        const auto *FD = dyn_cast<FunctionDecl>(DRE->getDecl());
        if (!FD)
            return noEdits();

        auto canonicalName = FD->getQualifiedNameAsString();
        auto writtenText = getSourceText(DRE->getSourceRange(), SM, LO);

        for (const auto &rule : kFreeFuncRules) {
            if (canonicalName != rule.from &&
                writtenText != rule.from)
                continue;

            auto oldFunc = rule.from.substr(
                rule.from.rfind(':') + 1);
            auto newFunc = rule.to.substr(
                rule.to.rfind(':') + 1);
            bool name_changes = (oldFunc != newFunc);

            if (name_changes) {
                std::string suggestion = std::string(rule.to) +
                    " (function name changes — adjust arguments manually)";
                rep.addFinding(FindingKind::FreeFunc, SM, CE->getBeginLoc(),
                               writtenText, suggestion,
                               FindingAction::Flag);
                return noEdits();
            }

            rep.addFinding(FindingKind::FreeFunc, SM, CE->getBeginLoc(),
                           writtenText, rule.to);
            if (!rewrite)
                return noEdits();

            return singleEdit(DRE->getSourceRange(),
                              std::string(rule.to));
        }
        return noEdits();
    };

    std::vector<std::string> funcNames;
    for (const auto &rule : kFreeFuncRules)
        funcNames.push_back(std::string(rule.from));

    rules.push_back(makeRule(
        callExpr(loc.stmt,
            callee(functionDecl(hasAnyNameFromVec(std::move(funcNames)))))
            .bind("freeFuncCall"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// .nbytes() → .numel() * .element_size()
// ---------------------------------------------------------------------------

static void addNbytesRule(std::vector<RewriteRule> &rules, Reporter &rep,
                          bool rewrite, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CXXMemberCallExpr>("nbytesCall");
        if (!CE || !isTensorType(CE->getImplicitObjectArgument()))
            return noEdits();

        auto rewriteLoc = getRewritableLoc(CE->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        const auto &SM = *R.SourceManager;
        const auto &LO = R.Context->getLangOpts();
        const auto *obj = CE->getImplicitObjectArgument();
        auto objText = getSourceText(obj->getSourceRange(), SM, LO);
        auto fullText = getSourceText(CE->getSourceRange(), SM, LO);

        std::string replacement = objText + ".numel() * " +
                                  objText + ".element_size()";

        const bool sideEffectFree =
            isa<DeclRefExpr>(obj->IgnoreParenImpCasts()) ||
            isa<MemberExpr>(obj->IgnoreParenImpCasts());

        if (!sideEffectFree) {
            rep.addFinding(FindingKind::MethodToFunc, SM, CE->getBeginLoc(),
                           fullText, replacement, FindingAction::Flag);
            return noEdits();
        }

        rep.addFinding(FindingKind::MethodToFunc, SM, CE->getBeginLoc(),
                       fullText, replacement);
        if (!rewrite)
            return noEdits();

        return singleEdit(CE->getSourceRange(), replacement);
    };

    rules.push_back(makeRule(
        cxxMemberCallExpr(loc.stmt,
            callee(cxxMethodDecl(hasName("nbytes"))))
            .bind("nbytesCall"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// c10::nullopt → std::nullopt
// ---------------------------------------------------------------------------

static void addNulloptRule(std::vector<RewriteRule> &rules, Reporter &rep,
                           bool rewrite, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *DRE = R.Nodes.getNodeAs<DeclRefExpr>("nulloptRef");
        if (!DRE)
            return noEdits();

        const auto &SM = *R.SourceManager;
        auto rewriteLoc = getRewritableLoc(DRE->getBeginLoc(), SM, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        auto text = getSourceText(DRE->getSourceRange(), SM,
                                  R.Context->getLangOpts());
        if (text != "c10::nullopt" && text != "::c10::nullopt")
            return noEdits();

        rep.addFinding(FindingKind::Type, SM, DRE->getBeginLoc(),
                       text, "std::nullopt");
        if (!rewrite)
            return noEdits();

        return singleEdit(*rewriteLoc,
                          static_cast<unsigned>(text.size()),
                          "std::nullopt");
    };

    rules.push_back(makeRule(
        declRefExpr(loc.stmt,
                    to(namedDecl(hasName("std::nullopt"))))
            .bind("nulloptRef"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// Catch-all detectors for unstable API usage
// ---------------------------------------------------------------------------

static bool isUnstableNamespace(llvm::StringRef qualName) {
    if (qualName.starts_with("at::") || qualName.starts_with("c10::"))
        return true;
    if (qualName.starts_with("torch::") &&
        !qualName.starts_with("torch::stable::") &&
        !qualName.starts_with("torch::headeronly::"))
        return true;
    return false;
}

static void addUnstableTypeCatchAll(std::vector<RewriteRule> &rules,
                                     Reporter &rep, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto dedup = std::make_shared<
        llvm::DenseSet<std::pair<const NamedDecl *, unsigned>>>();
    auto editGen = [&rep, projectRoot, dedup](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *TL = R.Nodes.getNodeAs<TypeLoc>("catchallType");
        if (!TL)
            return noEdits();

        const auto &SM = *R.SourceManager;
        auto spelling = SM.getSpellingLoc(TL->getBeginLoc());
        if (!isInProjectScope(SM, spelling, projectRoot))
            return noEdits();

        const auto *ND = R.Nodes.getNodeAs<NamedDecl>("catchallTypeDecl");
        if (!ND)
            return noEdits();

        auto qualName = ND->getQualifiedNameAsString();
        if (!isUnstableNamespace(qualName))
            return noEdits();

        unsigned line = SM.getSpellingLineNumber(spelling);
        if (!dedup->insert({ND, line}).second)
            return noEdits();

        bool inMacroBody = SM.isMacroBodyExpansion(TL->getBeginLoc());
        std::string text = inMacroBody
            ? qualName
            : getSourceText(TL->getSourceRange(), SM, R.Context->getLangOpts());
        std::string msg = "no stable equivalent — rewrite or remove " + qualName;
        if (inMacroBody) msg += " (inside macro body)";
        rep.addFinding(FindingKind::Flag, SM, spelling,
                       text, msg, FindingAction::Flag);
        return noEdits();
    };

    rules.push_back(makeRule(
        typeLoc(loc.typeLoc,
                ast_matchers::loc(qualType(hasDeclaration(
                    namedDecl().bind("catchallTypeDecl")))))
            .bind("catchallType"),
        EditGenerator(editGen)));
}

static void addUnstableRefCatchAll(std::vector<RewriteRule> &rules,
                                    Reporter &rep, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *DRE = R.Nodes.getNodeAs<DeclRefExpr>("catchallRef");
        if (!DRE)
            return noEdits();

        const auto &SM = *R.SourceManager;
        auto spelling = SM.getSpellingLoc(DRE->getBeginLoc());
        if (!isInProjectScope(SM, spelling, projectRoot))
            return noEdits();

        const auto *ND = R.Nodes.getNodeAs<NamedDecl>("catchallRefDecl");
        if (!ND)
            return noEdits();
        if (isa<CXXMethodDecl>(ND))
            return noEdits();
        if (const auto *FD = dyn_cast<FunctionDecl>(ND))
            if (FD->isOverloadedOperator())
                return noEdits();

        auto qualName = ND->getQualifiedNameAsString();
        if (!isUnstableNamespace(qualName))
            return noEdits();

        bool inMacroBody = SM.isMacroBodyExpansion(DRE->getBeginLoc());
        std::string text = inMacroBody
            ? qualName
            : getSourceText(DRE->getSourceRange(), SM, R.Context->getLangOpts());
        std::string msg = "no stable equivalent — rewrite or remove " + qualName;
        if (inMacroBody) msg += " (inside macro body)";
        rep.addFinding(FindingKind::Flag, SM, spelling,
                       text, msg, FindingAction::Flag);
        return noEdits();
    };

    rules.push_back(makeRule(
        declRefExpr(loc.stmt,
                    to(namedDecl().bind("catchallRefDecl")))
            .bind("catchallRef"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

RewriteRule buildTransformerRules(Reporter &rep, bool rewrite_mode,
                                  const std::string &projectRoot) {
    auto loc = buildLocFilter(projectRoot);
    std::vector<RewriteRule> rules;
    addTypeRules(rules, rep, rewrite_mode, loc);
    addScalarTypeRules(rules, rep, rewrite_mode, loc);
    addDeviceTypeRules(rules, rep, rewrite_mode, loc);
    addDataPtrRule(rules, rep, rewrite_mode, loc);
    addMethodToFuncRules(rules, rep, rewrite_mode, loc);
    addMethodRenameRules(rules, rep, rewrite_mode, loc);
    addElementSizeRules(rules, rep, rewrite_mode, loc);
    addFreeFuncRules(rules, rep, rewrite_mode, loc);
    addNbytesRule(rules, rep, rewrite_mode, loc);
    addNulloptRule(rules, rep, rewrite_mode, loc);
    addUnstableTypeCatchAll(rules, rep, loc);
    addUnstableRefCatchAll(rules, rep, loc);
    return applyFirst(rules);
}

// ---------------------------------------------------------------------------
// Manual AST matcher callbacks (DeviceGuard, CudaStream)
// ---------------------------------------------------------------------------

void DeviceGuardCallback::run(const MatchFinder::MatchResult &Result) {
    const auto *DS = Result.Nodes.getNodeAs<DeclStmt>("guardDeclStmt");
    const auto *VD = Result.Nodes.getNodeAs<VarDecl>("guardDecl");
    if (!VD)
        return;

    const auto &SM = *Result.SourceManager;
    const auto &LO = Result.Context->getLangOpts();
    auto loc = VD->getBeginLoc();

    if (!isInProjectScope(SM, SM.getSpellingLoc(loc), project_root_))
        return;
    if (SM.isMacroBodyExpansion(loc)) {
        auto text = getSourceText(VD->getSourceRange(), SM, LO);
        if (text.find("CUDAGuard") != std::string::npos) {
            reporter_.addFinding(FindingKind::DeviceGuard, SM,
                                 SM.getSpellingLoc(loc), text,
                                 "DeviceGuard usage inside macro body",
                                 FindingAction::Flag);
        }
        return;
    }

    auto text = getSourceText(VD->getSourceRange(), SM, LO);

    if (text.find("CUDAGuard") == std::string::npos)
        return;

    auto varName = VD->getNameAsString();

    std::string deviceExpr;

    if (const auto *init = VD->getInit()) {
        if (const auto *CE = dyn_cast<CXXConstructExpr>(
                init->IgnoreImplicit())) {
            if (CE->getNumArgs() > 0) {
                auto argText =
                    getSourceText(CE->getArg(0)->getSourceRange(), SM, LO);
                if (const auto *callArg = dyn_cast<CallExpr>(
                        CE->getArg(0)->IgnoreUnlessSpelledInSource())) {
                    if (auto *callee = callArg->getDirectCallee()) {
                        if (callee->getName() == "device_of" &&
                            callArg->getNumArgs() > 0) {
                            auto tensorText = getSourceText(
                                callArg->getArg(0)->getSourceRange(), SM, LO);
                            deviceExpr = tensorText + ".get_device_index()";
                        }
                    }
                    if (deviceExpr.empty()) {
                        if (const auto *memberCall =
                                dyn_cast<CXXMemberCallExpr>(callArg)) {
                            if (auto *method = memberCall->getMethodDecl()) {
                                if (method->getName() == "device") {
                                    auto objText = getSourceText(
                                        memberCall->getImplicitObjectArgument()
                                            ->getSourceRange(), SM, LO);
                                    deviceExpr = objText + ".get_device_index()";
                                }
                            }
                        }
                    }
                }
                if (deviceExpr.empty()) {
                    deviceExpr = argText;
                }
            }
        }
    }

    if (deviceExpr.empty()) {
        auto deviceOfPos = text.find("device_of(");
        if (deviceOfPos != std::string::npos) {
            auto argStart = deviceOfPos + 10;
            int depth = 1;
            auto argEnd = argStart;
            for (; argEnd < text.size() && depth > 0; ++argEnd) {
                if (text[argEnd] == '(')
                    ++depth;
                else if (text[argEnd] == ')')
                    --depth;
            }
            if (depth == 0 && argEnd > argStart + 1) {
                auto tensorName = text.substr(argStart, argEnd - argStart - 1);
                deviceExpr = tensorName + ".get_device_index()";
            }
        }
    }

    if (deviceExpr.empty()) {
        reporter_.addFinding(
            FindingKind::DeviceGuard, SM, loc, text,
            "torch::stable::accelerator::DeviceGuard " + varName +
                "(tensor.get_device_index())",
            FindingAction::Flag);
        return;
    }

    last_device_expr_ = deviceExpr;

    std::string replacement = "const torch::stable::accelerator::DeviceGuard " +
                              varName + "(" + deviceExpr + ");";

    reporter_.addFinding(FindingKind::DeviceGuard, SM, loc, text, replacement);
    if (rewrite_mode_) {
        SourceRange replaceRange = DS ? DS->getSourceRange() : VD->getSourceRange();
        auto stmtEnd = Lexer::findLocationAfterToken(
            replaceRange.getEnd(), tok::semi, SM, LO, false);
        if (stmtEnd.isValid()) {
            addReplacement(file_repls_, SM,
                CharSourceRange::getCharRange(replaceRange.getBegin(), stmtEnd),
                replacement, LO);
        } else {
            addReplacement(file_repls_, SM,
                CharSourceRange::getTokenRange(replaceRange),
                replacement, LO);
        }
    }
}

void CudaStreamCallback::run(const MatchFinder::MatchResult &Result) {
    const auto *DS = Result.Nodes.getNodeAs<DeclStmt>("streamDeclStmt");
    const auto *CE = Result.Nodes.getNodeAs<CallExpr>("streamCall");
    if (!CE)
        return;

    const auto &SM = *Result.SourceManager;
    const auto &LO = Result.Context->getLangOpts();
    auto loc = CE->getBeginLoc();

    if (!isInProjectScope(SM, SM.getSpellingLoc(loc), project_root_))
        return;
    if (SM.isMacroBodyExpansion(loc)) {
        auto text = getSourceText(CE->getSourceRange(), SM, LO);
        bool isStream =
            text.find("getCurrentCUDAStream") != std::string::npos ||
            text.find("getCurrentStream") != std::string::npos;
        if (isStream) {
            reporter_.addFinding(FindingKind::CudaStream, SM,
                                 SM.getSpellingLoc(loc), text,
                                 "CUDA stream usage inside macro body",
                                 FindingAction::Flag);
        }
        return;
    }

    auto text = getSourceText(CE->getSourceRange(), SM, LO);

    bool isCurrentStream =
        text.find("getCurrentCUDAStream") != std::string::npos ||
        text.find("getCurrentStream") != std::string::npos;

    if (!isCurrentStream)
        return;

    if (DS && DS->isSingleDecl()) {
        if (const auto *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
            auto varName = VD->getNameAsString();
            auto stmtText = getSourceText(DS->getSourceRange(), SM, LO);
            auto indent = getIndent(DS->getBeginLoc(), SM);

            const auto &device_expr = guard_cb_.getLastDeviceExpr();
            std::string deviceExpr = device_expr.empty() ? "-1" : device_expr;

            std::string replacement =
                "void* " + varName + "_ptr = nullptr;\n" +
                indent + "aoti_torch_get_current_cuda_stream(" + deviceExpr +
                ", &" + varName + "_ptr);\n" +
                indent + "const cudaStream_t " + varName +
                " = reinterpret_cast<cudaStream_t>(" + varName + "_ptr);";

            reporter_.addFinding(FindingKind::CudaStream, SM, loc, stmtText,
                                 replacement);
            if (rewrite_mode_) {
                auto stmtEnd = Lexer::findLocationAfterToken(
                    DS->getEndLoc(), tok::semi, SM, LO, false);
                if (stmtEnd.isValid()) {
                    addReplacement(file_repls_, SM,
                        CharSourceRange::getCharRange(DS->getBeginLoc(),
                                                      stmtEnd),
                        replacement, LO);
                } else {
                    addReplacement(file_repls_, SM,
                        CharSourceRange::getTokenRange(DS->getSourceRange()),
                        replacement, LO);
                }
            }
            return;
        }
    }

    reporter_.addFinding(
        FindingKind::CudaStream, SM, loc, text,
        "aoti_torch_get_current_cuda_stream(device_index, &stream_ptr)",
        FindingAction::Flag);
}

void registerManualMatchers(MatchFinder &finder,
                            CudaStreamCallback &streamCallback,
                            DeviceGuardCallback &guardCallback) {
    finder.addMatcher(
        declStmt(containsDeclaration(
                     0, varDecl(hasType(hasUnqualifiedDesugaredType(recordType(
                                    hasDeclaration(cxxRecordDecl(hasAnyName(
                                        "OptionalCUDAGuard", "CUDAGuard",
                                        "InferenceMode")))))))
                            .bind("guardDecl")))
            .bind("guardDeclStmt"),
        &guardCallback);

    finder.addMatcher(
        declStmt(containsDeclaration(
                     0, varDecl(hasInitializer(hasDescendant(
                            callExpr(callee(functionDecl(hasAnyName(
                                         "getCurrentCUDAStream",
                                         "getCurrentStream"))))
                                .bind("streamCall"))))))
            .bind("streamDeclStmt"),
        &streamCallback);

    finder.addMatcher(
        callExpr(callee(functionDecl(
                     hasAnyName("getCurrentCUDAStream", "getCurrentStream"))),
                 unless(hasAncestor(declStmt())))
            .bind("streamCall"),
        &streamCallback);
}

} // namespace stable_abi
