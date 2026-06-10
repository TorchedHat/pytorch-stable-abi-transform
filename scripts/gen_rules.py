#!/usr/bin/env python3
"""
Generate Rules.h from the PyTorch stable ABI headers.

Uses clang -ast-dump=json to parse C++ declarations (functions, methods,
enums) from the stable headers. Regex is used only for macro detection
(#define), which is invisible to the AST.

This makes the rule set a derived artifact — not a hand-maintained one.
When PyTorch adds new stable APIs, re-running this script picks them up.
"""

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StableOp:
    """A function in torch::stable:: namespace."""
    name: str
    first_param_is_tensor: bool
    is_method_like: bool


@dataclass
class StableMacro:
    """A macro defined in stable headers."""
    name: str
    unstable_name: str = ""


@dataclass
class EnumConstant:
    """An enum constant in the stable headers."""
    qualified: str
    short_forms: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AST parsing
# ---------------------------------------------------------------------------

def parse_ast(header: Path, include_dir: Path) -> dict:
    """Run clang -ast-dump=json on a header and return the parsed AST."""
    clang = shutil.which('clang++')
    if not clang:
        print("error: clang++ not found. Install LLVM/Clang 19.", file=sys.stderr)
        sys.exit(1)

    if not header.exists():
        print(f"error: header not found: {header}", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        [clang, '-Xclang', '-ast-dump=json', '-fsyntax-only',
         '-std=c++20', f'-I{include_dir}', str(header)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"warning: clang parse errors in {header.name} "
              f"(extracting what we can)", file=sys.stderr)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"error: clang produced invalid JSON for {header}", file=sys.stderr)
        sys.exit(1)


def walk_ast(node: dict, visitors: list, namespace: str = ''):
    """Walk the clang AST JSON, calling visitors on each node."""
    kind = node.get('kind', '')
    name = node.get('name', '')

    if kind == 'NamespaceDecl' and name:
        child_ns = f'{namespace}::{name}' if namespace else name
        for child in node.get('inner', []):
            walk_ast(child, visitors, child_ns)
        return

    for v in visitors:
        v(node, kind, name, namespace)

    for child in node.get('inner', []):
        walk_ast(child, visitors, namespace)


# ---------------------------------------------------------------------------
# AST-based extractors
# ---------------------------------------------------------------------------

def extract_functions(ast: dict) -> list[StableOp]:
    """Extract free functions from torch::stable namespace (not detail)."""
    ops: list[StableOp] = []

    def visitor(node, kind, name, namespace):
        if kind != 'FunctionDecl' or node.get('isImplicit'):
            return
        if namespace != 'torch::stable' and not namespace.startswith('torch::stable::'):
            return
        if '::detail' in namespace:
            return
        if not name or name.startswith('_'):
            return
        if 'includedFrom' in node.get('loc', {}):
            return

        params = [c for c in node.get('inner', [])
                  if c.get('kind') == 'ParmVarDecl']
        first_is_tensor = bool(
            params and 'Tensor' in params[0].get('type', {}).get('qualType', '')
        )
        ops.append(StableOp(
            name=name,
            first_param_is_tensor=first_is_tensor,
            is_method_like=first_is_tensor,
        ))

    walk_ast(ast, [visitor])
    return ops


def extract_methods(ast: dict, class_names: set[str]) -> list[str]:
    """Extract public method names from named classes."""
    methods: list[str] = []

    def visit_class(node, kind, name, namespace):
        if kind == 'CXXRecordDecl' and name in class_names:
            for child in node.get('inner', []):
                visit_method(child)

    def visit_method(node):
        if node.get('kind') != 'CXXMethodDecl':
            return
        if node.get('isImplicit'):
            return
        name = node.get('name', '')
        if not name:
            return
        if name.startswith('operator') or name.startswith('~'):
            return
        access = node.get('access', 'public')
        if access != 'public':
            return
        methods.append(name)

    walk_ast(ast, [visit_class])
    return sorted(set(methods))


def extract_enum_constants(ast: dict, enum_name: str) -> list[str]:
    """Extract constant names from a named enum."""
    constants: list[str] = []

    def visitor(node, kind, name, namespace):
        if kind == 'EnumDecl' and name == enum_name:
            for child in node.get('inner', []):
                if child.get('kind') == 'EnumConstantDecl':
                    cname = child.get('name', '')
                    if cname:
                        constants.append(cname)

    walk_ast(ast, [visitor])
    return constants


# ---------------------------------------------------------------------------
# Enum shorthand derivation
# ---------------------------------------------------------------------------

# Aliases where the shorthand name differs from k{EnumName}
_SCALAR_TYPE_ALIASES = {
    'Float': ['kFloat', 'kFloat32'],
    'Double': ['kDouble', 'kFloat64'],
    'Half': ['kHalf', 'kFloat16'],
    'BFloat16': ['kBFloat16'],
    'Bool': ['kBool'],
    'Byte': ['kByte', 'kUInt8'],
    'Char': ['kChar', 'kInt8'],
    'Short': ['kShort', 'kInt16'],
    'Int': ['kInt', 'kInt32'],
    'Long': ['kLong', 'kInt64'],
    'ComplexFloat': ['kComplexFloat'],
    'ComplexDouble': ['kComplexDouble'],
    'ComplexHalf': ['kComplexHalf'],
}

_SKIP_SCALAR_TYPES = re.compile(r'^(Q|Bits|UInt[1-7]$|Int[1-7]$)')


def derive_scalar_types(enum_values: list[str]) -> list[EnumConstant]:
    """Derive ScalarType shorthands from enum constant names."""
    constants = []
    for name in enum_values:
        if _SKIP_SCALAR_TYPES.match(name):
            continue
        qualified = f"torch::headeronly::ScalarType::{name}"
        if name in _SCALAR_TYPE_ALIASES:
            short_forms = _SCALAR_TYPE_ALIASES[name]
        elif name.startswith('Float8_') or name.startswith('Float4_'):
            short_forms = [f'k{name}']
        else:
            continue
        constants.append(EnumConstant(qualified=qualified, short_forms=short_forms))
    return constants


def derive_device_types(enum_values: list[str]) -> list[EnumConstant]:
    """Derive DeviceType shorthands from enum constant names."""
    constants = []
    for name in enum_values:
        if name == 'COMPILE_TIME_MAX_DEVICE_TYPES':
            continue
        constants.append(EnumConstant(
            qualified=f"torch::headeronly::DeviceType::{name}",
            short_forms=[f'k{name}'],
        ))
    return constants


# ---------------------------------------------------------------------------
# Macro detection (must stay as text search — macros are invisible to AST)
# ---------------------------------------------------------------------------

def extract_macros(stable_dir: Path) -> list[StableMacro]:
    """Detect stable macros by checking header text for #define."""
    macros = []

    macros_h = stable_dir / 'macros.h'
    if macros_h.exists():
        text = macros_h.read_text()
        if 'STD_CUDA_CHECK' in text:
            macros.append(StableMacro('STD_CUDA_CHECK', 'C10_CUDA_CHECK'))
            macros.append(StableMacro('STD_CUDA_CHECK', 'AT_CUDA_CHECK'))
        if 'STD_CUDA_KERNEL_LAUNCH_CHECK' in text:
            macros.append(StableMacro('STD_CUDA_KERNEL_LAUNCH_CHECK',
                                      'C10_CUDA_KERNEL_LAUNCH_CHECK'))

    library_h = stable_dir / 'library.h'
    if library_h.exists():
        text = library_h.read_text()
        if 'STABLE_TORCH_LIBRARY_IMPL' in text:
            macros.append(StableMacro('STABLE_TORCH_LIBRARY_IMPL',
                                      'TORCH_LIBRARY_IMPL'))
        if 'STABLE_TORCH_LIBRARY_FRAGMENT' in text:
            macros.append(StableMacro('STABLE_TORCH_LIBRARY_FRAGMENT', ''))
        if re.search(r'#define\s+STABLE_TORCH_LIBRARY\b', text):
            macros.append(StableMacro('STABLE_TORCH_LIBRARY', 'TORCH_LIBRARY'))
        if 'TORCH_BOX' in text:
            macros.append(StableMacro('TORCH_BOX', ''))

    return macros


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def generate_rules_h(
    ops: list[StableOp],
    tensor_methods: list[str],
    scalar_types: list[EnumConstant],
    device_types: list[EnumConstant],
    macros: list[StableMacro],
    stable_methods: list[str],
    pytorch_dir: Path,
) -> str:
    """Generate the complete Rules.h file."""
    lines = []
    lines.append('#pragma once')
    lines.append('')
    lines.append('// AUTO-GENERATED by scripts/gen_rules.py from stable ABI headers.')
    lines.append(f'// Source: {pytorch_dir}')
    lines.append('// Do not edit manually. Re-run gen_rules.py to update.')
    lines.append('')
    lines.append('#include <array>')
    lines.append('#include <concepts>')
    lines.append('#include <ranges>')
    lines.append('#include <string_view>')
    lines.append('')
    lines.append('namespace stable_abi {')
    lines.append('')
    lines.append('template <typename R>')
    lines.append('concept MappingRule = requires(const R &r) {')
    lines.append('    { r.from } -> std::convertible_to<std::string_view>;')
    lines.append('    { r.to   } -> std::convertible_to<std::string_view>;')
    lines.append('};')
    lines.append('')
    lines.append('template <typename T>')
    lines.append('concept MappingRuleRange = std::ranges::range<T> &&')
    lines.append('    MappingRule<std::ranges::range_value_t<T>>;')
    lines.append('')

    # --- Include rules (structural, hand-maintained) ---
    lines.append('// Include rules are structural and hand-maintained.')
    lines.append('// The stable headers don\'t encode which unstable headers map to them.')
    lines.append('struct IncludeRule {')
    lines.append('    std::string_view old_path;')
    lines.append('    std::string_view new_paths[5];')
    lines.append('    bool remove_only;')
    lines.append('};')
    lines.append('')
    lines.append('inline constexpr std::array kIncludeRules = {')
    include_rules = [
        ('torch/all.h', ['torch/csrc/stable/tensor.h', 'torch/csrc/stable/ops.h', 'torch/csrc/stable/accelerator.h', 'torch/headeronly/core/ScalarType.h', 'torch/csrc/stable/device.h'], False),
        ('torch/torch.h', ['torch/csrc/stable/tensor.h', 'torch/csrc/stable/ops.h', 'torch/csrc/stable/accelerator.h', 'torch/headeronly/core/ScalarType.h', 'torch/csrc/stable/device.h'], False),
        ('torch/extension.h', ['torch/csrc/stable/tensor.h', 'torch/csrc/stable/ops.h', 'torch/csrc/stable/accelerator.h', 'torch/headeronly/core/ScalarType.h', 'torch/csrc/stable/device.h'], False),
        ('torch/cuda.h', ['torch/csrc/stable/accelerator.h', 'cuda_runtime.h', '', '', ''], False),
        ('torch/library.h', ['torch/csrc/stable/library.h', '', '', '', ''], False),
        ('ATen/cuda/CUDAContext.h', ['torch/csrc/stable/accelerator.h', 'cuda_runtime.h', '', '', ''], False),
        ('c10/cuda/CUDAGuard.h', ['torch/csrc/stable/accelerator.h', 'cuda_runtime.h', '', '', ''], False),
        ('ATen/Dispatch.h', ['torch/headeronly/core/Dispatch.h', 'torch/headeronly/core/Dispatch_v2.h', '', '', ''], False),
        ('ATen/ATen.h', ['torch/csrc/stable/tensor.h', 'torch/csrc/stable/ops.h', 'torch/headeronly/core/ScalarType.h', '', ''], False),
        ('ATen/cuda/Exceptions.h', ['torch/csrc/stable/macros.h', '', '', '', ''], False),
        ('c10/cuda/CUDAException.h', ['torch/csrc/stable/macros.h', '', '', '', ''], False),
        ('c10/cuda/CUDAStream.h', ['torch/csrc/stable/accelerator.h', 'cuda_runtime.h', '', '', ''], False),
        ('c10/util/Optional.h', ['', '', '', '', ''], True),
    ]
    for old, new_paths, remove in include_rules:
        new_str = ', '.join(f'"{p}"' for p in (new_paths + [''] * 5)[:5])
        lines.append(f'    IncludeRule{{"{old}", {{{new_str}}}, {"true" if remove else "false"}}},')
    lines.append('};')
    lines.append('')

    # --- Type rules ---
    lines.append('struct TypeRule {')
    lines.append('    std::string_view from;')
    lines.append('    std::string_view to;')
    lines.append('};')
    lines.append('')

    type_rules = [
        ('at::Tensor', 'torch::stable::Tensor'),
        ('torch::Tensor', 'torch::stable::Tensor'),
        ('c10::Device', 'torch::stable::Device'),
        ('at::Device', 'torch::stable::Device'),
        ('torch::Device', 'torch::stable::Device'),
        ('at::ScalarType', 'torch::headeronly::ScalarType'),
        ('c10::ScalarType', 'torch::headeronly::ScalarType'),
        ('torch::Dtype', 'torch::headeronly::ScalarType'),
        ('at::DeviceType', 'torch::headeronly::DeviceType'),
        ('c10::DeviceType', 'torch::headeronly::DeviceType'),
        ('at::Layout', 'torch::headeronly::Layout'),
        ('c10::Layout', 'torch::headeronly::Layout'),
        ('at::MemoryFormat', 'torch::headeronly::MemoryFormat'),
        ('c10::MemoryFormat', 'torch::headeronly::MemoryFormat'),
        ('at::Half', 'torch::headeronly::Half'),
        ('c10::Half', 'torch::headeronly::Half'),
        ('at::BFloat16', 'torch::headeronly::BFloat16'),
        ('c10::BFloat16', 'torch::headeronly::BFloat16'),
        ('c10::Float8_e4m3fn', 'torch::headeronly::Float8_e4m3fn'),
        ('c10::Float8_e4m3fnuz', 'torch::headeronly::Float8_e4m3fnuz'),
        ('c10::Float8_e5m2', 'torch::headeronly::Float8_e5m2'),
        ('c10::Float8_e5m2fnuz', 'torch::headeronly::Float8_e5m2fnuz'),
        ('c10::optional', 'std::optional'),
        ('c10::ArrayRef', 'torch::headeronly::HeaderOnlyArrayRef'),
        ('c10::IntArrayRef', 'torch::headeronly::IntHeaderOnlyArrayRef'),
        ('at::IntArrayRef', 'torch::headeronly::IntHeaderOnlyArrayRef'),
        ('c10::string_view', 'std::string_view'),
        ('c10::CppTypeToScalarType', 'torch::headeronly::CppTypeToScalarType'),
        ('torch::TensorOptions', ''),
        ('at::TensorOptions', ''),
        ('c10::TensorOptions', ''),
    ]

    lines.append('inline constexpr std::array kTypeRules = {')
    for old, new in type_rules:
        lines.append(f'    TypeRule{{"{old}", "{new}"}},')
    lines.append('};')
    lines.append('')

    # --- Macro rules ---
    lines.append('struct MacroRule {')
    lines.append('    std::string_view from;')
    lines.append('    std::string_view to;')
    lines.append('    bool flag_only;')
    lines.append('};')
    lines.append('')

    macro_rules = [
        ('TORCH_CHECK', 'STD_TORCH_CHECK', False),
        ('TORCH_CHECK_NOT_IMPLEMENTED', 'STD_TORCH_CHECK_NOT_IMPLEMENTED', False),
        ('TORCH_LIBRARY', 'STABLE_TORCH_LIBRARY', False),
        ('TORCH_LIBRARY_EXPAND', 'STABLE_TORCH_LIBRARY_FRAGMENT', False),
        ('TORCH_LIBRARY_IMPL', 'STABLE_TORCH_LIBRARY_IMPL', False),
        ('AT_DISPATCH_SWITCH', 'THO_DISPATCH_SWITCH', False),
        ('AT_DISPATCH_CASE', 'THO_DISPATCH_CASE', False),
        ('AT_DISPATCH_CASE_TYPE', '', True),
        ('C10_CUDA_CHECK', 'STD_CUDA_CHECK', False),
        ('AT_CUDA_CHECK', 'STD_CUDA_CHECK', False),
        ('C10_CUDA_KERNEL_LAUNCH_CHECK', 'STD_CUDA_KERNEL_LAUNCH_CHECK', False),
    ]

    lines.append('inline constexpr std::array kMacroRules = {')
    for old, new, flag in macro_rules:
        lines.append(f'    MacroRule{{"{old}", "{new}", {"true" if flag else "false"}}},')
        if old == 'TORCH_CHECK_NOT_IMPLEMENTED':
            lines.append('    // TORCH_CHECK_EQ/NE/LT/GT/GE/LE handled via kComparisonMacroRules below')
    lines.append('};')
    lines.append('')

    # --- Scalar type shorthands ---
    lines.append('struct ScalarTypeShorthand {')
    lines.append('    std::string_view from;')
    lines.append('    std::string_view to;')
    lines.append('};')
    lines.append('')
    lines.append('inline constexpr std::array kScalarTypeShorthands = {')
    for sc in scalar_types:
        for short in sc.short_forms:
            for ns in ['at', 'torch']:
                lines.append(f'    ScalarTypeShorthand{{"{ns}::{short}", "{sc.qualified}"}},')
    lines.append('};')
    lines.append('')

    # --- Device type shorthands ---
    lines.append('struct DeviceTypeShorthand {')
    lines.append('    std::string_view from;')
    lines.append('    std::string_view to;')
    lines.append('};')
    lines.append('')
    lines.append('inline constexpr std::array kDeviceTypeShorthands = {')
    for dt in device_types:
        for short in dt.short_forms:
            for ns in ['at', 'torch']:
                lines.append(f'    DeviceTypeShorthand{{"{ns}::{short}", "{dt.qualified}"}},')
    lines.append('};')
    lines.append('')

    # --- Method-to-free-function rules ---
    lines.append('struct MethodToFreeFunc {')
    lines.append('    std::string_view from;')
    lines.append('    std::string_view to;')
    lines.append('};')
    lines.append('')

    method_ops = [op for op in ops if op.is_method_like]
    method_ops = [op for op in method_ops if op.name not in tensor_methods]
    method_ops = [op for op in method_ops if not op.name.endswith('_out')]
    seen = set()
    deduped = []
    for op in method_ops:
        if op.name not in seen:
            seen.add(op.name)
            deduped.append(op)
    method_ops = deduped

    lines.append('inline constexpr std::array kMethodToFreeFuncRules = {')
    for op in method_ops:
        lines.append(f'    MethodToFreeFunc{{"{op.name}", "torch::stable::{op.name}"}},')
    lines.append('};')
    lines.append('')

    # --- Method rename rules ---
    lines.append('struct MethodRenameRule {')
    lines.append('    std::string_view from;')
    lines.append('    std::string_view to;')
    lines.append('};')
    lines.append('')
    lines.append('inline constexpr std::array kMethodRenameRules = {')
    lines.append('    MethodRenameRule{"dtype", "scalar_type"},')
    lines.append('    MethodRenameRule{"itemsize", "element_size"},')
    lines.append('    MethodRenameRule{"data_ptr", "mutable_data_ptr"},')
    lines.append('};')
    lines.append('')

    # --- Comparison macro rules ---
    lines.append('struct ComparisonMacroRule {')
    lines.append('    std::string_view name;')
    lines.append('    std::string_view op;')
    lines.append('};')
    lines.append('')

    comparison_rules = [
        ('TORCH_CHECK_EQ', '=='),
        ('TORCH_CHECK_NE', '!='),
        ('TORCH_CHECK_LT', '<'),
        ('TORCH_CHECK_GT', '>'),
        ('TORCH_CHECK_GE', '>='),
        ('TORCH_CHECK_LE', '<='),
    ]

    lines.append('inline constexpr std::array kComparisonMacroRules = {')
    for name, op in comparison_rules:
        lines.append(f'    ComparisonMacroRule{{"{name}", "{op}"}},')
    lines.append('};')
    lines.append('')

    # --- Dispatch conversion rules ---
    lines.append('struct DispatchConvRule {')
    lines.append('    std::string_view old_name;')
    lines.append('    std::string_view type_collection;')
    lines.append('};')
    lines.append('')

    dispatch_rules = [
        ('AT_DISPATCH_FLOATING_TYPES', 'AT_FLOATING_TYPES'),
        ('AT_DISPATCH_ALL_TYPES', 'AT_ALL_TYPES'),
        ('AT_DISPATCH_ALL_TYPES_AND_COMPLEX', 'AT_ALL_TYPES_AND_COMPLEX'),
        ('AT_DISPATCH_INTEGRAL_TYPES', 'AT_INTEGRAL_TYPES'),
        ('AT_DISPATCH_COMPLEX_TYPES', 'AT_COMPLEX_TYPES'),
        ('AT_DISPATCH_FLOAT8_TYPES', 'AT_FLOAT8_TYPES'),
    ]

    lines.append('inline constexpr std::array kDispatchConvRules = {')
    for old_name, type_coll in dispatch_rules:
        lines.append(f'    DispatchConvRule{{"{old_name}", "{type_coll}"}},')
    lines.append('};')
    lines.append('')

    # --- Free function rules ---
    lines.append('struct FreeFuncRule {')
    lines.append('    std::string_view from;')
    lines.append('    std::string_view to;')
    lines.append('};')
    lines.append('')

    all_free = []
    seen_free = set()
    for op in ops:
        if not op.is_method_like and op.name not in seen_free:
            seen_free.add(op.name)
            all_free.append(op)
    for op in ops:
        if op.is_method_like and op.name not in seen_free:
            seen_free.add(op.name)
            all_free.append(op)

    lines.append('inline constexpr std::array kFreeFuncRules = {')
    for op in all_free:
        for ns in ['torch', 'at']:
            lines.append(f'    FreeFuncRule{{"{ns}::{op.name}", "torch::stable::{op.name}"}},')
    lines.append('    FreeFuncRule{"torch::zeros", "torch::stable::full"},')
    lines.append('    FreeFuncRule{"at::zeros", "torch::stable::full"},')
    lines.append('};')
    lines.append('')

    # --- Stable method whitelist (auto-generated from AST) ---
    lines.append('// Methods that exist on stable types unchanged. Auto-generated from')
    lines.append('// tensor_struct.h, device_struct.h, library.h via clang AST.')
    lines.append('// Used by the method call catch-all to avoid false positives.')
    lines.append('inline constexpr std::array kStableMethodWhitelist = {')
    for m in stable_methods:
        lines.append(f'    std::string_view("{m}"),')
    lines.append('};')
    lines.append('')

    # --- Namespace rules (derived from include rules) ---
    lines.append('// Unstable namespace prefixes and their stable equivalents.')
    lines.append('struct NamespaceRule {')
    lines.append('    std::string_view from;')
    lines.append('    std::string_view to;')
    lines.append('};')
    lines.append('')
    # Derive from include rules: ATen/cuda/ and c10/cuda/ → accelerator
    ns_rules = []
    for old, new_paths, _ in include_rules:
        if 'accelerator.h' in str(new_paths):
            if old.startswith('ATen/cuda/'):
                ns_rules.append(('at::cuda::', 'torch::stable::accelerator::'))
            elif old.startswith('c10/cuda/'):
                ns_rules.append(('c10::cuda::', 'torch::stable::accelerator::'))
    ns_rules = list(dict.fromkeys(ns_rules))  # dedup preserving order
    lines.append('inline constexpr std::array kNamespaceRules = {')
    for from_ns, to_ns in ns_rules:
        lines.append(f'    NamespaceRule{{"{from_ns}", "{to_ns}"}},')
    lines.append('};')
    lines.append('')

    lines.append('} // namespace stable_abi')
    lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: gen_rules.py <pytorch-dir> [output-file]", file=sys.stderr)
        sys.exit(1)
    pytorch_dir = Path(sys.argv[1])
    stable_dir = pytorch_dir / 'torch' / 'csrc' / 'stable'
    headeronly_dir = pytorch_dir / 'torch' / 'headeronly'
    include_dir = pytorch_dir / 'torch' / 'include'
    if not include_dir.exists():
        include_dir = pytorch_dir

    if not stable_dir.exists():
        print(f"Error: {stable_dir} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing stable headers from {pytorch_dir}...", file=sys.stderr)

    # Parse ASTs
    ops_ast = parse_ast(stable_dir / 'ops.h', include_dir)
    tensor_ast = parse_ast(stable_dir / 'tensor_struct.h', include_dir)
    device_struct_ast = parse_ast(stable_dir / 'device_struct.h', include_dir)
    library_ast = parse_ast(stable_dir / 'library.h', include_dir)
    scalar_ast = parse_ast(headeronly_dir / 'core' / 'ScalarType.h', include_dir)
    device_type_ast = parse_ast(headeronly_dir / 'core' / 'DeviceType.h', include_dir)

    # Extract via AST
    ops = extract_functions(ops_ast)
    print(f"  Found {len(ops)} ops in ops.h (AST)", file=sys.stderr)

    tensor_methods = extract_methods(tensor_ast, {'Tensor'})
    print(f"  Found {len(tensor_methods)} tensor methods (AST)", file=sys.stderr)

    stable_methods = sorted(set(
        extract_methods(tensor_ast, {'Tensor'}) +
        extract_methods(device_struct_ast, {'Device'}) +
        extract_methods(library_ast, {'StableLibrary'})
    ))
    print(f"  Found {len(stable_methods)} stable methods total (AST)", file=sys.stderr)

    scalar_values = extract_enum_constants(scalar_ast, 'ScalarType')
    scalar_types = derive_scalar_types(scalar_values)
    print(f"  Found {len(scalar_types)} scalar types (AST)", file=sys.stderr)

    device_values = extract_enum_constants(device_type_ast, 'DeviceType')
    device_types = derive_device_types(device_values)
    print(f"  Found {len(device_types)} device types (AST)", file=sys.stderr)

    # Macros (text search — only thing that can't use AST)
    macros = extract_macros(stable_dir)
    print(f"  Found {len(macros)} stable macros (text)", file=sys.stderr)

    # Verify disjointness: methods in rewrite rules must not be in whitelist
    method_op_names = {op.name for op in ops
                       if op.is_method_like and op.name not in tensor_methods
                       and not op.name.endswith('_out')}
    overlap = method_op_names & set(stable_methods)
    if overlap:
        print(f"error: methods in both rewrite rules and whitelist: {overlap}",
              file=sys.stderr)
        sys.exit(1)

    # Generate
    output = generate_rules_h(
        ops, tensor_methods, scalar_types, device_types, macros,
        stable_methods, pytorch_dir,
    )

    if len(sys.argv) > 2:
        outpath = Path(sys.argv[2])
        outpath.write_text(output)
        print(f"Written to {outpath}", file=sys.stderr)
    else:
        print(output)

    # Coverage report
    method_ops = [op for op in ops
                  if op.is_method_like
                  and op.name not in tensor_methods
                  and not op.name.endswith('_out')]
    all_deduped = {op.name for op in ops}

    print(f"\n--- Coverage Report ---", file=sys.stderr)
    print(f"Method→FreeFunc rules: {len(set(op.name for op in method_ops))}", file=sys.stderr)
    print(f"Free function rules:   {len(all_deduped) * 2 + 2}", file=sys.stderr)
    print(f"Stable method whitelist: {stable_methods}", file=sys.stderr)


if __name__ == '__main__':
    main()
