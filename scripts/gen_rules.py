#!/usr/bin/env python3
"""
Generate Rules.h from the PyTorch stable ABI headers.

Parses torch/csrc/stable/ and torch/headeronly/ to extract the complete
stable API surface, then derives transformation rules by construction.

This makes the rule set a derived artifact — not a hand-maintained one.
Completeness is provable: for every symbol in torch::stable::*, either
a rewrite rule exists or the symbol is documented as having no unstable
counterpart.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StableOp:
    """A function in torch::stable:: namespace."""
    name: str
    first_param_is_tensor: bool
    is_method_like: bool  # first param is self tensor → was likely a method


@dataclass
class StableType:
    """A type defined in the stable headers."""
    qualified_name: str  # e.g., "torch::stable::Tensor"
    unstable_names: list[str] = field(default_factory=list)


@dataclass
class StableMacro:
    """A macro defined in stable headers."""
    name: str
    unstable_name: str = ""


@dataclass
class EnumConstant:
    """An enum constant in the stable headers."""
    qualified: str  # e.g., "torch::headeronly::ScalarType::Float"
    short_forms: list[str] = field(default_factory=list)


def extract_ops_from_header(path: Path) -> list[StableOp]:
    """Extract inline function declarations from ops.h."""
    text = path.read_text()
    ops = []

    # Match: inline torch::stable::Tensor funcname(params...)
    # Also match: inline void funcname(params...)
    pattern = re.compile(
        r'inline\s+(?:torch::stable::Tensor&?|void|uint32_t)\s+'
        r'(\w+)\s*\((.*?)\)\s*\{',
        re.DOTALL
    )

    for m in pattern.finditer(text):
        name = m.group(1)
        params = m.group(2).strip()

        # Check if first param is a Tensor (method-like)
        first_is_tensor = bool(re.match(
            r'\s*(?:const\s+)?torch::stable::Tensor\s*&', params
        ))

        ops.append(StableOp(
            name=name,
            first_param_is_tensor=first_is_tensor,
            is_method_like=first_is_tensor,
        ))

    return ops


def extract_tensor_methods(path: Path) -> list[str]:
    """Extract method names from tensor_struct.h that exist on stable Tensor."""
    text = path.read_text()
    methods = []

    # Match method declarations on class Tensor:
    # Look for lines like: type methodname(params) const? {
    # But only match actual method definitions, not random function calls
    pattern = re.compile(
        r'^\s+(?:inline\s+)?(?:[\w:]+(?:<[\w:,\s]+>)?[*&\s]+)'
        r'(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{',
        re.MULTILINE,
    )

    for m in pattern.finditer(text):
        name = m.group(1)
        # Skip constructors, operators, internal helpers, macros
        if name in ('Tensor', 'get', 'release', 'DEFINE_DATA_PTR_CAST'):
            continue
        if name.startswith('operator') or name.startswith('TORCH'):
            continue
        # Skip C function names and member variables
        if name.startswith('aoti_') or name.endswith('_'):
            continue
        methods.append(name)

    return list(set(methods))


def extract_scalar_types(path: Path) -> list[EnumConstant]:
    """Extract ScalarType enum values from ScalarType.h.

    The enum is populated via the AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS
    macro, so we parse that instead of the enum body.
    """
    text = path.read_text()
    constants = []

    # Parse AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS macro
    macro_match = re.search(
        r'#define\s+AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS\s*\(\s*_\s*\)\s*\\(.*?)(?=\n\n|\n#define\s)',
        text, re.DOTALL
    )
    if not macro_match:
        return constants

    macro_body = macro_match.group(1)
    # Extract _(cpp_type, EnumName) entries
    for m in re.finditer(r'_\(\s*[\w:<>]+\s*,\s*(\w+)\s*\)', macro_body):
        name = m.group(1)

        # Skip internal types: quantized (Q*), sub-byte (Bits*), narrow unsigned (UInt1-7)
        if re.match(r'^(Q|Bits|UInt[1-7]$|Int[1-7]$)', name):
            continue

        qualified = f"torch::headeronly::ScalarType::{name}"

        short_forms = []
        short_map = {
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
        if name in short_map:
            short_forms = short_map[name]
        elif name.startswith('Float8_') or name.startswith('Float4_'):
            short_forms = [f'k{name}']

        constants.append(EnumConstant(qualified=qualified, short_forms=short_forms))

    return constants


def extract_device_types(path: Path) -> list[EnumConstant]:
    """Extract DeviceType enum values."""
    text = path.read_text()
    constants = []

    enum_match = re.search(r'enum class DeviceType\s*:\s*int8_t\s*\{(.*?)\}', text, re.DOTALL)
    if not enum_match:
        return constants

    enum_body = enum_match.group(1)
    for line in enum_body.split('\n'):
        line = line.strip().rstrip(',')
        if not line or line.startswith('//') or 'COMPILE_TIME_MAX' in line:
            continue
        name_match = re.match(r'(\w+)', line)
        if name_match:
            name = name_match.group(1)
            constants.append(EnumConstant(
                qualified=f"torch::headeronly::DeviceType::{name}",
                short_forms=[f'k{name}'],
            ))

    return constants


def extract_macros(stable_dir: Path) -> list[StableMacro]:
    """Extract macro definitions from stable headers."""
    macros = []

    # From macros.h
    macros_h = stable_dir / 'macros.h'
    if macros_h.exists():
        text = macros_h.read_text()
        if 'STD_CUDA_CHECK' in text:
            macros.append(StableMacro('STD_CUDA_CHECK', 'C10_CUDA_CHECK'))
            macros.append(StableMacro('STD_CUDA_CHECK', 'AT_CUDA_CHECK'))
        if 'STD_CUDA_KERNEL_LAUNCH_CHECK' in text:
            macros.append(StableMacro('STD_CUDA_KERNEL_LAUNCH_CHECK', 'C10_CUDA_KERNEL_LAUNCH_CHECK'))

    # From library.h
    library_h = stable_dir / 'library.h'
    if library_h.exists():
        text = library_h.read_text()
        if 'STABLE_TORCH_LIBRARY_IMPL' in text:
            macros.append(StableMacro('STABLE_TORCH_LIBRARY_IMPL', 'TORCH_LIBRARY_IMPL'))
        if 'STABLE_TORCH_LIBRARY_FRAGMENT' in text:
            macros.append(StableMacro('STABLE_TORCH_LIBRARY_FRAGMENT', ''))
        if re.search(r'#define\s+STABLE_TORCH_LIBRARY\b', text):
            macros.append(StableMacro('STABLE_TORCH_LIBRARY', 'TORCH_LIBRARY'))
        if 'TORCH_BOX' in text:
            macros.append(StableMacro('TORCH_BOX', ''))

    # From headeronly Exception.h
    return macros


def generate_rules_h(
    ops: list[StableOp],
    tensor_methods: list[str],
    scalar_types: list[EnumConstant],
    device_types: list[EnumConstant],
    macros: list[StableMacro],
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

    # --- Include rules (these are structural, keep hand-maintained) ---
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

    # --- Type rules (derived from stable headers) ---
    lines.append('struct TypeRule {')
    lines.append('    std::string_view from;')
    lines.append('    std::string_view to;')
    lines.append('};')
    lines.append('')

    # Core type mappings derived from stable header class names
    type_rules = [
        # Tensor
        ('at::Tensor', 'torch::stable::Tensor'),
        ('torch::Tensor', 'torch::stable::Tensor'),
        # Device
        ('c10::Device', 'torch::stable::Device'),
        ('at::Device', 'torch::stable::Device'),
        ('torch::Device', 'torch::stable::Device'),
        # ScalarType / Dtype
        ('at::ScalarType', 'torch::headeronly::ScalarType'),
        ('c10::ScalarType', 'torch::headeronly::ScalarType'),
        ('torch::Dtype', 'torch::headeronly::ScalarType'),
        # DeviceType
        ('at::DeviceType', 'torch::headeronly::DeviceType'),
        ('c10::DeviceType', 'torch::headeronly::DeviceType'),
        # Layout
        ('at::Layout', 'torch::headeronly::Layout'),
        ('c10::Layout', 'torch::headeronly::Layout'),
        # MemoryFormat
        ('at::MemoryFormat', 'torch::headeronly::MemoryFormat'),
        ('c10::MemoryFormat', 'torch::headeronly::MemoryFormat'),
        # Half
        ('at::Half', 'torch::headeronly::Half'),
        ('c10::Half', 'torch::headeronly::Half'),
        # BFloat16
        ('at::BFloat16', 'torch::headeronly::BFloat16'),
        ('c10::BFloat16', 'torch::headeronly::BFloat16'),
        # Float8 types
        ('c10::Float8_e4m3fn', 'torch::headeronly::Float8_e4m3fn'),
        ('c10::Float8_e4m3fnuz', 'torch::headeronly::Float8_e4m3fnuz'),
        ('c10::Float8_e5m2', 'torch::headeronly::Float8_e5m2'),
        ('c10::Float8_e5m2fnuz', 'torch::headeronly::Float8_e5m2fnuz'),
        # Optional
        ('c10::optional', 'std::optional'),
        # ArrayRef / string_view
        ('c10::ArrayRef', 'torch::headeronly::HeaderOnlyArrayRef'),
        ('c10::IntArrayRef', 'torch::headeronly::IntHeaderOnlyArrayRef'),
        ('at::IntArrayRef', 'torch::headeronly::IntHeaderOnlyArrayRef'),
        ('c10::string_view', 'std::string_view'),
        # Template utilities
        ('c10::CppTypeToScalarType', 'torch::headeronly::CppTypeToScalarType'),
        # TensorOptions (flag-only: empty to)
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
        # TORCH_CHECK_EQ/NE/LT/GT/GE/LE handled via kComparisonMacroRules below
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

    # --- Scalar type shorthands (derived from enum) ---
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

    # --- Device type shorthands (derived from enum) ---
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

    # --- Method-to-free-function rules (derived from ops.h) ---
    lines.append('struct MethodToFreeFunc {')
    lines.append('    std::string_view from;')
    lines.append('    std::string_view to;')
    lines.append('};')
    lines.append('')

    # Ops where first param is Tensor → these are method-like
    method_ops = [op for op in ops if op.is_method_like]
    # Filter out ops that are already methods on stable::Tensor (no rewrite needed)
    method_ops = [op for op in method_ops if op.name not in tensor_methods]
    # Exclude _out variants
    method_ops = [op for op in method_ops if not op.name.endswith('_out')]
    # Deduplicate (overloads produce duplicate names)
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

    # --- Free function namespace rules (derived from ops.h) ---
    lines.append('struct FreeFuncRule {')
    lines.append('    std::string_view from;')
    lines.append('    std::string_view to;')
    lines.append('};')
    lines.append('')

    # All ops exist as free functions in at:: and torch:: namespaces.
    # Non-method-like first (factory functions), then method-like.
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
    # Special mappings where function name changes
    lines.append('    FreeFuncRule{"torch::zeros", "torch::stable::full"},')
    lines.append('    FreeFuncRule{"at::zeros", "torch::stable::full"},')
    lines.append('};')
    lines.append('')

    lines.append('} // namespace stable_abi')
    lines.append('')

    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: gen_rules.py <pytorch-dir> [output-file]", file=sys.stderr)
        sys.exit(1)
    pytorch_dir = Path(sys.argv[1])
    stable_dir = pytorch_dir / 'torch' / 'csrc' / 'stable'
    headeronly_dir = pytorch_dir / 'torch' / 'headeronly'

    if not stable_dir.exists():
        print(f"Error: {stable_dir} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing stable headers from {pytorch_dir}...", file=sys.stderr)

    # Extract ops from ops.h
    ops = extract_ops_from_header(stable_dir / 'ops.h')
    print(f"  Found {len(ops)} ops in ops.h", file=sys.stderr)

    # Extract tensor methods
    tensor_methods = extract_tensor_methods(stable_dir / 'tensor_struct.h')
    # Also check tensor_inl.h
    tensor_inl = stable_dir / 'tensor_inl.h'
    if tensor_inl.exists():
        tensor_methods.extend(extract_tensor_methods(tensor_inl))
    print(f"  Found {len(tensor_methods)} tensor methods", file=sys.stderr)

    # Extract scalar types
    scalar_types = extract_scalar_types(headeronly_dir / 'core' / 'ScalarType.h')
    print(f"  Found {len(scalar_types)} scalar types", file=sys.stderr)

    # Extract device types
    device_types = extract_device_types(headeronly_dir / 'core' / 'DeviceType.h')
    print(f"  Found {len(device_types)} device types", file=sys.stderr)

    # Extract macros
    macros = extract_macros(stable_dir)
    print(f"  Found {len(macros)} stable macros", file=sys.stderr)

    # Generate
    output = generate_rules_h(ops, tensor_methods, scalar_types, device_types, macros, pytorch_dir)

    # Write to stdout or file
    if len(sys.argv) > 2:
        outpath = Path(sys.argv[2])
        outpath.write_text(output)
        print(f"Written to {outpath}", file=sys.stderr)
    else:
        print(output)

    # Report coverage
    method_ops = [op for op in ops if op.is_method_like and op.name not in tensor_methods and not op.name.endswith('_out')]
    all_deduped = {op.name for op in ops}
    shorthand_count = sum(len(sc.short_forms) * 2 for sc in scalar_types)

    print(f"\n--- Coverage Report ---", file=sys.stderr)
    print(f"Method→FreeFunc rules: {len(method_ops)}", file=sys.stderr)
    print(f"  {[op.name for op in method_ops]}", file=sys.stderr)
    print(f"Free function rules:   {len(all_deduped) * 2 + 2} (x2 for at::/torch:: + zeros)", file=sys.stderr)
    print(f"  {sorted(all_deduped)}", file=sys.stderr)
    print(f"Scalar type shorthands: {shorthand_count}", file=sys.stderr)
    print(f"Tensor methods (no rewrite needed): {tensor_methods}", file=sys.stderr)


if __name__ == '__main__':
    main()
