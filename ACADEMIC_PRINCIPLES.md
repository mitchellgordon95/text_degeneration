# Academic Principles for Text Degeneration Experiments

## 🎯 **ZERO TOLERANCE POLICY**

This codebase follows **ACADEMIC-GRADE RIGOR** with zero tolerance for:
- ❌ **Silent fallbacks** or approximations
- ❌ **Hardcoded capabilities** that don't match reality
- ❌ **Parameter contamination** between methods
- ❌ **Configuration errors** that could invalidate results
- ❌ **API assumptions** without verification

## 🔬 **CORE PRINCIPLES**

### 1. **FAIL FAST AND LOUD**
```python
# ✅ CORRECT: Explicit error with actionable fix
if not model.supports_method(method):
    raise UnsupportedMethodError(
        f"Model {model_name} does not support {method}. "
        f"Fix: Remove '{method}' from {model_name} in models.yaml"
    )

# ❌ WRONG: Silent fallback or approximation
if not model.supports_method(method):
    print("Warning: Using approximation...")
    return approximate_method()
```

### 2. **STRICT PARAMETER ISOLATION**
Each decoding method gets **ONLY** its designated parameters:
```python
# ✅ CORRECT: Strict isolation
"greedy": {
    "temperature": 0.0,    # Greedy = no randomness
    "top_p": 1.0,         # Disabled
    "top_k": None,        # Disabled
    "num_beams": 1,       # No beam search
    "do_sample": False
}

# ❌ WRONG: Parameter contamination
"greedy": {
    "temperature": 0.0,
    "top_p": 0.95,        # BUG: Nucleus param in greedy!
    "num_beams": 1
}
```

### 3. **CONFIGURATION-DRIVEN CAPABILITIES**
Model capabilities MUST be defined in `config/models.yaml`, not hardcoded:
```yaml
# ✅ CORRECT: YAML-driven capabilities
gpt-4:
  capabilities:
    supported_methods: ["greedy", "temperature", "nucleus"]
    supports_logprobs: true
    supports_beam_search: false
    limitations: ["no_beam_search", "top_5_logprobs_only"]

# ❌ WRONG: Hardcoded in Python
class GPT4Model:
    def supported_methods(self):
        return ["greedy", "beam"]  # BUG: Claims beam support!
```

### 4. **COMPREHENSIVE VERIFICATION**
Every capability claim MUST be tested against reality:
```python
# ✅ CORRECT: Test actual capability
try:
    result = model.generate("test", method="beam_10")
    if not result:
        raise VerificationError(f"Claims beam support but fails")
except UnsupportedMethodError:
    raise VerificationError(f"Claims beam support but rejects it")

# ❌ WRONG: Trust configuration without testing
if "beam" in config["supported_methods"]:
    return True  # BUG: Assumes config is correct!
```

## 🛠️ **IMPLEMENTATION REQUIREMENTS**

### Model Classes Must:
1. **Use CapabilityManager** for all capability queries
2. **Implement strict parameter enforcement** in `_generate_impl()`
3. **Raise UnsupportedMethodError** for unsupported methods
4. **Never approximate or fallback** silently

### Configuration Files Must:
1. **Accurately reflect reality** - no aspirational capabilities
2. **Specify exact limitations** (e.g., "top_5_logprobs_only")
3. **Include actionable notes** for each model's constraints
4. **Be validated** by verification scripts

### Verification Must:
1. **Test every claimed capability** with actual generation
2. **Verify parameter isolation** is working correctly
3. **Test error handling** for unsupported methods
4. **Check memory requirements** against available hardware
5. **Fail immediately** on any discrepancy

## 📋 **MANDATORY VERIFICATION CHECKLIST**

Before any experiment run, the following MUST pass:

### ✅ **Dependency Verification**
- [ ] All required packages installed
- [ ] GPU/CPU resources available
- [ ] API keys valid and working

### ✅ **Capability Verification**
- [ ] Every model capability tested against reality
- [ ] Parameter isolation verified for all methods
- [ ] Unsupported methods properly rejected
- [ ] Memory requirements within limits

### ✅ **Configuration Validation**
- [ ] All experiment configs valid against model capabilities
- [ ] No invalid model/method combinations
- [ ] All parameterized methods properly parsed

### ✅ **Implementation Verification**
- [ ] No hardcoded capabilities in model classes
- [ ] Strict parameter enforcement in all models
- [ ] OpenAI models use Chat Completions API
- [ ] All error messages actionable

## 🚨 **COMMON PITFALLS TO AVOID**

### 1. **Silent Parameter Contamination**
```python
# ❌ WRONG: Other parameters leak into greedy
model.generate(prompt, method="greedy", top_p=0.9)  # BUG!

# ✅ CORRECT: Strict isolation enforced
params = get_method_parameters("greedy")  # Returns top_p=1.0
model.generate(prompt, method="greedy", **params)
```

### 2. **API Assumptions**
```python
# ❌ WRONG: Assume all OpenAI models support logprobs
if model_type == "openai":
    return model.get_logprobs()  # BUG: Not all support this!

# ✅ CORRECT: Check actual capability
if model.supports_logprobs:
    return model.get_logprobs()
else:
    raise UnsupportedMethodError("Model doesn't support logprobs")
```

### 3. **Configuration Drift**
```python
# ❌ WRONG: Config claims don't match implementation
# models.yaml: supports_beam_search: true
# But model implementation rejects beam search!

# ✅ CORRECT: Verification catches this discrepancy
verify_setup.py  # Will fail with actionable error message
```

## 🔄 **VERIFICATION WORKFLOW**

1. **After ANY configuration change**: Run `python verify_setup.py`
2. **Before experiments**: Verification must pass completely
3. **When adding models**: Test all claimed capabilities
4. **When updating APIs**: Verify no capabilities lost

## 💡 **FOR AI AGENTS**

When working on this codebase:

1. **NEVER** add silent fallbacks or approximations
2. **ALWAYS** update `models.yaml` when changing capabilities
3. **MUST** run verification after changes
4. **REQUIRE** actionable error messages for failures
5. **ENFORCE** strict parameter isolation
6. **TEST** all capabilities against reality

Remember: **Academic experiments require perfect reliability**. A single silent fallback can invalidate months of research.

## 🎯 **SUCCESS CRITERIA**

The verification script passing means:
- ✅ **Zero configuration discrepancies**
- ✅ **All capabilities work as claimed**
- ✅ **Perfect parameter isolation**
- ✅ **Reliable error handling**
- ✅ **Academic-grade rigor achieved**

When in doubt, **fail fast with actionable errors** rather than approximate!