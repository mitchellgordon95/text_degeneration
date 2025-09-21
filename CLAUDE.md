# Claude Code Instructions for Text Degeneration Project

## 🎯 **PROJECT CONTEXT**
This is an **academic research project** testing whether findings from "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019) still apply to modern RLHF-trained language models. **Academic rigor is paramount**.

## 🔬 **CORE PHILOSOPHY: ZERO TOLERANCE**
This codebase has **ZERO TOLERANCE** for:
- Silent fallbacks or approximations
- Hardcoded assumptions about model capabilities
- Parameter contamination between decoding methods
- Configuration errors that could invalidate results

## 🛠️ **MANDATORY PRACTICES**

### 1. **ALWAYS Run Verification First**
Before any work:
```bash
python verify_setup.py
```
If this fails, **fix the issues immediately**. Never proceed with broken verification.

### 2. **Capability Changes Require Full Verification**
When modifying model capabilities:
1. Update `config/models.yaml` with accurate capabilities
2. Run verification to ensure claims match reality
3. Fix any discrepancies immediately

### 3. **Strict Parameter Isolation**
Each decoding method gets **only** its designated parameters:
- `greedy`: temp=0, top_p=1.0, top_k=None, beams=1
- `beam_N`: temp=1.0, top_p=1.0, top_k=None, beams=N
- `nucleus_P`: temp=1.0, top_p=P, top_k=None, beams=1
- `top_k_K`: temp=1.0, top_p=1.0, top_k=K, beams=1

### 4. **Error Messages Must Be Actionable**
```python
# ✅ GOOD: Actionable error
raise UnsupportedMethodError(
    f"Model {model_name} does not support {method}. "
    f"Fix: Remove '{method}' from supported_methods in models.yaml"
)

# ❌ BAD: Vague error
raise Error("Method not supported")
```

### 5. **No Hardcoded Capabilities**
Use `CapabilityManager` for all capability queries:
```python
# ✅ CORRECT
capabilities = CapabilityManager()
if capabilities.supports_method(model_name, method):
    # proceed

# ❌ WRONG
if model_type == "openai":
    supported = ["greedy", "nucleus"]  # Hardcoded!
```

## 📋 **CHECKLIST FOR ANY CHANGES**

### Before Making Changes:
- [ ] Read `ACADEMIC_PRINCIPLES.md` thoroughly
- [ ] Run `python verify_setup.py` to ensure starting state is clean
- [ ] Understand which models/methods the change affects

### While Making Changes:
- [ ] Update `config/models.yaml` if capabilities change
- [ ] Use `CapabilityManager` for all capability queries
- [ ] Implement strict parameter isolation
- [ ] Add actionable error messages

### After Making Changes:
- [ ] Run `python verify_setup.py` and ensure it passes
- [ ] Fix any verification failures with actionable solutions
- [ ] Test the specific functionality you changed
- [ ] Commit only after verification passes

## 🚨 **NEVER DO THESE THINGS**

1. **Silent Fallbacks**: Never approximate or substitute methods
2. **Parameter Contamination**: Never let other method params leak in
3. **Hardcoded Assumptions**: Never assume model capabilities
4. **Skip Verification**: Never commit without running verification
5. **Vague Errors**: Never give errors without actionable fixes

## 🎯 **COMMON TASKS & CORRECT APPROACHES**

### Adding a New Model:
1. Add complete capability specification to `models.yaml`
2. Ensure all claimed capabilities are accurate
3. Run verification to test actual vs claimed capabilities
4. Fix any discrepancies found

### Adding a New Decoding Method:
1. Add to `get_method_parameters()` with strict isolation
2. Update model capabilities in `models.yaml` as appropriate
3. Test that unsupported models properly reject it
4. Verify parameter isolation is working

### Fixing a Bug:
1. Identify root cause (often capability mismatch)
2. Fix in configuration and/or implementation
3. Add verification to prevent regression
4. Ensure error messages are actionable

## 📊 **VERIFICATION INTERPRETATION**

### When Verification Passes:
- ✅ All model capabilities accurate
- ✅ Parameter isolation working
- ✅ Error handling correct
- ✅ Ready for experiments

### When Verification Fails:
- 🔧 Read error message for specific fix
- 🔧 Update `models.yaml` if capability mismatch
- 🔧 Fix implementation if parameter isolation broken
- 🔧 Remove unsupported methods from experiments

## 🔄 **EXPERIMENTAL WORKFLOW**

### Planning Experiments:
1. Check which models support required methods
2. Use `validate_config.py` to verify experiment configs
3. Plan based on actual capabilities, not assumptions

### Running Experiments:
1. Verification must pass first
2. Use only verified model/method combinations
3. Monitor for any capability-related errors

### Analyzing Results:
1. Document any capability limitations that affected results
2. Note which models couldn't be tested due to limitations
3. Include capability constraints in research conclusions

## 💡 **DEBUGGING TIPS**

### "Method not supported" errors:
1. Check `models.yaml` for actual supported methods
2. Verify method name spelling (e.g., `nucleus_0.95` not `nucleus_95`)
3. Ensure model type supports the method category

### Parameter isolation issues:
1. Check `get_method_parameters()` function
2. Verify method parsing logic for parameterized methods
3. Test with `python -c "from src.utils.capabilities import get_method_parameters; print(get_method_parameters('method_name'))"`

### Verification failures:
1. Read the specific error message for actionable fix
2. Never ignore or work around verification failures
3. Fix root cause, don't mask symptoms

## 🎉 **SUCCESS INDICATORS**

- `python verify_setup.py` passes completely
- All experiment configurations are valid
- No silent fallbacks or approximations
- Error messages provide clear fix instructions
- Academic rigor maintained throughout

Remember: **This is academic research**. Perfect reliability is more important than convenience. When in doubt, fail fast with clear error messages rather than approximate!