# API Limitations for Decoding Experiments

## Critical Limitations

### OpenAI API
- **No native beam search support**: The API doesn't expose beam search as a decoding option
- **Limited logprobs**: Only returns top 5 logprobs per token (via `logprobs` parameter)
- **No access to full vocabulary distribution**: Cannot get probabilities for all tokens
- **Workaround**: Can simulate beam search using repeated API calls with logprobs, but this is:
  - Expensive (multiple API calls)
  - Slow (network latency)
  - Limited to top-5 tokens at each step

### Anthropic API (Claude)
- **No native beam search support**: No beam search parameter available
- **No logprobs at all**: The API doesn't return token probabilities
- **No vocabulary distribution access**: Cannot implement custom decoding strategies
- **Temperature/top_p only**: Limited to basic sampling parameters

## Implications for Experiments

### Beam Search Experiments
**Cannot use API models for beam search experiments:**
- ❌ GPT-3.5, GPT-4, GPT-5
- ❌ Claude-3.5-Sonnet, Claude-4-Opus
- ✅ Must use open source models with full access to logits

### Perplexity Calculations
- **OpenAI**: Limited accuracy due to top-5 restriction
- **Anthropic**: Cannot calculate perplexity at all
- **Open source**: Full access allows accurate perplexity computation

### Tail Analysis
- **OpenAI**: Cannot properly analyze tail distribution (only top-5)
- **Anthropic**: Impossible without probability access
- **Open source**: Can analyze full probability distribution

## Recommended Models for Full Experiments

### State-of-the-art open source models (2025):
1. **Llama 3.3 70B** - Matches GPT-4 performance, full logit access
2. **DeepSeek-V3 671B** - Top performer, but very large
3. **Qwen2.5 72B** - Strong multilingual, good for diverse prompts
4. **Mistral Small 3 24B** - Efficient, good balance of size/performance
5. **Mixtral 8x7B** - MoE architecture, efficient inference

### Practical considerations:
- Use smaller variants (7B, 13B) for initial testing
- 70B+ models require significant VRAM (80GB+)
- Consider quantization (8-bit, 4-bit) for larger models

## Experimental Design Adjustments

### For API models, limit to:
- ✅ Greedy decoding (temperature=0)
- ✅ Nucleus sampling (top_p)
- ✅ Temperature sampling
- ❌ Beam search
- ❌ Full perplexity analysis
- ❌ Tail distribution analysis

### For open source models, full suite:
- ✅ All decoding methods including beam search
- ✅ Complete perplexity calculations
- ✅ Full probability distribution analysis
- ✅ Custom decoding strategies