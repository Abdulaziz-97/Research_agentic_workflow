# API Billing & Cost Management Guide

## Error: "Insufficient budget available"

This error means your LLM provider (OpenAI or Google Gemini) has run out of credits or hit a spending limit.

## Quick Fixes

### 1. Check Your Billing Status
- **Google Gemini**: https://aistudio.google.com/app/apikey
- **OpenAI/Vocareum**: https://platform.openai.com/account/billing
- Verify your current balance and spending limits
- Add credits if needed

### 2. Increase Spending Limit
- **OpenAI**: Settings → Billing → Usage limits
- **Gemini**: Review quota in Google AI Studio and upgrade if needed

### 3. Verify API Key
- Check your `.env` file has the correct key (`GEMINI_API_KEY` or `OPENAI_API_KEY`)
- **Gemini**: https://aistudio.google.com/app/apikey
- **OpenAI**: https://platform.openai.com/api-keys
- If using a custom OpenAI endpoint (Vocareum), check that service's billing

## Cost Optimization Strategies

### Option 1: Use Cost-Saving Mode
The system supports a cost-saving mode that uses cheaper models for some agents.

Add to your `.env`:
```
COST_SAVING_MODE=true
```

This will use `gpt-3.5-turbo` for some non-critical agents while keeping `gpt-4o` for synthesis.

### Option 2: Reduce Domain Agents
- Use 1-2 domain agents instead of 3
- Each agent makes multiple API calls, so fewer agents = lower costs

### Option 3: Shorter Outputs
- The system is configured for comprehensive outputs (2000-4000 words)
- You can reduce `max_tokens` in synthesis if needed

### Option 4: Use Cheaper Model
- **Gemini**: set `GEMINI_MODEL=gemini-1.5-flash` in `.env`
- **OpenAI**: set `OPENAI_MODEL=gpt-3.5-turbo`

**Note:** This will reduce output quality but significantly lower costs.

## Cost Breakdown (Approximate)

Per research query with 3 domain agents:
- **Knowledge Graph Sampling**: ~$0.01
- **Ontologist Agent**: ~$0.05-0.10
- **Scientist I Agent**: ~$0.10-0.20
- **Scientist II Agent**: ~$0.10-0.20
- **Critic Agent**: ~$0.05-0.10
- **Planner Agent**: ~$0.05-0.10
- **Novelty Checker**: ~$0.05-0.10 (includes tool calls)
- **Domain Agents (3x)**: ~$0.30-0.60
- **Synthesis**: ~$0.20-0.40

**Total per query: ~$0.90-1.80** (with gpt-4o)

With `gpt-3.5-turbo`: ~$0.10-0.20 per query  
With `gemini-1.5-flash`: similar low cost structure (see Google AI Studio pricing)

## Monitoring Usage

1. **Google Gemini Usage**: https://aistudio.google.com/app/apikey  
2. **OpenAI Usage**: https://platform.openai.com/usage  
3. Check daily/weekly spending and set usage alerts

## Alternative Solutions

If you're using a custom OpenAI endpoint (Vocareum):
- Check that service's billing dashboard
- Verify credits/limits on that platform
- Contact their support if needed

## Emergency: Continue Without API

If you need to test the system without API calls:
- The knowledge graph and UI will still work
- Domain agents will fail but you can see the workflow
- Consider using mock responses for testing

---

**Need Help?** Check your `.env` file and verify all API keys are correct.

