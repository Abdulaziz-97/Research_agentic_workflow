# API Key Rotation Guide

## Overview

The system now supports **automatic API key rotation**. When one key fails (due to rate limits, budget issues, etc.), the system automatically tries the next available key.

## How to Configure Multiple Keys

### Option 1: Comma-Separated (Recommended)

In your `.env` file, add all your keys separated by commas. Examples:

```env
# Google Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=gemini-key1,gemini-key2,gemini-key3
GEMINI_MODEL=gemini-3.0-pro

# OpenAI / Vocareum
LLM_PROVIDER=openai
OPENAI_API_KEY=voc-key1,voc-key2,voc-key3,voc-key4,voc-key5,voc-key6
OPENAI_BASE_URL=https://openai.vocareum.com/v1
OPENAI_MODEL=gpt-4o
```

### Option 2: Multiple Lines (Alternative)

You can also put keys on separate lines (the system will read them):

```env
GEMINI_API_KEY=gemini-key1
GEMINI_API_KEY=gemini-key2
GEMINI_API_KEY=gemini-key3
```

**Note:** The comma-separated format is recommended and easier to manage.

## Example Configuration

Here's an example with OpenAI Vocareum keys:

```env
# OpenAI Configuration (Vocareum)
OPENAI_API_KEY=voc-585290941690924505358691f4ae1c0f449.28122897,voc-1516140538166950450535869075353dacd19.45717922,voc-141343078168865450535868fbb7344683e6.02198483,voc-8454431591729974505358691f48d99ced25.03191901,voc-17475177091587664505358690383b6526c04.96412439,voc-1142372955159874450535868fa2c751ba597.80273989
OPENAI_BASE_URL=https://openai.vocareum.com/v1
OPENAI_MODEL=gpt-4o
```

## How It Works

1. **Key Rotation**: The system starts with the first key
2. **Automatic Fallback**: If a key fails, it automatically tries the next one
3. **Smart Disabling**: Failed keys are temporarily disabled based on error type:
   - **Rate Limit (429)**: Disabled for 1 minute
   - **Budget/Insufficient Credits**: Disabled for 1 hour
   - **Authentication Error (401)**: Disabled permanently (until restart)
   - **Other Errors**: Disabled for 5 minutes
4. **Auto-Recovery**: Disabled keys are automatically re-enabled after the timeout period

## Benefits

- ✅ **No Manual Intervention**: Keys rotate automatically
- ✅ **Resilient**: System continues working even if some keys fail
- ✅ **Smart**: Different error types get different timeout periods
- ✅ **Efficient**: Only working keys are used

## Monitoring Key Status

You can check key status programmatically:

```python
from config.key_manager import get_key_manager

key_manager = get_key_manager()
if key_manager:
    status = key_manager.get_status()
    print(f"Total keys: {status['total_keys']}")
    print(f"Available keys: {status['available_keys']}")
    print(f"Current key: {status['current_key']}")
```

## Troubleshooting

### All Keys Failing

If all keys are failing:
1. Check that keys are correctly formatted (no extra spaces)
2. Verify the base URL is correct
3. Check that keys have sufficient credits
4. Review error messages in the logs

### Keys Not Rotating

If keys aren't rotating:
1. Ensure multiple keys are in the `.env` file (comma-separated)
2. Check that the key manager is initialized (should happen automatically)
3. Verify keys are different (no duplicates)

### Rate Limits Still Occurring

If you're still hitting rate limits:
1. Add more keys to the rotation pool
2. Reduce the number of parallel agents
3. Add delays between agent calls (future enhancement)

## Best Practices

1. **Use Multiple Keys**: The more keys you have, the more resilient the system
2. **Monitor Usage**: Keep an eye on which keys are being used
3. **Distribute Load**: Keys will be used in rotation, distributing load
4. **Keep Keys Updated**: Replace expired or invalid keys in the `.env` file

## Technical Details

- Keys are stored in memory (not persisted)
- Key health is tracked per session
- Rotation happens automatically on failure
- Thread-safe for concurrent agent calls

