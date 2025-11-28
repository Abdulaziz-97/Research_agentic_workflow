# BGE-M3 Embeddings Setup Guide

## Overview

BGE-M3 is a **free, open-source** multilingual embedding model that provides excellent performance for RAG systems. It's now the default embeddings provider for the research lab.

### Key Benefits

✅ **Free** - No API costs  
✅ **Multilingual** - Supports 100+ languages  
✅ **Long Documents** - Handles up to 8192 tokens  
✅ **High Performance** - Top-tier results on embedding benchmarks  
✅ **Local Processing** - Runs on your machine, no API calls  

## Installation

BGE-M3 is already installed via `FlagEmbedding`:

```bash
pip install FlagEmbedding
```

The model will be automatically downloaded from Hugging Face on first use (~2.3GB).

## Configuration

### Default Setup (Recommended)

The system is configured to use BGE-M3 by default. No additional configuration needed!

```env
# In your .env file
EMBEDDINGS_PROVIDER=bge-m3
```

### Optional Settings

```env
# Customize BGE-M3 model (default: BAAI/bge-m3)
BGE_M3_MODEL_NAME=BAAI/bge-m3

# Use FP16 for faster computation (default: true)
BGE_M3_USE_FP16=true
```

## Testing

Run the test script to verify BGE-M3 is working:

```bash
python test_bge_m3.py
```

This will:
1. Initialize the BGE-M3 model (downloads on first run)
2. Test query embedding generation
3. Test document embedding generation
4. Test similarity calculations

## Model Specifications

- **Dimensions**: 1024
- **Max Sequence Length**: 8192 tokens
- **Languages**: 100+
- **Model Size**: ~2.3GB

## Performance Notes

- **First Load**: The model downloads and loads on first use (~2-3 minutes)
- **Subsequent Loads**: Much faster (~10-30 seconds) as model is cached
- **FP16 Mode**: Enabled by default for faster computation with minimal quality loss
- **Memory Usage**: ~4-6GB RAM during inference

## Switching Back to OpenAI

If you need to use OpenAI embeddings instead:

```env
EMBEDDINGS_PROVIDER=openai
OPENAI_EMBEDDINGS_API_KEY=sk-your-key-here
OPENAI_EMBEDDINGS_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small
```

## Troubleshooting

### Model Download Issues

If the model fails to download:
- Check your internet connection
- Ensure you have ~3GB free disk space
- Try manually downloading: `huggingface-cli download BAAI/bge-m3`

### Memory Issues

If you run out of memory:
- Set `BGE_M3_USE_FP16=true` (reduces memory usage)
- Close other applications
- Consider using a smaller model variant

### Slow Performance

- First run is slow due to model download
- Subsequent runs are faster
- FP16 mode improves speed
- GPU acceleration (if available) significantly speeds up inference

## References

- [BGE-M3 on Hugging Face](https://huggingface.co/BAAI/bge-m3)
- [FlagEmbedding GitHub](https://github.com/FlagOpen/FlagEmbedding)
- [BGE-M3 Paper](https://arxiv.org/abs/2402.03216)

