# Danory GPT v8

A Persian-focused decoder-only language model with approximately 100M parameters. See [README_FA.md](README_FA.md) for complete Persian setup, dataset provenance/license notes, training commands, GitHub Actions, and checkpoint recovery.

Architecture: 99,922,560 parameters, 16 layers, 640 hidden size, 10 query heads / 2 KV heads, 24k BPE vocabulary, 2,048 context, GQA, RoPE, RMSNorm, SwiGLU, tied embeddings.

This is a fresh architecture: v7 checkpoints are not compatible. Public dataset ingestion is streaming and never reads local/private user data.
