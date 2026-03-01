# Tiny GPT From Scratch v2

A root-level, from-scratch character-level GPT notebook that trains on Tiny Shakespeare and supports:
- Jetson Orin Nano (CUDA)
- MacBook Air M4 (PyTorch MPS/Metal)

Main notebook: `tiny_gpt_from_scratch_v2.ipynb`

## Prerequisites

### Jetson (JetPack 6 + CUDA)
- Verify PyTorch can see CUDA:
  ```bash
  python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
  ```
- Optional monitor while training:
  ```bash
  tegrastats
  ```

### MacBook Air M4 (MPS)
- Verify MPS is available:
  ```bash
  python3 -c "import torch; print(torch.backends.mps.is_available())"
  ```

## Run

```bash
jupyter notebook tiny_gpt_from_scratch_v2.ipynb
```

or

```bash
jupyter lab tiny_gpt_from_scratch_v2.ipynb
```

## Train Once, Then Chat

1. Run the notebook training sections through **Save Checkpoint** to produce `gpt_char_ckpt.pt`.
2. Restart kernel.
3. Run only the **ONE-CELL: Load + Chat** cell.
4. Chat immediately from terminal-style input in that cell.

## Sample Prompts

Use plain ASCII when possible (no emojis/special symbols) to avoid vocab errors.

- `User: Who are you?\nAssistant:`
- `User: Explain gravity in simple words.\nAssistant:`
- `User: Write a short poem about rain.\nAssistant:`
- `User: Give me three tips to learn Python.\nAssistant:`
- `KING RICHARD III: What news?\nMESSENGER:`
- `HAMLET: To be, or not to be?\nHORATIO:`
- `ROMEO: But, soft! what light through yonder window breaks?\nJULIET:`

## Troubleshooting

- Missing checkpoint (`gpt_char_ckpt.pt`):
  - Run training + save checkpoint cells first.
  - Confirm notebook working directory matches the checkpoint location.

- `KeyError` in `encode`:
  - Your prompt contains characters not present in training vocab.
  - Use simpler ASCII prompts, avoid emojis/special chars, or retrain with broader text.

- Out of memory (OOM):
  - Reduce in order: `batch_size` -> `block_size` -> `d_model`.
  - Restart kernel after changing memory-heavy settings.
