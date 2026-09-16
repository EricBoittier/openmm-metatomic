# PET-MAD-XS (lab-cosmo/upet)

`pet-mad-xs-v1.5.0.pt` (~20 MB) is the only model file in git. Hugging Face
remains the source of truth; S/M checkpoints are 110–450 MB and stay on the
Hub.

```bash
# regenerate (metatrain 2026.3.1 or 2026.4 — not 2026.5.dev)
python -c "import upet; upet.save_upet(model='pet-mad', size='xs', version='1.5.0', output='models/pet-mad-xs-v1.5.0.pt')"
```

The gallery loads this file and does not download or convert unless it is missing.
