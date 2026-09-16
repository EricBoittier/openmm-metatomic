# PET-MAD (lab-cosmo/upet)

Checkpoints and exported TorchScript files are downloaded / converted locally.
They are not committed.

```bash
hf download lab-cosmo/upet models/pet-mad-xs-v1.5.0.ckpt --local-dir models/hf-upet
# metatrain 2026.3.1 or 2026.4 — not the 2026.5.dev dipole-head regression
mtt export models/hf-upet/models/pet-mad-xs-v1.5.0.ckpt -o models/pet-mad-xs-v1.5.0.converted.pt
python -c "import upet; upet.save_upet(model='pet-mad', size='xs', version='1.5.0', output='models/pet-mad-xs-v1.5.0.pt')"
```

The gallery example `plot_05_petmad.py` runs those steps if the files are missing.
