# Models

`pet-mad-xs-v1.5.0.pt` (~20 MB) is the PET-MAD extra-small TorchScript export
from `lab-cosmo/upet`. Hugging Face remains the source of truth; S/M
checkpoints are 110–450 MB and stay on the Hub.

`soap-bpnn-tiny.ckpt` (~76 KB) is a metatrain `soap_bpnn` checkpoint with the
tiny hypers used by the C++ twin (Laplacian eigenstates, `max_angular=1`,
`max_radial=2`, one SiLU layer of 8 neurons). The gallery loads it through
`metatrain.utils.io.load_model`. If the file is missing, `_bpnn.ensure_soap_checkpoint`
fetches from Hugging Face when `OPENMM_METATOMIC_SOAP_HF` is set, otherwise it
writes a matching checkpoint from `SoapBpnn.get_checkpoint()`.

```bash
# regenerate PET-MAD (metatrain 2026.3.1 or 2026.4 — not 2026.5.dev)
python -c "import upet; upet.save_upet(model='pet-mad', size='xs', version='1.5.0', output='models/pet-mad-xs-v1.5.0.pt')"

# regenerate the SOAP-BPNN checkpoint
python examples/_bpnn.py --emit openmmapi/include/openmmmetatomic/internal/BpnnWeights.h
```

The PET-MAD gallery example does not download unless the vendored `.pt` is missing.
SOAP-BPNN can download a `.ckpt` from the Hub; execution still needs neighbor lists
(vesin, via metatrain).
