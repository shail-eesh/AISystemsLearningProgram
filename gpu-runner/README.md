# GPU runner — the RTX 4070 lane

Scripts here are the parts of the course that need a real GPU. They are written
in the cloud sandbox, verified there on CPU at reduced scale, and *run by you* on
the 4070. Each writes a `results.json` back into its topic's `bench/` directory,
which is what flips that topic's ledger row from 🖥️ *awaiting-4070* to ✅.

The contract every runner follows:

- **It is the same code.** A runner imports the topic's own modules; it does not
  reimplement anything. If a run on the 4070 disagrees with the CPU verification,
  that is a finding about the hardware or the config, never about two codebases
  drifting apart.
- **It is restartable.** Every runner checkpoints and resumes, because a run that
  cannot survive a closed laptop is a run you will not finish.
- **It states its budget up front** — VRAM, wall clock, disk — and refuses to
  start if the device cannot meet it, rather than dying at hour four with an OOM.
- **It writes `results.json` in the same shape as the CPU bench**, so the two are
  directly comparable.

## Setup (once)

```bash
git clone https://github.com/shail-eesh/AISystemsLearningProgram.git
cd AISystemsLearningProgram
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[torch,dev]'          # a CUDA build of torch, on this machine
python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"
```

## Runners

| script | topic | what it does | budget |
|:--|:--|:--|:--|
| `t15_alphaslm_40m.py` | T15 | pretrains AlphaSLM-15M and AlphaSLM-40M on the packed FinTok shards | ~6 GB VRAM, a few hours |

```bash
python3 gpu-runner/t15_alphaslm_40m.py --rung alphaslm-40m           # the overnight run
python3 gpu-runner/t15_alphaslm_40m.py --rung alphaslm-15m --hours 1 # a shorter first pass
python3 gpu-runner/t15_alphaslm_40m.py --rung alphaslm-40m --resume  # after an interruption
python3 gpu-runner/t15_alphaslm_40m.py --dry-run                     # plan only, no training
```

Copy the printed `results.json` path into the topic's `bench/` folder (the script
offers to do it) and commit — the ledger regenerates from `EXECUTION/status.json`.
