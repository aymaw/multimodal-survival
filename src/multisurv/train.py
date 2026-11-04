import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .data import generate, split
from .metrics import bootstrap_cindex, concordance, ipcw_brier
from .model import FusionSurvival, discrete_nll, survival


def run(n, epochs, seeds, modes, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cohort = generate(n)
    data, manifest = split(cohort)
    (out / "split.json").write_text(json.dumps(manifest, indent=2))
    report = {
        "synthetic_only": True,
        "data_sha256": cohort.fingerprint(),
        "n": n,
        "epochs": epochs,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "censor_fraction": float((~cohort.event).float().mean()),
        "runs": [],
    }
    for seed in seeds:
        for mode in modes:
            started = time.perf_counter()
            torch.manual_seed(seed)
            model = FusionSurvival(mode)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
            best, checkpoint, history = float("inf"), None, []
            for epoch in range(epochs):
                model.train()
                order = torch.randperm(len(data["train"].time))
                losses = []
                for batch in order.split(96):
                    rows = data["train"].subset(batch)
                    loss = discrete_nll(model(rows), rows.time, rows.event)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(loss.item())
                model.eval()
                with torch.no_grad():
                    val = discrete_nll(
                        model(data["validation"]), data["validation"].time, data["validation"].event
                    ).item()
                history.append(
                    {"epoch": epoch + 1, "train_nll": float(np.mean(losses)), "val_nll": val}
                )
                if val < best:
                    best, checkpoint = val, copy.deepcopy(model.state_dict())
            model.load_state_dict(checkpoint)
            model.eval()
            with torch.no_grad():
                logits = model(data["test"])
                curves = survival(logits).numpy()
                test_nll = discrete_nll(logits, data["test"].time, data["test"].event).item()
            t, e = data["test"].time.numpy(), data["test"].event.numpy()
            risk = -curves.sum(1)  # negative restricted mean survival over 12 bins
            run_result = {
                "seed": seed,
                "mode": mode,
                "best_val_nll": best,
                "test_nll": test_nll,
                "c_index": concordance(t, e, risk),
                "c_index_bootstrap_95": bootstrap_cindex(t, e, risk),
                "brier": {
                    str(h): ipcw_brier(
                        data["train"].time, data["train"].event, t, e, curves[:, h], h
                    )
                    for h in (3, 7, 9)
                },
                "seconds": time.perf_counter() - started,
                "history": history,
            }
            report["runs"].append(run_result)
            stem = f"{mode}-{seed}"
            torch.save({"state_dict": checkpoint, "mode": mode}, out / f"{stem}.pt")
            np.savez_compressed(out / f"{stem}-predictions.npz", survival=curves, time=t, event=e)
            print(f"{stem}: c-index={run_result['c_index']:.3f}, NLL={test_nll:.3f}", flush=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 31])
    parser.add_argument("--modes", nargs="+", default=["tabular", "image", "sequence", "fusion"])
    parser.add_argument("--out", default="runs/reference")
    args = parser.parse_args()
    run(args.n, args.epochs, args.seeds, args.modes, args.out)
