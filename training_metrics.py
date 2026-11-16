"""Record evaluation losses and export a headless, README-ready chart."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from tensorboardX import SummaryWriter


def plot_loss_curve(csv_path: str | Path, output_path: str | Path) -> Path:
    """Plot measured train/validation cross-entropy without smoothing."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No loss measurements in the CSV yet.")

    steps = [int(row["step"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    validation = [float(row["val_loss"]) for row in rows]

    figure = Figure(figsize=(10, 5.5), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(steps, train, color="#2563eb", linewidth=2.2, label="Train", marker="." if len(rows) < 3 else None)
    axis.plot(steps, validation, color="#ea580c", linewidth=2.2, label="Validation", marker="." if len(rows) < 3 else None)
    axis.set_title("MiniGPT learning curves", loc="left", fontsize=18, fontweight="bold", pad=16)
    axis.set_xlabel("Completed training steps")
    axis.set_ylabel("Cross-entropy loss (nats / character)")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.supxlabel(
        f"Final evaluation at step {steps[-1]:,}: train {train[-1]:.4f} | validation {validation[-1]:.4f}",
        fontsize=10,
        color="#475569",
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        figure.savefig(temporary, format=output.suffix.lstrip(".") or "png", dpi=180)
        temporary.replace(output)
    finally:
        figure.clear()
        temporary.unlink(missing_ok=True)
    return output


class TrainingLogger:
    """One run per directory; flush each evaluation so interrupted runs survive."""

    def __init__(self, run_dir: str | Path | None = None):
        self.run_dir = Path(run_dir) if run_dir is not None else Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.run_dir / "losses.csv"
        self.csv_file = self.csv_path.open("x", newline="", encoding="utf-8")
        try:
            self.writer = SummaryWriter(logdir=str(self.run_dir))
        except BaseException:
            self.csv_file.close()
            raise
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=["step", "train_loss", "val_loss"])
        self.csv_writer.writeheader()
        self.csv_file.flush()

    def log(self, step: int, losses: dict[str, float]) -> None:
        self.csv_writer.writerow({"step": step, "train_loss": losses["train"], "val_loss": losses["val"]})
        self.csv_file.flush()
        self.writer.add_scalar("Loss/train", losses["train"], step)
        self.writer.add_scalar("Loss/validation", losses["val"], step)
        self.writer.flush()
        plot_loss_curve(self.csv_path, self.run_dir / "loss_curve.png")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.writer.close()
        finally:
            self.csv_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a measured MiniGPT loss CSV to a PNG or SVG.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/loss_curve.png"))
    args = parser.parse_args()
    output = plot_loss_curve(args.csv_path, args.output)
    print(f"Saved loss curve: {output.resolve()}")


if __name__ == "__main__":
    main()
