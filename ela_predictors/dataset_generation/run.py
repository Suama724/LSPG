from __future__ import annotations

from total_files.config import config_ela_data_pipeline
from pipeline.run_generate import run_generate


def main():
    resume = True
    final_cfg = config_ela_data_pipeline
    run_dir = run_generate(final_cfg, resume=resume)
    print(f"generation finished: {run_dir}")


if __name__ == "__main__":
    main()
