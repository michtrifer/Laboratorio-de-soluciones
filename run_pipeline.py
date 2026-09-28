"""Punto de entrada único del proyecto (#21: reproducibilidad de punta a punta).

Ejemplos (desde la raíz del repo, en Git Bash o la terminal de PyCharm):

    python run_pipeline.py                     # todo desde cero (descarga incluida)
    python run_pipeline.py --skip-download     # usa los CSV que ya están en data/raw
    python run_pipeline.py --steps models      # solo re-entrena/evalúa modelos
    python run_pipeline.py --refit-every 5     # GARCH re-estimado cada 5 días (más rápido)
    python run_pipeline.py --synthetic         # prueba offline con datos simulados
"""
import argparse
import logging

from src import config
from src.pipeline import run


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", nargs="+", default=["data", "eda", "models"],
                   choices=["data", "eda", "models"])
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--refit-every", type=int, default=config.GARCH_REFIT_EVERY)
    p.add_argument("--egarch", action="store_true", help="incluir EGARCH (más lento, simulación)")
    p.add_argument("--no-lstm", action="store_true")
    p.add_argument("--no-banxico", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    run(steps=args.steps, download=not args.skip_download, synthetic=args.synthetic,
        refit_every=args.refit_every, run_egarch=args.egarch, run_lstm=not args.no_lstm,
        use_banxico=not args.no_banxico)


if __name__ == "__main__":
    main()
