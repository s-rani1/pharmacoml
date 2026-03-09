# pharmacoml

**pharmacoml is a benchmark-backed hybrid AI/ML covariate screening toolkit for population PK/PD, combining explainable ML discovery, penalized confirmation, and SCM-style bridging in an estimation-tool-agnostic Python workflow.**

## What It Is

`pharmacoml` helps pharmacometricians use a **hybrid AI/ML screening workflow**
to identify and prioritize likely covariates from subject-level EBEs or
individual parameters before formal model confirmation. It is designed to work
with outputs from `NONMEM`, `nlmixr2`, `Monolix`, `Pumas`, or similar
mixed-effects workflows.

The current release is evaluated against a fixed public benchmark suite that
includes real public PK examples and paper-style benchmark scenarios.

## What It Is Not

`pharmacoml` is **not** a replacement for final NLME estimation, full model
search, or pharmacometric confirmation in the current release. It is a
**hybrid AI/ML covariate screening and preselection** tool designed to reduce
search space before SCM, backward elimination, or final model fitting.

## Why It Is Different

- Uses a hybrid AI/ML screening workflow that combines explainable ML discovery, penalized confirmation, and SCM-style bridging instead of relying on a single method.
- Works with EBEs or individual parameters from any solver, including `NONMEM`, `nlmixr2`, `Monolix`, and `Pumas`, so screening is not tied to a single estimation engine.
- Supports many screening backends, including explainable boosting, `AALASSO`, `STG`, and an SCM-style bridge, rather than relying on a single screening model.
- Includes pharmacometric screening features such as shrinkage-aware logic, biology-aware proxy preservation, and optional interaction screening.
- Ships with a public benchmark suite, pinned baselines, and generated benchmark reports so workflow changes can be evaluated against fixed reference cases.

## Installation

From PyPI:

```bash
pip install pharmacoml
```

For development:

```bash
git clone https://github.com/s-rani1/pharmacoml.git
cd pharmacoml
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[dev,dl,symbolic]"
```

## Quick Start

```python
import pandas as pd
from pharmacoml.covselect import HybridScreener

ebes = pd.read_csv("individual_parameters.csv")
covariates = pd.read_csv("covariates.csv")

report = HybridScreener(
    include_scm=True,
).fit(
    ebes=ebes,
    covariates=covariates,
    parameter_shrinkage={"CL": 0.12, "V": 0.28},
)

report.confirmed_covariates()   # recommended daily-use answer
report.candidate_covariates()   # shortlist to carry forward
report.core_covariates()        # strongest ML-supported signals
report.proxy_groups()           # correlated alternatives
print(report.to_nonmem_candidates())
```

Example `confirmed_covariates()` output:

```text
  parameter covariate functional_form confirmation_status
0        CL        WT           power                 scm
1         V        WT           power                 scm
```

## Typical Outputs

- `confirmed_covariates()`: compact answer after SCM-style confirmation
- `candidate_covariates()`: practical shortlist for downstream PMx confirmation
- `core_covariates()`: strongest ML-supported signals
- `proxy_groups()`: correlated or overlapping covariate groups
- `interaction_covariates()`: screened interactions when enabled
- `to_nonmem_candidates()`: export-ready candidate block for downstream workflows

## Benchmarks

`pharmacoml` includes a fixed public benchmark suite for release calibration:

- `pheno` (Pharmpy phenobarbital example)
- `Eleveld/Wahlquist` public propofol data
- `ggPMX` Monolix theophylline example
- `Asiimwe-style` correlated-covariate simulation
- `Shap-Cov-style` collinear simulation
- optional `Kekic` public synthetic scenarios when available locally

Run the benchmark suite:

```bash
PYTHONPATH=. python benchmarks/run_public_benchmarks.py --check
```

That command generates a reusable report bundle under
`benchmarks/reports/fixed_public/` by default:

- `public_benchmark_report.md`
- `public_benchmark_summary.csv`
- `public_benchmark_details.csv`
- `public_benchmark_report.json`

Use `--no-report` to skip artifact generation, or `--report-dir <path>` to
write the bundle somewhere else.

## Experimental Consensus

For advanced benchmarking and model-family comparison, the experimental
namespace exposes a curated multi-model consensus workflow:

```python
from pharmacoml.covselect.experimental import MultiModelConsensusScreener

report = MultiModelConsensusScreener(
    top_k=3,
    n_bootstrap=8,
    include_neural=False,
).fit(ebes, covariates)

report.consensus_covariates()
report.selection_frequency_table()
report.compare_with_hybrid(ebes, covariates)
```

## Documentation

Static docs pages live in `docs/` and are suitable for GitHub Pages:

- [Overview](docs/index.html)
- [Tutorial](docs/tutorial.html)
- [Benchmarks](docs/benchmarks.html)

## Methodological References

The default hybrid workflow implements and combines approaches described in
recent pharmacometric ML literature on covariate screening, including
Sibieude et al. (2021), Asiimwe et al. (2024), Brooks et al. (2025), Karlsen
et al. (2025), and Kekic et al. (2026).
The broader package also includes additional experimental screening and
benchmarking capabilities.

## How to Cite

If you use `pharmacoml` in your work, please cite the software repository.
GitHub will also expose citation metadata directly via the repository citation
panel.

Suggested citation:

```text
Rani S. pharmacoml: Benchmark-backed hybrid AI/ML covariate screening toolkit
for population PK/PD. Version 0.1.1. GitHub.
https://github.com/s-rani1/pharmacoml
```

When relevant, also cite the methodological papers that informed the workflow,
especially Sibieude et al. (2021), Asiimwe et al. (2024), Brooks et al. (2025),
Karlsen et al. (2025), and Kekic et al. (2026).

## Roadmap

Potential future expansion includes:

- backend integration for formal model-confirmation workflows such as `nlmixr2` and `NONMEM`
- estimation-driven SCM and backward elimination
- simulation and reporting layers for broader pharmacometric workflows
- possible R integration paths via subprocess-based execution or `rpy2`

## License

MIT
