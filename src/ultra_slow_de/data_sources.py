from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceRecord:
    name: str
    kind: str
    source_url: str
    version: str
    license: str
    provenance: str
    checksum: str | None = None


def built_in_sources() -> dict[str, SourceRecord]:
    return {
        "pantheon_plus": SourceRecord(
            name="Pantheon+ SN Ia",
            kind="sn",
            source_url="https://doi.org/10.17909/T95Q4X",
            version="v1 (2022)",
            license="Public; Scolnic et al. 2022, ApJ, 938, 113",
            provenance=(
                "Pantheon+ release: 1701 SN Ia light-curve distances. "
                "GitHub: github.com/PantheonPlusSH0ES/DataRelease"
            ),
        ),
        "desi_bao": SourceRecord(
            name="DESI BAO DR2",
            kind="bao",
            source_url="https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/README.html",
            version="DR2 / Year-3 BAO (2025)",
            license="Public; DESI Collaboration 2025, Phys. Rev. D 112, 083515",
            provenance=(
                "DESI DR2 BAO cosmology catalog (Y3 paper series), "
                "including low-z galaxy BAO plus Ly-alpha BAO companion constraints. "
                "Numerical likelihood assets follow Cobaya desi_bao_dr2 files."
            ),
        ),
        "eboss_bao_rsd": SourceRecord(
            name="eBOSS DR16 BAO/RSD",
            kind="rsd",
            source_url="https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/",
            version="DR16 v1_0_1 (2020)",
            license="Public; Alam et al. 2021, PRD, 103, 083533",
            provenance=(
                "SDSS-IV eBOSS final BAO+RSD consensus. "
                "Includes LRG, ELG, QSO tracers."
            ),
        ),
        "planck_cmb": SourceRecord(
            name="Planck 2018 likelihood",
            kind="cmb",
            source_url="https://pla.esac.esa.int/",
            version="PR3 / 2018",
            license="Public; Planck Collaboration 2020, A&A, 641, A5",
            provenance=(
                "Planck PR3 likelihood products (plik TTTEEE+lowl+lowE). "
                "Baseline compressed distance priors usable for "
                "background-level analysis."
            ),
        ),
    }


def acquire_dataset(dataset_id: str, cache_dir: str | Path) -> Path:
    """Offline-friendly acquisition guard.

    This function does not download data. It checks whether expected local assets
    exist and raises a helpful error with provenance metadata otherwise.
    """
    sources = built_in_sources()
    if dataset_id not in sources:
        raise KeyError(f"Unknown dataset id: {dataset_id}")

    target = Path(cache_dir) / dataset_id
    if target.exists():
        return target

    src = sources[dataset_id]
    raise FileNotFoundError(
        "Dataset not found locally. "
        f"id={dataset_id}, url={src.source_url}, version={src.version}, "
        f"license={src.license}, provenance={src.provenance}"
    )