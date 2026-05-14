"""ace_xs.py
Author: Emil Mammadzada

GitHub:
https://github.com/emilmammadzada99

Contact:
emilmemmedzade23@outlook.com
egoemil32@gmail.com

Reads nuclear data files in ACE format and retrieves cross-section (XS) values.


Supported data types:
  - Standard neutron data (continuous-energy)
  - Dosimetry data (XS_TYPE_DOSIMETRY)
  - Thermal induced scattering - S(a,b) (XS_TYPE_SAB)

Usage example:
    from ace_xs import AceLibrary, MaterialXS

    lib = AceLibrary()
    lib.read_ace_file("U235.710nc", "92235.710nc")

    mat_xs = MaterialXS(lib)
    mat_xs.add_isotope("92235.710nc", atom_density=0.02)
    mat_xs.build()

    sigma_total = mat_xs.get_total_xs(energy_eV=1.0)"""

import numpy as np
import struct
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ────────────────────────────────────── ─────────────────── ────────────────────
# Constants (equivalent to #define in C code)
# ────────────────────────────────────── ─────────────────── ────────────────────

XS_TYPE_NEUTRON   = 0   # Standard neutron data
XS_TYPE_DOSIMETRY = 1   # Dosimetry
XS_TYPE_SAB       = 2   # Bound S(a,b)
XS_TYPE_DECAY     = 3   # Degradation data (not readable)

# Material reaction types (materialxs.c → MT constants)
MT_TOTAL    = 1     # Total cross section
MT_FISSION  = 18   # fission
MT_FISSE    = 901  # Fission energy production (kappa-fission)
MT_NSF      = 452  # Nu*Sigma_f
MT_ELASTIC  = 2    # Elastic scattering
MT_N2N      = 16   # (n,2n)
MT_CAPTURE  = 102  # Radiative capture (absorption)

CACHE_MODE = 0


# ────────────────────────────────────── ─────────────────── ────────────────────
# Data structures
# ────────────────────────────────────── ─────────────────── ────────────────────

@dataclass
class AceReaction:
    """Cross-sectional data of a single reaction channel.
    Corresponds to the REA_BLOCK structure in C."""
    mt:       int             # Reaction MT number
    Q:        float = 0.0    # Q-value (MeV)
    awr:      float = 0.0    # Atomic weight ratio
    T:        float = 0.0    # Temperature (K)
    adens:    float = 1.0    # Atomic density (atom/barn·cm)
    emin:     float = 0.0    # Lower energy limit (MeV)
    emax:     float = 1e10   # Upper energy limit (MeV)
    p0:       int   = 0      # First point index in the energy grid
    energy:   Optional[np.ndarray] = None   # Energy points (MeV)
    xs:       Optional[np.ndarray] = None   # Section values ​​(barn)
    # Cache (getxs.c → REA_IE0 / REA_XS0)
    _cache_E: float = field(default=-1.0, repr=False)
    _cache_xs: float = field(default=-1.0, repr=False)


@dataclass
class AceIsotope:
    """ACE data of a single isotope.
    It corresponds to the XSDATA + ACE block structure in C."""
    zaid:    str                          # Ex. "92235.710nc"
    awr:     float = 1.0                  # Atomic weight ratio
    T:       float = 293.6               # Temperature (K)
    xs_type: int   = XS_TYPE_NEUTRON

    # NXS/JXS sequences (ACE format metadata)
    NXS: np.ndarray = field(default_factory=lambda: np.zeros(16))
    JXS: np.ndarray = field(default_factory=lambda: np.zeros(32))

    # Raw XSS string
    XSS: Optional[np.ndarray] = None

    # Processed reaction channels
    reactions: Dict[int, AceReaction] = field(default_factory=dict)

    # Unified energy grid (unionized grid)
    energy_grid: Optional[np.ndarray] = None


@dataclass
class MaterialXSData:
    """Table of macroscopic sections of a material.
    Corresponds to the output of materialxs.c in C."""
    name:   str = "unnamed"
    T:      float = 293.6
    adens:  float = 1.0        # Total atomic density
    emin:   float = 1e-11      # MeV
    emax:   float = 20.0       # MeV

    energy_grid: Optional[np.ndarray] = None   # unified energy grid

    # Arrays of macroscopic sections (barn·atom/barn·cm = 1/cm)
    sigma_total:    Optional[np.ndarray] = None
    sigma_fission:  Optional[np.ndarray] = None
    sigma_capture:  Optional[np.ndarray] = None
    sigma_elastic:  Optional[np.ndarray] = None
    sigma_n2n:      Optional[np.ndarray] = None
    nu_sigma_f:     Optional[np.ndarray] = None    # ν·Σ_f

    # Isotope list: (AceIsotope, atom_fraction)
    isotopes: List[Tuple['AceIsotope', float]] = field(default_factory=list)


# ────────────────────────────────────── ─────────────────── ────────────────────
#  AceLibrary
# ────────────────────────────────────── ─────────────────── ────────────────────

class AceLibrary:
    """Reads ACE files and returns AceIsotope objects.
    It implements readacefiles.c logic in Python."""

    def __init__(self):
        self._isotopes: Dict[str, AceIsotope] = {}

    # ── General reading ───────────────────────────── ─────────────────────────────

    def read_ace_file(self, filepath: str, target_zaid: str) -> AceIsotope:
        """Reads data for the specified ZAID from the ACE file.

        parameters
        ----------
        filepath : path to the ACE file
        target_zaid : The desired isotope ZAID code (e.g. "92235.710nc")

        returns
        -------
        AceIsotope: Filled isotope object"""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"ACE not find!!!: {filepath}")

        print(f"ACE reading: {filepath}  ->  {target_zaid} ...")

        iso = None
        with open(filepath, 'r', errors='replace') as fp:
            iso = self._parse_ace_text(fp, target_zaid)

        if iso is None:
            raise ValueError(
                f"{target_zaid} izotopu {filepath} file not find!!!"
            )

        self._isotopes[target_zaid] = iso
        print(f"  Okay: {target_zaid}  AWR={iso.awr:.4f}  T={iso.T:.1f} K")
        return iso

    def get(self, zaid: str) -> Optional[AceIsotope]:
        return self._isotopes.get(zaid)

    # ── ACE text format parser ──────────────────── ────────────────────

    def _parse_ace_text(self, fp, target_zaid: str) -> Optional[AceIsotope]:
        """Parses the text-based ACE file line by line."""
        def _read_tokens_from_lines(lines):
            for line in lines:
                for tok in line.split():
                    yield tok

        lines = fp.readlines()
        if len(lines) < 12:
            return None

        header = lines[0].split()
        if len(header) < 3:
            return None

        hz = header[0]
        if hz != target_zaid:
            return None

        try:
            awr = float(header[1])
            T = float(header[2])
        except ValueError:
            return None

        # Line 2 is the comment line; is skipped directly.
        # Rows 3-6: 16 IZ-AW pairs, Rows 7-8: NXS, Rows 9-12: JXS
        data_tokens = _read_tokens_from_lines(lines[2:])

        # 16 IZ-AW pairs = 32 tokens
        for _ in range(32):
            try:
                next(data_tokens)
            except StopIteration:
                return None

        NXS = np.zeros(16, dtype=int)
        for i in range(16):
            try:
                NXS[i] = int(float(next(data_tokens)))
            except (StopIteration, ValueError):
                return None

        sz = NXS[0]
        if sz <= 0:
            return None

        JXS = np.zeros(32, dtype=int)
        for i in range(32):
            try:
                JXS[i] = int(float(next(data_tokens)))
            except (StopIteration, ValueError):
                return None

        XSS = np.zeros(sz)
        for i in range(sz):
            try:
                XSS[i] = float(next(data_tokens))
            except (StopIteration, ValueError):
                return None

        return AceIsotope(
            zaid=hz, awr=awr, T=T,
            NXS=NXS, JXS=JXS, XSS=XSS,
        )


# ────────────────────────────────────── ─────────────────── ────────────────────
#  AceProcessor
# ────────────────────────────────────── ─────────────────── ────────────────────

class AceProcessor:
    """It processes the raw ACE data (XSS sequence) and separates it into reaction channels.
    It implements processxsdata.c logic in Python.

    Supported types:
      - Standard neutron (continuous-energy)
      - Dosimetry"""

    def __init__(self, main_energy_grid: np.ndarray):
        """parameters
        ----------
        main_energy_grid : Unionized energy grid (in MeV)."""
        self.energy_grid = main_energy_grid

    # ── Main processing function ──────────────────────── ────────────────────────

    def process(self, iso: AceIsotope) -> AceIsotope:
        """Populates the iso.reactions dictionary by parsing the isotope's XSS string."""
        if iso.xs_type == XS_TYPE_DOSIMETRY:
            self._process_dosimetry(iso)
        elif iso.xs_type == XS_TYPE_SAB:
            self._process_sab(iso)
        else:
            self._process_neutron(iso)

        iso.energy_grid = self.energy_grid
        return iso

    # ── Standard neutron data ─────────────────────── ────────────────────────

    def _process_neutron(self, iso: AceIsotope):
        """Standard ACE processes neutron data.
        processxsdata.c → else branch (standard type) logic."""
        XSS = iso.XSS
        JXS = iso.JXS
        NXS = iso.NXS

        # Number of energy points (NXS[2])
        NES  = NXS[2]
        # Start of ESZ block (JXS[0] → 0-based)
        esz0 = JXS[0] - 1

        if NES <= 0 or esz0 < 0 or esz0 + NES > len(XSS):
            return   # invalid data

        # Isotope's own energy grid
        iso_energy = XSS[esz0 : esz0 + NES]           # MeV

        # Total cross section (MT=1) — ESZ block, 2nd column
        sig_total = XSS[esz0 + NES : esz0 + 2*NES]

        # Absorption cross section — 3rd column
        sig_abs   = XSS[esz0 + 2*NES : esz0 + 3*NES]

        # Elastic scattering — column 4
        sig_ela   = XSS[esz0 + 3*NES : esz0 + 4*NES]

        # Interpolation to composite grid
        eg = self.energy_grid
        mask = (eg >= iso_energy[0]) & (eg <= iso_energy[-1])

        def _interp(xs_raw):
            result = np.zeros(len(eg))
            result[mask] = np.interp(eg[mask], iso_energy, xs_raw)
            return result

        # Total
        rea_total = AceReaction(
            mt=MT_TOTAL, awr=iso.awr, T=iso.T,
            emin=iso_energy[0], emax=iso_energy[-1],
            energy=eg, xs=_interp(sig_total),
        )
        iso.reactions[MT_TOTAL] = rea_total

        # Absorption (capture)
        rea_cap = AceReaction(
            mt=MT_CAPTURE, awr=iso.awr, T=iso.T,
            emin=iso_energy[0], emax=iso_energy[-1],
            energy=eg, xs=_interp(sig_abs),
        )
        iso.reactions[MT_CAPTURE] = rea_cap

        # Elastic scattering
        rea_ela = AceReaction(
            mt=MT_ELASTIC, awr=iso.awr, T=iso.T,
            emin=iso_energy[0], emax=iso_energy[-1],
            energy=eg, xs=_interp(sig_ela),
        )
        iso.reactions[MT_ELASTIC] = rea_ela

        # ── Fission (optional) ─────────────────────── ───────────────────────
        # Nu*Sigma_f: ESZ 5th column (if JXS[1] > 0)
        nubar_ptr = JXS[1] - 1   # NU block
        fiss_interp = None
        if nubar_ptr >= 0:
            sig_fiss_ptr = esz0 + 4*NES
            if sig_fiss_ptr + NES <= len(XSS):
                sig_fiss = XSS[sig_fiss_ptr : sig_fiss_ptr + NES]
                fiss_interp = _interp(sig_fiss)
                rea_fiss = AceReaction(
                    mt=MT_FISSION, awr=iso.awr, T=iso.T,
                    emin=iso_energy[0], emax=iso_energy[-1],
                    energy=eg, xs=fiss_interp,
                )
                iso.reactions[MT_FISSION] = rea_fiss

        # If MT=452 is not in ACE, generate it with tabulated nu(E) from NU block.
        if MT_NSF not in iso.reactions and fiss_interp is not None and nubar_ptr >= 0:
            nu_times_f = None

            # Common layout in ENDF/B-VII ACE files:
            # [LNU, NR, ..., NP, E(1..NP), NU(1..NP)] (LNU < 0)
            if nubar_ptr + 4 < len(XSS):
                lnu = XSS[nubar_ptr]
                if lnu < 0.0:
                    try:
                        np_nu = int(XSS[nubar_ptr + 3])
                    except (TypeError, ValueError):
                        np_nu = 0

                    start = nubar_ptr + 4
                    stop_e = start + np_nu
                    stop_nu = stop_e + np_nu

                    if (
                        np_nu > 1
                        and stop_nu <= len(XSS)
                    ):
                        e_nu = XSS[start:stop_e]
                        nu_vals = XSS[stop_e:stop_nu]
                        if np.all(np.diff(e_nu) > 0.0):
                            nu_interp = np.zeros(len(eg))
                            m = (eg >= e_nu[0]) & (eg <= e_nu[-1])
                            nu_interp[m] = np.interp(eg[m], e_nu, nu_vals)
                            nu_times_f = nu_interp * fiss_interp

            # Last resort: approximate the typical value of U-235 in the thermal zone.
            if nu_times_f is None:
                nu_times_f = 2.43 * fiss_interp

            rea_nsf = AceReaction(
                mt=MT_NSF, awr=iso.awr, T=iso.T,
                emin=iso_energy[0], emax=iso_energy[-1],
                energy=eg, xs=nu_times_f,
            )
            iso.reactions[MT_NSF] = rea_nsf

        # ── Partial reactions (NTR units) ──────────────────────────────────
        NTR  = NXS[3]    # number of reactions
        if NTR > 0 and JXS[2] > 0:
            mtr_ptr = JXS[2] - 1   # MTR block (MT list)
            lsig_ptr = JXS[5] - 1  # LSIG block (section offsets)
            sig_ptr  = JXS[6] - 1  # SIG block (section data)

            for nr in range(NTR):
                if mtr_ptr + nr >= len(XSS):
                    break
                mt = int(XSS[mtr_ptr + nr])
                if lsig_ptr + nr >= len(XSS):
                    break
                L0 = int(XSS[lsig_ptr + nr]) + sig_ptr - 1
                if L0 < 0 or L0 + 2 >= len(XSS):
                    continue

                IE  = int(XSS[L0])       # First energy point index (based 1)
                NP  = int(XSS[L0 + 1])  # Number of points

                p0 = IE - 1             # 0-based

                if p0 < 0 or p0 + NP > NES:
                    continue

                xs_partial = XSS[L0 + 2 : L0 + 2 + NP]
                eg_partial  = iso_energy[p0 : p0 + NP]

                xs_interp = np.zeros(len(eg))
                m = (eg >= eg_partial[0]) & (eg <= eg_partial[-1])
                xs_interp[m] = np.interp(eg[m], eg_partial, xs_partial)

                rea = AceReaction(
                    mt=mt, awr=iso.awr, T=iso.T,
                    emin=eg_partial[0], emax=eg_partial[-1],
                    p0=p0,
                    energy=eg, xs=xs_interp,
                )
                iso.reactions[mt] = rea

    # ── Dosimetry data ────────────────────────── ───────────────────────────

    def _process_dosimetry(self, iso: AceIsotope):
        """Dosimetry processes ACE data.
        processxsdata.c → XS_TYPE_DOSIMETRY branch."""
        XSS = iso.XSS
        JXS = iso.JXS
        NXS = iso.NXS
        NTR = NXS[3]
        eg  = self.energy_grid

        for nr in range(NTR):
            mt = int(XSS[JXS[2] - 1 + nr])
            L0 = int(XSS[JXS[5] - 1 + nr]) + JXS[6] - 2

            # Non-linear interpolation warning
            if int(XSS[L0]) > 0:
                print(f"  Warning: Non-lineer interpolation  ({iso.zaid} MT={mt})")

            NES = int(XSS[L0 + 1])
            if NES <= 0:
                continue

            e_raw  = XSS[L0 + 2          : L0 + 2 + NES]
            xs_raw = XSS[L0 + 2 + NES    : L0 + 2 + 2*NES]

            if len(e_raw) == 0 or e_raw[0] >= eg[-1]:
                continue

            xs_interp = np.zeros(len(eg))
            m = (eg >= e_raw[0]) & (eg <= e_raw[-1])
            xs_interp[m] = np.interp(eg[m], e_raw, xs_raw)

            rea = AceReaction(
                mt=mt, awr=iso.awr, T=iso.T,
                emin=e_raw[0], emax=e_raw[-1],
                energy=eg, xs=xs_interp,
            )
            iso.reactions[mt] = rea

    # ── S(a,b) thermal scattering ──────────────────────── ────────────────────────

    def _process_sab(self, iso: AceIsotope):
        """Bound processes data S(a,b) (simplified).
        processxsdata.c → XS_TYPE_SAB branch.
        It only extracts the inelastic total cross section (MT=1007)."""
        XSS = iso.XSS
        JXS = iso.JXS
        eg  = self.energy_grid

        # Inelastic thermal scattering (JXS[1]: ITIE pointer)
        if JXS[1] <= 0:
            return

        ptr = JXS[1] - 1
        NES = int(XSS[ptr])
        if NES <= 0 or ptr + 1 + 2*NES > len(XSS):
            return

        e_raw  = XSS[ptr + 1       : ptr + 1 + NES]
        xs_raw = XSS[ptr + 1 + NES : ptr + 1 + 2*NES]

        xs_interp = np.zeros(len(eg))
        m = (eg >= e_raw[0]) & (eg <= e_raw[-1])
        xs_interp[m] = np.interp(eg[m], e_raw, xs_raw)

        rea = AceReaction(
            mt=1007, awr=iso.awr, T=iso.T,
            emin=e_raw[0], emax=e_raw[-1],
            energy=eg, xs=xs_interp,
        )
        iso.reactions[1007] = rea


# ────────────────────────────────────── ─────────────────── ────────────────────
#  interpolate_xs
# ────────────────────────────────────── ─────────────────── ────────────────────

def interpolate_xs(reaction: AceReaction, E: float) -> float:
    """Linearly calculate the cross section of the reaction at a given energy point (MeV).
    Calculates by interpolation.

    parameters
    ----------
    reaction : AceReaction
    E : Neutron energy (MeV)

    returns
    -------
    float : Macroscopic section value (1/cm) = microscopic (barn) × adensity"""
    # ── Cache control (equivalent to REA_IE0 / REA_XS0) ────────────────────
    if E == reaction._cache_E and reaction._cache_E >= 0:
        return reaction._cache_xs * reaction.adens

    # out of energy range
    if reaction.energy is None or len(reaction.energy) == 0:
        return 0.0
    if E < reaction.energy[0] or E > reaction.energy[-1]:
        return 0.0

    # ── Linear interpolation ──────────────────────── ─────────────────────────
    xs = float(np.interp(E, reaction.energy, reaction.xs))

    # Save to cache
    reaction._cache_E  = E
    reaction._cache_xs = xs

    return xs * reaction.adens


# ────────────────────────────────────── ─────────────────── ────────────────────
# MaterialXS
# ────────────────────────────────────── ─────────────────── ────────────────────

class MaterialXS:
    """Creates macroscopic cross-section tables of a material.
    It implements materialxs.c logic in Python.

    Features:
      - Total (sigma_total)
      - Fission (sigma_fission)
      - Fission energy production (sigma_fisse)
      - Nu*Sigma_f (nu_sigma_f)
      - Elastic scattering (sigma_elastic)
      - (n,2n) (sigma_n2n)
      - Absorption / capture (sigma_capture)
      - Majorant (maximum total) (sigma_majorant)"""

    def __init__(self, library: AceLibrary, name: str = "mat"):
        self.library  = library
        self.name     = name
        self._entries: List[Tuple[str, float]] = []   # (zaid, atom_density)
        self.data: Optional[MaterialXSData] = None

    # ── Adding isotopes ──────────────────────────── ────────────────────────────

    def add_isotope(self, zaid: str, atom_density: float):
        """Adds isotopes to the material.

        parameters
        ----------
        zaid : ACE ZAID code (e.g. "92235.710nc")
        atom_density : Atomic density (atom/barn·cm)"""
        self._entries.append((zaid, atom_density))

    # ── Macroscopic section construction ( MaterialXS) ────

    def build(
        self,
        energy_grid: Optional[np.ndarray] = None,
        emin: float = 1e-11,
        emax: float = 20.0,
        n_points: int = 10000,
    ) -> MaterialXSData:
        """It creates macroscopic cross-section tables by collecting all isotopes.

        parameters
        ----------
        energy_grid : Custom energy grid (MeV). If not specified, log-uniform is produced.
        emin, emax : Energy limits (MeV)
        n_points : Number of energy grid points (used if there is no energy_grid)

        returns
        -------
        MaterialXSData"""
        # energy grid
        if energy_grid is None:
            energy_grid = np.logspace(
                np.log10(max(emin, 1e-15)),
                np.log10(emax),
                n_points,
            )

        ne = len(energy_grid)

        # Reset macroscopic slice arrays
        sig_total   = np.zeros(ne)
        sig_fiss    = np.zeros(ne)
        sig_fisse   = np.zeros(ne)    # Fission energy production
        nu_sig_f    = np.zeros(ne)
        sig_ela     = np.zeros(ne)
        sig_n2n     = np.zeros(ne)
        sig_cap     = np.zeros(ne)

        processor = AceProcessor(energy_grid)
        total_adens = 0.0

        for zaid, adens in self._entries:
            iso = self.library.get(zaid)
            if iso is None:
                print(f"  Warning: {zaid} not find in library, skipped.")
                continue

            # Process the isotope
            iso = processor.process(iso)
            total_adens += adens

            # Add contribution for each reaction channel
            def _add(mt, target):
                if mt in iso.reactions:
                    rea = iso.reactions[mt]
                    rea.adens = adens
                    for i, E in enumerate(energy_grid):
                        target[i] += interpolate_xs(rea, E)

            _add(MT_TOTAL,    sig_total)
            _add(MT_FISSION,  sig_fiss)
            _add(MT_ELASTIC,  sig_ela)
            _add(MT_N2N,      sig_n2n)
            _add(MT_CAPTURE,  sig_cap)

            # Nu*Sigma_f (MT=452 or fission × nu)
            if MT_NSF in iso.reactions:
                rea = iso.reactions[MT_NSF]
                rea.adens = adens
                for i, E in enumerate(energy_grid):
                    nu_sig_f[i] += interpolate_xs(rea, E)

        # Fission energy production = sigma_fission × average energy (approximately)
        sig_fisse = sig_fiss * 200.0   # ~200 MeV/fission, rough approximation

        self.data = MaterialXSData(
            name        = self.name,
            adens       = total_adens,
            emin        = emin,
            emax        = emax,
            energy_grid = energy_grid,
            sigma_total   = sig_total,
            sigma_fission = sig_fiss,
            sigma_capture = sig_cap,
            sigma_elastic = sig_ela,
            sigma_n2n     = sig_n2n,
            nu_sigma_f    = nu_sig_f,
        )
        print(f"MaterialXS '{self.name}' generated: {len(self._entries)} isotopes.")
        return self.data

    # ── Instant section query ──────────────────────── ─────────────────────────

    def get_total_xs(self, energy_eV: float) -> float:
        """Total macroscopic cross section (1/cm) — energy in eV."""
        return self._query(self.data.sigma_total, energy_eV * 1e-6)

    def get_fission_xs(self, energy_eV: float) -> float:
        """Fission macroscopic section (1/cm)."""
        return self._query(self.data.sigma_fission, energy_eV * 1e-6)

    def get_capture_xs(self, energy_eV: float) -> float:
        """Absorption macroscopic section (1/cm)."""
        return self._query(self.data.sigma_capture, energy_eV * 1e-6)

    def get_elastic_xs(self, energy_eV: float) -> float:
        """Elastic scattering macroscopic section (1/cm)."""
        return self._query(self.data.sigma_elastic, energy_eV * 1e-6)

    def get_nu_sigma_f(self, energy_eV: float) -> float:
        """Nu*Sigma_f (1/cm)."""
        return self._query(self.data.nu_sigma_f, energy_eV * 1e-6)

    def _query(self, xs_array: np.ndarray, E_MeV: float) -> float:
        if self.data is None or xs_array is None:
            raise RuntimeError("build() henüz çağrılmadı.")
        return float(np.interp(
            E_MeV,
            self.data.energy_grid,
            xs_array,
            left=0.0, right=0.0,
        ))

    # ── Majorant (Woodcock delta-tracking) ───────────────────────────────────

    def majorant(self) -> float:
        """The maximum total cross section in the entire energy grid.
        sigma_ext account for Woodcock delta-tracking.
        Same role as material.py → calculate_sigma_ext."""
        if self.data is None or self.data.sigma_total is None:
            return 0.0
        return float(np.max(self.data.sigma_total))