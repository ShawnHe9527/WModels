# L5PC target-segment figures (`apical_51`)

## Source
- Model: official L5PC NEURON model in `neuron/`
  (`L5PCtemplate_2.hoc`, `L5PCbiophys5b.hoc`)
- Morphology: `neuron/morphologies/cell1.asc`
- Mechanisms: `neuron/mods/x86_64/libnrnmech.so`
- Geometry is read directly from NEURON (`x3d/y3d/z3d/diam3d`,
  `sec.L`, `sec.nseg`); no hand-drawn coordinates.

## Sections shown
| section | nseg | L (µm) | diam (µm) | parent | children |
|---|---:|---:|---:|---|---|
| apic[50] | 3 | 61.927 | 1.80 | apic[36] | apic[60], apic[51] |
| apic[51] | 11 | 215.480 | 1.29 | apic[50] | apic[53], apic[52] |
| apic[52] | 5 | 113.916 | 0.482 | apic[51] | — |
| apic[53] | 9 | 190.289 | 1.03 | apic[51] | apic[57], apic[54] |

`apic[60]` is a sibling of `apic[51]` (also a child of `apic[50]`);
it is omitted from Fig. 1 to avoid clutter.

## Figures
- `fig1_target_and_neighbors.png/.pdf`
  Target `apic[51]` (red) plus parent `apic[50]` (blue) and children
  `apic[52]` (green) / `apic[53]` (orange); horizontal, proximal → distal.
  Inset: whole cell with the local region boxed. True scale; 50 µm bar.
- `fig2_apical51_compartments.png/.pdf`
  Top: `apical_51` with its 11 compartments (color-coded, numbered).
  Bottom: true diameter profile along arc length (11 compartments).
  `L = 215.48 µm`, `nseg = 11`, each compartment = 19.589 µm.
  Top panel thickness is exaggerated for visibility; bottom panel is true diameter.
- `fig4_ih_calva_profiles.png/.pdf`
  Spatial profiles of `Ih` (exponential gradient, saturating at 0.65·Lmax)
  and `Ca_LVAst` (high only in the 685–885 µm path-distance band; 1% outside).
- `fig3_single_compartment_cable.png/.pdf`
  Compartment 6: `L = 19.59 µm`, `d = 1.29 µm` (constant).
  Shaded 2D cable schematic with a cut end; inset shows its location
  within `apical_51`.
  Diameter is visually exaggerated; `L` and `d` values are from NEURON geometry.

## Data tables
- `geometry_sections.csv` — section-level L, nseg, diam, parent/children.
- `geometry_compartments.csv` — compartment boundaries and diameters for `apical_51`.

## Notes
- These figures are morphology/geometry visualizations, not simulation traces.
- The 2D cable in Fig. 3 is a schematic of a NEURON compartment (cylinder/frustum),
  not a 3D reconstruction of the membrane surface.
- All panels use the official model’s 3D points; no smoothing or artificial branching.
