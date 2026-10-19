# dsh4SCI

A library of domain logic for physics, chemistry, and materials science: the
rules a computational materials scientist applies when building and judging
atomistic structures. It is not an agent framework, a service, or a storage
layer — consumers supply those.

## Language

### Core objects

**Structure**:
A concrete atomistic geometry: atoms with coordinates plus the periodic lattice
that contains them. It is a specific object with specific numbers, not a chemical
identity.
_Avoid_: Material, system, model, compound

**Material**:
The chemical identity of a composition, such as `YbFe4Sb12`. One material has
indefinitely many structures. It is a label on a structure, not an object this
library operates on.
_Avoid_: Structure, compound, phase

### Surfaces and slabs

**Surface**:
The physical two-dimensional boundary exposed when a crystal is cut along a
plane. It is a physical object and is not directly computable.
_Avoid_: Slab, face, plane

**Slab**:
A finite computable model of a surface: a stack of atomic layers plus a vacuum
region, periodic in the two in-plane directions. One surface admits indefinitely
many slabs.
_Avoid_: Surface, thin film, layer

**Miller index**:
The three-integer label `(hkl)` naming which crystal plane is cut. It selects a
plane family, not a specific cut within that family.
_Avoid_: Plane, facet, orientation

**Termination**:
A specific cut within one Miller-index family, distinguished by which atomic
layer the exposed face ends on. It must be identified by reproducible content,
not by an enumeration index alone.
_Avoid_: Slice, variant, surface type, `termination` as a bare integer

**Vacuum**:
The empty gap added above and below a slab, in ångström, making the periodic
repeat of the model physically non-interacting. It expresses a modelling choice,
not a material property.
_Avoid_: Gap, spacing, padding

### Adsorption

**Adsorbate**:
The molecule sitting on a surface, such as `CO` or `OH`. It is the species, not
its placement.
_Avoid_: Adsorbed molecule, ligand, species

**Adsorption site**:
The geometric location on a surface where an adsorbate sits, classified by how
many surface atoms it contacts — atop one, bridging two, or hollow over three or
more.
_Avoid_: Site, position, binding site

**Coverage**:
The number of adsorbates per unit surface area. It is derived from the size of
the surface cell and the count of adsorbates placed in it, never passed as an
input in its own right.
_Avoid_: Monolayer fraction, ML, loading

### Identity and provenance

**Structure reference**:
A path to a stored structure file, used only to load one. It is an entry point,
not the library's principal type — once loaded, a structure is passed as a
`Structure` object. A structure pasted in as text is never accepted.
_Avoid_: CIF string, inline structure, payload

**Termination fingerprint**:
The reproducible content-based identity of a termination — the composition of
the terminating layer together with its in-plane offset — so that a cut can be
recognised again after the underlying enumeration order changes.
_Avoid_: Termination index, enumeration order, slab number
