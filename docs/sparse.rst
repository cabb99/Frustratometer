Sparse Calculations
===================

For large proteins, the standard dense Potts model representation stores
a coupling tensor :math:`J` of shape :math:`(L, L, 21, 21)`, which grows
as :math:`O(L^2)` and can require gigabytes of memory. The dense representation
can provide a more straight-forward view of the interactions, but it is computationally expensive. 
The sparse framework avoids this by storing only the couplings at positions where the contact mask
is nonzero — typically a small fraction of all pairs.

Sparse Potts Model
------------------

The sparse Potts model is a dictionary with the following keys:

- ``h`` — shape :math:`(L, 21)`: fields (identical to dense)
- ``J`` — shape :math:`(N_\text{contacts}, 21, 21)`: couplings only at contact positions
- ``contact_i``, ``contact_j`` — shape :math:`(N_\text{contacts},)`: row and column indices of each contact
- ``L`` — sequence length

For a protein of length :math:`L = 163` with :math:`N_\text{contacts} \approx 2{,}200`,
the dense :math:`J` tensor uses :math:`163^2 \times 21^2 \approx 12\text{M}` floats
(:math:`\sim 94\text{ MB}`), while the sparse representation uses only
:math:`2{,}200 \times 441 \approx 970\text{K}` floats (:math:`\sim 7.6\text{ MB}`).

Conversion
^^^^^^^^^^

.. autofunction:: frustratometer.frustration.frustration.potts_model_dense_to_sparse
.. autofunction:: frustratometer.frustration.frustration.potts_model_sparse_to_dense

Contact Lookup
^^^^^^^^^^^^^^

To efficiently compute per-residue sums over contacts (e.g. for single-residue
decoy fluctuations), a CSR-like lookup structure groups contacts by position:

.. autofunction:: frustratometer.frustration.frustration.build_contact_lookup

Sparse Mask
^^^^^^^^^^^

Computes the mask without materializing a full :math:`(L, L)` matrix:

.. autofunction:: frustratometer.frustration.frustration.compute_mask_sparse


Sparse Energy Functions
-----------------------

These functions compute the same energies as their dense counterparts but
operate directly on the sparse Potts model, avoiding any :math:`(L, L)` or
:math:`(L, L, 21, 21)` intermediate arrays.

.. autofunction:: frustratometer.frustration.frustration.compute_native_energy_sparse
.. autofunction:: frustratometer.frustration.frustration.compute_couplings_energy_sparse
.. autofunction:: frustratometer.frustration.frustration.compute_sequences_energy_sparse


Sparse Decoy Fluctuations
--------------------------

Decoy fluctuations measure how the energy changes under hypothetical mutations.
The dense versions produce tensors of shape :math:`(L, 21)` (single-residue) or
:math:`(L, L, 21, 21)` (pair). The sparse versions produce :math:`(L, 21)` for
single-residue and :math:`(N_\text{contacts}, 21, 21)` for pair kinds — only
storing values at contact positions.

.. autofunction:: frustratometer.frustration.frustration.compute_singleresidue_decoy_energy_fluctuation_sparse
.. autofunction:: frustratometer.frustration.frustration.compute_mutational_decoy_energy_fluctuation_sparse
.. autofunction:: frustratometer.frustration.frustration.compute_pseudoconfigurational_decoy_energy_fluctuation_sparse
.. autofunction:: frustratometer.frustration.frustration.compute_contact_decoy_energy_fluctuation_sparse


Sparse Frustration
------------------

.. autofunction:: frustratometer.frustration.frustration.compute_pair_frustration_sparse
.. autofunction:: frustratometer.frustration.frustration.sparse_frustration_to_dense


Electrostatic Corrections
--------------------------

The sparse Potts model stores couplings only at contact positions,
:math:`(N_\text{contacts}, Q, Q)` instead of :math:`(L, L, Q, Q)`, which saves
memory when :math:`N_\text{contacts} \ll L^2`. Electrostatics, however, is a
long-ranged interaction. Distant residues interact through the screened Coulomb
potential. In the dense AWSEM formulation, the electrostatic contribution is
folded directly into the coupling tensor as

.. math::

   V_{ij}^{\mathrm{elec}}(a,b)
   =
   \Gamma^{\mathrm{elec}}_{ij}\, q(a)\, q(b),

where :math:`Q = 21` is the number of amino-acid types (including the gap
state). If we tried to include electrostatics in the sparse couplings, the
number of nonzero contacts would grow to :math:`N_\text{contacts} \sim L^2`
(every pair interacts), defeating the purpose of a sparse representation.

However the amino-acid dependence enters only through the charge :math:`q(a)`,
a single number per amino acid, and the geometric dependence enters only through
:math:`\Gamma^{\mathrm{elec}}_{ij}`, a single number per residue pair. So the
full electrostatic coupling is determined by an :math:`(L, L)` matrix of
geometric factors and a :math:`(Q,)` vector of charges instead of a
four-dimensional tensor. This reduces the storage from :math:`L^2 Q^2` to
:math:`L^2 + Q`: the :math:`(L, L)` indicator matrix is still needed
(electrostatics is long-ranged and position-dependent, so it cannot be
compressed further without the coordinates), but it is :math:`Q^2 = 441` times
smaller than the dense coupling tensor.

For the current electrostatic model each amino-acid type :math:`a` is assigned a
charge :math:`q(a)`, where D and E (aspartate, glutamate) have a charge of
:math:`-1` and K and R have a charge of :math:`+1` with all others having 0
charge.

The strength of the electrostatic interaction between residues :math:`i` and
:math:`j` is described by the screened Coulomb (Debye--Hückel) kernel:

.. math::

   \Gamma^{\mathrm{elec}}_{ij}
   =
   -k_{\mathrm{elec}}
   \frac{\exp(-r_{ij}/l_D)}{r_{ij}},

where :math:`r_{ij}` is the inter-residue distance,
:math:`k_{\mathrm{elec}}` is the electrostatic coupling constant, and
:math:`l_D` is the Debye screening length. Each value is always negative
(attractive for opposite charges). Pairs with sequence separation below a
minimum threshold are usually excluded from this interaction.

.. autofunction:: frustratometer.frustration.frustration.compute_elec_indicator


Definitions
^^^^^^^^^^^

Before computing the sparse electrostatics, we precompute a small set of summary
quantities from the native structure. These are stored in a dictionary and reused
by all correction functions.

- :math:`\mathcal{M}_{ij}`: the frustration contact mask. When electrostatics is
  enabled, this mask enforces only a minimum sequence separation (no distance
  cutoff), so the sums below include all long-range electrostatic interactions.
- :math:`q_i^N`: the charge of the native amino acid at position :math:`i`.
- :math:`\Gamma^{\mathrm{elec}}_{ij}`: the full :math:`(L \times L)` pairwise
  electrostatic indicator matrix. This matrix must be computed at all residue
  pairs because the potentials :math:`\phi_i` and :math:`\phi_i^{\mathrm{raw}}`
  require summing over distant residues.
- :math:`\phi_i^{\mathrm{raw}} = \sum_j \Gamma^{\mathrm{elec}}_{ij}\,q_j^N`:
  the electrostatic potential at position :math:`i` without the contact mask
  (only needed for pseudo-configurational decoys, which average over masked
  configurations).
- :math:`\phi_i = \sum_j \Gamma^{\mathrm{elec}}_{ij}\,\mathcal{M}_{ij}\,q_j^N`:
  the electrostatic potential felt by residue :math:`i` from its native
  environment.
- :math:`\bar{m}`: the mean value of :math:`\mathcal{M}_{ij}`, representing the
  average contact density.

.. autofunction:: frustratometer.frustration.frustration.build_elec_data


Native energy correction
^^^^^^^^^^^^^^^^^^^^^^^^

The electrostatic contribution to the native energy is

.. math::

   V^{\mathrm{elec},N}
   =
   -\frac{1}{2}\sum_{i,j}
   \Gamma^{\mathrm{elec}}_{ij}\,\mathcal{M}_{ij}\,q_i^N\,q_j^N.

This is added to the base native energy :math:`V^{(0),N}` from the other terms.
The factor of :math:`\tfrac{1}{2}` avoids double-counting the symmetric
:math:`(i,j)` and :math:`(j,i)` contributions.

.. autofunction:: frustratometer.frustration.frustration.compute_native_energy_elec


Decoy corrections
^^^^^^^^^^^^^^^^^

Each decoy type requires a different correction formula, because each type
varies different degrees of freedom.

**Single-residue decoys.**
In single-residue decoys, we compute the change in energy if we mutate position
:math:`i` from its native amino acid (charge :math:`q_i^N`) to type :math:`a`
(charge :math:`q(a)`), while every other position keeps its native identity.
The change in electrostatic energy is the charge difference times the
electrostatic potential at that position:

.. math::

   \delta V_i^{\mathrm{elec}}(a)
   =
   -(q(a) - q_i^N)\,\phi_i.

.. autofunction:: frustratometer.frustration.frustration.apply_elec_correction_singleresidue

**Mutational decoys.**
Both positions :math:`i` and :math:`j` are mutated simultaneously to types
:math:`a` and :math:`b`, while the rest of the protein stays native. This is
more complex because mutating :math:`i` and :math:`j` changes not only their
direct interaction, but also each residue's interaction with the entire native
electrostatic environment.

We define the charge shifts :math:`\delta q_i(a) = q(a) - q_i^N` and
:math:`\delta q_j(b) = q(b) - q_j^N`. The correction has three terms:

.. math::

   \delta V_{ij}^{\mathrm{elec,mut}}(a,b)
   =
   -\delta q_i(a)\,\phi_i
   -\,\delta q_j(b)\,\phi_j
   -\,\Gamma^{\mathrm{elec}}_{ij}\,\delta q_i(a)\,\delta q_j(b)

The three terms are:

1. **Environment of** :math:`i`: Changing :math:`i`'s charge by
   :math:`\delta q_i(a)` in the electrostatic field :math:`\phi_i`.
2. **Environment of** :math:`j`: Changing :math:`j`'s charge by
   :math:`\delta q_j(b)` in the electrostatic field :math:`\phi_j`.
3. **Direct interaction**: The potential :math:`\phi_i` already included the
   contribution from :math:`j` (with its native charge) and vice versa. This
   term corrects for the missing new :math:`i`--:math:`j` interaction and
   removes the double-counted one.

The result has shape :math:`(N_\text{contacts}, Q, Q)`.

.. autofunction:: frustratometer.frustration.frustration.apply_elec_correction_mutational

**Contact decoys.**
The amino-acid pair :math:`(a,b)` at contact :math:`(i,j)` is drawn randomly,
but the rest of the protein stays native. Only the direct electrostatic
interaction between positions :math:`i` and :math:`j` changes; the interactions
of :math:`i` and :math:`j` with the rest of the chain are unaffected. The
correction is simply the difference between the native and decoy direct
interactions:

.. math::

   \delta V_{ij}^{\mathrm{elec,ct}}(a,b)
   =
   \Gamma^{\mathrm{elec}}_{ij}\,(q_i^N q_j^N - q(a)\, q(b)).

When the native pair is uncharged (:math:`q_i^N q_j^N = 0`), the correction
reduces to :math:`-\Gamma^{\mathrm{elec}}_{ij}\,q(a)\,q(b)`. The result has
shape :math:`(N_\text{contacts}, Q, Q)`.

.. autofunction:: frustratometer.frustration.frustration.apply_elec_correction_contact

**Pseudo-configurational decoys.**
Both the amino-acid pair and the contact geometry change: the native contact mask
:math:`\mathcal{M}_{ij}` is replaced by its average value :math:`\bar{m}`,
simulating a random contact geometry at each position. (This is an analytical
approximation; the real configurational frustration in AWSEM uses explicit Monte
Carlo decoys.) Under the decoy mask, the relevant potential becomes
:math:`\bar{m}\,\phi_i^{\mathrm{raw}}` — the unmasked potential scaled by the
average contact density.

The correction decomposes into four coefficients in the basis
:math:`\{1, q(a), q(b), q(a)\,q(b)\}`:

.. math::

   \delta V_{ij}^{\mathrm{elec,cfg}}(a,b)
   =
   c_{00} + c_{10}\,q(a) + c_{01}\,q(b) + c_{11}\,q(a)\,q(b),

where:

.. math::

   c_{00} &= q_i^N \phi_i + q_j^N \phi_j
   - \Gamma^{\mathrm{elec}}_{ij}\, q_i^N q_j^N\, \mathcal{M}_{ij}, \\
   c_{10} &= \bar{m} \left(\Gamma^{\mathrm{elec}}_{ij}\, q_j^N - \phi_i^{\mathrm{raw}}\right), \\
   c_{01} &= \bar{m} \left(\Gamma^{\mathrm{elec}}_{ij}\, q_i^N - \phi_j^{\mathrm{raw}}\right), \\
   c_{11} &= -\bar{m}\, \Gamma^{\mathrm{elec}}_{ij}.

- :math:`c_{00}` is a constant offset: the electrostatic energy of the native
  charges evaluated under the native mask.
- :math:`c_{10}` multiplies :math:`q(a)` and captures how the decoy charge at
  position :math:`i` interacts with the averaged environment.
- :math:`c_{01}` is the symmetric counterpart for position :math:`j`.
- :math:`c_{11}` multiplies :math:`q(a)\,q(b)` and represents the direct
  :math:`i`--:math:`j` electrostatic interaction under the averaged mask.

The result has shape :math:`(N_\text{contacts}, Q, Q)`.

.. autofunction:: frustratometer.frustration.frustration.apply_elec_correction_pseudoconfigurational


Corrected frustration statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The frustration index at contact :math:`(i,j)` is

.. math::

   F_{ij} = -\frac{V_{ij}^N - \langle V_{ij}^U \rangle}{\sigma_{ij}^U},

where :math:`V_{ij}^N` is the native energy at the contact,
:math:`\langle V_{ij}^U \rangle` is the mean over the decoy (unfolded) ensemble,
and :math:`\sigma_{ij}^U` is the corresponding standard deviation. Adding
electrostatic corrections means the total decoy energy is
:math:`V_{ij}^{\mathrm{tot}}(a,b) = V_{ij}^{(0)}(a,b) + \delta V_{ij}^{\mathrm{elec}}(a,b)`,
where :math:`V_{ij}^{(0)}` comes from the base Potts model. The goal is to
compute the mean and variance of this sum exactly, without creating
:math:`(Q \times Q)` correction arrays.

**Bilinear structure.**
All four corrections depend on amino-acid identity only through the charge
:math:`q(a)`. Their statistics can be computed exactly from a few scalar or
:math:`(2 \times 2)` operations, without enumerating all :math:`Q` (or
:math:`Q^2`) decoy types.

*Single-residue* is the simplest case. The correction
:math:`\delta V_i^{\mathrm{elec}}(a) = -(q(a) - q_i^N)\,\phi_i` depends on
only one amino-acid index, so it is linear rather than bilinear. Its mean and
variance over the decoy ensemble are:

.. math::

   \mu_i^{\mathrm{elec}} = -(\bar{q} - q_i^N)\,\phi_i,
   \qquad
   \sigma_i^{2,\mathrm{elec}} = \sigma_q^2\,\phi_i^2,

where :math:`\bar{q} = \sum_a p(a)\,q(a)` and
:math:`\sigma_q^2 = \sum_a p(a)\,q(a)^2 - \bar{q}^2` are the mean and variance
of the charge under the amino-acid frequency distribution :math:`p(a)`. The
covariance with the base single-residue energy :math:`V_i^{(0)}(a)` is
:math:`\mathrm{Cov}_i^{(0,\mathrm{elec})} = -\phi_i \, \mathrm{Cov}_a(q(a),\, V_i^{(0)}(a))`.
All of these are scalars computed directly from the precomputed potential
:math:`\phi_i` and the charge statistics.

When decoys are drawn uniformly (:math:`p(a) = 1/Q`), the charges sum to zero
(D, E contribute :math:`-2`; K, R contribute :math:`+2`), giving
:math:`\bar{q} = 0` and :math:`\sigma_q^2 = 4/Q`.

*Pair decoy types* (mutational, contact, pseudo-configurational) depend on two
amino-acid indices and share a bilinear structure. Each correction can be written
as

.. math::

   \delta V_{ij}^{\mathrm{elec}}(a,b)
   =
   \mathbf{f}_i(a)^\top \, \mathbf{A}_{ij}^{\mathrm{elec}} \, \mathbf{f}_j(b),

where :math:`\mathbf{f}_i(a)` is a small feature vector that depends only on
the amino-acid type (dimension 2: a constant and a charge), and
:math:`\mathbf{A}_{ij}^{\mathrm{elec}}` is a :math:`(2 \times 2)` coefficient
matrix that depends on the geometry and native environment at the contact. The
feature vectors and coefficient matrices for each decoy type are:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Decoy type
     - :math:`\mathbf{f}_i(a)`
     - :math:`\mathbf{f}_j(b)`
     - :math:`\mathbf{A}_{ij}^{\mathrm{elec}}`
   * - Mutational
     - :math:`[1,\; \delta q_i(a)]^\top`
     - :math:`[1,\; \delta q_j(b)]^\top`
     - :math:`\begin{bmatrix} 0 & -\phi_j \\ -\phi_i & -\Gamma^{\mathrm{elec}}_{ij} \end{bmatrix}`
   * - Contact
     - :math:`[1,\; q(a)]^\top`
     - :math:`[1,\; q(b)]^\top`
     - :math:`\begin{bmatrix} q_i^N q_j^N\,\Gamma^{\mathrm{elec}}_{ij} & 0 \\ 0 & -\Gamma^{\mathrm{elec}}_{ij} \end{bmatrix}`
   * - Pseudo-config.
     - :math:`[1,\; q(a)]^\top`
     - :math:`[1,\; q(b)]^\top`
     - :math:`\begin{bmatrix} c_{00} & c_{01} \\ c_{10} & c_{11} \end{bmatrix}`

where :math:`\delta q_i(a) = q(a) - q_i^N` as before, and the :math:`c`
coefficients are those defined in the pseudo-configurational section above.

As a quick verification, expanding the mutational case gives
:math:`\mathbf{f}_i^\top \mathbf{A} \mathbf{f}_j = 1 \cdot (-\phi_j) \cdot \delta q_j + \delta q_i \cdot (-\phi_i) \cdot 1 + \delta q_i \cdot (-\Gamma^{\mathrm{elec}}_{ij}) \cdot \delta q_j`,
which recovers the three-term formula derived earlier.

This bilinear structure lets us compute the exact mean and variance of the
correction using only :math:`(2 \times 2)` matrix operations, because decoy
amino acids :math:`a` and :math:`b` are drawn independently.


Step 1: Feature moments
""""""""""""""""""""""""

The decoy amino acids :math:`a` and :math:`b` are drawn from a distribution
(typically uniform over the :math:`Q` types), so the feature vectors become
random variables. We need their first and second moments.

**First moments** (means):

.. math::

   \mathbf{m}_i = \mathbb{E}_a[\mathbf{f}_i(a)],
   \qquad
   \mathbf{m}_j = \mathbb{E}_b[\mathbf{f}_j(b)].

For mutational decoys with
:math:`\mathbf{f}_i(a) = [1,\; \delta q_i(a)]^\top`:
:math:`\mathbf{m}_i = [1,\; \bar{q} - q_i^N]^\top`, where
:math:`\bar{q} = \sum_a p(a)\,q(a)` is the mean charge under the amino-acid
frequency distribution :math:`p(a)`.

**Second moments** (uncentered):

.. math::

   \mathbf{M}_i = \mathbb{E}_a[\mathbf{f}_i(a)\,\mathbf{f}_i(a)^\top],
   \qquad
   \mathbf{M}_j = \mathbb{E}_b[\mathbf{f}_j(b)\,\mathbf{f}_j(b)^\top].

Continuing the mutational example:

.. math::

   \mathbf{M}_i
   =
   \begin{bmatrix}
   1 & \bar{q} - q_i^N \\
   \bar{q} - q_i^N & \overline{q^2} - 2q_i^N\bar{q} + (q_i^N)^2
   \end{bmatrix},

where :math:`\overline{q^2} = \sum_a p(a)\,q(a)^2` is the second moment of the
charge. The :math:`(2,2)` entry is
:math:`\mathbb{E}[\delta q_i(a)^2] = \overline{q^2} - 2q_i^N\bar{q} + (q_i^N)^2`.


Step 2: Mean of the correction
"""""""""""""""""""""""""""""""

Because :math:`a` and :math:`b` are drawn independently, the expectation of the
bilinear form factors cleanly:

.. math::

   \mu_{ij}^{\mathrm{elec}}
   =
   \mathbb{E}_{a,b}\!\left[\mathbf{f}_i(a)^\top \mathbf{A}_{ij}^{\mathrm{elec}}\, \mathbf{f}_j(b)\right]
   =
   \mathbf{m}_i^\top \, \mathbf{A}_{ij}^{\mathrm{elec}} \, \mathbf{m}_j.

This is a single scalar computed from a :math:`(2 \times 2)` matrix sandwiched
between two 2-vectors.


Step 3: Variance of the correction
"""""""""""""""""""""""""""""""""""

For the variance, we need :math:`\mathbb{E}[(\delta V_{ij}^{\mathrm{elec}})^2]`.
Squaring the bilinear form:

.. math::

   (\delta V_{ij}^{\mathrm{elec}})^2
   =
   (\mathbf{f}_i^\top \mathbf{A}_{ij}^{\mathrm{elec}} \mathbf{f}_j)\,
   (\mathbf{f}_j^\top \mathbf{A}_{ij}^{\mathrm{elec}\top} \mathbf{f}_i)
   =
   \mathbf{f}_i^\top \mathbf{A}_{ij}^{\mathrm{elec}}\,
   \mathbf{f}_j \mathbf{f}_j^\top\,
   \mathbf{A}_{ij}^{\mathrm{elec}\top} \mathbf{f}_i.

Taking the expectation over :math:`b` first replaces
:math:`\mathbf{f}_j \mathbf{f}_j^\top` with :math:`\mathbf{M}_j`:

.. math::

   \mathbb{E}_b\!\left[(\delta V_{ij}^{\mathrm{elec}})^2\right]
   =
   \mathbf{f}_i^\top \mathbf{A}_{ij}^{\mathrm{elec}}\, \mathbf{M}_j\, \mathbf{A}_{ij}^{\mathrm{elec}\top} \mathbf{f}_i.

This is now a quadratic form :math:`\mathbf{f}_i^\top \mathbf{B} \, \mathbf{f}_i`
where :math:`\mathbf{B} = \mathbf{A}_{ij}^{\mathrm{elec}}\, \mathbf{M}_j\, \mathbf{A}_{ij}^{\mathrm{elec}\top}`.
Using the standard identity
:math:`\mathbb{E}[\mathbf{f}^\top \mathbf{B} \, \mathbf{f}] = \operatorname{tr}(\mathbf{B}\,\mathbb{E}[\mathbf{f}\mathbf{f}^\top])`,
the expectation over :math:`a` gives:

.. math::

   \mathbb{E}[(\delta V_{ij}^{\mathrm{elec}})^2]
   =
   \operatorname{tr}\!\left(
   \mathbf{A}_{ij}^{\mathrm{elec}}\, \mathbf{M}_j\, \mathbf{A}_{ij}^{\mathrm{elec}\top}\, \mathbf{M}_i
   \right).

The variance of the correction is therefore:

.. math::

   \sigma_{ij}^{2,\mathrm{elec}}
   =
   \operatorname{tr}\!\left(
   \mathbf{A}_{ij}^{\mathrm{elec}}\, \mathbf{M}_j\, \mathbf{A}_{ij}^{\mathrm{elec}\top}\, \mathbf{M}_i
   \right)
   -
   (\mu_{ij}^{\mathrm{elec}})^2.

All matrices involved are :math:`(2 \times 2)`.


Step 4: Covariance with the base energy
""""""""""""""""""""""""""""""""""""""""

The total decoy energy is
:math:`V_{ij}^{\mathrm{tot}} = V_{ij}^{(0)} + \delta V_{ij}^{\mathrm{elec}}`,
so its variance decomposes as:

.. math::

   \mathrm{Var}(V_{ij}^{\mathrm{tot}})
   =
   \sigma_{ij}^{2,(0)}
   +
   \sigma_{ij}^{2,\mathrm{elec}}
   +
   2\,\mathrm{Cov}_{ij}^{(0,\mathrm{elec})}.

The first term is already computed by the sparse decoy fluctuation code. The
second was derived above. The remaining piece is the cross-covariance between the
base Potts energy and the electrostatic correction.

Define the cross-moment matrix:

.. math::

   \mathbf{T}_{ij}
   =
   \mathbb{E}_{a,b}\!\left[
   V_{ij}^{(0)}(a,b)\,
   \mathbf{f}_i(a)\,\mathbf{f}_j(b)^\top
   \right].

This :math:`(2 \times 2)` matrix captures how the base energy correlates with
the charge features. In practice it is computed by a weighted average over the
:math:`Q` amino-acid types. Using this:

.. math::

   \mathrm{Cov}_{ij}^{(0,\mathrm{elec})}
   =
   \operatorname{tr}(\mathbf{A}_{ij}^{\mathrm{elec}\top} \, \mathbf{T}_{ij})
   -
   \mu_{ij}^{(0)} \, \mu_{ij}^{\mathrm{elec}},

where :math:`\mu_{ij}^{(0)} = \mathbb{E}[V_{ij}^{(0)}(a,b)]` is the mean base
decoy energy.


Step 5: Corrected frustration index
""""""""""""""""""""""""""""""""""""

Putting it all together:

.. math::

   F_{ij}
   =
   -\frac{\mu_{ij}^{(0)} + \mu_{ij}^{\mathrm{elec}}}
   {\sqrt{
   \sigma_{ij}^{2,(0)}
   +
   \sigma_{ij}^{2,\mathrm{elec}}
   +
   2\,\mathrm{Cov}_{ij}^{(0,\mathrm{elec})}
   }}.

Every quantity is computed exactly from the precomputed electrostatic data and
the sparse Potts model.
