# SpaDiff paper-aligned implementation

This package is a separate implementation derived from the finalized Methods
and Supplementary Information. The original `SpaDiff_improved` package is not
modified.

## Implemented objective

The training objective is

```text
L = lambda_dsm * L_DSM
  + lambda_batch * (L_ratio + lambda_posterior * L_q(b|x0))
  + lambda_prior * KL(q(H|b) || p(H)).
```

- `L_DSM` is the topology- and batch-conditional diffusion loss.
- The three outer coefficients correspond exactly to the manuscript's DSM,
  batch-alignment, and latent-prior terms. `lambda_posterior` is only an
  internal scale for identifying `q(b|x0)`, not a fourth manuscript loss.
- `q(b|x0)` is a categorical posterior trained from the observed technical
  labels.
- `p(b|H)` is a categorical topology predictor. A gradient-reversal layer
  trains the predictor to estimate the technical label while training the
  topology encoder to remove that label from `H`.
- The optional prior term approximates each batch-specific embedding
  distribution and the pooled embedding distribution as diagonal Gaussians.
  It is disabled by default because the SI explicitly says this term was
  removed in practical training for efficiency.

The SI calls the discrete `p(b|H)` a normal distribution and also describes
continuous latent distributions as uniform, so Eq. (19) is not a complete
implementable likelihood as written. The categorical adversarial formulation
above is the smallest operational interpretation that preserves the stated
goal of weakening batch information in `H`.

## DSM weighting

`dsm_weighting` is explicit:

- `score`: literal unweighted score error from Eq. (18)/(19).
- `variance`: variance-weighted score error, exactly epsilon-MSE; this is the
  stable practical default used by the revised tutorials.
- `likelihood`: continuous-time diffusion-squared likelihood weighting.

## Deliberate differences from the old package

- DEC is not part of the training objective. Mclust/Louvain remain downstream
  clustering methods, as stated in the Methods.
- Technical groups are averaged equally in the loss, rather than weighting a
  slice solely by its number of spots.
- Classifier-free condition dropout defaults to zero because it is absent from
  the manuscript.
- Loss coefficients no longer change the topology encoder architecture.
- Diffusion training is never stopped by clustering-label stability.

Coordinate alignment, Moran's I feature selection, and the alignment MLP are
outside this package. Revised tutorials assume their input coordinates are
already in the desired common coordinate system.

## Maximum simplex-order ablation

Every revised notebook defines one ``MAX_ORDER`` value. The corresponding
encoder channels are derived automatically:

- ``0`` -> ``simplex_orders=(0,)``: node-only identity baseline;
- ``1`` -> ``simplex_orders=(1,)``: edges only;
- ``2`` -> ``simplex_orders=(1, 2)``: edges and triangles;
- ``3`` -> ``simplex_orders=(1, 2, 3)``;
- ``4`` -> ``simplex_orders=(1, 2, 3, 4)``.

``build_simplicial_operators`` prints the number of simplices at every order.
Orders three and four enumerate 4-cliques and 5-cliques respectively, so their
time and memory requirements can grow combinatorially.
