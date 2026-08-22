# Legacy manuscript sources

`main_elsevier.tex` is the `elsarticle` version of this manuscript, kept as a historical record and
no longer built. It was the source for the Neurocomputing / earlier submissions; the
submitted packages themselves are preserved unchanged under
`superseded-submission/`.

It is **out of date and must not be reused**. Two of its readings did not survive the threshold
defect KD6 and are corrected in the current manuscript: the claim that the two pre-registered
estimands disagree, and the claim that a pre-ReLU probe beats the network by a full sparsity level.
See `paper/sections/app_withdrawn.tex` and `docs/known_defects.md`.

The live manuscript is `paper/tmlr.tex`, built from `paper/sections/*.tex`. It is the only source
`scripts/check_manuscript_numbers.py` verifies, so a number changed here would be checked by
nothing.
