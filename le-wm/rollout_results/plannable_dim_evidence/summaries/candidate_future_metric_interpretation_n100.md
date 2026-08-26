# Candidate Future Metric Summary, 100 windows

This is mechanism evidence for the plannable-latent-dimension story.

- state8 mean false shortcut rate: 0.026107459118571034
- baseline192 mean false shortcut rate: 0.01869136881674846
- state8 / baseline192 false-shortcut ratio: 1.40x

Interpretation: state8 has the highest mean false shortcut rate, while 16D and above form a better plateau. Confidence intervals overlap, so this should not be phrased as a statistical-significance claim. The result supports the mechanism-level claim that too-small latent width can produce more graph-far / score-near candidate errors; it does not imply a universal monotonic relationship between nominal dimension and planning performance.
