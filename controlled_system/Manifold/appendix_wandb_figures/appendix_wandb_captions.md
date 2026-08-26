# Appendix W&B Figure Captions

**Figure A1. Training convergence.** Prediction loss curves are shown with a centred rolling median over 51 logged points. Lines stop at their actual final logged training step, without extrapolation or forward-filling. Panel (a) shows baseline latent-dimension runs. The State-8 prediction-loss export uses run ID `baseline_state8_no_bottleneck_TBA`, whereas the other State-8 metrics use the fresh-seed run; both are labelled State-8. Panel (b) compares State-32, State-tangent (k=32), and Global-linear (k=32). Training lengths differ, so this comparison is descriptive rather than a controlled convergence-speed comparison.

**Figure A2. Local representation geometry.** Plateau statistics are tail10 medians over the final 10% of logged points for baseline dimension runs only. Larger ambient dimensions are associated with lower top-eigenvalue concentration and higher local effective rank.

**Figure A3. Full versus shuffled transition geometry.** Plateau statistics are tail10 medians for baseline dimension runs only. Full transition pairings remain much closer to a delta-norm ratio of 1 than shuffled-transition controls across dimensions; directional alignment is shown only for the full transition pairing.

**Figure A4. Structured predictor ablation.** Plateau statistics are tail10 medians. State-tangent (k=32) has prediction loss similar to State-32, a lower top-eigenvalue fraction, a higher local participation-ratio rank, and the best directional alignment. Global-linear (k=32) has a norm ratio close to 1 but poorer directional alignment and higher prediction loss, indicating that matching transition magnitude alone is not sufficient; transition direction also matters.
