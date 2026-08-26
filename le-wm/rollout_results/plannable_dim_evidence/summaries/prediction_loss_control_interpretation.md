# Prediction Loss Negative Control

This experiment is a negative control for the plannable-latent-geometry story.

If prediction-loss curves are similar across latent widths while candidate-level false shortcuts, ranking metrics, or closed-loop success differ, then prediction loss alone does not explain planning performance.

The intended conclusion is not that prediction loss is irrelevant. Prediction accuracy is necessary for a useful world model. The narrower claim is that one-step latent prediction loss is insufficient as a diagnostic for distance-based latent planning, because the planner also depends on whether Euclidean latent distance metricizes transition-to-goal geometry.

Suggested paper sentence:

> Although all models achieve comparable latent prediction losses, their candidate-level transition false shortcuts and closed-loop performance differ substantially. This suggests that one-step predictive accuracy is not sufficient to diagnose whether the latent metric is suitable as a planner cost.
