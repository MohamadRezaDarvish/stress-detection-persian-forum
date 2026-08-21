# Next iteration plan

The current members do not need to rerun their accepted base pipelines for the present
project package. The next improvement cycle should be a new experiment:

1. Freeze the current test as historical evidence; never use it for tuning.
2. Build nested/grouped cross-validation for fusion architecture and threshold selection.
3. Select new annotation candidates using:
   - proximity to the two Moderate boundaries,
   - Member 1/Member 2 disagreement,
   - high temporal risk,
   - underrepresented authors/threads.
4. Obtain a fresh, dual-annotated confirmation set with enough Moderate and Very-High
   examples.
5. Test an ordinal text/fusion head with outputs for stress >3, >5, and >7.
6. Re-evaluate on the new untouched set.

The practical target is to raise Moderate recall without sacrificing Very-High recall or
precision.
