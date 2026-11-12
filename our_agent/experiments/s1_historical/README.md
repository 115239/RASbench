# S1 configuration comparison

S1 is fixed-base block pick-and-place. Qualification records report 183/200 and 188/200 successful episodes for GR00T (91.5% and 94.0%). The training report records 150/200 for π0.5 full250 (75.0%) and 163/200 for a separate canary checkpoint (81.5%).

[results.csv](results.csv) contains these counts and descriptive Wilson intervals. The two GR00T qualification records are in [evidence](evidence). The [π0.5 source excerpts](evidence/pi05_report_excerpt.json) preserve the reported counts and source-document hash.

GR00T used 449 demonstrations and 20,000 updates. The π0.5 full250 run also lists 449 demonstrations, but continued from a canary initialization for 250 steps; the canary used 17 demonstrations and additional hard replay. Learning rates, batch sizes, and executed action chunks also differed. These results describe the evaluated configurations rather than a matched-budget architecture comparison.

The additional [Q17–Q24 experiment](../market_offline/README.md) measures offline action prediction on a different task set.
