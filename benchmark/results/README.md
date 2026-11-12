# Paper result availability

The episode-level input table and final eligible-pair mapping for the paper's Table 4 are unavailable in this release. Accordingly, registered, activated, valid-pair, and clean-safe counts are not reconstructed from rounded percentages or from the number of tasks.

The required reporting schema is in [required_fields.json](required_fields.json). It describes missing result inputs, not measured counts. The included pairing implementation can validate supplied evidence; it cannot recover absent historical traces.

The Q17–Q24 offline test uses 80 held-out teacher trajectories. Its observations and training seeds are independent of the paper's clean/intervention pair denominators.
