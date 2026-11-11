# Rebuild the matched dataset from PhysioNet

## Run the extraction

1. Open [dataset_splits_one_shot.sql](dataset_splits_one_shot.sql).
2. Replace every `YOUR_PROJECT_ID` with your own Google Cloud project ID.
   Leave the `physionet-data` source paths unchanged.
3. Run the entire file in BigQuery using GoogleSQL and the **US** location.
   You need access to the MIMIC source tables.

The script creates `mimic_project` if needed and **replaces
`YOUR_PROJECT_ID.mimic_project.dataset_splits`**. It reads raw PhysioNet tables
without intermediate project tables and produces full notes, not filtered notes.

## Cohort rules

- Keep the latest ICU stay **per hospital admission**, not one admission per patient.
- Exclude hospital stays with missing discharge time or discharge before admission.
- Summarize ten selected lab/vital concepts within that ICU stay and before hospital discharge.
- Include admission diagnoses; retain admissions with missing diagnosis records.
- Require a nonempty discharge note stored by **discharge +24 hours**.
- Exclude in-hospital deaths and recorded death dates through the discharge +1-day calendar date.
- Label recorded deaths through discharge +30 days positive; missing death dates count as negative under the original convention.
- Assign patients to train/validation/test using the original hash seed; all their admissions stay together.

## Expected result and reproducibility

Previously checked sources yielded **45,482 admissions**, **2,248 positive labels**,
and **4.9426% mortality** for these eligibility rules. Verify the actual results.
The existing local 52,027-row exports use the old three-day cutoff; export the new
cohort and regenerate local filtered datasets/reports before training.

FLOAT64 averages can have tiny rounding differences between executions. Tied
latest timestamps can select different last values, and units are selected without
ordering. These original rules are preserved. The earlier three-day comparison
confirmed matching admissions, notes, labels and diagnoses; mean differences were
within numerical tolerance, and all 28 last-value differences were explained by
raw timestamp ties. Do not assume byte-identical output on each rebuild.
