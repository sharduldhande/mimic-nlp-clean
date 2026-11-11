-- TEAM TEMPLATE: replace YOUR_PROJECT_ID with your Google Cloud project ID.
-- Creates/replaces YOUR_PROJECT_ID.mimic_project.dataset_splits.
-- Raw PhysioNet sources; one-day note/survival cutoff; invalid hospital times excluded.
CREATE SCHEMA IF NOT EXISTS `YOUR_PROJECT_ID.mimic_project`
OPTIONS(location = 'US');

CREATE OR REPLACE TABLE `YOUR_PROJECT_ID.mimic_project.dataset_splits` AS
WITH latest_icu AS (
    SELECT subject_id, hadm_id, stay_id, intime, outtime
    FROM `physionet-data.mimiciv_3_1_icu.icustays`
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY hadm_id ORDER BY intime DESC, stay_id DESC
    ) = 1
), cohort AS (
    SELECT
        a.subject_id, a.hadm_id, a.admittime, a.dischtime,
        i.stay_id, i.intime, i.outtime,
        p.gender,
        p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year AS age,
        a.admission_type, a.admission_location, a.insurance,
        DATETIME_DIFF(CAST(a.dischtime AS DATETIME), CAST(a.admittime AS DATETIME), HOUR) / 24.0
            AS length_of_stay,
        p.dod, a.deathtime, a.hospital_expire_flag
    FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
    JOIN latest_icu i ON a.subject_id = i.subject_id AND a.hadm_id = i.hadm_id
    JOIN `physionet-data.mimiciv_3_1_hosp.patients` p ON a.subject_id = p.subject_id
    -- Exclude invalid hospital timestamps, including admission 23252384.
    WHERE a.dischtime IS NOT NULL AND a.dischtime >= a.admittime
), diagnosis_features AS (
    SELECT c.hadm_id, d.icd_code, d.icd_version, d.seq_num, i.long_title AS diagnosis
    FROM cohort c
    JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
        ON c.subject_id = d.subject_id AND c.hadm_id = d.hadm_id
    LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` i
        ON d.icd_code = i.icd_code AND d.icd_version = i.icd_version
), diagnosis_aggregated AS (
    SELECT hadm_id,
        ARRAY_AGG(STRUCT(icd_code, icd_version, diagnosis) ORDER BY seq_num) AS diagnoses,
        COUNT(*) AS diagnosis_count
    FROM diagnosis_features
    GROUP BY hadm_id
), lab_measurements AS (
    SELECT c.hadm_id,
        CASE l.itemid
            WHEN 50912 THEN 'creatinine'
            WHEN 51222 THEN 'hemoglobin'
            WHEN 50983 THEN 'sodium'
            WHEN 50931 THEN 'glucose'
        END AS feature,
        l.charttime, l.valuenum AS value, l.valueuom AS unit
    FROM cohort c
    JOIN `physionet-data.mimiciv_3_1_hosp.labevents` l
        ON c.subject_id = l.subject_id AND c.hadm_id = l.hadm_id
        AND l.charttime BETWEEN c.intime AND c.outtime
    JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` d ON l.itemid = d.itemid
    WHERE l.itemid IN (50912, 51222, 50983, 50931)
        AND l.valuenum > 0 AND l.charttime <= c.dischtime
), vital_measurements AS (
    SELECT c.hadm_id,
        CASE v.itemid
            WHEN 220045 THEN 'heart_rate'
            WHEN 220179 THEN 'systolic_bp'
            WHEN 220180 THEN 'diastolic_bp'
            WHEN 220210 THEN 'respiratory_rate'
            WHEN 223761 THEN 'temperature'
            WHEN 223762 THEN 'temperature'
            WHEN 220277 THEN 'spo2'
        END AS feature,
        v.charttime,
        CASE WHEN v.itemid = 223761 THEN (v.valuenum - 32) / 1.8 ELSE v.valuenum END AS value,
        CASE WHEN v.itemid IN (223761, 223762) THEN '°C' ELSE v.valueuom END AS unit
    FROM cohort c
    JOIN `physionet-data.mimiciv_3_1_icu.chartevents` v
        ON c.stay_id = v.stay_id AND c.subject_id = v.subject_id
        AND v.charttime BETWEEN c.intime AND c.outtime
    JOIN `physionet-data.mimiciv_3_1_icu.d_items` d ON v.itemid = d.itemid
    WHERE v.itemid IN (220045, 220179, 220180, 220210, 223761, 223762, 220277)
        AND v.valuenum > 0 AND v.charttime <= c.dischtime
), all_measurements AS (
    SELECT * FROM lab_measurements
    UNION ALL
    SELECT * FROM vital_measurements
), measurement_summary AS (
    SELECT hadm_id, feature,
        AVG(value) AS mean_value,
        MIN(value) AS min_value,
        MAX(value) AS max_value,
        -- Preserve legacy choices: no additional tie-breaker or unit ordering.
        ARRAY_AGG(value ORDER BY charttime DESC LIMIT 1)[OFFSET(0)] AS last_value,
        MAX(charttime) AS last_time,
        ARRAY_AGG(unit IGNORE NULLS LIMIT 1)[SAFE_OFFSET(0)] AS unit,
        COUNT(*) AS measurement_count
    FROM all_measurements
    GROUP BY hadm_id, feature
), feature_list AS (
    SELECT feature FROM UNNEST([
        'creatinine', 'hemoglobin', 'sodium', 'glucose', 'heart_rate',
        'systolic_bp', 'diastolic_bp', 'respiratory_rate', 'temperature', 'spo2'
    ]) AS feature
), complete_features AS (
    SELECT c.hadm_id, f.feature, s.mean_value, s.min_value, s.max_value,
        s.last_value, s.last_time, s.unit,
        IF(s.measurement_count IS NULL, 1, 0) AS is_missing
    FROM cohort c
    CROSS JOIN feature_list f
    LEFT JOIN measurement_summary s ON c.hadm_id = s.hadm_id AND f.feature = s.feature
), measurement_arrays AS (
    SELECT hadm_id,
        ARRAY_AGG(STRUCT(feature, mean_value, min_value, max_value,
                         last_value, last_time, unit, is_missing) ORDER BY feature) AS measurements
    FROM complete_features
    GROUP BY hadm_id
), mortality_labeled AS (
    SELECT subject_id, hadm_id, dischtime, dod,
        CASE
            WHEN dischtime IS NULL THEN NULL
            WHEN hospital_expire_flag = 1 OR deathtime IS NOT NULL THEN NULL
            WHEN dod IS NULL THEN 0
            WHEN DATE(dod) = DATE(dischtime) THEN NULL
            WHEN DATE(dod) > DATE(dischtime)
                AND DATE(dod) <= DATE_ADD(DATE(dischtime), INTERVAL 30 DAY) THEN 1
            ELSE 0
        END AS mortality_30d
    FROM cohort
), mortality_labels AS (
    SELECT subject_id, hadm_id, mortality_30d
    FROM mortality_labeled
    WHERE mortality_30d IS NOT NULL
        AND (dod IS NULL OR DATE(dod) >= DATE(dischtime))
), discharge_notes AS (
    SELECT note_id, subject_id, hadm_id, charttime, storetime, text
    FROM `physionet-data.mimiciv_note.discharge`
    WHERE text IS NOT NULL AND LENGTH(TRIM(text)) > 0
), matched AS (
    SELECT
        c.subject_id, c.hadm_id, c.admittime, c.dischtime, c.stay_id,
        c.gender, c.age, c.admission_type, c.admission_location, c.insurance, c.length_of_stay,
        d.diagnoses, d.diagnosis_count, a.measurements,
        m.mortality_30d, n.note_id, n.charttime AS note_charttime,
        n.storetime AS note_storetime, n.text AS discharge_note
    FROM cohort c
    LEFT JOIN diagnosis_aggregated d ON c.hadm_id = d.hadm_id
    LEFT JOIN measurement_arrays a ON c.hadm_id = a.hadm_id
    JOIN mortality_labels m ON c.subject_id = m.subject_id AND c.hadm_id = m.hadm_id
    JOIN discharge_notes n ON c.subject_id = n.subject_id AND c.hadm_id = n.hadm_id
    WHERE CAST(n.storetime AS DATETIME) <= DATETIME_ADD(CAST(c.dischtime AS DATETIME), INTERVAL 1 DAY)
        AND (c.dod IS NULL OR DATE(c.dod) > DATE(DATETIME_ADD(CAST(c.dischtime AS DATETIME), INTERVAL 1 DAY)))
), patients AS (
    SELECT DISTINCT subject_id FROM matched
), patient_hashes AS (
    SELECT subject_id,
        MOD(MOD(FARM_FINGERPRINT(CONCAT(CAST(subject_id AS STRING), '_1')), 10000) + 10000, 10000)
            AS random_value
    FROM patients
), patient_splits AS (
    SELECT subject_id,
        CASE WHEN random_value < 7000 THEN 'train'
             WHEN random_value < 8500 THEN 'validation'
             ELSE 'test' END AS split
    FROM patient_hashes
)
SELECT m.*, s.split
FROM matched m
JOIN patient_splits s ON m.subject_id = s.subject_id;
