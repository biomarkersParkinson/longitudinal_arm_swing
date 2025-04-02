from collections import defaultdict
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path("src/").resolve()))

from longitudinal_arm_swing.constants import *
from longitudinal_arm_swing.util import convert_defaultdict_to_dict, extract_med_info, get_med_ids, \
    get_no_med_ids, get_start_med_ids, add_updrs_columns, set_data_types, \
        determine_affected_side_watch_side_mapping, generate_ids_bins, determine_excluded_ids_med_split

# Configuration

# ---------
# Load data
# ---------

# IDs processed per step
step_ids = {}
with open(os.path.join(PATH_IDS, PROCESSED_IDS_FILENAME), 'r') as f:
    step_ids = json.load(f)

# Statistics on processed data
stats = {}
for dataset in DATASETS:
    with open(os.path.join(PATH_STATS, STATS_FILENAMES_PER_DATASET[dataset]), 'r') as f:
        stats[dataset] = json.load(f)

# Clinical data
clinical_data = defaultdict(lambda: defaultdict(dict))
for dataset in DATASETS:
    for visit_nr, visit_filename in VISIT_FILENAMES_PER_DATASET[dataset].items():
        clinical_data[dataset]['visits'][visit_nr] = pd.read_csv(os.path.join(PATHS_PER_DATASET[dataset]['clinical'], visit_filename))
        
        if dataset == 'ppp':
            clinical_data[dataset]['visits'][visit_nr]['id'] = clinical_data[dataset]['visits'][visit_nr]['id'].apply(
                lambda x: f'POMU{x[:16]}'
            )

clinical_data['ppp']['ledd'] = pd.read_csv(
    os.path.join(PATHS_PER_DATASET['ppp']['clinical'], LEDD_PPP_FILENAME)
)

clinical_data = convert_defaultdict_to_dict(clinical_data)

# Visit-week combinations
with open(os.path.join(PATH_CLINICAL_DATA, VISIT_WEEKS_FILENAME), 'r') as f:
    visit_weeks = json.load(f)

# Start medication week
start_med_week_dict = {
    'ppp': pd.read_csv(os.path.join(PATH_CLINICAL_DATA, 'ppp', PPP_START_MED_FILENAME)),
    'denovo': pd.read_csv(os.path.join(PATH_CLINICAL_DATA, 'denovo', DENOVO_START_MED_WEEK_FILENAME))
}

# Digital measures
digital_measures = defaultdict(lambda: defaultdict(dict))
for dataset in DATASETS:
    for week_nr in WEEKS_PER_DATASET[dataset]:
        aggregation_week_path = os.path.join(PATHS_PER_DATASET[dataset]['aggregations'], str(week_nr))
        for filename in os.listdir(aggregation_week_path):
            subject = filename.split('.')[0]
            with open(os.path.join(aggregation_week_path, filename), 'r') as f:
                digital_measures[dataset][week_nr][subject] = json.load(f)

digital_measures = convert_defaultdict_to_dict(digital_measures)

# ------------
# Prepare data
# ------------

# Determine whether participants have sufficient sensor data
sufficient_sensor_data = defaultdict(lambda: defaultdict(dict))
for dataset in DATASETS:
    for week, week_data in stats[dataset].items():
        sufficient_sensor_data[dataset][week] = {
            subject: sum(
                week_data[subject]['hours_data'][weekday]['hours_between_8_and_22'] >= 10
                for weekday in week_data[subject]['hours_data']
            ) >= 3
            for subject in week_data.keys()
        }

sufficient_sensor_data = convert_defaultdict_to_dict(sufficient_sensor_data)

# Prepare medication info
clinical_data['ppp']['visits'][3].loc[clinical_data['ppp']['visits'][3]['id'] == 'POMU30163E78B5A9CCAB', 'ParkinMedUser'] = 0  # Starts medication AFTER study
clinical_data['ppp']['visits'][2].loc[clinical_data['ppp']['visits'][2]['id'] == 'POMU40FDD170B7D6C89E', 'ParkinMedUser'] = 0  # Does not start medication
clinical_data['ppp']['visits'][2].loc[clinical_data['ppp']['visits'][2]['id'] == 'POMU42A89EF7BECED9AD', 'ParkinMedUser'] = 0  # Does not start medication

med_info_ids = {'visits': {}, 'groups': {}}
denovo_med_colnames = {1: 'ParkinMedUser', 2: 'Up3OfParkMedic', 3: 'Up3OfParkMedic'}

# Extract medication status per visit for each dataset
for visit in [1, 2, 3]:
    med_info_ids['visits'][visit] = {
        'ppp': extract_med_info(clinical_data['ppp']['visits'][visit], 'ParkinMedUser'),
        'denovo': extract_med_info(clinical_data['denovo']['visits'][visit], denovo_med_colnames[visit])
    }

# For each dataset, assign participants to groups based on medication status
med_info_ids['groups'] = {
    'med': {group: get_med_ids(med_info_ids, group) for group in PD_DATASETS},
    'no_med': {group: get_no_med_ids(med_info_ids, group) for group in PD_DATASETS},
    'start_med': {group: get_start_med_ids(med_info_ids, group) for group in PD_DATASETS},
}

# Move participants who start medication after study to the no_med group
for dataset in PD_DATASETS:
    start_med_ids_dataset = start_med_week_dict[dataset]['ID'].values
    for id in start_med_ids_dataset:
        final_visit_week = visit_weeks[dataset][id]["visit3"]
        if final_visit_week is None:
            final_visit_week = 104
        start_med_week = start_med_week_dict[dataset].loc[start_med_week_dict[dataset]['ID'] == id, 'StartWeek'].values[0]
        if start_med_week >= final_visit_week:
            start_med_week_dict[dataset] = start_med_week_dict[dataset][start_med_week_dict[dataset]['ID'] != id]
            if id in med_info_ids['groups']['start_med'][dataset]:
                med_info_ids['groups']['start_med'][dataset].remove(id)
            if id not in med_info_ids['groups']['no_med'][dataset]:
                med_info_ids['groups']['no_med'][dataset].append(id)

med_info_full_path = os.path.join(PATH_IDS, 'med_info_ids.json')
with open(med_info_full_path, 'w') as f:
    json.dump(med_info_ids, f)

# Preprocess per dataset
dataset_ids = {}
affected_side_ids = {}
watch_side_dict = defaultdict(lambda: defaultdict(dict))
final_dfs = []

for dataset in DATASETS:
    # Load IDs
    dataset_ids[dataset] = list(pd.concat(
        [clinical_data[dataset]['visits'][visit_nr]['id'] for visit_nr in VISITS_PER_DATASET[dataset]]
    ).unique())

    # Set data types
    for visit_nr in VISITS_PER_DATASET[dataset]:
        df_visit = clinical_data[dataset]['visits'][visit_nr] 

        if dataset != 'controls':
            if visit_nr == 1:
                numeric_cols = UPDRS_COLS_PER_DATASET[dataset] + NUMERIC_COLS_VISIT_1_PER_DATASET[dataset]
            else:
                numeric_cols = UPDRS_COLS_PER_DATASET[dataset] + NUMERIC_COLS_VISITS

            df_visit = set_data_types(df_visit, numeric_cols)

        # Apply mappings
        df_visit['WatchSide'] = df_visit['WatchSide'].apply(pd.to_numeric, errors='coerce').apply(lambda x: WATCH_SIDE_MAPPING[x] if pd.notna(x) else np.nan)
        df_visit.loc[:, 'PrefHand'] = df_visit['PrefHand'].apply(pd.to_numeric, errors='coerce')

        if dataset != 'controls' and visit_nr == 1:
            df_visit['MostAffSide'] = df_visit['MostAffSide'].apply(pd.to_numeric, errors='coerce').map(AFFECTED_SIDE_MAPPING)      

        # Determine the watch side of participants per visit
        for subject in dataset_ids[dataset]:
            df_subject = df_visit.loc[df_visit['id'] == subject]
        
            if df_subject.shape[0] > 0:
                watch_side_dict[dataset][subject][str(visit_nr)] = df_subject['WatchSide'].values[0]
            else:
                watch_side_dict[dataset][subject][str(visit_nr)] = np.nan

        clinical_data[dataset]['visits'][visit_nr] = df_visit

    # Years since diagnosis (float)
    clinical_data['ppp']['visits'][1]['YearsSinceDiagFloat'] = clinical_data['ppp']['visits'][1]['MonthSinceDiag'] / 12
    clinical_data['denovo']['visits'][1].loc[:, 'YearsSinceDiagFloat'] = clinical_data['denovo']['visits'][1].apply(
        lambda x: (x['AssessYear'] + x['AssessMonth'] / 12) -  (x['DiagParkYear'] + x['DiagParkMonth'] / 12), axis=1 
    )

    # Determine the watch side of participants based on the three visits
    for subject, watch_side_vals in watch_side_dict[dataset].items():
        # Option 1: all three are the same and either 'left' or 'right'
        if len(set(watch_side_vals.values())) == 1 and watch_side_vals['1'] in ['left', 'right']:
            watch_side_dict[dataset][subject]['final'] = watch_side_vals['1']
        # Option 2: all three are the same but all nan
        elif all(pd.isna(val) for val in watch_side_vals.values()):
            watch_side_dict[dataset][subject]['final'] = 'unknown'
        # Option 3: excluding nans, all are the same
        elif len(set([val for val in watch_side_vals.values() if pd.notna(val)])) == 1:
            watch_side_dict[dataset][subject]['final'] = [val for val in watch_side_vals.values() if pd.notna(val)][0]
        # Option 4: excluding nans, there are two values
        elif len(set([val for val in watch_side_vals.values() if pd.notna(val)])) == 2:
            watch_side_dict[dataset][subject]['final'] = 'switched'

    with open(os.path.join(PATH_CLINICAL_DATA, 'watch_side_dict.json'), 'w') as f:
        json.dump(watch_side_dict, f, indent=2)

    for visit_nr in VISITS_PER_DATASET[dataset]:
        df_visit = clinical_data[dataset]['visits'][visit_nr] 

        # Change watch side to the final value if it is not 'switched'
        df_visit['WatchSide'] = df_visit['id'].apply(
            lambda x: watch_side_dict[dataset][x]['final'] 
            if watch_side_dict[dataset][x]['final'] != 'switched'
        else watch_side_dict[dataset][x][str(visit_nr)]
        )

        # Create UPDRS columns
        if dataset in PD_DATASETS:
            df_visit = add_updrs_columns(df_visit)

        clinical_data[dataset]['visits'][visit_nr] = df_visit

    # Determine if watch is worn on most affected side
    if dataset in PD_DATASETS:
        visit_1_df = clinical_data[dataset]['visits'][1]
        affected_side_ids[dataset] = determine_affected_side_watch_side_mapping(visit_1_df)

    for visit_nr in VISITS_PER_DATASET[dataset]:
        df_visit = clinical_data[dataset]['visits'][visit_nr].copy()

        cols_visit_dataset = STORE_CLINICAL_COLS_PER_DATASET[dataset][visit_nr]
        if dataset in PD_DATASETS:
            cols_visit_dataset += [col for col in df_visit.columns if col.startswith('updrs_')]

        df_visit = df_visit[cols_visit_dataset]

        df_visit.loc[:, 'dataset'] = dataset
        df_visit.loc[:, 'visit'] = visit_nr

        if dataset == 'ppp':
            # Add LEDD to ppp data
            ledd_df = clinical_data[dataset]['ledd'][['id', f'Visit{visit_nr}', f'Visit{visit_nr}_missing_med']]
            ledd_df.columns = ['id', 'ledd', 'missing_med']
            df_visit = df_visit.merge(ledd_df, on='id', how='left') 

        final_dfs.append(df_visit)

with open(os.path.join(PATH_IDS, 'affected_side_ids.json'), 'w') as f:
    json.dump(affected_side_ids, f, indent=2)

with open(os.path.join(PATH_IDS, 'dataset_ids.json'), 'w') as f:
    json.dump(dataset_ids, f, indent=2)

df_clinical = pd.concat(final_dfs, ignore_index=True).reset_index(drop=True)

for dataset in PD_DATASETS:
    df_clinical.loc[df_clinical['dataset'] == dataset, 'affected_side'] = df_clinical.loc[df_clinical['dataset'] == dataset, 'id'].map(
        affected_side_ids[dataset]
    )

# Participants using a walking aid
df_clinical['walking_aid'] = df_clinical['Updrs2It25'] >= 2

# Participants with significant dyskinesia
df_clinical['at_least_slight_dyskinesia'] = df_clinical['MotComDysKinTime'].isin(AT_LEAST_SLIGHT_DYSKINESIA_VALS)
df_clinical['at_least_significant_dyskinesia'] = df_clinical['MotComDysKinTime'].isin(AT_LEAST_SIGNIFICANT_DYSKINESIA_VALS)

# Save data
df_clinical.to_parquet(os.path.join(PATH_CLINICAL_DATA, 'clinical_data.parquet'))
        
# Prepare digital measures
dysk_start_exclude = 'significant'  # slight or significant

min_valid_days = 2
valid_arm_swing_days = {}

for dataset in DATASETS:
    valid_arm_swing_days[dataset] = {}
    for week in digital_measures[dataset]:
        valid_arm_swing_days[dataset][week] = {}
        for subject in digital_measures[dataset][week].keys():
            n_valid_days = len([day for day, n_minutes in digital_measures[dataset][week][subject]['filtered']['minutes_of_data'].items() if n_minutes >= 2])
            valid_arm_swing_days[dataset][week][subject] = n_valid_days

store_ids_by_category = {step: {dataset: {} for dataset in DATASETS} for step in ANALYSIS_STEPS}
ids_srm_weeks = {}

excluded_ids_by_category = {
    step: {
        'clinical': {dataset: {} for dataset in DATASETS},
        'measurement': {dataset: {} for dataset in DATASETS}
    }
    for step in ANALYSIS_STEPS
}

ids_remaining_after_exclusions = {}
for analysis in ANALYSIS_STEPS:
    ids_remaining_after_exclusions[analysis] = {}

start_med_ids = {}
for dataset in DATASETS:
    ids_remaining_after_exclusions[dataset] = {}

    df_clinical_dataset = df_clinical[df_clinical['dataset'] == dataset]
    start_ids = dataset_ids[dataset]

    ids_bins = generate_ids_bins(df_clinical_dataset, dataset, start_ids, watch_side_dict[dataset], dysk_start_exclude)

    excluded_ids_by_category['cs']['clinical'][dataset] = {
        'Walking aid': ids_bins['walking_aid_visit_1'],
        f'At least {dysk_start_exclude} dyskinesia': ids_bins[f'at_least_{dysk_start_exclude}_dyskinesia_visit_1'],
        'Watch side unknown': ids_bins['watch_side_unknown'],
        'No clinical data': ids_bins['no_clinical_data_visit_1'],
    }

    # Calculate exclusions for clinical data
    cs_excluded_participants = set([x for y in excluded_ids_by_category['cs']['clinical'][dataset].values() for x in y])
    cs_ids_remaining_after_clinical_exclusion = set(start_ids) - cs_excluded_participants

    # Excluded for measurement data reasons
    ids_remaining = cs_ids_remaining_after_clinical_exclusion

    for analysis, week in CS_ANALYSIS_WEEKS.items():
        # Perform exclusion checks based on availability of measurement data (sequential)
        excluded_ids_cs_analysis_measurement = {
            'No converted data': lambda x: x not in step_ids[f'Week{week}']['converted_data'],
            'No preprocessed data': lambda x: x not in step_ids[f'Week{week}']['preprocessed_data'],
            'Insufficient sensor data': lambda x: x in [x for x, y in sufficient_sensor_data[dataset][str(week)].items() if not y],
            'No gait predicted': lambda x: x not in step_ids[f'Week{week}']['predicted_gait_data'],
            'No arm swing predicted': lambda x: x not in digital_measures[dataset][week],
            'No arm swing predicted in vlong gait segments': lambda x: 'very_long' not in digital_measures[dataset][week].get(x, {}).get('filtered', ''),
            'Less than 2 valid arm swing days': lambda x: valid_arm_swing_days[dataset][week][x] < 2,
        }

        for category, condition in excluded_ids_cs_analysis_measurement.items():
            ids_excluded = [x for x in ids_remaining if condition(x)]
            ids_remaining = [x for x in ids_remaining if x not in ids_excluded]
            excluded_ids_by_category[analysis]['measurement'][dataset][category] = ids_excluded

        ids_remaining_after_exclusions[analysis][dataset] = ids_remaining

    # Longitudinal analyses
    # From here on, participants starting with medication are split into med and no_med parts by
    # adding "_med" for medicated and "_no_med" for non-medicated stages to the ID.
    # These participants are only included in the L1TF, not in the SRM or regression analyses.

    start_ids = dataset_ids[dataset]
    ids_bins = generate_ids_bins(df_clinical_dataset, dataset, start_ids, watch_side_dict[dataset], dysk_start_exclude)

    # L1TF
    # Clinical: Exclude participant with clinical reasons of CS but also in other visits
    # Measurement: Require at minimum one valid week in each N weeks
    #     NOTE: start_med should be split into med and no_med for measurement data
    
    base_exclude_l1tf_clinical = {
        'Walking aid': list(set(sum([ids_bins[f'walking_aid_visit_{i}'] for i in VISITS_PER_DATASET[dataset]], []))),
        f'At least {dysk_start_exclude} dyskinesia': list(set(sum([ids_bins[f'at_least_{dysk_start_exclude}_dyskinesia_visit_{i}'] for i in VISITS_PER_DATASET[dataset]], []))),
        'Watch side unknown': ids_bins['watch_side_unknown'],
        'Watch side switched': ids_bins['watch_side_switched'],
        'Alternative diagnosis': [x for x in IDS_ALTERNATIVE_DIAGNOSIS if x in dataset_ids[dataset]],
        'LEDD missing in visit 1': [x for x in IDS_LEDD_MISSING_VISIT_1 if x in dataset_ids[dataset]],
        'Start med time unknown': [x for x in IDS_START_MED_TIME_UNKNOWN if x in dataset_ids[dataset]],
        'Detailed med info missing visit 1': [x for x in IDS_MED_INFO_PARTICIPANT_MISSING if x in dataset_ids[dataset]],
    }

    excluded_ids_by_category['l1tf']['clinical'][dataset], l1tf_remaining_ids_after_clinical_exclusion = determine_excluded_ids_med_split(
        dataset_ids[dataset], med_info_ids, dataset, base_exclude_l1tf_clinical
    )

    # Measurement
    
    # Subjects should have at least 8 valid weeks (of sensor data and arm swing)
    # for start_med: split into med and no_med and then apply requirement
    minimum_valid_weeks = 8
    measurement_ids = l1tf_remaining_ids_after_clinical_exclusion.copy()

    # Split start_med participants into med and non-med parts for l1tf
    if dataset in PD_DATASETS:
        for row in start_med_week_dict[dataset].iterrows():
            start_med_participant = row[1]['ID']
            start_med_week = row[1]['StartWeek']

            range_no_med = range(1, start_med_week)
            range_med = range(start_med_week, 104)

            if start_med_participant in measurement_ids:
                for week in range_no_med:
                    if str(week) in sufficient_sensor_data[dataset] and start_med_participant in sufficient_sensor_data[dataset][str(week)]:
                        sufficient_sensor_data[dataset][str(week)][f'{start_med_participant}_no_med'] = sufficient_sensor_data[dataset][str(week)][start_med_participant]
                        del sufficient_sensor_data[dataset][str(week)][start_med_participant]

                    if week in valid_arm_swing_days[dataset] and start_med_participant in valid_arm_swing_days[dataset][week]:
                        valid_arm_swing_days[dataset][week][f'{start_med_participant}_no_med'] = valid_arm_swing_days[dataset][week][start_med_participant]
                        del valid_arm_swing_days[dataset][week][start_med_participant]

                for week in range_med:
                    if str(week) in sufficient_sensor_data[dataset] and start_med_participant in sufficient_sensor_data[dataset][str(week)]:
                        sufficient_sensor_data[dataset][str(week)][f'{start_med_participant}_med'] = sufficient_sensor_data[dataset][str(week)][start_med_participant]
                        del sufficient_sensor_data[dataset][str(week)][start_med_participant]

                    if week in valid_arm_swing_days[dataset] and start_med_participant in valid_arm_swing_days[dataset][week]:
                        valid_arm_swing_days[dataset][week][f'{start_med_participant}_med'] = valid_arm_swing_days[dataset][week][start_med_participant]
                        del valid_arm_swing_days[dataset][week][start_med_participant]

                # Split l1tf remaining ids into med and no_med parts
                if len(range_no_med) > 0:
                    measurement_ids.append(f'{start_med_participant}_no_med')
                if len(range_med) > 0:
                    measurement_ids.append(f'{start_med_participant}_med')
                if len(range_no_med) > 0 or len(range_med) > 0:
                    measurement_ids.remove(start_med_participant)
                    l1tf_remaining_ids_after_clinical_exclusion.remove(start_med_participant)
    
    all_sensor_data_ids = [
        x for week in sufficient_sensor_data[dataset] 
        for x in sufficient_sensor_data[dataset][str(week)] 
        if x in measurement_ids
    ]
    accumulated_ids_by_valid_sensor_data_weeks = [
        x for week in sufficient_sensor_data[dataset] 
        for x, y in sufficient_sensor_data[dataset][str(week)].items() 
        if y and x in measurement_ids
    ]
    no_sensor_data_ids = [
        id for id in measurement_ids if id not in all_sensor_data_ids
    ]

    insufficient_sensor_data_ids = [
        id for id in set(accumulated_ids_by_valid_sensor_data_weeks) 
        if accumulated_ids_by_valid_sensor_data_weeks.count(id) < minimum_valid_weeks 
        and id in all_sensor_data_ids
    ]

    l1tf_remaining_ids_after_invalid_sensor_data = [
        id for id in measurement_ids 
        if id not in insufficient_sensor_data_ids and id in all_sensor_data_ids
    ]

    # If either 'med' or 'no_med' part of a participant has sufficient sensor data, keep the participant (but only that part)
    no_arm_swing_ids = [
        id for id in l1tf_remaining_ids_after_invalid_sensor_data
        if id not in [x for week in valid_arm_swing_days[dataset] for x in valid_arm_swing_days[dataset][week]]
    ]
    accumulated_ids_by_valid_arm_swing_weeks = [
        x for week in valid_arm_swing_days[dataset] 
        for x, y in valid_arm_swing_days[dataset][week].items() 
        if y >= min_valid_days and x in l1tf_remaining_ids_after_invalid_sensor_data
    ]
    insufficient_arm_swing_ids = [
        id for id in set(accumulated_ids_by_valid_arm_swing_weeks) 
        if accumulated_ids_by_valid_arm_swing_weeks.count(id) < minimum_valid_weeks
        and id not in no_arm_swing_ids
    ]

    base_exclude_l1tf_measurement = {
        f'No converted data in any week': sorted(no_sensor_data_ids),
        f'Insufficient preprocessed data in less than {minimum_valid_weeks} valid weeks': sorted(insufficient_sensor_data_ids),
        f'No valid arm swing data in any week': sorted(no_arm_swing_ids),
        f'Insufficient arm swing data in less than {minimum_valid_weeks} valid weeks': sorted(insufficient_arm_swing_ids)
    }

    excluded_ids_by_category['l1tf']['measurement'][dataset], ids_remaining_after_exclusions['l1tf'][dataset] = determine_excluded_ids_med_split(
        measurement_ids, med_info_ids, dataset, base_exclude_l1tf_measurement
    )

    # Exclude participants which due to split were missed in the previous step


    # ---
    # SRM
    # ---

    # From here on, we exclude participants starting medication during the study.
    # We set them to no_med, and exclude them for starting medication during the study.
    if dataset in PD_DATASETS:
        start_med_ids[dataset] = med_info_ids['groups']['start_med'][dataset].copy()
        for subject in start_med_ids[dataset]:
            med_info_ids['groups']['no_med'][dataset].append(subject)
            med_info_ids['groups']['start_med'][dataset].remove(subject)

        del med_info_ids['groups']['start_med'][dataset]
    else:
        start_med_ids[dataset] = []

    # Clinical: Similar to L1TF, but also exclude those starting medication in study
    start_ids = dataset_ids[dataset]
    ids_bins = generate_ids_bins(df_clinical_dataset, dataset, start_ids, watch_side_dict[dataset], dysk_start_exclude)

    base_exclude_srm_clinical = {
        'Walking aid': list(set(sum([ids_bins[f'walking_aid_visit_{i}'] for i in VISITS_PER_DATASET[dataset]], []))),
        f'At least {dysk_start_exclude} dyskinesia': list(set(sum([ids_bins[f'at_least_{dysk_start_exclude}_dyskinesia_visit_{i}'] for i in VISITS_PER_DATASET[dataset]], []))),
        'Watch side unknown': ids_bins['watch_side_unknown'],
        'Watch side switched': ids_bins['watch_side_switched'],
        'Starting medication during study': [x for x in start_med_ids[dataset] if x in dataset_ids[dataset]],
        'Alternative diagnosis': [x for x in IDS_ALTERNATIVE_DIAGNOSIS if x in dataset_ids[dataset]],
        'LEDD missing in visit 1': [x for x in IDS_LEDD_MISSING_VISIT_1 if x in dataset_ids[dataset]],
        'Start med time unknown': [x for x in IDS_START_MED_TIME_UNKNOWN if x in dataset_ids[dataset]],
        'Detailed med info missing visit 1': [x for x in IDS_MED_INFO_PARTICIPANT_MISSING if x in dataset_ids[dataset]],
    }
    
    excluded_ids_by_category['srm']['clinical'][dataset], srm_remaining_ids_after_clinical_exclusion = determine_excluded_ids_med_split(
        dataset_ids[dataset], med_info_ids, dataset, base_exclude_srm_clinical
    )

    # Measurement
    #   Subjects should have valid arm swing in any of weeks 2, 4, 6, and any of 
    #   weeks 96, 98, 100 (PPP, DeNovo) / 48, 50, 52 (PSP)
    ids_with_valid_start_weeks = set([x for week in [2, 4, 6] for x, y in valid_arm_swing_days[dataset][week].items() if y >= min_valid_days and x in srm_remaining_ids_after_clinical_exclusion])
    ids_without_valid_start_weeks = [id for id in srm_remaining_ids_after_clinical_exclusion if id not in ids_with_valid_start_weeks]

    srm_remaining_ids_after_start_weeks_exclusion = [ids for ids in srm_remaining_ids_after_clinical_exclusion if ids not in ids_without_valid_start_weeks]

    if dataset == 'controls':
        ids_with_valid_end_weeks = set([x for week in [48, 50, 52] for x, y in valid_arm_swing_days[dataset][week].items() if y >= min_valid_days and x in srm_remaining_ids_after_start_weeks_exclusion])
    else:
        ids_with_valid_end_weeks = set([x for week in [96, 98, 100] for x, y in valid_arm_swing_days[dataset][week].items() if y >= min_valid_days and x in srm_remaining_ids_after_start_weeks_exclusion])

    ids_without_valid_end_weeks = [ids for ids in srm_remaining_ids_after_start_weeks_exclusion if ids not in ids_with_valid_end_weeks]
    srm_remaining_ids_after_final_weeks_exclusion = [id for id in srm_remaining_ids_after_start_weeks_exclusion if id not in ids_without_valid_end_weeks]

    base_exclude_srm_measurement = {
        'No valid arm swing data in weeks 2, 4, 6': sorted(ids_without_valid_start_weeks),
        'No valid arm swing data in weeks 96, 98, 100 (PPP, DeNovo) / 48, 50, 52 (Controls)': sorted(ids_without_valid_end_weeks),
    }

    excluded_ids_by_category['srm']['measurement'][dataset], ids_remaining_after_exclusions['srm'][dataset] = determine_excluded_ids_med_split(
        srm_remaining_ids_after_clinical_exclusion, med_info_ids, dataset, base_exclude_srm_measurement
    )

    # ----------
    # Regression
    # ----------

    # Clinical: Similar to SRM, but add additional doubtful LEDDs and diagnosis dates
    base_exclude_regr_clinical = base_exclude_srm_clinical.copy()
    base_exclude_regr_clinical.update({
        'Use anticholinergic meds': sorted([x for x in IDS_USE_ANTICHOLINERGIC_MEDS if x in dataset_ids[dataset]]),
        'LEDD doubtful': sorted([x for x in IDS_LEDD_DOUBTFUL if x in dataset_ids[dataset]]),
        'Diagnosis date doubtful': sorted([x for x in IDS_DIAGNOSIS_DATE_DOUBTFUL if x in dataset_ids[dataset]]),
    })

    excluded_ids_by_category['regr']['clinical'][dataset], regr_remaining_ids_after_clinical_exclusion = determine_excluded_ids_med_split(
        dataset_ids[dataset], med_info_ids, dataset, base_exclude_regr_clinical
    )

    base_exclude_regr_measurement = base_exclude_srm_measurement.copy()
    excluded_ids_by_category['regr']['measurement'][dataset], ids_remaining_after_exclusions['regr'][dataset] = determine_excluded_ids_med_split(
        regr_remaining_ids_after_clinical_exclusion, med_info_ids, dataset, base_exclude_regr_measurement
    )

with open(os.path.join(PATH_IDS, 'ids_excluded_by_category.json'), 'w') as f:
    json.dump(excluded_ids_by_category, f, indent=2)

with open(os.path.join(PATH_IDS, 'ids_remaining_after_exclusions.json'), 'w') as f:
    json.dump(ids_remaining_after_exclusions, f, indent=2)

# ------------------------
# Prepare digital measures
# ------------------------

digital_measures_pre_df = []
for dataset, dataset_vals in digital_measures.items():
    for week_nr, week_vals in dataset_vals.items():
        for subject, subject_vals in week_vals.items():
            cross_sectional = True if subject in ids_remaining_after_exclusions['cs'][dataset] else False
            icc = True if subject in ids_remaining_after_exclusions['icc'][dataset] else False
            l1tf = True if subject in ids_remaining_after_exclusions['l1tf'][dataset] else False
            srm = True if subject in ids_remaining_after_exclusions['srm'][dataset] else False
            regr = True if subject in ids_remaining_after_exclusions['regr'][dataset] else False

            try:
                sufficient_arm_swing = False if valid_arm_swing_days[dataset][week_nr][subject] < min_valid_days else True
            except KeyError:
                # Participant started medication during the study
                start_med_week = start_med_week_dict[dataset].loc[start_med_week_dict[dataset]['ID'] == subject, 'StartWeek'].values[0]
                if start_med_week > week_nr:
                    suffix = '_no_med'
                else:
                    suffix = '_med'
                sufficient_arm_swing = False if valid_arm_swing_days[dataset][week_nr][f'{subject}{suffix}'] < min_valid_days else True

            if dataset != 'controls':
                try:
                    aff_side = 'mas' if subject in affected_side_ids[dataset]['mas'] else ('las' if subject in affected_side_ids[dataset]['las'] else 'unknown')
                except IndexError:
                    aff_side = np.nan

            for filter_type, filter_vals in subject_vals.items():
                for segment_category, category_vals in filter_vals.items():
                    if segment_category != 'minutes_of_data':
                        # Create a base dictionary with the metadata
                        row = {
                            'dataset': dataset,
                            'week': week_nr,
                            'id': subject,
                            'filter_type': filter_type,
                            'segment_category': segment_category,
                            'include_in_cross_sectional_analysis': cross_sectional,
                            'include_in_icc_analysis': icc,
                            'include_in_l1tf_analysis': l1tf,
                            'include_in_srm_analysis': srm,
                            'include_in_regression_analysis': regr,
                            'sufficient_arm_swing_this_week': sufficient_arm_swing
                        }

                        if dataset != 'controls': 
                            row['affected_side'] = aff_side

                        # Add all measures dynamically
                        row.update(category_vals)  # This adds all measure key-value pairs

                        digital_measures_pre_df.append(row)

                # Not very long segments
                avoid_segments = ['minutes_of_data', 'all_segment_categories', 'very_long', 'not_very_long']
                vals = {}
                duration_per_segment = [filter_vals[segment_length]['duration_s'] for segment_length in filter_vals if segment_length not in avoid_segments]
                if len(duration_per_segment) == 0:
                    continue

                duration_s = np.sum(duration_per_segment) 
                vals['duration_s'] = duration_s
                for measure in ['median_range_of_motion', '95p_range_of_motion', 'median_peak_velocity', '95p_peak_velocity']:
                    vals[measure] = np.sum([np.multiply(filter_vals[segment_length]['duration_s'], filter_vals[segment_length][measure]) / duration_s for segment_length in filter_vals if segment_length not in avoid_segments])

                row = {
                    'dataset': dataset,
                    'week': week_nr,
                    'id': subject,
                    'filter_type': filter_type,
                    'segment_category': 'not_very_long',
                    'include_in_cross_sectional_analysis': cross_sectional,
                    'include_in_icc_analysis': icc,
                    'include_in_l1tf_analysis': l1tf,
                    'include_in_srm_analysis': srm,
                    'include_in_regression_analysis': regr,
                    'sufficient_arm_swing_this_week': sufficient_arm_swing,
                }  

                if dataset != 'controls':
                    row['affected_side'] = aff_side

                row.update(vals)

                digital_measures_pre_df.append(row)              

# Convert to DataFrame
df_digital_measures = pd.DataFrame(digital_measures_pre_df)
df_digital_measures.loc[df_digital_measures['dataset'].isin(['ppp', 'denovo']), 'population'] = 'pd'
df_digital_measures.loc[df_digital_measures['dataset']=='controls', 'population'] = 'controls'

os.makedirs(os.path.join(PATH_PREPARED_DATA, 'measures'), exist_ok=True)

df_digital_measures.to_parquet(os.path.join(PATH_PREPARED_DATA, 'measures', 'digital_measures.parquet'), index=False)

l1tf_ids = ids_remaining_after_exclusions['l1tf']['ppp'] + ids_remaining_after_exclusions['l1tf']['denovo'] + ids_remaining_after_exclusions['l1tf']['controls']

for filter_type in ['filtered', 'unfiltered']:
    for segment_length in ['very_long', 'not_very_long']:

        os.makedirs(os.path.join(PATH_PREPARED_DATA, 'measures', f'{filter_type}_gait', f'{segment_length}_gait_segments'), exist_ok=True)

        # Store subset for L1-trend filter in Matlab
        for side in ['mas', 'las']:
            df_l1tf = df_digital_measures.loc[
                (df_digital_measures['id'].isin(l1tf_ids)) & 
                (df_digital_measures['filter_type']==filter_type) &
                (df_digital_measures['segment_category']==segment_length) &
                (df_digital_measures['affected_side']==side) &
                (df_digital_measures['week'] != 1) &
                (df_digital_measures['dataset'].isin(['denovo', 'ppp'])) &
                (df_digital_measures['sufficient_arm_swing_this_week'])
            ]

            for measure in ['median', '95p']:
                df_pivot = df_l1tf.pivot(index="id", columns="week", values=f'{measure}_range_of_motion').reset_index(drop=False).rename_axis(None, axis=1)

                ids = df_pivot['id'].values.tolist()

                # Ensure all columns are numeric
                df_pivot = df_pivot.drop(columns=['id']).apply(pd.to_numeric, errors='coerce')

                df_pivot.to_csv(os.path.join(PATH_PREPARED_DATA, 'measures', f'{filter_type}_gait', f'{segment_length}_gait_segments', f'pd_{side}_measure_{measure}.csv'), index=False)

            with open(os.path.join(PATH_PREPARED_DATA, 'measures', f'{filter_type}_gait', f'{segment_length}_gait_segments', f'pd_{side}_ids.txt'), 'w') as f:
                f.write('\n'.join(ids))

        df_l1tf = df_digital_measures.loc[
            (df_digital_measures['id'].isin(l1tf_ids)) & 
            (df_digital_measures['filter_type']==filter_type) &
            (df_digital_measures['segment_category']==segment_length) &
            (df_digital_measures['week'] != 1) &
            (df_digital_measures['dataset'] == 'controls') &
            (df_digital_measures['sufficient_arm_swing_this_week'])
        ]

        for measure in ['median', '95p']:
            df_pivot = df_l1tf.pivot(index="id", columns="week", values=f'{measure}_range_of_motion').reset_index(drop=False).rename_axis(None, axis=1)

            ids = df_pivot['id'].values.tolist()

            # Ensure all columns are numeric
            df_pivot = df_pivot.drop(columns=['id']).apply(pd.to_numeric, errors='coerce')

            df_pivot.to_csv(os.path.join(PATH_PREPARED_DATA, 'measures', f'{filter_type}_gait', f'{segment_length}_gait_segments', f'controls_measure_{measure}.csv'), index=False)

        with open(os.path.join(PATH_PREPARED_DATA, 'measures', f'{filter_type}_gait', f'{segment_length}_gait_segments', 'controls_ids.txt'), 'w') as f:
            f.write('\n'.join(ids))
