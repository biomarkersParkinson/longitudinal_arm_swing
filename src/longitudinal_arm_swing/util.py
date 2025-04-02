from collections import defaultdict
import pandas as pd
from pathlib import Path
import sys
from tabulate import tabulate
import warnings

sys.path.append(str(Path("src/").resolve()))

from longitudinal_arm_swing.constants import *

def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d

def extract_med_info(visit_data, med_colname):
    """Allocate participants into groups based on medication status."""
    return {
        'med': [id for id in visit_data.loc[visit_data[med_colname].isin([1, '1']), 'id'].tolist()],
        'no_med': [id for id in visit_data.loc[visit_data[med_colname].isin([0, '0']), 'id'].tolist()]
    }

def get_med_ids(med_info_ids, group):
    """Extract the participants who have started medication before the study."""
    return med_info_ids['visits'][1][group]['med']

def get_no_med_ids(med_info_ids, group):
    """Extract the participants who have never started medication."""
    return list(set([x for x in med_info_ids['visits'][1][group]['no_med'] if x in med_info_ids['visits'][2][group]['no_med'] and x in med_info_ids['visits'][3][group]['no_med']]))

def get_start_med_ids(med_info_ids, group):
    """Extract the participants who started levodopa medication between visit 1 and visit 3."""
    return list(set([x for x in med_info_ids['visits'][1][group]['no_med'] if x in med_info_ids['visits'][2][group]['med'] or x in med_info_ids['visits'][3][group]['med']]))

def set_data_types(df, numeric_cols):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        
        # Replace USER_MISSING with NaN
        df = df.replace(to_replace=r'.*USER_MISSING*.', value=np.nan, regex=True)

        # Convert numeric columns
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].apply(pd.to_numeric, errors='coerce')

        return df

def add_updrs_columns(visit_df):
    # Create masks for 'right' and 'left'
    right_mask = visit_df['WatchSide'] == 'right'
    left_mask = visit_df['WatchSide'] == 'left'

    # Initialize Series for new scores
    subscore_off_ws = pd.Series(index=visit_df.index, dtype=int)
    subscore_off_non_ws = pd.Series(index=visit_df.index, dtype=int)
    subscore_on_ws = pd.Series(index=visit_df.index, dtype=int)
    subscore_on_non_ws = pd.Series(index=visit_df.index, dtype=int)

    # Compute score for watch and non-watch side
    subscore_off_ws[right_mask] = visit_df.loc[right_mask, UPDRS_HYPOKINESIA_SIDE_MAPPING['Of']['hypokinesia']['right_side'] + UPDRS_HYPOKINESIA_SIDE_MAPPING['Of']['hypokinesia']['watch_side']].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
    subscore_off_non_ws[right_mask] = visit_df.loc[right_mask, UPDRS_HYPOKINESIA_SIDE_MAPPING['Of']['hypokinesia']['left_side'] + UPDRS_HYPOKINESIA_SIDE_MAPPING['Of']['hypokinesia']['non_watch_side']].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)         

    # Compute for left watch side
    subscore_off_ws[left_mask] = visit_df.loc[left_mask, UPDRS_HYPOKINESIA_SIDE_MAPPING['Of']['hypokinesia']['left_side'] + UPDRS_HYPOKINESIA_SIDE_MAPPING['Of']['hypokinesia']['watch_side']].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
    subscore_off_non_ws[left_mask] = visit_df.loc[left_mask, UPDRS_HYPOKINESIA_SIDE_MAPPING['Of']['hypokinesia']['right_side'] + UPDRS_HYPOKINESIA_SIDE_MAPPING['Of']['hypokinesia']['non_watch_side']].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
        
    # Compute all new columns at once
    new_columns = {
        'updrs_3_subscore_off_ws': subscore_off_ws,
        'updrs_3_subscore_off_non_ws': subscore_off_non_ws,
    }

    if dataset == 'ppp': 
        subscore_on_ws[right_mask] = visit_df.loc[right_mask, UPDRS_HYPOKINESIA_SIDE_MAPPING['On']['hypokinesia']['right_side'] + UPDRS_HYPOKINESIA_SIDE_MAPPING['On']['hypokinesia']['watch_side']].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
        subscore_on_non_ws[right_mask] = visit_df.loc[right_mask, UPDRS_HYPOKINESIA_SIDE_MAPPING['On']['hypokinesia']['left_side'] + UPDRS_HYPOKINESIA_SIDE_MAPPING['On']['hypokinesia']['non_watch_side']].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
        subscore_on_ws[left_mask] = visit_df.loc[left_mask, UPDRS_HYPOKINESIA_SIDE_MAPPING['On']['hypokinesia']['left_side'] + UPDRS_HYPOKINESIA_SIDE_MAPPING['On']['hypokinesia']['watch_side']].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
        subscore_on_non_ws[left_mask] = visit_df.loc[left_mask, UPDRS_HYPOKINESIA_SIDE_MAPPING['On']['hypokinesia']['right_side'] + UPDRS_HYPOKINESIA_SIDE_MAPPING['On']['hypokinesia']['non_watch_side']].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)

        new_columns['updrs_3_subscore_on_ws'] = subscore_on_ws
        new_columns['updrs_3_subscore_on_non_ws'] = subscore_on_non_ws

        new_columns['updrs_1_total'] = visit_df[UPDRS_PART_1_COLS].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
        new_columns['updrs_2_total'] = visit_df[UPDRS_PART_2_COLS].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
        new_columns['updrs_3_off_total'] = visit_df[UPDRS_PART_3_OFF_COLS].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
        new_columns['updrs_3_on_total'] = visit_df[UPDRS_PART_3_ON_COLS].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
    else:
        new_columns['updrs_1_total'] = visit_df[UPDRS_PART_1_COLS].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
        new_columns['updrs_2_total'] = visit_df[UPDRS_PART_2_COLS].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
        new_columns['updrs_3_off_total'] = visit_df[UPDRS_PART_3_OFF_COLS].apply(
            lambda row: row.sum() if row.isna().sum() == 0 else float('nan'), axis=1)
        new_columns['updrs_3_on_total'] = np.nan

    return pd.concat([visit_df, pd.DataFrame(new_columns)], axis=1)

def determine_affected_side_watch_side_mapping(df):
    mas_mask = df['updrs_3_subscore_off_ws'] > df['updrs_3_subscore_off_non_ws']
    las_mask = df['updrs_3_subscore_off_ws'] < df['updrs_3_subscore_off_non_ws']
    unknown_mask = df['updrs_3_subscore_off_ws'] == df['updrs_3_subscore_off_non_ws']

    mas_ids = list(df[mas_mask]['id'].values)
    las_ids = list(df[las_mask]['id'].values)
    rest_ids = list(df[unknown_mask]['id'].values)

    side_ids = {
        'mas': mas_ids,
        'las': las_ids,
        'rest': rest_ids,
    }

    mas_ids_subjective = [
        subject for subject in side_ids['rest'] 
        if (
            df.loc[df['id']==subject, 'MostAffSide'].values[0] in AFFECTED_SIDE_CATEGORIES['right'] and 
            df.loc[df['id']==subject, 'WatchSide'].values[0]=='right'
        ) or (
            df.loc[df['id']==subject,'MostAffSide'].values[0] in AFFECTED_SIDE_CATEGORIES['left'] and 
            df.loc[df['id']==subject, 'WatchSide'].values[0]=='left'
        )
    ]

    las_ids_subjective = [
        subject for subject in side_ids['rest'] 
        if (
            df.loc[df['id']==subject,'MostAffSide'].values[0] in AFFECTED_SIDE_CATEGORIES['right'] and 
            df.loc[df['id']==subject, 'WatchSide'].values[0]=='left'
        ) or (
            df.loc[df['id']==subject,'MostAffSide'].values[0] in AFFECTED_SIDE_CATEGORIES['left'] and 
            df.loc[df['id']==subject, 'WatchSide'].values[0]=='right'
        )
    ]

    rest_ids = [
        subject for subject in side_ids['rest'] 
        if subject not in mas_ids_subjective and subject not in las_ids_subjective
    ]

    side_ids['mas'] = side_ids['mas'] + mas_ids_subjective
    side_ids['las'] = side_ids['las'] + las_ids_subjective
    side_ids['rest'] = rest_ids

    return side_ids

def generate_ids_bins(df, dataset, start_ids, watch_side_dict, dysk_start_exclude):

    # Identify participants with unknown or switched watch sides
    watch_side_unknown, watch_side_switched = [], []
    for subject in watch_side_dict:
        watch_side_subject = watch_side_dict[subject]['final']
        if pd.isna(watch_side_subject):
            watch_side_unknown.append(subject)
        elif watch_side_subject == 'switched':
            watch_side_switched.append(subject)

    # Initialize dictionary to store per exclusion criteria the participants
    ids_bins = {}
    for visit_nr in VISITS_PER_DATASET[dataset]:
        df_visit = df.loc[df['visit'] == visit_nr]
        visit_ids = df_visit['id'].unique().tolist()
        walking_aid_ids = df_visit.loc[df_visit['walking_aid'] == True, 'id'].unique().tolist()
        dysk_ids = df_visit.loc[df_visit[f'at_least_{dysk_start_exclude}_dyskinesia'] == True, 'id'].unique().tolist()

        # Bin participants by clinical conditions
        ids_bins[f'no_clinical_data_visit_{visit_nr}'] = [x for x in start_ids if x not in visit_ids]
        ids_bins[f'walking_aid_visit_{visit_nr}'] = walking_aid_ids
        ids_bins[f'at_least_{dysk_start_exclude}_dyskinesia_visit_{visit_nr}'] = dysk_ids

    # Add watch side exclusions
    ids_bins['watch_side_unknown'] = watch_side_unknown
    ids_bins['watch_side_switched'] = watch_side_switched

    return ids_bins

def determine_excluded_ids_med_split(start_ids, med_info_ids, dataset, base_exclude_by_category_dict):
        excluded_ids_split_by_med = {}
        if dataset != 'controls':  
            med_groups = [group for group in med_info_ids['groups'] if dataset in med_info_ids['groups'][group]]       
            excluded_ids = []
            ids_remaining = []
            for med_status in med_groups:
                lng_ids_med_status = [id for id in start_ids if id.replace('_no_med', '').replace('_med', '') in med_info_ids['groups'][med_status][dataset]]

                excluded_ids_split_by_med[med_status] = {
                    k: sorted([
                        subject for subject in v 
                        if subject in lng_ids_med_status
                    ]) 
                    for k, v in base_exclude_by_category_dict.items()
                }

                # Flatten the excluded IDs and add to the total excluded list
                excluded_ids.extend([y for z in excluded_ids_split_by_med[med_status].values() for y in z])

                # Filter remaining IDs that are not excluded
                ids_remaining.extend([
                    x for x in lng_ids_med_status if 
                    all(x not in v for v in excluded_ids_split_by_med[med_status].values())
                ])
            
            ids_remaining = sorted(set(ids_remaining))

        else:
            excluded_ids_split_by_med = base_exclude_by_category_dict
            excluded_ids = [y for z in excluded_ids_split_by_med.values() for y in z]

            ids_remaining = sorted([x for x in start_ids if x not in excluded_ids])

        return excluded_ids_split_by_med, ids_remaining


def allocate_start_meds(med_info_ids):
    for dataset in PD_DATASETS:
        for subject in med_info_ids['groups']['start_med'][dataset]:
            med_info_ids['groups']['med'][dataset].append(subject)
            med_info_ids['groups']['no_med'][dataset].append(subject)

    return med_info_ids


def print_clinical_exclusions_table(excluded_ids_by_category, dataset_ids, med_info_ids, analyses):

    # Add start_med participants to med and no_med groups
    med_info_ids = allocate_start_meds(med_info_ids)

    for analysis in analyses:
        print(f"---- Analysis: {ANALYSIS_MAPPING[analysis]} ----\n")
        if analysis == 'l1tf':
            print("Note: number between brackets is the number of participants starting medication throughout the study excluded.\n")

        if analysis in ['l1tf', 'srm', 'regr']:
            datasets_analysis = {f'{pd_dataset} {med_status}': [x for x in dataset_ids[pd_dataset] if x in med_info_ids['groups'][med_status][pd_dataset]] for pd_dataset in PD_DATASETS for med_status in ['med', 'no_med']}
            datasets_analysis['controls'] = dataset_ids['controls']
        else:
            datasets_analysis = {dataset: dataset_ids[dataset] for dataset in DATASETS}
        
        headers = ['Category', *datasets_analysis] 

        if analysis != 'icc':
            remaining_ids_per_dataset = datasets_analysis.copy()
        
            table_data_clinical = []

            if analysis in ['l1tf', 'srm', 'regr']:
                row = ['Starting number of participants with\nknown medication status']
            else:
                row = ['Starting number of participants']
                
            for dataset_med in datasets_analysis:
                n_start = len(set(remaining_ids_per_dataset[dataset_med]))
                row.append(n_start)
            table_data_clinical.append(row)

            dashed_row = ['-'*len(category) for category in ['Category'] + list(datasets_analysis.keys())]
            table_data_clinical.append(dashed_row) 

            for category in excluded_ids_by_category[analysis]['clinical']['controls'].keys():
                row = [category]
                for dataset_med in datasets_analysis:
                    dataset = dataset_med.split(' ')[0]
                    if analysis in ['l1tf', 'srm', 'regr'] and dataset in PD_DATASETS:
                        med_status = dataset_med.split(' ')[1]
                        n = len(set(excluded_ids_by_category[analysis]['clinical'][dataset][med_status].get(category, [])))
                        if analysis == 'l1tf':
                            n_start_med = len(set(excluded_ids_by_category[analysis]['clinical'][dataset]['start_med'].get(category, [])))
                            n += n_start_med
                            row.append(f'{n} ({n_start_med})')
                        else:
                            row.append(n)
                    else:
                        n = len(set(excluded_ids_by_category[analysis]['clinical'][dataset].get(category, [])))
                        row.append(n)
                table_data_clinical.append(row)

            dashed_row = ['-'*len(category) for category in ['Category'] + list(datasets_analysis.keys())]
            table_data_clinical.append(dashed_row) 

            # Add row for remaining participants after clinical exclusion
            row = ['Remaining after clinical exclusion']
            for dataset_med in datasets_analysis:
                dataset = dataset_med.split(' ')[0]
                if analysis in ['l1tf', 'srm', 'regr'] and dataset in PD_DATASETS:
                    med_status = dataset_med.split(' ')[1]
                    excluded_ids = list(set([x for y in excluded_ids_by_category[analysis]['clinical'][dataset][med_status].values() for x in y]))
                    if analysis == 'l1tf':
                        excluded_ids_start_med = list(set([x for y in excluded_ids_by_category[analysis]['clinical'][dataset]['start_med'].values() for x in y]))
                        excluded_ids = list(set(excluded_ids + excluded_ids_start_med))
                        remaining_ids = list(set([x for x in remaining_ids_per_dataset[dataset_med] if x not in excluded_ids]))
                        row.append(f'{len(remaining_ids)} ({len(excluded_ids_start_med)})')
                    else:
                        remaining_ids = list(set([x for x in remaining_ids_per_dataset[dataset_med] if x not in excluded_ids]))
                        row.append(len(remaining_ids))
                else:
                    excluded_ids = list(set([x for y in excluded_ids_by_category[analysis]['clinical'][dataset].values() for x in y]))
                    remaining_ids = list(set([x for x in remaining_ids_per_dataset[dataset] if x not in excluded_ids]))
                    row.append(len(remaining_ids))

                remaining_ids_per_dataset[dataset_med] = remaining_ids

            table_data_clinical.append(row)

            # Print the first table: Clinical exclusions
            print("Clinical Exclusions:")
            print(tabulate(table_data_clinical, headers=headers, tablefmt='pretty'))
            print("\n")

        table_data_measurement = []

        row = ['Starting number of participants']
        for dataset_med in datasets_analysis:
            n_start = len(set(remaining_ids_per_dataset[dataset_med]))
            row.append(n_start)
        table_data_measurement.append(row)

        dashed_row = ['-'*len(category) for category in ['Category'] + list(datasets_analysis.keys())]
        table_data_measurement.append(dashed_row) 
        
        for category in excluded_ids_by_category[analysis]['measurement']['controls'].keys():
            row = [category]
            for dataset_med in datasets_analysis:
                dataset = dataset_med.split(' ')[0]
                if analysis in ['l1tf', 'srm', 'regr'] and dataset in PD_DATASETS:
                    med_status = dataset_med.split(' ')[1]
                    n = len(set(excluded_ids_by_category[analysis]['measurement'][dataset][med_status].get(category, [])))
                    if analysis == 'l1tf':
                        n_start_med = len(set([x for x in excluded_ids_by_category[analysis]['measurement'][dataset]['start_med'].get(category, []) if x.endswith(med_status) and not x.endswith(f'_no_{med_status}')]))
                        n += n_start_med
                        row.append(f'{n} ({n_start_med})')
                    else:
                        row.append(n)
                else:
                    n = len(set(excluded_ids_by_category[analysis]['measurement'][dataset].get(category, [])))
                    row.append(n)
            table_data_measurement.append(row)

        dashed_row = ['-'*len(category) for category in ['Category'] + list(datasets_analysis.keys())]
        table_data_measurement.append(dashed_row) 

        # Add row for remaining participants after clinical exclusion
        row = ['Remaining after measurement exclusion']
        for dataset_med in datasets_analysis:
            dataset = dataset_med.split(' ')[0]
            if analysis in ['l1tf', 'srm', 'regr'] and dataset in PD_DATASETS:
                med_status = dataset_med.split(' ')[1]
                excluded_ids = list(set([x for y in excluded_ids_by_category[analysis]['measurement'][dataset][med_status].values() for x in y]))
                if analysis == 'l1tf':
                    excluded_ids_start_med = list(set([x for y in excluded_ids_by_category[analysis]['measurement'][dataset]['start_med'].values() for x in y if x.endswith(med_status) and not x.endswith(f'_no_{med_status}')]))
                    excluded_ids = list(set([x.replace(f'_{med_status}', '') for x in excluded_ids + excluded_ids_start_med]))
                    remaining_ids = list(set([x for x in remaining_ids_per_dataset[dataset_med] if x not in excluded_ids]))
                    row.append(f'{len(remaining_ids)} ({len(excluded_ids_start_med)})')
                else:
                    remaining_ids = list(set([x for x in remaining_ids_per_dataset[dataset_med] if x not in excluded_ids]))
                    row.append(len(remaining_ids))
            else:
                excluded_ids = list(set([x for y in excluded_ids_by_category[analysis]['measurement'][dataset].values() for x in y]))
                remaining_ids = [x for x in remaining_ids_per_dataset[dataset] if x not in excluded_ids]
                row.append(len(remaining_ids))

            remaining_ids_per_dataset[dataset_med] = remaining_ids

        table_data_measurement.append(row)

        # Print the first table: Clinical exclusions
        print("Measurements Exclusions:")
        print(tabulate(table_data_measurement, headers=headers, tablefmt='pretty'))
        print("\n")