import pandas as pd
import numpy as np

def clean_feature_data(df):
    
    df_clean = df.copy()
    
    #Grouping small columns together
    columns_to_merge = {
        'smoker': [7, 8, 9],
        'class_of_worker': [7, 8, 9],
        'more_than_one_job': [7, 8, 9],
        'kidney_condition': [7, 8, 9],
        'liver_condition': [7, 8, 9],
        'cant_afford_meds': [7, 8, 9],
        'skipped_meds': [7, 8, 9],
        'ever_worked': [7, 8, 9],
    }
    
    for column, values in columns_to_merge.items():
        df_clean[column] = df_clean[column].replace(values, 'OTHER')
    
    #creating meaninful names for the column vales
    
    mapping_dict = {
        
    "diabetes":{1:'YES', 0:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "smoker":{1:'YES', 0:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "coronary_heart_disease":{1:'YES', 0:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "had_high_cholesterol":{1:'YES', 0:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "hypertension":{1:'YES', 0:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "heart_condition":{1:'YES', 0:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "cancer":{1:'YES', 0:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "family_history_diabetes":{1:'YES', 0:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "kidney_condition":{1:'YES', 0:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "liver_condition":{1:'YES', 0:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},   
    
    "sex":{1:'MALE', 2:'FEMALE','OTHER':'OTHER', 'MISSING':'MISSING'},    
    "ever_worked":{1:'YES', 2:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "more_than_one_job":{1:'YES', 2:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "cant_afford_meds":{1:'YES', 2:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    "skipped_meds":{1:'YES', 2:'NO','OTHER':'OTHER', 'MISSING':'MISSING'},
    
    "non_hispanic_race":{1:'White',2: 'Black/African American',3:'Indian (American), Alaska Native',9: 'Asian Indian' , 
                         10: 'Chinese' ,11 :'Filipino' ,15 :'Other Asian',16 :'Primary race not releasable' ,
                         17 :'Multiple race, no primary race selected','OTHER':'OTHER', 'MISSING':'MISSING'},
    
    "hispanic":{0: 'Multiple Hispanic',1 :'Puerto Rico',2: 'Mexican',3 :'Mexican-American',4 :'Cuban/Cuban American',
                5: 'Dominican (Republic)',6: 'Central or South American',7: 'Other Latin American, not specified',8: 'Other Spanish',
                9: 'Hispanic/Latino/Spanish, non-specific',10: 'Hispanic/Latino/Spanish, type refused',
                11: 'Hispanic/Latino/Spanish, type not ascertained', 12: 'Not Hispanic/Spanish origin',
                'OTHER':'OTHER', 'MISSING':'MISSING',},
    
    'class_of_worker':{1:'PRIVATE COMPANY', 2:'FEDERAL GOVERNMENT', 3:'STATE GOVERNMENT',
                        4:'LOCAL GOVERNMENT', 5:'SELF-EMPLOYED', 6:'IN FAMLY-OWED WITHOUT PAY', 
                        7:'Refused',8:'Not ascertained',9:"Don't know",'OTHER':'OTHER', 'MISSING':'MISSING',},
    'primary_care':{0:"Doesn't get preventive care anywhere", 1:'Clinic/health center', 2:'Doctor office',
                        3:'ER', 4:'Outpatient', 5:'Other places', 6:'Inconsistent places',
                        7:'OTHER',8:'OTHER',9:"OTHER",'OTHER':'OTHER', 'MISSING':'MISSING',}
    }

    #Nan for weight and height
    df_clean.loc[df_clean['weight'] >= 300, 'weight'] = np.nan
    df_clean.loc[df_clean['height'] >= 77, 'height'] = np.nan
    
    #imputing Nan values weight and height
    #stratified median for imputation based on 'sex', 'non_hispanic_race', 'hispanic' for more accurate representation of skewed values
    df_clean['weight'] = df_clean.groupby(['sex', 'non_hispanic_race', 'hispanic'])['weight'].transform(lambda x: x.fillna(x.median()))
    df_clean['height'] = df_clean.groupby(['sex', 'non_hispanic_race', 'hispanic'])['height'].transform(lambda x: x.fillna(x.median()))
    
    # New BMI
    df_clean['bmi'] = df_clean['weight'] / (df_clean['height'] ** 2)*703
    df_clean['bmi'] = df_clean['bmi'].round(2)
    
    for key, value in mapping_dict.items():
        df_clean[key] = df_clean[key].map(value)

    age_bins = [18, 25, 35, 45, 55, 65, 75, 85]
    age_labels = ['Young Adult (18-25)', 'Emerging Adulthood (26-35)', 
                  'Early Middle Age (36-45)', 'Midlife (46-55)', 
                  'Late Middle Age (56-65)', 'Early Senior (66-75)', 
                  'Senior (76-85)']
    
    # Apply life stage categorization
    df_clean['age_group'] = pd.cut(df_clean['age'], bins=age_bins, labels=age_labels)

    return df_clean