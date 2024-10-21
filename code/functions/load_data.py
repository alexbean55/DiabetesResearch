import pandas as pd
import numpy as np

def load_diabetes_data(path = "../data/samadult.csv"):
    # load in the original data
    diabetes_orig = pd.read_csv(path)

    # take just one person from each household
    diabetes = diabetes_orig.groupby("HHX") \
      .sample(1, random_state=24648765) \
      .reset_index() \
      .copy()
    # add an id column
    diabetes["id"] = np.arange(len(diabetes.index))
    # create the house_family_person_id column by joining together three ID columns
    diabetes["house_family_person_id"] = diabetes.apply(lambda x: "_".join(x[["HHX", "FMX", "FPX"]].astype(int).astype(str)), 
                                                        axis=1)
    # create the diabetes column
    diabetes["diabetes"] = (diabetes["DIBEV1"] == 1).astype(int)
    # create coronary heart disease column
    diabetes["coronary_heart_disease"] = (diabetes["CHDEV"] == 1).astype(int)
    # create hypertension column
    diabetes["hypertension"] = (diabetes["HYPEV"] == 1).astype(int)
    # create heart_condition column
    diabetes["heart_condition"] = (diabetes["HRTEV"] == 1).astype(int)
    # create cancer column
    diabetes["cancer"] = (diabetes["CANEV"] == 1).astype(int)
    # create family_history_diabetes column
    diabetes["family_history_diabetes"] = (diabetes["DIBREL"] == 1).astype(int)
    # create had_high_cholesterol
    diabetes["had_high_cholesterol"] = (diabetes["CHLYR"] == 1).astype(int)
    # create kidney condition
    diabetes["kidney_condition"] = (diabetes["KIDWKYR"] == 1).astype(int)
    # create liver condition
    diabetes["liver_condition"] = (diabetes["LIVYR"] == 1).astype(int)
    # create smoker
    diabetes["smoker"] = (diabetes["SMKEV"] == 1).astype(int)
    
    
    # rename remaining relevant columns
    diabetes = diabetes.rename(columns={"AGE_P": "age",
                                          "SEX": "sex",
                                          "AWEIGHTP": "weight",
                                          "AHEIGHT": "height",
                                          "WRKCATA": "class_of_worker",
                                          "YRSWRKPA": "years_on_job",
                                          "EVERWRK": "ever_worked",
                                          "ONEJOB": "more_than_one_job",
                                          "AHCAFYR1": "cant_afford_meds",
                                          "ARX12_1": "skipped_meds",
                                          "MRACRPI2": "non_hispanic_race",
                                          "HISPAN_I": "hispanic",
                                          "AHCPLKND": "primary_care"})

    # select just the relevant columns
    diabetes = diabetes[["house_family_person_id",
                        "diabetes",
                        "age",
                        "smoker",
                        "sex",
                        "coronary_heart_disease",
                        "weight",
                        "had_high_cholesterol",
                        "class_of_worker",
                        "years_on_job", 
                        "height",
                        "hypertension",
                        "heart_condition",
                        "cancer",
                        "family_history_diabetes",
                        "ever_worked",
                        "more_than_one_job",
                        "kidney_condition",
                        "liver_condition",
                        "cant_afford_meds",
                        "skipped_meds",
                        "non_hispanic_race",
                        "hispanic",
                        "primary_care"]]

    diabetes.fillna("MISSING", inplace = True)
    
    return(diabetes)