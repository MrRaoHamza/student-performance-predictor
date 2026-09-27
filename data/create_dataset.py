"""
UCI Student Performance Dataset - Generator
Creates a realistic dataset matching the UCI Student Performance dataset structure.
Features match the original paper: Cortez & Silva, 2008.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 649

def generate_dataset():
    school = np.random.choice(['GP', 'MS'], n, p=[0.75, 0.25])
    sex = np.random.choice(['F', 'M'], n, p=[0.53, 0.47])
    age = np.random.choice(range(15, 23), n, p=[0.15, 0.25, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02])
    address = np.random.choice(['U', 'R'], n, p=[0.68, 0.32])
    famsize = np.random.choice(['LE3', 'GT3'], n, p=[0.33, 0.67])
    Pstatus = np.random.choice(['T', 'A'], n, p=[0.78, 0.22])
    Medu = np.random.choice([0, 1, 2, 3, 4], n, p=[0.03, 0.16, 0.26, 0.27, 0.28])
    Fedu = np.random.choice([0, 1, 2, 3, 4], n, p=[0.03, 0.22, 0.24, 0.28, 0.23])
    Mjob = np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n, p=[0.10, 0.10, 0.18, 0.20, 0.42])
    Fjob = np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n, p=[0.10, 0.07, 0.22, 0.08, 0.53])
    reason = np.random.choice(['home', 'reputation', 'course', 'other'], n, p=[0.27, 0.25, 0.33, 0.15])
    guardian = np.random.choice(['mother', 'father', 'other'], n, p=[0.54, 0.38, 0.08])
    traveltime = np.random.choice([1, 2, 3, 4], n, p=[0.47, 0.37, 0.10, 0.06])
    studytime = np.random.choice([1, 2, 3, 4], n, p=[0.20, 0.46, 0.22, 0.12])
    failures = np.random.choice([0, 1, 2, 3], n, p=[0.67, 0.18, 0.08, 0.07])
    schoolsup = np.random.choice(['yes', 'no'], n, p=[0.15, 0.85])
    famsup = np.random.choice(['yes', 'no'], n, p=[0.61, 0.39])
    paid = np.random.choice(['yes', 'no'], n, p=[0.36, 0.64])
    activities = np.random.choice(['yes', 'no'], n, p=[0.49, 0.51])
    nursery = np.random.choice(['yes', 'no'], n, p=[0.81, 0.19])
    higher = np.random.choice(['yes', 'no'], n, p=[0.93, 0.07])
    internet = np.random.choice(['yes', 'no'], n, p=[0.79, 0.21])
    romantic = np.random.choice(['yes', 'no'], n, p=[0.37, 0.63])
    famrel = np.random.choice([1, 2, 3, 4, 5], n, p=[0.02, 0.05, 0.20, 0.45, 0.28])
    freetime = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.17, 0.36, 0.30, 0.12])
    goout = np.random.choice([1, 2, 3, 4, 5], n, p=[0.06, 0.20, 0.32, 0.27, 0.15])
    Dalc = np.random.choice([1, 2, 3, 4, 5], n, p=[0.53, 0.24, 0.12, 0.07, 0.04])
    Walc = np.random.choice([1, 2, 3, 4, 5], n, p=[0.33, 0.22, 0.20, 0.14, 0.11])
    health = np.random.choice([1, 2, 3, 4, 5], n, p=[0.09, 0.08, 0.19, 0.24, 0.40])
    absences = np.random.negative_binomial(2, 0.3, n)
    absences = np.clip(absences, 0, 75)

    # G1 influenced by studytime, failures, Medu, Fedu
    G1_base = (
        8
        + studytime * 1.2
        - failures * 2.5
        + Medu * 0.4
        + Fedu * 0.3
        - absences * 0.05
        + np.where(higher == 'yes', 1.5, 0)
        + np.where(sex == 'F', 0.3, 0)
        + np.random.normal(0, 2, n)
    )
    G1 = np.clip(np.round(G1_base).astype(int), 0, 20)

    G2_base = G1 * 0.85 + np.random.normal(0, 1.5, n) + np.where(paid == 'yes', 0.5, 0)
    G2 = np.clip(np.round(G2_base).astype(int), 0, 20)

    G3_base = (G1 * 0.4 + G2 * 0.5
               + studytime * 0.6
               - failures * 1.5
               - absences * 0.04
               + np.where(higher == 'yes', 1.0, 0)
               + np.random.normal(0, 1.5, n))
    G3 = np.clip(np.round(G3_base).astype(int), 0, 20)

    df = pd.DataFrame({
        'school': school, 'sex': sex, 'age': age, 'address': address,
        'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
        'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason, 'guardian': guardian,
        'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
        'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
        'activities': activities, 'nursery': nursery, 'higher': higher,
        'internet': internet, 'romantic': romantic, 'famrel': famrel,
        'freetime': freetime, 'goout': goout, 'Dalc': Dalc, 'Walc': Walc,
        'health': health, 'absences': absences, 'G1': G1, 'G2': G2, 'G3': G3
    })

    return df


if __name__ == '__main__':
    df = generate_dataset()
    df.to_csv('student-mat.csv', index=False)
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Pass rate (G3 >= 10): {(df['G3'] >= 10).mean():.1%}")
    print(df.head())
