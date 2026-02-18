import numpy as np

def preprocess_data(df):
    df = df.dropna()
    
    df['Heat_Stress_Index'] = (
        0.5 * df['Temperature'] +
        0.3 * df['Heart_Rate'] +
        0.2 * df['Humidity']
    )
    
    df['Risk_Level'] = np.where(
        df['Heat_Stress_Index'] > 75, 'High',
        np.where(df['Heat_Stress_Index'] > 60, 'Medium', 'Low')
    )
    
    return df
