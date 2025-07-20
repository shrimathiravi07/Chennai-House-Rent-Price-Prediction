def clean_input(form_data):
    import pandas as pd
    df = pd.DataFrame([form_data])
    # Example cleaning
    df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
    df['bath'] = pd.to_numeric(df['bath'], errors='coerce')
    df['balcony'] = pd.to_numeric(df['balcony'], errors='coerce')
    df.fillna(0, inplace=True)
    return df
