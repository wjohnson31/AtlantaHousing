import pandas as pd

try:
    # Load the full Zillow dataset
    df = pd.read_csv('ZipcodeHouseValue.csv')
    print("Dataset loaded successfully!")

    # Check if 'City' column exists
    if 'City' not in df.columns:
        print("The 'City' column is not in the dataset. Please check the file.")
    else:
        # Filter the dataframe for rows where the city is 'Atlanta'
        df_atlanta_clean = df[df['City'].str.contains('Atlanta', case=False, na=False)]
        print(f"Filtered data: {len(df_atlanta_clean)} rows match the 'Atlanta' city filter.")

        # Drop unnecessary columns
       

        # Save the cleaned data to a new CSV file
        df_atlanta_clean.to_csv('atlanta_housing_data_clean.csv', index=False)
        print("Cleaned data for Atlanta saved as 'atlanta_housing_data_clean.csv'")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the file 'zillow_home_value_data.csv' is in the correct directory.")
except Exception as e:
    print(f"An error occurred: {e}")
