import pandas as pd
from fuzzywuzzy import fuzz

# Load Excel file into a Pandas DataFrame
excel_file_path = "chip_images.xlsx"  # Replace with the path to your Excel file
df_excel = pd.read_excel(excel_file_path)

# Load CSV file into a Pandas DataFrame
csv_file_path = "info_detected_cccd_v3_20k.csv"  # Replace with the path to your CSV file
df_csv = pd.read_csv(csv_file_path)

# Merge DataFrames based on the "IMAGE_ID" column
merged_df = pd.merge(df_excel, df_csv, on="IMAGE_ID", how="inner")
# Select columns from both tables
selected_columns = [
    "IMAGE_ID",
    "ID_x", "NAME_x", "DOB_x", "NATIONALITY_x", "SEX_x", "HOME_x", "ADDRESS_x", "DOE_x",
    "ID_y", "NAME_y", "DOB_y", "NATIONALITY_y", "SEX_y", "HOME_y", "ADDRESS_y", "DOE_y"
]

# Only take the selected columns
merged_df = merged_df[selected_columns]

# Calculate Levenshtein distance for each pair of columns _x and _y
for column in selected_columns[1:]:  # Skip the "IMAGE_ID" column
    if column.endswith("_x"):
        corresponding_column_y = column.replace("_x", "_y")
        merged_df[f"{column}_difference"] = merged_df.apply(
            lambda row: fuzz.ratio(str(row[column]).lower(), str(row[corresponding_column_y]).lower()),
            axis=1
        )

# Calculate the mean of all difference columns
mean_row = merged_df.filter(like='_difference').mean().to_dict()

# Append the mean row to the DataFrame
merged_df = pd.concat([merged_df,pd.DataFrame([mean_row])], ignore_index=True)

# Save the merged DataFrame to a new Excel file
output_csv_file_path = "detection_cccd_v3.1_20k.xlsx"  # Replace with the desired output file path
merged_df.to_excel(output_csv_file_path, index=False)

print(f"Saved merged data to {output_csv_file_path}")
