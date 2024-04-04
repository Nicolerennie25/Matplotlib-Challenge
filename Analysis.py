import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
mouse_metadata_path = r'C:\Users\n.rennie\Documents\MatplotLib-challenge\Starter_Code\Pymaceuticals\data\mouse_metadata.csv'
study_results_path = r'C:\Users\n.rennie\Documents\MatplotLib-challenge\Starter_Code\Pymaceuticals\data\study_results.csv'

# Load mouse_metadata.csv into a DataFrame
mouse_metadata = pd.read_csv(mouse_metadata_path)

# Load study_results.csv into a DataFrame
study_results = pd.read_csv(study_results_path)

# Merge mouse_metadata and study_results into a single DataFrame
merged_df = pd.merge(mouse_metadata, study_results, on="Mouse ID", how="outer")

# Display the number of unique mice IDs in the data
unique_mice_count = merged_df['Mouse ID'].nunique()
print("Number of unique mice IDs before cleaning:", unique_mice_count)

# Check for any mouse IDs with duplicate time points
duplicate_mouse_ids = merged_df[merged_df.duplicated(subset=['Mouse ID', 'Timepoint'], keep=False)]['Mouse ID'].unique()

if len(duplicate_mouse_ids) > 0:
    print("\nMouse IDs with duplicate time points:", duplicate_mouse_ids)
    print("\nData associated with duplicate time points:")
    for mouse_id in duplicate_mouse_ids:
        print(merged_df[merged_df['Mouse ID'] == mouse_id])
else:
    print("\nNo mouse IDs with duplicate time points.")

# Create a new DataFrame where data associated with duplicate time points is removed
cleaned_df = merged_df.drop_duplicates(subset=['Mouse ID', 'Timepoint'], keep='first')

# Display the updated number of unique mice IDs after cleaning
updated_unique_mice_count = cleaned_df['Mouse ID'].nunique()
print("\nUpdated number of unique mice IDs after cleaning:", updated_unique_mice_count)

# Verify the cleaning by displaying the first few rows of the cleaned DataFrame
print("\nFirst few rows of the cleaned DataFrame:")
print(cleaned_df.head())

# Group the cleaned DataFrame by drug regimen and calculate summary statistics
summary_statistics = cleaned_df.groupby('Drug Regimen')['Tumor Volume (mm3)'].agg(['mean', 'median', 'var', 'std', 'sem'])

# Rename the columns for clarity
summary_statistics.columns = ['Mean', 'Median', 'Variance', 'Std Deviation', 'SEM']

# Display the summary statistics DataFrame
print("Summary Statistics of Tumor Volume by Drug Regimen:")
print(summary_statistics)

# Group the cleaned DataFrame by drug regimen and count the number of rows (Mouse ID/Timepoints)
bar_data = cleaned_df.groupby('Drug Regimen').size()

# Plot the bar chart using Pandas DataFrame.plot() method
bar_data.plot(kind='bar', color='skyblue', alpha=0.7, figsize=(10, 6))
plt.title('Total Number of Rows by Drug Regimen (Pandas)')
plt.xlabel('Drug Regimen')
plt.ylabel('Total Number of Rows')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Extract drug regimens and corresponding row counts
drug_regimens = bar_data.index
row_counts = bar_data.values

# Plot the bar chart using Matplotlib's pyplot methods
plt.figure(figsize=(10, 6))
plt.bar(drug_regimens, row_counts, color='salmon', alpha=0.7)
plt.title('Total Number of Rows by Drug Regimen (Matplotlib)')
plt.xlabel('Drug Regimen')
plt.ylabel('Total Number of Rows')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Count the number of male and female mice
gender_counts = cleaned_df['Sex'].value_counts()

# Plot the pie chart using Pandas DataFrame.plot() method
gender_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], figsize=(8, 8))
plt.title('Distribution of Female vs. Male Mice (Pandas)')
plt.ylabel('')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Plot the pie chart using Matplotlib's pyplot methods
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
plt.title('Distribution of Female vs. Male Mice (Matplotlib)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Step 1: Create a grouped DataFrame that shows the last (greatest) time point for each mouse
last_timepoint_df = cleaned_df.groupby('Mouse ID')['Timepoint'].max().reset_index()

# Step 2: Merge this grouped DataFrame with the original cleaned DataFrame
merged_last_timepoint_df = pd.merge(last_timepoint_df, cleaned_df, on=['Mouse ID', 'Timepoint'], how='inner')

# Step 3: Create a list to hold the treatment names
treatments = ['Capomulin', 'Ramicane', 'Infubinol', 'Ceftamin']

# Create an empty list to hold the tumor volume data
tumor_volume_data = []

# Step 4: Loop through each drug in the treatment list
for treatment in treatments:
    # Locate the rows in the merged DataFrame that correspond to each treatment
    tumor_volume = merged_last_timepoint_df.loc[merged_last_timepoint_df['Drug Regimen'] == treatment, 'Tumor Volume (mm3)']
    # Append the resulting final tumor volumes for each drug to the empty list
    tumor_volume_data.append(tumor_volume)

# Step 5: Calculate the quartiles and interquartile range (IQR) for each treatment regimen
quartiles = [tumor_volume.quantile([0.25, 0.5, 0.75]) for tumor_volume in tumor_volume_data]
lowerq = [q[0.25] for q in quartiles]
upperq = [q[0.75] for q in quartiles]
iqr = [q[0.75] - q[0.25] for q in quartiles]

# Step 6: Determine outliers by using the upper and lower bounds
lower_bound = [lower - (1.5 * i) for lower, i in zip(lowerq, iqr)]
upper_bound = [upper + (1.5 * i) for upper, i in zip(upperq, iqr)]

# Step 7: Print the results
for treatment, tumor_volume, lb, ub in zip(treatments, tumor_volume_data, lower_bound, upper_bound):
    outliers = tumor_volume[(tumor_volume < lb) | (tumor_volume > ub)]
    print(f"Outliers for {treatment}: {outliers.tolist()}")

# Step 8: Create a box plot
plt.figure(figsize=(10, 6))
plt.boxplot(tumor_volume_data, labels=treatments)
plt.title('Final Tumor Volume Across Four Treatment Regimens')
plt.xlabel('Treatment Regimen')
plt.ylabel('Final Tumor Volume (mm3)')
plt.show()

import matplotlib.pyplot as plt

# Create a box plot
plt.figure(figsize=(10, 6))
plt.boxplot(tumor_volume_data, labels=treatments, patch_artist=True)

# Highlight potential outliers
for i in range(len(treatments)):
    outliers = tumor_volume_data[i][(tumor_volume_data[i] < lower_bound[i]) | (tumor_volume_data[i] > upper_bound[i])]
    if len(outliers) > 0:
        plt.scatter([i + 1] * len(outliers), outliers, color='red', marker='o', s=50, label='Outliers' if i == 0 else None)

# Add labels and title
plt.title('Final Tumor Volume Across Four Treatment Regimens')
plt.xlabel('Treatment Regimen')
plt.ylabel('Final Tumor Volume (mm3)')
plt.legend()

# Show plot
plt.show()

# Step 1: Filter the cleaned DataFrame to select data for mice treated with Capomulin
capomulin_data = cleaned_df[cleaned_df['Drug Regimen'] == 'Capomulin']

# Step 2: Choose a single mouse ID from the filtered data
single_mouse_id = capomulin_data['Mouse ID'].iloc[0]

# Step 3: Filter the data further to include only the selected mouse
single_mouse_data = capomulin_data[capomulin_data['Mouse ID'] == single_mouse_id]

# Step 4: Plot tumor volume versus time point
plt.figure(figsize=(10, 6))
plt.plot(single_mouse_data['Timepoint'], single_mouse_data['Tumor Volume (mm3)'], marker='o', color='skyblue')
plt.title(f'Tumor Volume vs. Time Point for Mouse ID {single_mouse_id} (Capomulin)')
plt.xlabel('Timepoint (Days)')
plt.ylabel('Tumor Volume (mm3)')
plt.grid(True)
plt.show()

# Step 1: Group the Capomulin data by mouse ID to calculate the average tumor volume and weight for each mouse
capomulin_grouped = capomulin_data.groupby('Mouse ID').agg({'Tumor Volume (mm3)': 'mean', 'Weight (g)': 'mean'}).reset_index()

# Step 2: Plot the average tumor volume versus mouse weight
plt.figure(figsize=(10, 6))
plt.scatter(capomulin_grouped['Weight (g)'], capomulin_grouped['Tumor Volume (mm3)'], color='skyblue', alpha=0.7)
plt.title('Mouse Weight vs. Average Tumor Volume (Capomulin)')
plt.xlabel('Weight (g)')
plt.ylabel('Average Tumor Volume (mm3)')
plt.grid(True)
plt.show()

from scipy.stats import linregress

# Calculate the correlation coefficient
correlation_coefficient = capomulin_grouped['Weight (g)'].corr(capomulin_grouped['Tumor Volume (mm3)'])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(capomulin_grouped['Weight (g)'], capomulin_grouped['Tumor Volume (mm3)'])

# Print correlation coefficient
print("Correlation coefficient:", correlation_coefficient)

# Print linear regression model
print("Linear Regression Model:")
print("Slope:", slope)
print("Intercept:", intercept)
print("R-squared value:", r_value ** 2)
print("p-value:", p_value)
print("Standard error:", std_err)

from scipy.stats import linregress

# Calculate the correlation coefficient
correlation_coefficient = capomulin_grouped['Weight (g)'].corr(capomulin_grouped['Tumor Volume (mm3)'])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(capomulin_grouped['Weight (g)'], capomulin_grouped['Tumor Volume (mm3)'])

# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(capomulin_grouped['Weight (g)'], capomulin_grouped['Tumor Volume (mm3)'], color='skyblue', alpha=0.7)

# Plot the linear regression line
regression_values = capomulin_grouped['Weight (g)'] * slope + intercept
plt.plot(capomulin_grouped['Weight (g)'], regression_values, color='red')

# Add labels and title
plt.title('Mouse Weight vs. Average Tumor Volume (Capomulin)')
plt.xlabel('Weight (g)')
plt.ylabel('Average Tumor Volume (mm3)')
plt.grid(True)

# Show plot
plt.show()
