# Matplotlib-Challenge
Module 5 challenge

# Pymaceuticals Data Analysis

This repository contains a Python script for analyzing data related to a pharmaceutical study conducted by Pymaceuticals, Inc. The script utilizes the Pandas library for data manipulation and analysis, as well as the Matplotlib library for data visualization.

## Overview

The script performs the following tasks:

1. **Data Loading**: Loads mouse metadata and study results from CSV files into Pandas DataFrames.
2. **Data Cleaning**: Removes any duplicate entries and prepares the data for analysis.
3. **Summary Statistics**: Calculates summary statistics of tumor volume for different drug regimens.
4. **Data Visualization**:
    - Generates bar charts to visualize the total number of rows by drug regimen.
    - Generates pie charts to show the distribution of female versus male mice.
    - Generates box plots to visualize the final tumor volume across four treatment regimens and highlight potential outliers.
    - Generates line plots to show tumor volume versus time point for a single mouse treated with Capomulin.
    - Generates scatter plots to show mouse weight versus average observed tumor volume for the entire Capomulin treatment regimen.
    - Calculates correlation coefficients and performs linear regression to analyze the relationship between mouse weight and tumor volume, then visualizes the results.

## Sources
- [Link 1] https://matplotlib.org/stable/gallery/index.html
- BSC Learning Assistant
