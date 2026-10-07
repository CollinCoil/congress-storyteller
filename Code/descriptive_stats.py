"""
Descriptive statistics on storyteller witnesses (Figures 1 and 2).

Input:   Data/testimonies.xlsx
Output:  Percent of storytellers overall and by gender, committee, and occupation (printed);
         number of hearings and committees (printed);
         Results/Storytellers by Occupation.png (Figure 1);
         Results/Storytellers by Committee.png (Figure 2)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

os.makedirs("Results", exist_ok=True)

# Load the data
file_path = "Data/testimonies.xlsx"
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])

# Define a function to calculate and print percentages
def calculate_percentages(group_field, df):
    """Calculates and prints the percentage of narratives by a given field."""
    percentages = df.groupby(group_field)['Narrative'].mean() * 100
    print(f"\nPercent of witnesses providing narratives by {group_field}:")
    print(percentages.round(2).to_string())
    return percentages

# Overall percentage
overall_percent = df['Narrative'].mean() * 100
print(f"Overall percent of witnesses providing narratives: {overall_percent:.2f}%")

# By gender
calculate_percentages('Gender', df)

# By committee *with branch to disambiguate*
calculate_percentages(['Branch', 'Committee'], df)

# By occupation
calculate_percentages('Occupation', df)

# Count unique hearings (based on Hearing, Date, Committee)
unique_hearings = df.drop_duplicates(subset=['Hearing', 'Date', 'Committee'])
print(f"\nNumber of unique hearings: {len(unique_hearings)}")

# Count unique committees (with branch, since the House and Senate share committee names)
unique_committees = df.groupby(['Branch', 'Committee']).ngroups
print(f"Number of unique committees: {unique_committees}")

# --- Figure 1: Storytellers by occupation ---
# Group the data to get the count of narratives and the total count for each occupation
grouped_data = df.groupby('Occupation')['Narrative'].agg(['sum', 'count'])
grouped_data.columns = ['narrative_count', 'total_count']
grouped_data['percentage'] = (grouped_data['narrative_count'] / grouped_data['total_count']) * 100

# Sort the data by percentage from largest to smallest
grouped_data.sort_values(by='percentage', ascending=False, inplace=True)

# Plotting the bar chart
# The style is applied only to this figure so the committee figure keeps the default style.
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.figure(figsize=(16, 9))
    plt.bar(grouped_data.index, grouped_data['percentage'], color='teal', edgecolor='black')
    plt.xlabel('Occupation', fontsize=14)
    plt.ylabel('Percent of Witnesses', fontsize=14)
    plt.xticks(rotation=75, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("Results/Storytellers by Occupation.png", dpi=300)

# --- Figure 2: Storytellers by committee, House and Senate ---

# Filter data for House and Senate
house_df = df[df['Branch'] == 'House']
senate_df = df[df['Branch'] == 'Senate']

# Calculate percentages for each committee within the House and sort
house_committee_percentages = house_df.groupby('Committee')['Narrative'].mean() * 100
house_committee_percentages = house_committee_percentages.sort_values(ascending=False)

# Calculate percentages for each committee within the Senate and sort
senate_committee_percentages = senate_df.groupby('Committee')['Narrative'].mean() * 100
senate_committee_percentages = senate_committee_percentages.sort_values(ascending=False)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 22), sharey=True, constrained_layout=True)
plt.style.use('seaborn-v0_8-whitegrid')

# Wrap long labels for x-axis
def wrap_labels(labels, width=25):
    """Wraps long labels for better readability on plots."""
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

# First subplot for House Committees
ax1.bar(house_committee_percentages.index, house_committee_percentages.values,
        color='cornflowerblue', edgecolor='black')
ax1.set_title('House', fontsize=16, fontweight='bold')
ax1.set_ylabel('Percent of Witnesses', fontsize=14)
wrapped_house_labels = wrap_labels(house_committee_percentages.index)
ax1.set_xticks(range(len(wrapped_house_labels)))
ax1.set_xticklabels(wrapped_house_labels, rotation=45, ha='right', fontsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# Second subplot for Senate Committees
ax2.bar(senate_committee_percentages.index, senate_committee_percentages.values,
        color='darkseagreen', edgecolor='black')
ax2.set_title('Senate', fontsize=16, fontweight='bold')
ax2.set_xlabel('Committee', fontsize=14)
ax2.set_ylabel('Percent of Witnesses', fontsize=14)
wrapped_senate_labels = wrap_labels(senate_committee_percentages.index)
ax2.set_xticks(range(len(wrapped_senate_labels)))
ax2.set_xticklabels(wrapped_senate_labels, rotation=45, ha='right', fontsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

# Increase space at the bottom for long labels
fig.subplots_adjust(bottom=0.2)

fig.savefig("Results/Storytellers by Committee.png", dpi=300)
plt.show()
