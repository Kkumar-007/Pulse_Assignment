import pandas as pd
import json

# Paths
json_path = 'output_data/topic_trends.json'
csv_path = 'output_data/topic_trends.csv'

# Load JSON
with open(json_path, 'r') as f:
    trend = json.load(f)

df = pd.DataFrame(trend['topics'], index=trend['dates']).T
df.index.name = 'Topic'

# Save to CSV
df.to_csv(csv_path)
print(f"Saved: {csv_path}")
