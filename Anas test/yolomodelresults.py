import pandas as pd

# Load the CSV file
df = pd.read_csv("C:/Users/anasm/OneDrive/Documents/Projectss/Main-Project-1/CurrencyModule/Outputs/results.csv")

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Print available columns (optional)
print("cleaned Columns for checking:")
print(df.columns.tolist())

# Print final epoch metrics
print("\nFinal Epoch Metrics:")
print(df.iloc[-1][[
    'metrics/precision(B)',
    'metrics/recall(B)',
    'metrics/mAP50(B)',
    'metrics/mAP50-95(B)'
]])
