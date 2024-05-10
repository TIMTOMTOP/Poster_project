import pandas as pd

# Read the price consumption file
df = pd.read_excel('Electricity_price_consumption.xlsx')

# Convert the file into a CSV file
df.to_csv('price_consumption_file.csv', index=False)
