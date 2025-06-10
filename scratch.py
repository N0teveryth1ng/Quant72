# import pandas as pd
#
#
# df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
#
# # Shift down by one row
# shifted_down = df.shift(periods=1)
#
# # Shift up by one row
# shifted_up = df.shift(periods=-1)
#
# # Shift down by one row and fill NaN with 0
# shifted_down_filled = df.shift(periods=1, fill_value=0)
#
# print("Original DataFrame:\n", df)
# print("\nShifted Down:\n", shifted_down)
# print("\nShifted Up:\n", shifted_up)
# print("\nShifted Down Filled:\n", shifted_down_filled)





import pandas as pd

# Sample time series data
data = {'values': [10, 12, 15, 13, 18, 20, 17, 22, 25, 23]}
index = pd.date_range('2025-01-01', periods=10, freq='D')
df = pd.DataFrame(data, index=index)

# Calculate the 3-day rolling mean
df['rolling_mean'] = df['values'].rolling(3).mean()

# Calculate the 3-day rolling sum
df['rolling_sum'] = df['values'].rolling(3).sum()

print(df)
