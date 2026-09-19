# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename       | Description                                                      |
| -------------- | -----------------------------------------------------------------|
| "daily_pv.h5"  | Adjusted daily price and volume data.                            |


# For different data, We have some basic knowledge for them

## Daily price and volume data
$open: adjusted open price of the stock on that day.
$close: adjusted close price of the stock on that day.
$high: adjusted high price of the stock on that day.
$low: adjusted low price of the stock on that day.
$volume: adjusted traded volume, equal to the day's raw traded shares divided by $factor.
$factor: the complete price-adjustment multiplier, including any instrument-specific normalization.

For these QLib exports, adjusted_price = nominal_price * $factor for each OHLC field.
Actual nominal close in CNY per share is $close / $factor; no additional instrument-specific scale is required.
Use nominal prices for cross-stock price comparisons and adjusted prices directly for return calculations.
The reconstruction requires finite, strictly positive prices and $factor.

Raw traded volume in shares is $volume * $factor; volume is inversely adjusted by the same complete multiplier as prices.
For cross-stock liquidity comparisons, the nominal close-times-shares proxy in CNY is ($close / $factor) * ($volume * $factor) = $close * $volume.
This is a closing-price approximation, not the actual transaction amount.
