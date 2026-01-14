import pandas as pd
from src.wrtds import to_decimal_date

dates = ["2000-01-01", "2000-02-29", "2000-03-01", "2001-01-01"]
decs = to_decimal_date(pd.Series(dates))
print(pd.DataFrame({'Date': dates, 'Dec': decs}))
