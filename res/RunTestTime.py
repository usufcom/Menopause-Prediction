# %%
from datetime import datetime
from pytz import timezone
fmt = "%d-%m-%Y %H:%M:%S"
now_time = datetime.now(timezone('Africa/Douala')).strftime(fmt)
today_name = datetime.today().strftime("%A")
todayDateTime = today_name +',' + now_time
print(todayDateTime)
# %%