import shutil
import requests
import os
from datetime import date, timedelta


today = date.today()
formatted_date = today.strftime("%Y-%m-%d") #todays date
year = formatted_date[:4]

year_directory = "C:\\SWAT\\Data\\TLE\\" + year
date_directory = os.path.join(year_directory, formatted_date)
if os.path.exists(year_directory): #check if year folder exists
    if os.path.exists(date_directory):
        #if already pulled data for today stop
        pass
    else:
        #haven't pulled data today
        os.makedirs(date_directory, exist_ok=True)
else:
    #new year, new folder
    os.makedirs(year_directory, exist_ok=True)
    os.makedirs(date_directory, exist_ok=True)   



#######IMPORT 3LE########
url = "https://www.space-track.org/basicspacedata/query/class/gp/EPOCH/%3Enow-30/orderby/NORAD_CAT_ID,EPOCH/format/3le"

response = requests.get(url)

# Set the file path and name
file_path = r"C:\SWAT\Data\TLE\Import\3le.txt"

with open(file_path, "r+") as file:
    lines = file.readlines()
    for i, line in enumerate(lines):
        if line.startswith("0 TBA"):
            file.seek(0)
            file.truncate()
            file.writelines(lines[:i])
            break

# Check if the directory exists, create it if it doesn't
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
    
# Move the file to the new directory
new_file_path = os.path.join(directory, "3le.txt")
os.rename(file_path, new_file_path)
# Copy the file to the date directory
shutil.copy(new_file_path, date_directory)


#######IMPORT TLE#######

one_day = timedelta(days=1)
tomorrow = today + one_day
formatted_tomorrow = tomorrow.strftime("%Y-%m-%d") #tmrws date

url ="https://www.space-track.org/basicspacedata/query/class/gp_history/CREATION_DATE/" + formatted_date + "--" + formatted_tomorrow + "/orderby/NORAD_CAT_ID,EPOCH/format/tle/emptyresult/show"
response = requests.get(url)

# Set the file path and name
file_path = r"C:\SWAT\Data\TLE\Import\tle.txt"

# Write the response content to the file
with open(file_path, "wb") as file:
    file.write(response.content)

# Copy the file to the date directory
shutil.copy(file_path, date_directory)

    


####IMPORT SATELLITE DATA.CSV AND XLSX
url = "https://www.space-track.org/basicspacedata/query/class/satcat/predicates/OBJECT_ID,OBJECT_NAME,NORAD_CAT_ID,COUNTRY,PERIOD,INCLINATION,APOGEE,PERIGEE,RCS_SIZE,RCSVALUE,LAUNCH,COMMENT/DECAY/null-val/CURRENT/Y/orderby/NORAD_CAT_ID%20desc/format/csv/emptyresult/show"
response = requests.get(url)

# Set the file path and name
file_path = r"C:\SWAT\Data\TLE\Import\Satellite Data.csv"

# Write the response content to the file
with open(file_path, "wb") as file:
    file.write(response.content)

# Copy the file to the date directory
shutil.copy(file_path, date_directory)




file_path = r"C:\SWAT\Data\TLE\Import\Satellite Data.xlsx"
# Write the response content to the file
with open(file_path, "wb") as file:
    file.write(response.content)

# Copy the file to the date directory
shutil.copy(file_path, date_directory)
