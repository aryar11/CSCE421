import shutil
import requests
import os
from datetime import date, timedelta
import win32com.client
from scrapy import Selector



today = date.today()
formatted_date = today.strftime("%Y-%m-%d") #todays date
year = formatted_date[:4]

year_directory = "C:\\SWAT\\Data\\TLE\\" + year
date_directory = os.path.join(year_directory, formatted_date)
if os.path.exists(year_directory): #check if year folder exists
    if os.path.exists(date_directory):
        #if already pulled data for today do nothing
        pass
    else:
        #haven't pulled data today
        os.makedirs(date_directory, exist_ok=True)
else:
    #new year, new folder
    os.makedirs(year_directory, exist_ok=True)
    os.makedirs(date_directory, exist_ok=True)   



LOGIN_URL = "https://www.space-track.org/auth/login"

headers = {
    "content-type": "application/x-www-form-urlencoded",
    "origin": "https://www.space-track.org",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
}

payload = "spacetrack_csrf_token={}&identity=arya.r11223%40gmail.com&password=Password123456%21&btnLogin=LOGIN"



sess = requests.session()
r = sess.get(LOGIN_URL, headers={"User-Agent": headers.get("User-Agent")})
response = Selector(text=r.content)
csrf_token = response.xpath("//input[@name='spacetrack_csrf_token']/@value").get()
sess.post(LOGIN_URL, headers=headers, data=payload.format(csrf_token))

#######IMPORT 3LE########
print("Downloading 3le")
url = "https://www.space-track.org/basicspacedata/query/class/gp/EPOCH/%3Enow-30/orderby/NORAD_CAT_ID,EPOCH/format/3le"

# Set the file path and name
file_path = r"C:\SWAT\Data\TLE\Import\3le.txt"
r = sess.get(url, headers={"User-Agent": headers.get("User-Agent")})
if r.status_code == requests.codes.ok:
    print("Request successful")
else:
    print(f"Request failed with status code {r.status_code}")
# Write the TLEs to the file
with open(file_path, "wb") as file:
    file.write(r.content)

# Remove all lines from the file starting from the first line that starts with "0 TBA"
with open(file_path, "r") as file:
    lines = file.readlines()

with open(file_path, "w") as file:
    i = 0
    while i < len(lines):
        if lines[i].startswith("0 TBA"):
            del lines[i:i+3]  # Remove the current line and the two following lines
        else:
            i += 1

    file.writelines(lines)

# Copy the file to the date directory
shutil.copy(file_path, date_directory)


#######IMPORT TLE#######

one_day = timedelta(days=1)
tomorrow = today + one_day
yesterday = today - one_day
formatted_tomorrow = tomorrow.strftime("%Y-%m-%d") #tmrws date
formatted_yesterday = yesterday.strftime("%Y-%m-%d") #yesterday date
url ="https://www.space-track.org/basicspacedata/query/class/gp_history/CREATION_DATE/" + formatted_yesterday + "--" + formatted_date + "/orderby/NORAD_CAT_ID,EPOCH/format/tle/emptyresult/show"
print("Downloading TLE")
r = sess.get(url, headers={"User-Agent": headers.get("User-Agent")})
if r.status_code == requests.codes.ok:
    print("Request successful")
else:
    print(f"Request failed with status code {r.status_code}")
# Set the file path and name
file_path = r"C:\SWAT\Data\TLE\Import\tle.txt"

# Write the response content to the file
with open(file_path, "wb") as file:
    file.write(r.content)

# Copy the file to the date directory
shutil.copy(file_path, date_directory)


####IMPORT SATELLITE DATA.CSV AND XLSX
print("Downloading Satellite Data")
url = "https://www.space-track.org/basicspacedata/query/class/satcat/predicates/OBJECT_ID,OBJECT_NAME,NORAD_CAT_ID,COUNTRY,PERIOD,INCLINATION,APOGEE,PERIGEE,RCS_SIZE,RCSVALUE,LAUNCH,COMMENT/DECAY/null-val/CURRENT/Y/orderby/NORAD_CAT_ID%20desc/format/csv/emptyresult/show"
r = sess.get(url, headers={"User-Agent": headers.get("User-Agent")})

if r.status_code == requests.codes.ok:
    print("Request successful")
else:
    print(f"Request failed with status code {r.status_code}")
# Set the file path and name
file_path = r"C:\SWAT\Data\TLE\Import\Satellite Data.csv"

# Write the response content to the file
with open(file_path, "wb") as file:
    file.write(r.content)

# Copy the file to the date directory
shutil.copy(file_path, date_directory)


file_path = r"C:\SWAT\Data\TLE\Import\Satellite Data.xlsx"
# Write the response content to the file
with open(file_path, "wb") as file:
    file.write(r.content)

# Copy the file to the date directory
shutil.copy(file_path, date_directory)



##RUN UPDATE Command##
print("Updating Microsoft Databases")
accessApp = win32com.client.Dispatch("Access.Application")
accessApp.OpenCurrentDatabase("C:\\SWAT\\Data\\SID\\SWAT-Data-TLE.mdb")
accessApp.DoCmd.RunMacro("SSC-Import-TLE")
accessApp.CloseCurrentDatabase()
accessApp.Quit()
print("Done with updating satellite data")