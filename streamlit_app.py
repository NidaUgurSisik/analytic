import streamlit as st
import requests
import pandas as pd
import numpy as np
from streamlit_tags import st_tags  # to add labels on the fly!
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
    

############ 2. SETTING UP THE PAGE LAYOUT AND TITLE ############

# `st.set_page_config` is used to display the default layout width, the title of the app, and the emoticon in the browser tab.


st.set_page_config(
    layout="centered", page_title="Analytic", page_icon="❄️"
)

############ CREATE THE LOGO AND HEADING ############

# We create a set of columns to display the logo and the heading next to each other.


c1, c2 = st.columns([0.25,10])

# The snowflake logo will be displayed in the first column, on the left.

with c2:
    c31, c32 = st.columns([8, 2])
    with c31:
        st.caption("")
        st.title("Hubspot Analytics")
    with c32:
        st.image(
            "images/logo.png",
            width=200,
        )
    
    uploaded_file = st.file_uploader(
        "",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

    if uploaded_file is not None:
        file_container = st.expander("Check your uploaded .csv")
        shows = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(shows)

    else:
        st.info(
            f"""
                👆 Upload a .csv file first. Sample to try: [biostats.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
                """
        )

        st.stop()

    c21, c22 = st.columns([4.5,4.5])
    with c21:
        countjob = shows['Job Title'].value_counts()
        st.bar_chart(countjob[:15])
    with c22:
        countcountry= shows['Country/Region'].value_counts()
        st.bar_chart(countcountry[:15])

        

        
    list = []
    output = []
    keywords = pd.DataFrame(columns=['key', 'value'])
    for i in range(len(shows)):
        list.append(str(shows.loc[i]['Keywords']).split(','))

    for element in list:
        for i in element:
            output.append(i)

    for i in output:
        new= {'key': i, 'value': output.count(i)}
        keywords = keywords.append(new, ignore_index = True)
    keywords = keywords.sort_values(by=['value'],ascending=False)
    keywords = keywords.drop_duplicates(subset=['key'])
    keywords = keywords[keywords.key != 'nan']
    
    st.bar_chart(data = keywords, x = 'key', y = 'value')

    longitude = []
    latitude = []

    data = {'City': shows['City'].unique()}
    df = pd.DataFrame(data)
    coord = pd.DataFrame(columns=['longitude', 'latitude'])
    # function to find the coordinate
    # of a given city

    # declare an empty list to store
    # latitude and longitude of values
    # of city column
    longitude = []
    latitude = []

    # function to find the coordinate
    # of a given city
    def findGeocode(city):
        
        # try and catch is used to overcome
        # the exception thrown by geolocator
        # using geocodertimedout
        try:
            
            # Specify the user_agent as your
            # app name it should not be none
            geolocator = Nominatim(user_agent="your_app_name")
            location = geolocator.geocode(city)
            return location
        
        except GeocoderTimedOut:
            
            return findGeocode(city)	

    # each value from city column
    # will be fetched and sent to
    # function find_geocode
    for i in (df['City']):
        if i != None:
            st.write(i)
            loc = findGeocode(i)
            
            # coordinates returned from
            # function is stored into
            # two separate list
            #latitude.append(loc.latitude)
            #longitude.append(loc.longitude)
            if loc != None:
                new_row = {'longitude': loc.longitude, 'latitude': loc.latitude}
                coord = coord.append(new_row, ignore_index = True)
        
        # if coordinate for a city not
        # found, insert "NaN" indicating
        # missing value
        else:
            latitude.append(np.nan)
            longitude.append(np.nan)

    st.map(coord)
   

# We need to set up session state via st.session_state so that app interactions don't reset the app.

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False



