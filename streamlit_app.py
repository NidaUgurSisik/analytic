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


c1, c2 = st.columns([2,10])

# The snowflake logo will be displayed in the first column, on the left.

with c2:
    
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

    countjob = shows['Job Title'].value_counts()
    st.bar_chart(countjob[:15])

    countkey = shows['Keywords'].value_counts()
    st.bar_chart(countkey[:10])

    countcountry= shows['Country/Region'].value_counts()
    st.bar_chart(countcountry[:15])

    longitude = []
    latitude = []

    data = {'Count': shows['City'].value_counts()}
    df = pd.DataFrame(data)
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
            
            return geolocator.geocode(city)
        
        except GeocoderTimedOut:
            
            return findGeocode(city)	

    # each value from city column
    # will be fetched and sent to
    # function find_geocode
    for i in (df.index):
        
        if findGeocode(i) != None:
            
            loc = findGeocode(i)
            
            # coordinates returned from
            # function is stored into
            # two separate list
            latitude.append(loc.latitude)
            longitude.append(loc.longitude)
        
        # if coordinate for a city not
        # found, insert "NaN" indicating
        # missing value
        else:
            latitude.append(np.nan)
            longitude.append(np.nan)

    # now add this column to dataframe
    df["Longitude"] = longitude
    df["Latitude"] = latitude

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=37.76,
            longitude=-122.4,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
            'HexagonLayer',
            data=df,
            get_position='[Longitude, Latitude]',
            radius=200,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=df,
                get_position='[Longitude, Latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
        ],
    ))
   

# We need to set up session state via st.session_state so that app interactions don't reset the app.

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False


############ TABBED NAVIGATION ############

# First, we're going to create a tabbed navigation for the app via st.tabs()
# tabInfo displays info about the app.
# tabMain displays the main app.

MainTab, InfoTab = st.tabs(["Main", "Info"])

with InfoTab:

    st.subheader("What is Streamlit?")
    st.markdown(
        "[Streamlit](https://streamlit.io) is a Python library that allows the creation of interactive, data-driven web applications in Python."
    )

    st.subheader("Resources")
    st.markdown(
        """
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [Cheat sheet](https://docs.streamlit.io/library/cheatsheet)
    - [Book](https://www.amazon.com/dp/180056550X) (Getting Started with Streamlit for Data Science)
    """
    )

    st.subheader("Deploy")
    st.markdown(
        "You can quickly deploy Streamlit apps using [Streamlit Community Cloud](https://streamlit.io/cloud) in just a few clicks."
    )


with MainTab:

    # Then, we create a intro text for the app, which we wrap in a st.markdown() widget.

    st.write("")
    st.markdown(
        """
    Classify keyphrases on the fly with this mighty app. No training needed!
    """
    )

    st.write("")

    # Now, we create a form via `st.form` to collect the user inputs.

    # All widget values will be sent to Streamlit in batch.
    # It makes the app faster!

    with st.form(key="my_form"):

        ############ ST TAGS ############

        # We initialize the st_tags component with default "labels"

        # Here, we want to classify the text into one of the following user intents:
        # Transactional
        # Informational
        # Navigational

        labels_from_st_tags = st_tags(
            value=["Transactional", "Informational", "Navigational"],
            maxtags=3,
            suggestions=["Transactional", "Informational", "Navigational"],
            label="",
        )

        # The block of code below is to display some text samples to classify.
        # This can of course be replaced with your own text samples.

        # MAX_KEY_PHRASES is a variable that controls the number of phrases that can be pasted:
        # The default in this app is 50 phrases. This can be changed to any number you like.

        MAX_KEY_PHRASES = 50

        new_line = "\n"

        pre_defined_keyphrases = [
            "I want to buy something",
            "We have a question about a product",
            "I want a refund through the Google Play store",
            "Can I have a discount, please",
            "Can I have the link to the product page?",
        ]

        # Python list comprehension to create a string from the list of keyphrases.
        keyphrases_string = f"{new_line.join(map(str, pre_defined_keyphrases))}"

        # The block of code below displays a text area
        # So users can paste their phrases to classify

        text = st.text_area(
            # Instructions
            "Enter keyphrases to classify",
            # 'sample' variable that contains our keyphrases.
            keyphrases_string,
            # The height
            height=200,
            # The tooltip displayed when the user hovers over the text area.
            help="At least two keyphrases for the classifier to work, one per line, "
            + str(MAX_KEY_PHRASES)
            + " keyphrases max in 'unlocked mode'. You can tweak 'MAX_KEY_PHRASES' in the code to change this",
            key="1",
        )

        # The block of code below:

        # 1. Converts the data st.text_area into a Python list.
        # 2. It also removes duplicates and empty lines.
        # 3. Raises an error if the user has entered more lines than in MAX_KEY_PHRASES.

        text = text.split("\n")  # Converts the pasted text to a Python list
        linesList = []  # Creates an empty list
        for x in text:
            linesList.append(x)  # Adds each line to the list
        linesList = list(dict.fromkeys(linesList))  # Removes dupes
        linesList = list(filter(None, linesList))  # Removes empty lines

        if len(linesList) > MAX_KEY_PHRASES:
            st.info(
                f"❄️ Note that only the first "
                + str(MAX_KEY_PHRASES)
                + " keyphrases will be reviewed to preserve performance. Fork the repo and tweak 'MAX_KEY_PHRASES' in the code to increase that limit."
            )

            linesList = linesList[:MAX_KEY_PHRASES]

        submit_button = st.form_submit_button(label="Submit")

    ############ CONDITIONAL STATEMENTS ############

    # Now, let us add conditional statements to check if users have entered valid inputs.
    # E.g. If the user has pressed the 'submit button without text, without labels, and with only one label etc.
    # The app will display a warning message.

    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and not text:
        st.warning("❄️ There is no keyphrases to classify")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button and not labels_from_st_tags:
        st.warning("❄️ You have not added any labels, please add some! ")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button and len(labels_from_st_tags) == 1:
        st.warning("❄️ Please make sure to add at least two labels for classification")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.valid_inputs_received:

        if submit_button:

            # The block of code below if for our session state.
            # This is used to store the user's inputs so that they can be used later in the app.

            st.session_state.valid_inputs_received = True

        ############ MAKING THE API CALL ############

        # First, we create a Python function to construct the API call.

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        # The function will send an HTTP POST request to the API endpoint.
        # This function has one argument: the payload
        # The payload is the data we want to send to HugggingFace when we make an API request

        # We create a list to store the outputs of the API call

        list_for_api_output = []

        # We create a 'for loop' that iterates through each keyphrase
        # An API call will be made every time, for each keyphrase

        # The payload is composed of:
        #   1. the keyphrase
        #   2. the labels
        #   3. the 'wait_for_model' parameter set to "True", to avoid timeouts!

        for row in linesList:
            api_json_output = query(
                {
                    "inputs": row,
                    "parameters": {"candidate_labels": labels_from_st_tags},
                    "options": {"wait_for_model": True},
                }
            )

            # Let's have a look at the output of the API call
            # st.write(api_json_output)

            # All the results are appended to the empty list we created earlier
            list_for_api_output.append(api_json_output)

            # then we'll convert the list to a dataframe
            df = pd.DataFrame.from_dict(list_for_api_output)

        st.success("✅ Done!")

        st.caption("")
        st.markdown("### Check the results!")
        st.caption("")

        # st.write(df)

        ############ DATA WRANGLING ON THE RESULTS ############
        # Various data wrangling to get the data in the right format!

        # List comprehension to convert the score from decimals to percentages
        f = [[f"{x:.2%}" for x in row] for row in df["scores"]]

        # Join the classification scores to the dataframe
        df["classification scores"] = f

        # Rename the column 'sequence' to 'keyphrase'
        df.rename(columns={"sequence": "keyphrase"}, inplace=True)

        # The API returns a list of all labels sorted by score. We only want the top label.

        # For that, we need to select the first element in the 'labels' and 'classification scores' lists
        df["label"] = df["labels"].str[0]
        df["accuracy"] = df["classification scores"].str[0]

        # Drop the columns we don't need
        df.drop(["scores", "labels", "classification scores"], inplace=True, axis=1)

        # st.write(df)

        # We need to change the index. Index starts at 0, so we make it start at 1
        df.index = np.arange(1, len(df) + 1)

        # Display the dataframe
        st.write(df)

        cs, c1 = st.columns([2, 2])




        # The code below is for the download button
        # Cache the conversion to prevent computation on every rerun

        with cs:

            @st.experimental_memo
            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(df)

            st.caption("")

            st.download_button(
                label="Download results",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv",
            )