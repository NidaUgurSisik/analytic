import streamlit as st
import requests
import pandas as pd
import numpy as np
from streamlit_tags import st_tags


def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="✂️", page_title="Zero-shot Text")


c1, c2 = st.columns([1, 6])

with c2:

    st.caption("")
    st.title("Zero-Shot Text Classifier")

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


st.sidebar.write("")

# For elements to be displayed in the sidebar, we need to add the sidebar element in the widget.

# We create a text input field for users to enter their API key.

API_KEY = st.sidebar.text_input(
    "Enter your HuggingFace API key",
    help="Once you created you HuggingFace account, you can get your free API token in your settings page: https://huggingface.co/settings/tokens",
    type="password",
)

# Adding the HuggingFace API inference URL.
API_URL = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"

# Now, let's create a Python dictionary to store the API headers.
headers = {"Authorization": f"Bearer {API_KEY}"}


st.sidebar.markdown("---")


with st.form(key="my_form"):
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
    shows_list = shows.iloc[:, 0].tolist()
    shows_list_head = shows.iloc[:, 0].head(3).tolist()

    pre_defined_keyphrases = shows_list_head

    # Python list comprehension to create a string from the list of keyphrases.
    keyphrases_string = f"{new_line.join(map(str, shows_list))}"

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