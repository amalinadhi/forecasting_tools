import numpy as np
import pandas as pd
import utils
import yaml
import requests
import streamlit as st


# ===================================
# Get the data info
def extract_data(data, n=5):
    # Copy the data
    data = data.copy()
    
    # Get data info
    cols = data.columns
    types = data.dtypes
    contents = [data[col].head(n).values for col in cols]
    for i, content in enumerate(contents):
        contents[i] = [str(val) for val in content]
    contents = [', '.join(content) + ', ...' for content in contents]

    # Turn into dataframe
    info_df = pd.DataFrame({
        "Feature": cols,
        "Class": types,
        "Content": contents
    })
    info_df = info_df.style.hide()

    return info_df

def export_config(model_settings):
    # Buka default config
    with open("config/config_default.yaml", 'r') as file:
        config_data = yaml.safe_load(file)

    # Update config
    for key, val in model_settings.items():
        config_data[key] = val
    config_data["raw_dataset_path"] = "data/raw/data.csv"

    # Dump config
    with open("config/config.yaml", 'w') as file:
        yaml.dump(config_data, file, sort_keys=False)

    return config_data

@st.cache_data
def fetch_data(path, config_data):
    data = pd.read_csv(path, parse_dates=[config_data['datetime_col']])
    return data

# ====================================
# Pernak-pernik
st.title('Forecasting Tools')
st.caption('Forecasting tools sederhana untuk latihan @Pacmann')
tab_main, tab_doc = st.tabs(["Main", "Documentation"])


# FLAG
is_loaded = False
is_inspected = False
is_date_chosen = False
is_model_created = False


# Tab Main
with tab_main:
    # 1st container: File upload
    with st.container():
        st.subheader('File Upload', divider='gray')

        # Upload file
        uploaded_file = st.file_uploader('Choose a CSV file',
                                         accept_multiple_files=False)
        
        # Read the uploaded file
        if uploaded_file is not None:
            # Convert to dataframe
            data = pd.read_csv(uploaded_file)

            # Append status
            #is_loaded = st.button(label='Load data')
            is_loaded = True

            if is_loaded:
                st.success(f'Data successfully loaded', icon="✅")

    # 2nd container: Data Inspection
    with st.container():
        st.subheader('Data Inspection', divider='gray')

        if is_loaded is False:
            st.warning('Data is not loaded yet')
        else:
            try:
                # Get the info_df
                info_df = extract_data(data)

                # Write info
                st.info(f'File contains **{data.shape[0]}** rows', icon="ℹ️")
                st.write(info_df.to_html() + '<br>', unsafe_allow_html=True)

                # Flag
                is_inspected = True
            except Exception as e:
                st.error(f'Error!', icon="🚨")
                st.exception(e)

    # 3rd container: Model Setting
    if is_inspected:
        data = data.copy()
        with st.container():
            st.subheader('Model Settings', divider='gray')

            # Get the model setting
            datetime_col = st.selectbox(
                label = 'Date/datetime column:',
                options = data.columns.tolist(),
                index = None
            )
            if datetime_col:
                is_date_chosen = True

            target_col = st.selectbox(
                label = 'Feature to be forecasted:',
                options = data.columns.tolist(),
                index = None
            )

            if is_date_chosen:
                last_date_train = st.selectbox(
                label = 'Last date/datetime for training set:',
                options = sorted(set(data[datetime_col]))
                )

                # Advance setting
                with st.expander("Advanced setting"):
                    n_splits = st.text_input(
                        "Number of Time Series CV splits:",
                        10
                    )
                    test_size = st.text_input(
                        "Number of Time Series CV test size:",
                        1
                    )

                # Submit data untuk dimodelkan
                with st.form("model_setting_form"):
                    submitted = st.form_submit_button("Run!")
                    if submitted:
                        # Summarize model settings
                        model_settings = {
                            "datetime_col": datetime_col,
                            "target_col": target_col,
                            "last_date_train": last_date_train,
                            "n_splits": int(n_splits),
                            "test_size": int(test_size)
                        }

                        # Export YAML DATA
                        config_data = export_config(model_settings)
                        st.success("Config File has been created", icon="✅")

                        # Export import data
                        data_path = config_data["raw_dataset_path"]
                        data.to_csv(data_path, index=False)
                        st.success(f"Data has been dumped to *{data_path}*", icon="✅")

                        # Then, let's start modeling
                        st.info("Start modelling", icon="ℹ️")
                        with st.spinner():
                            # Request to perform training data
                            res = requests.post("http://localhost:8000/train").json()
                            if res["error"] is None:
                                st.success(res["message"], icon="✅")
                                is_model_created = True
                            else:
                                st.error(res["error"], icon="🚨")

    # 4th container: Results
    if is_model_created:
        with st.container():
            st.subheader("Forecasting ahead results", divider='gray')

            # Call for prediction
            st.info("Start predicting ahead", icon="ℹ️")
            with st.spinner():
                res_predict = requests.post("http://localhost:8000/predict").json()
                if res_predict["error"] is None:
                    st.success("Done prediction", icon="✅")

                    # Get the predicted data
                    res_results = res_predict["results"]
                    data_pred = pd.DataFrame(res_results["data"],
                                             columns = res_results["columns"],
                                             index = res_results["index"])
                    data_pred["Date"] = data_pred["Date"].astype('<M8[ns]')

                    # Load config data
                    with open("config/config.yaml", 'r') as file:
                        config_data = yaml.safe_load(file)

                    # Load data train & test
                    data_all = fetch_data(path=config_data['raw_dataset_path'],
                                          config_data=config_data)

                    # Gabungkan data untuk plotting
                    cols = [
                        config_data["datetime_col"],
                        config_data["target_col"]
                    ]
                    data_merge = data_all[cols].copy()
                    data_merge[cols[1]+'_Forecasted'] = np.nan
                    data_merge[cols[1]+'_Forecasted'].iloc[data_pred.index] = data_pred[cols[1]]

                    # Rename
                    data_merge = data_merge.rename(columns={
                        cols[1]: 'Actual',
                        cols[1]+'_Forecasted': 'Forecasted'
                    })

                    # Plot, finally
                    st.line_chart(
                        data_merge,
                        x = "Date",
                        y = ['Actual', 'Forecasted'],
                        color=["#FF0000", "#0000FF"]
                    )
                else:
                    st.error(res["error"], icon="🚨")