# Forecasting Tools
- Ini adalah auto-ml sederhana untuk melakukan forecasting
- Hanya mereplikasi dari artikel berikut: https://www.gojek.io/blog/under-the-hood-of-gojeks-automated-forecasting-tool
- Ini video demonstrasinya: https://youtu.be/u8k0UhhfbQM

# Cara kerja
- Lakukan forecasting sederhana
- Validasi model dengan timeseries CV (default melakukan 10x cv untuk predict 1 data future)
- Sekarang hanya menggunakan model XGBoost
- Data training akan ditransformasi menjadi data waktu (year, month, day)

# Cara pakai
- clone git ini
- buat python environment `python -b venv forecasting_tool_env`
- masuk ke environment tsb `source forecasting_tool_env/bin/activate`
- aktifkan API `python src/api.py`
- aktifkan streamlit `streamlit run src/streamlit.py`
- Lakukan training & prediction sederhana pada streamlit
