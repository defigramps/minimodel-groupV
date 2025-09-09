import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure page
st.set_page_config(page_title='Laptop Price Predictor', layout='wide')
st.title('Laptop Price Prediction App')
st.write('Enter the laptop specifications to predict the price in Euros. (No sliders)')

# Safe cache decorator (st.cache_resource if available, fallback to st.cache)
try:
    cache_decorator = st.cache_resource
except Exception:
    cache_decorator = st.cache


@cache_decorator
def load_resources():
    model = joblib.load('laptop_price.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature.joblib')
    return model, scaler, feature_names


model, scaler, feature_names = load_resources()

# Load the dataset to get unique values
df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')

# Get unique values for categorical inputs
companies = sorted(df['Company'].unique())
type_names = sorted(df['TypeName'].unique())
storage_types = ['SSD', 'HDD', 'Flash Storage', 'Hybrid']

# Common screen resolutions
resolutions = [
    '1366x768', '1600x900', '1920x1080', '2560x1440', '2560x1600',
    '2880x1800', '3200x1800', '3840x2160'
]

# Preset definitions
PRESETS = {
    'Custom': {},
    'Light Ultrabook': {
        'company': companies[0] if companies else None,
        'type_name': 'Ultrabook' if 'Ultrabook' in type_names else type_names[0],
        'inches': 13.3,
        'ram': 8,
        'touchscreen': False,
        'ips_panel': True,
        'resolution': '1920x1080',
        'storage_type': 'SSD',
        'storage_size': 256,
        'weight': 1.3
    },
    'Gaming': {
        'company': companies[0] if companies else None,
        'type_name': 'Gaming' if 'Gaming' in type_names else type_names[0],
        'inches': 15.6,
        'ram': 16,
        'touchscreen': False,
        'ips_panel': True,
        'resolution': '1920x1080',
        'storage_type': 'SSD',
        'storage_size': 512,
        'weight': 2.6
    },
    'Budget': {
        'company': companies[0] if companies else None,
        'type_name': type_names[0],
        'inches': 15.6,
        'ram': 4,
        'touchscreen': False,
        'ips_panel': False,
        'resolution': '1366x768',
        'storage_type': 'HDD',
        'storage_size': 500,
        'weight': 2.3
    }
}


def apply_preset(preset_name):
    return PRESETS.get(preset_name, {})


# UI: Tabs with an in-page form (no sidebar)
tabs = st.tabs(["Configure", "About"])

with tabs[0]:
    st.subheader('Configure laptop')
    with st.form('predict_form'):
        preset = st.selectbox('Preset', list(PRESETS.keys()), index=0)
        preset_vals = apply_preset(preset)

        inches_options = [11.6, 12.0, 12.5, 13.3, 14.0, 15.0, 15.6, 16.0, 17.3]

        col1, col2 = st.columns(2)
        with col1:
            company = st.selectbox('Company', companies, index=companies.index(preset_vals.get('company')) if preset_vals.get('company') in companies else 0)
            type_name = st.selectbox('Type Name', type_names, index=type_names.index(preset_vals.get('type_name')) if preset_vals.get('type_name') in type_names else 0)
            inches = st.selectbox('Screen Size (Inches)', inches_options, index=inches_options.index(preset_vals.get('inches')) if preset_vals.get('inches') in inches_options else inches_options.index(15.6))
            ram = st.radio('RAM (GB)', [4, 8, 16, 32, 64], index=[4,8,16,32,64].index(preset_vals.get('ram')) if preset_vals.get('ram') in [4,8,16,32,64] else 1)

        with col2:
            resolution = st.selectbox('Screen Resolution', resolutions, index=resolutions.index(preset_vals.get('resolution')) if preset_vals.get('resolution') in resolutions else 2)
            touchscreen = st.radio('Touchscreen', ['No', 'Yes'], index=1 if preset_vals.get('touchscreen') else 0) == 'Yes'
            ips_panel = st.radio('IPS Panel', ['No', 'Yes'], index=1 if preset_vals.get('ips_panel') else 0) == 'Yes'
            storage_type = st.selectbox('Storage Type', storage_types, index=storage_types.index(preset_vals.get('storage_type')) if preset_vals.get('storage_type') in storage_types else 0)

        with st.expander('Advanced'):
            storage_size = st.number_input('Storage Size (GB)', min_value=32, max_value=2000, value=preset_vals.get('storage_size', 256), step=32)
            weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=preset_vals.get('weight', 2.0), step=0.1)

        submitted = st.form_submit_button('Predict')

    if submitted:
        # Validation
        if storage_size < 32:
            st.error('Storage must be at least 32 GB')
        else:
            # Parse resolution to pixels
            try:
                width, height = map(int, resolution.split('x'))
                pixels = width * height
            except Exception:
                st.error('Invalid resolution format')
                pixels = 0

            # Create input dataframe
            input_data = pd.DataFrame({
                'Inches': [inches],
                'Ram': [ram],
                'Weight': [weight],
                'Touchscreen': [1 if touchscreen else 0],
                'IPS_Panel': [1 if ips_panel else 0],
                'Pixels': [pixels],
                'Storage_Size_GB': [storage_size]
            })

            # One-hot encode categorical (matching training drop_first behavior)
            for comp in companies[1:]:
                input_data[f'Company_{comp}'] = [1 if company == comp else 0]
            for typ in type_names[1:]:
                input_data[f'TypeName_{typ}'] = [1 if type_name == typ else 0]
            for stor in storage_types[1:]:
                input_data[f'Storage_Type_{stor}'] = [1 if storage_type == stor else 0]

            # Ensure all features are present
            for feat in feature_names:
                if feat not in input_data.columns:
                    input_data[feat] = 0

            # Reorder columns to match training
            try:
                input_ordered = input_data[feature_names]
            except Exception:
                # If ordering fails, fallback to columns in input_data
                input_ordered = input_data.reindex(columns=feature_names, fill_value=0)

            # Scale and predict
            input_scaled = scaler.transform(input_ordered)
            prediction = model.predict(input_scaled)[0]

            # Display results
            st.success(f'Predicted Price: €{prediction:.2f}')

            # Layout: summary + contributions
            left, right = st.columns([1, 2])
            with left:
                st.write('### Input Summary')
                summary = {
                    'Company': [company],
                    'Type': [type_name],
                    'Screen (in)': [inches],
                    'RAM (GB)': [ram],
                    'Storage': [f"{storage_size} GB ({storage_type})"],
                    'Weight (kg)': [weight]
                }
                st.table(pd.DataFrame(summary))

            with right:
                st.write('### Top Feature Contributions')
                # Try to compute linear contributions if coef_ exists
                try:
                    coeffs = np.ravel(model.coef_)
                    contributions = input_scaled[0] * coeffs
                    contrib_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Contribution': contributions
                    }).sort_values('Contribution', ascending=False)
                    st.dataframe(contrib_df.head(8).reset_index(drop=True))
                except Exception:
                    st.info('Model does not expose linear coefficients; contributions unavailable.')

with tabs[1]:
    st.subheader('About')
    st.write('This app predicts laptop prices using a pre-trained model. Change inputs on the Configure tab and press Predict. No sliders are used in this UI.')

