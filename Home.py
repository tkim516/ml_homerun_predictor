import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Home Run Predictor", 
                   page_icon="⚾",
                   layout="centered")

df = pd.read_csv('train.csv')
df = df.sort_values(by='strikes', ascending=True).copy()

df_park = pd.read_csv('park_dimensions.csv')
df_merged = df.merge(df_park, on="park", how="left").copy()
columns_to_drop = [
  'park',
  'bip_id',
  'game_date', # Include?
  'home_team', # Include?
  'away_team', # Include?
  'batter_team', # Include?
  'NAME',
  'batter_name', # Include?
  'pitcher_name', # Include?
  'batter_id',
  'pitcher_id',
  'bb_type',
  'is_home_run',
]

cat_vars = [
  'Cover',
  'is_batter_lefty',
  'is_pitcher_lefty',
  'bearing',
  'pitch_name',
  'inning',
  'outs_when_up',
  'balls',
  'strikes',
]

df_cleaned = df_merged.drop(columns=columns_to_drop).copy()
df_encoded = pd.get_dummies(df_cleaned, columns=cat_vars)

def get_encoded_row_from_df(i):
  return df_encoded.iloc[i]

def get_scenario_info(i):
  date = df_merged['game_date'].iloc[i]
  home_team = df_merged['home_team'].iloc[i]
  away_team = df_merged['away_team'].iloc[i]
  stadium = df_merged['NAME'].iloc[i]
  lf_D = df_merged['LF_Dim'].iloc[i]
  cf_D = df_merged['CF_Dim'].iloc[i]
  rf_D = df_merged['RF_Dim'].iloc[i]
  lf_W = df_merged['LF_W'].iloc[i]
  cf_W = df_merged['CF_W'].iloc[i]
  rf_W = df_merged['RF_W'].iloc[i]
  
  pitcher_name = df_merged['pitcher_name'].iloc[i]
  pitcher_name_split = pitcher_name.split(',')
  pitcher_name = pitcher_name_split[1].strip().capitalize() + ' ' + pitcher_name_split[0].strip().capitalize()

  batter_name = df_merged['batter_name'].iloc[i]
  batter_name_split = batter_name.split(',')
  batter_name = batter_name_split[1].strip().capitalize() + ' ' + batter_name_split[0].strip().capitalize()

  strikes = df_merged['strikes'].iloc[i]
  balls = df_merged['balls'].iloc[i]
  outs = df_merged['outs_when_up'].iloc[i]
  inning = df_merged['inning'].iloc[i]

  pitch_thrown = df_merged['pitch_name'].iloc[i]
  pitch_mph = df_merged['pitch_mph'].iloc[i]
  # strikes, balls, outs, inning
  
  scenario_dict = {'date': date, 
                   'home_team': home_team, 
                   'away_team': away_team, 
                   'stadium': stadium, 
                   'lf_D': lf_D,
                   'cf_D': cf_D, 
                   'rf_D': rf_D,
                   'lf_W': lf_W,
                   'cf_W': cf_W, 
                   'rf_W': rf_W,
                   'pitcher_name': pitcher_name, 
                   'batter_name': batter_name,
                   'strikes': strikes,
                   'balls': balls,
                   'outs': outs,
                   'inning': inning,
                   'pitch_thrown': pitch_thrown,
                   'pitch_mph': pitch_mph,}
  return scenario_dict 

ada_pickle = open('ada.pickle', 'rb') 
ada_clf = pickle.load(ada_pickle)
ada_pickle.close()

if 'homerun' not in st.session_state:
  st.session_state['homerun'] = None

if 'pred_proba' not in st.session_state:
  st.session_state['pred_proba'] = 0

if 'i' not in st.session_state:
  st.session_state['i'] = 0

if st.session_state['homerun'] == True:
  st.balloons()
  st.markdown("<h1 style='font-size: 60px; text-align: center; color: #57cfff;'>HOME RUN!!!</h1>", unsafe_allow_html=True)
  st.divider()
  st.image('homerun.gif', use_column_width=True)

if st.session_state['homerun'] == False:
  st.markdown("<h1 style='font-size: 60px; text-align: center; color: #eb5b42;'>TRY AGAIN</h1>", unsafe_allow_html=True)
  st.divider()
  st.image('try_again.gif', use_column_width=True)

elif st.session_state['homerun'] == None:

  scenario_data = get_scenario_info(st.session_state['i'])
  encoded_row = get_encoded_row_from_df(st.session_state['i'])

  st.markdown("<h1 style='text-align: center; color: #57cfff;'>Can you hit a home run?</h1>", unsafe_allow_html=True)
  st.markdown("<h3 style='text-align: center;'>Use machine learning models to predict home runs</h3>", unsafe_allow_html=True)
  st.divider( )
  st.image('ohtani.gif', use_column_width=True)

  with st.sidebar:
    st.header('Park info', divider='gray')
    st.subheader(f"LF wall distance: {scenario_data['lf_D']} feet")
    st.subheader(f"LF wall height: {scenario_data['lf_W']} feet")
    st.subheader(f"CF wall distance: {scenario_data['cf_D']} feet")
    st.subheader(f"CF wall height: {scenario_data['cf_W']} feet")
    st.subheader(f"RF wall distance: {scenario_data['rf_D']} feet")
    st.subheader(f"RF wall height: {scenario_data['rf_W']} feet")
    st.header('Game info', divider='gray')
    st.subheader(f'Inning number: {scenario_data['inning']}')
    st.subheader(f'Outs: {scenario_data['outs']}')
    st.subheader(f'Balls: {scenario_data['balls']}')
    st.subheader(f'Strikes: {scenario_data['strikes']}')
  
  st.divider()
  st.subheader(f'{scenario_data['home_team']} vs {scenario_data['away_team']} at {scenario_data['stadium']} on {scenario_data['date']}')
  st.write("")
  st.subheader(f'Batting as {scenario_data['batter_name']} and facing pitcher, {scenario_data['pitcher_name']}.')
  st.subheader(f'{scenario_data['pitcher_name']} throws you a {scenario_data['pitch_thrown']} at {scenario_data['pitch_mph']} MPH!')
  st.divider()
  
  with st.form(key='input_form'):
    input_speed = st.slider('Set launch speed (MPH)', min_value=0, max_value=105, value=50)
    input_angle = st.slider('Set launch angle (degrees)', min_value=-80, max_value=80, value=20)
    input_bearing = st.radio('Set ball flight direction', options=['Left', 'Center', 'Right'])
    
    encoded_row['launch_speed'] = input_speed
    encoded_row['launch_angle'] = input_angle

    if input_bearing == 'Left':
      encoded_row['bearing_left'] = True
      encoded_row['bearing_center'] = False
      encoded_row['bearing_right'] = False
    
    elif input_bearing == 'Center':
      encoded_row['bearing_left'] = False
      encoded_row['bearing_center'] = True
      encoded_row['bearing_right'] = False
    
    elif input_bearing == 'Right':
      encoded_row['bearing_left'] = False
      encoded_row['bearing_center'] = False
      encoded_row['bearing_right'] = True
    
    st.markdown("""
    <style>
    .center-button {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Wrap the button in a div with the custom class
    submit_form = st.form_submit_button(label='Hit', type='primary', use_container_width=True)

  encoded_row_df = encoded_row.to_frame().T

  if submit_form:
    pred = ada_clf.predict(encoded_row_df)[0]
    pred_proba = ada_clf.predict_proba(encoded_row_df).max()
    st.session_state['pred_proba'] = pred_proba
    
    if pred == 1:
      st.session_state['homerun'] = True
    
    if pred == 0:
      st.session_state['homerun'] = False
    
    st.rerun()

st.divider()
with st.expander('Game tips'):
  st.write('Higher launch speeds have a greater home run probability.')
  st.write('The ideal launch angle is between 20 and 25 degrees.')
  st.write('Each park has different dimensions. Try hitting the ball towards the wall with the shortest distance and height.')

if st.button('New at bat'):
  st.session_state['i'] += 1
  st.session_state['homerun'] = None
  st.session_state['pred_proba'] = 0

  st.rerun()