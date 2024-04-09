import streamlit as st
import pandas as pd
import pickle
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import base64

@st.cache_resource
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
# Load your data once at the beginning
file_path = 'final_ipl_df.csv'
merged_df = load_data(file_path)

most_runs_player_race = pd.read_csv('most_run_animation.csv')
most_wickets_player_race = pd.read_csv('most_wicket_animation.csv')



def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Win Predictor', 'Analysis by Venue', 'Player Stats', 'Match Analysis', 'Team Analysis', 'Versus', 'Seasonal Stats','Overall Stats'])

    if page == 'Win Predictor':
        show_page_1()
    elif page == 'Analysis by Venue':
        show_page_2()
    elif page == 'Player Stats':
        show_page_3()
    elif page == 'Match Analysis':
        show_page_4()
    elif page == 'Team Analysis':
        show_page_5()
    elif page == 'Versus':
        show_page_6()
    elif page == 'Seasonal Stats':
        show_page_7()
    elif page == 'Overall Stats':
        show_page_8()



def show_page_1():
    add_name_to_header('@abhishek_rawool')
    @ st.cache_data
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()


    img = get_img_as_base64("football-stadium-night-generative-ai.jpg")

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img}");
    background-size: 110% 110%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    
    /* Add semi-transparent overlay */
    [data-testid="stAppViewContainer"] > .main::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(rgba(5, 5, 5, 0.1), rgba(5, 5, 5, 0.1)); /* Semi-transparent white overlay */
    }}
    
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    
    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


    # Custom CSS styling
    custom_css = """
    <style>
    /* Change text color */
    body {
        color: #333333; /* Dark gray text color */
        font-family: Arial, sans-serif; /* Set font family */
    }
    
    /* Change select box background color */
    [data-baseweb="select"] > div {
        background-color: #FFFFFF; /* Light gray background color */
    }
    
    /* Change select box text color */
    [data-baseweb="select"] > div > div {
        color: #000000; /* Dark gray text color */
    
    }
    
    /* Change number input background color */
    .stNumberInput input {
        background-color: #FFFFFF; /* Light gray background color */
        color: #000000; /* Dark gray text color */
    }
    </style>
    """
    # Apply custom CSS styling
    st.markdown(custom_css, unsafe_allow_html=True)


    # Define custom CSS styling in a variable
    customize_css = """
    <style>
    /* Change text color for select box options */
    .stSelectbox label {
        color: #FFFFFF; /* Change color to blue */
    }
    </style>
    """
    # Apply custom CSS styling using st.markdown
    st.markdown(customize_css, unsafe_allow_html=True)



    teams = ['Mumbai Indians', 'Delhi Capitals', 'Kolkata Knight Riders',
             'Chennai Super Kings', 'Punjab Kings', 'Sunrisers Hyderabad',
             'Royal Challengers Bangalore', 'Rajasthan Royals',
             'Lucknow Super Giants', 'Gujarat Titans']

    cities = ['Ahmedabad', 'Chennai', 'Mumbai', 'Bengaluru', 'Bangalore',
       'Kolkata', 'Delhi', 'Dharamsala', 'Hyderabad', 'Lucknow', 'Jaipur',
       'Chandigarh', 'Guwahati', 'Navi Mumbai', 'Pune', 'Dubai',
       'Sharjah', 'Abu Dhabi', 'Visakhapatnam', 'Indore', 'Raipur',
       'Ranchi', 'Cuttack', 'Johannesburg', 'Centurion', 'Durban',
       'Bloemfontein', 'Port Elizabeth', 'Kimberley', 'East London',
       'Cape Town']

    pipe = pickle.load(open('pipe1.pkl', 'rb'))
    st.title(':orange[IPL Win Predictor]')

    col1,col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('**Select the Batting Team**', options=sorted(teams))

    with col2:
        balling_team = st.selectbox('**Select the Bowling Team**', options=sorted(teams))

    selected_city = st.selectbox('**Select host City**', sorted(cities))

    target = st.number_input('**Target**', min_value=0, max_value=1000, step=1)

    col3,col4,col5 = st.columns(3)

    with col3:
           score = st.number_input('**Score**', min_value=0, max_value=1000, step=1)
    with col4:
           overs = st.number_input('**Overs Completed**',min_value=0.0, max_value=19.6)
    with col5:
           wickets = st.number_input('**Wickets**', min_value=0, max_value=10, step=1)

    if st.button('Predict Probability'):
        if batting_team != balling_team:
            if target != 0:
               runs_left = target - score
               if overs != 0:
                   balls_left = 120 - (overs*6)
                   wickets_left = 10 - wickets
                   crr = score / (overs)
                   rrr = runs_left / (balls_left/6)

                   df =  pd.DataFrame({'batting_team':[batting_team],	'bowling_team':[balling_team],
                                 'city':[selected_city], 'runs_left':[runs_left],
                                 'balls_left':[balls_left]	,'wickets':[wickets_left]	,
                                 'total_runs':[target]	,'crr':[crr]	,'rrr':[rrr]})

                   st.header(f":black[{batting_team}  needs   {str(round(runs_left))}  runs in  {str(round(balls_left))}  balls]")
                   result = pipe.predict_proba(df)
                   loss = result[0][0]
                   win = result[0][1]
                   st.header(f':blue[{batting_team}]' + '-' + str(round(win*100)) + '%')
                   st.header(f':blue[{balling_team}]' + '-' + str(round(loss*100)) + '%')
               else:
                   st.subheader('**Enter the Completed Overs.**')
            else:
                st.subheader('**Enter the target score.**')
        else:
            st.subheader('**Select Different Bowling Team or Batting Team**')



def show_page_2():
    add_name_to_header('@abhishek_rawool')

    @st.cache_data
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    img = get_img_as_base64("marcus-wallis-mUtQXjjLPbw-unsplash.jpg")

    page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/png;base64,{img}");
        background-size: 110% 110%;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: local;
        }}

        /* Add semi-transparent overlay */
        [data-testid="stAppViewContainer"] > .main::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(rgba(5, 5, 5, 0.4), rgba(5, 5, 5, 0.3)); /* Semi-transparent white overlay */
        }}

        [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
        }}

        [data-testid="stToolbar"] {{
        right: 2rem;
        }}
        </style>
        """
    st.markdown(page_bg_img, unsafe_allow_html=True)



    st.title(':orange[Player and Team Analysis by Venue]')
    teams = ['Overall stats','Mumbai Indians', 'Delhi Capitals', 'Kolkata Knight Riders',
           'Chennai Super Kings', 'Punjab Kings', 'Sunrisers Hyderabad',
           'Royal Challengers Bangalore', 'Rajasthan Royals',
           'Lucknow Super Giants', 'Gujarat Titans']
    selected_team = st.selectbox('**Select Team**',teams)

    plot = st.button('Plot team Graph')
    if plot:
        if selected_team == 'Overall stats':
            st.write('**Click on circle to see Information**')
            venue_df = merged_df.copy()
            venue_total_match = venue_df.groupby(['venue', 'longitude', 'latitude'])['id'].nunique().reset_index(name='venue_total_match')
            venue_fours = venue_df[(venue_df['runs_off_bat'] == 4)].groupby(['venue'])[
                'runs_off_bat'].count().reset_index(name='Venue fours')
            venue_six = venue_df[(venue_df['runs_off_bat'] == 6)].groupby(['venue'])[
                'runs_off_bat'].count().reset_index(name='Venue six')
            venue_highest_score = \
            venue_df.groupby(['venue', 'id', 'innings', 'longitude', 'latitude'])['total_run'].sum().reset_index(
                name='Runs').groupby(['venue', 'longitude', 'latitude'])['Runs'].max().reset_index(
                name='Venue Team Runs')
            temp = venue_total_match.merge(venue_highest_score, on=['venue', 'longitude', 'latitude']).merge(
                venue_fours, on='venue', how='outer').merge(venue_six, on='venue', how='outer')

            st.write('•Bigger circle represents more matches at that venue')

            import plotly.express as px
            fig = px.scatter_mapbox(temp, lat="latitude", lon="longitude", size='venue_total_match',
                                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=15,
                                    zoom=5, mapbox_style='carto-positron', hover_name='venue',
                                    height=500, width=800,
                                    custom_data=['venue_total_match', 'Venue Team Runs','Venue fours','Venue six'])
            fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>" +
                                            "Total Matches: %{customdata[0]}<br>"
                                            "Highest Score: %{customdata[1]}<br>" +
                                            "Fours Hit: %{customdata[2]}<br>" +
                                            "Six Hit: %{customdata[3]}<extra></extra>")
            st.plotly_chart(fig)
        else:
            st.write('**Click on circle to see Information**')
            # win percentage based on venue
            temp_matches_win_venue = merged_df[(merged_df['batting_team'] == selected_team) | (merged_df['bowling_team'] == selected_team)]
            temp_matches_at_venue = temp_matches_win_venue.groupby(['venue', 'latitude', 'longitude'])['id'].nunique().reset_index(name='total matches')
            temp_matches_win_at_venue = temp_matches_win_venue[temp_matches_win_venue['winner'] == selected_team].groupby(['venue'])['id'].nunique().reset_index(name='wins')
            temp = temp_matches_at_venue.merge(temp_matches_win_at_venue, on='venue', how='left').fillna(0)
            temp['Win Percentage'] = ((temp['wins'] / temp['total matches']) * 100).fillna(0)
            highest_score = merged_df[(merged_df['batting_team'] == selected_team)].groupby(['venue', 'id'])[
                'total_run'].sum().reset_index(name='runs').groupby(['venue'])['runs'].max().reset_index(
                name='Highest Score')
            total_runs = merged_df[(merged_df['batting_team'] == selected_team)].groupby(['venue'])[
                'total_run'].sum().reset_index(name='Total runs')
            total_overs = merged_df[(merged_df['batting_team'] == selected_team)].groupby(['venue', 'id'])[
                'ball'].max().reset_index(name='overs').groupby(['venue'])['overs'].sum().reset_index(
                name='Total overs')
            temp = temp.merge(highest_score, on='venue', how='outer').merge(total_runs, on='venue', how='outer').merge(
                total_overs, on='venue', how='outer')
            temp['run rate'] = (temp['Total runs'] / temp['Total overs'])

            import plotly.express as px
            fig = px.scatter_mapbox(temp, lat="latitude", lon="longitude", color='Win Percentage',
                                    color_continuous_scale='jet',
                                    zoom=3, mapbox_style='carto-positron', hover_name='venue',
                                    height=500, width=800,
                                    custom_data=[ 'total matches', 'wins', 'Highest Score','Total runs','run rate','Win Percentage'])

            # Customize the hover tooltip
            fig.update_traces(marker=dict(size=10), hovertemplate="<b>%{hovertext}</b><br>" +
                                 "Total Matches Played: %{customdata[0]}<br>" +
                                 "Wins: %{customdata[1]}<br>" +
                                 "Highest Score: %{customdata[2]}<br>" +
                                 "Total runs: %{customdata[3]}<br>" +
                                 "run rate: %{customdata[4]:.1f}<br>" +
                                 "Win Percentage: %{customdata[5]:.2f}%<extra></extra>")

            st.plotly_chart(fig)

    st.write(
        '--------------------------------------------------------------------------------------------------------------------------')

    combined1 = pd.concat([merged_df['striker'], merged_df['non_striker'], merged_df['bowler']], axis=0)
    players = sorted(list(set(combined1)))
    selected_player = st.selectbox("**Select player, If you can't find the player's name, try searching by surname**",
                                   players)
    # playes stats according to venue
    player_total_match = merged_df[(merged_df['striker'] == selected_player)].groupby(['venue', 'latitude', 'longitude'])[
        'id'].nunique().reset_index(name='Total Matches Bat')
    player_total_runs =  merged_df[(merged_df['striker'] == selected_player)].groupby(['venue', 'latitude', 'longitude'])[
        'runs_off_bat'].sum().reset_index(name='Runs Scored')
    player_total_match_bowl = merged_df[(merged_df['bowler'] == selected_player)].groupby(['venue', 'latitude', 'longitude'])[
        'id'].nunique().reset_index(name='Total matches bowled')
    player_total_wicket =  merged_df[(merged_df['bowler'] == selected_player)].groupby(['venue', 'latitude', 'longitude'])[
        'is_wicket_delivery'].sum().reset_index(name='Wickets taken')
    temp1 = ( player_total_match.merge(player_total_runs, on=['venue', 'latitude', 'longitude'], how='outer').merge(
            player_total_match_bowl, on=['venue', 'latitude', 'longitude'], how='outer').merge(
            player_total_wicket, on=['venue', 'latitude', 'longitude'], how='outer')).fillna(0)
    temp1['combined_size'] = (temp1['Runs Scored'] + temp1['Wickets taken']).astype(int)
    balls_faced = merged_df[(merged_df['striker'] == selected_player) & ~(merged_df['wides'] > 0)].groupby(['venue'])[
        'ball'].count().reset_index(
        name='ball faced')
    highest_score = merged_df[(merged_df['striker'] == selected_player)].groupby(['venue', 'id'])[
        'runs_off_bat'].sum().reset_index(
        name='runs').groupby(['venue'])['runs'].max().reset_index(name='Highest score')
    temp1 = (temp1.merge(highest_score, on=['venue'], how='outer').merge(balls_faced, on=['venue'],
                                                                         how='outer')).fillna(0)
    temp1['Strike rate'] = (temp1['Runs Scored'] / temp1['ball faced']) * 100
    total_runs_conceded = merged_df[(merged_df['bowler'] == selected_player)].groupby(['venue'])[
        'total_run'].sum().reset_index(name='Total runs conceded')
    balls = merged_df[(merged_df['bowler'] == selected_player)].groupby(['venue'])['ball'].count().reset_index(
        name='Balls')
    temp1 = (temp1.merge(total_runs_conceded, on=['venue'], how='outer').merge(balls, on=['venue'],
                                                                               how='outer')).fillna(0)
    temp1['Economy rate'] = (temp1['Total runs conceded'] / (temp1['Balls'] / 6)).fillna(0)

    plot1 = st.button('Plot Batting Graph')
    plot2 = st.button('Plot Bowling Graph')

    if plot1:
        st.write('**Click on circle to see Information**')
        import plotly.express as px
        bat = temp1[temp1['ball faced']!=0]
        # Add batting graph to the figure
        batting_fig = px.scatter_mapbox(bat, lat="latitude", lon="longitude",
                                        color_continuous_scale='jet', color='venue',
                                        zoom=3, mapbox_style='carto-positron', hover_name='venue',
                                        height=500, width=1000,
                                        custom_data=['Total Matches Bat', 'Runs Scored', 'ball faced',
                                                     'Strike rate',
                                                     'Highest score'])

        batting_fig.update_traces(marker=dict(size=15), hovertemplate="<b>%{hovertext}</b><br>"
                                                                             "Total Matches Bat: %{customdata[0]}<br>" +
                                                                             "Runs Scored: %{customdata[1]}<br>" +
                                                                             "Ball faced: %{customdata[2]}<br>" +
                                                                             "Strike rate: %{customdata[3]:.2f}<br>" +
                                                                             "Higheset Score: %{customdata[4]}<extra></extra>")
        st.plotly_chart(batting_fig)

    if plot2:
        import plotly.express as px
        # Add bowling graph to the figure
        bowl = temp1[temp1['Balls']!=0]
        bowling_fig = px.scatter_mapbox(bowl ,lat="latitude", lon="longitude", color='venue',
                                        color_continuous_scale='jet',
                                        zoom=3, mapbox_style='carto-positron', hover_name='venue',
                                        height=500, width=1000,
                                        custom_data=['Total matches bowled', 'Balls', 'Total runs conceded',
                                                     'Wickets taken', 'Economy rate'])

        bowling_fig.update_traces(marker=dict(size=15), hovertemplate="<b>%{hovertext}</b><br>"
                                                                      "Total Matches Bowled: %{customdata[0]}<br>" +
                                                                      "Total balls bowled: %{customdata[1]}<br>" +
                                                                      "Total runs conceded: %{customdata[2]}<br>" +
                                                                      "Wickets taken: %{customdata[3]}<br>" +
                                                                      "Economy rate: %{customdata[4]:.2f}%<extra></extra>")
        st.plotly_chart(bowling_fig)




def show_page_3():
    add_name_to_header('@abhishek_rawool')
    st.title('Player Stats')
    # all players
    combined = pd.concat([merged_df['striker'], merged_df['non_striker']], axis=0)
    combined = list(set(combined))
    st.write("**If you can't find the player's name, try searching by surname**")
    col6, col7=st.columns(2)

    with col6:
        selected_player = st.selectbox('Select a Player', options=sorted(combined), key='batter')
        plot4 = st.button('Player batting stats')
        if plot4:
            # player total matches
            total_match = merged_df[(merged_df['striker'] == selected_player) | (merged_df['non_striker'] == selected_player)].groupby(['id'])[
                'id'].nunique().sum()
            # total runs
            total_runs = merged_df[(merged_df['striker'] == selected_player)]['runs_off_bat'].sum()
            # highest score
            highest_score = merged_df[(merged_df['striker'] == selected_player)].groupby('id')[
                'runs_off_bat'].sum().sort_values(ascending=False).values[0]
            # batting average
            total_dissmissals_aLL_season = len(merged_df[(merged_df['striker'] == selected_player) & (merged_df['is_wicket_delivery'] == 1)].groupby('id')['id'])
            total_runs_all_season = merged_df[(merged_df['striker'] == selected_player)]['runs_off_bat'].sum()
            batting_avg = round((total_runs_all_season / total_dissmissals_aLL_season), ndigits=2)
            ## Strike rate all season
            total_runs_all_season = merged_df[(merged_df['striker'] == selected_player)]['runs_off_bat'].sum()
            total_balls_faced_all_season = len(merged_df[(merged_df['striker'] == selected_player)& ~(merged_df['wides'] > 0)])
            strike_rate = round((total_runs_all_season / total_balls_faced_all_season) * 100, ndigits=2)
            # centuries and half centuries
            temp_centuries = merged_df[(merged_df['striker'] == selected_player)].groupby('id')[
                'runs_off_bat'].sum().reset_index()
            # half centuries
            half_centuries=temp_centuries[(temp_centuries['runs_off_bat'] < 100) & (temp_centuries['runs_off_bat'] > 49)]['runs_off_bat'].count()
            # centuries
            centuries=temp_centuries[(temp_centuries['runs_off_bat'] > 99)]['runs_off_bat'].count()
            ## boundries count
            fours = len(merged_df[(merged_df['striker'] == selected_player) & (merged_df['runs_off_bat'] == 4)])
            six = len(merged_df[(merged_df['striker'] == selected_player)  & ( merged_df['runs_off_bat'] == 6)])
            most_runs_by_season = merged_df.groupby(['season', 'striker'])['runs_off_bat'].sum().reset_index()
            most_runs_by_season = most_runs_by_season.sort_values(by=['season', 'runs_off_bat'],
                                                                  ascending=[True, False])
            most_runs_by_season = most_runs_by_season.drop_duplicates(subset='season')
            # player of the match
            merged_df['player_of_match'] = merged_df['player_of_match'].fillna(0)
            player_of_match= merged_df[(merged_df['player_of_match'] == selected_player)].groupby(['id'])['id'].nunique().sum()

            # Display the analysis using Streamlit
            st.subheader(f'{selected_player} Batting Stats')

            # Display total matches, runs, highest score, and batting average
            st.subheader('General Statistics')
            st.markdown(f"*Total Matches Played:* {total_match}")
            st.markdown(f"*Total Runs Scored:* {total_runs}")
            st.markdown(f"*Batting Average:* {batting_avg}")
            st.markdown(f"*Strike Rate:* {strike_rate}%")
            st.markdown(f"*Fours Hit:* {fours}")
            st.markdown(f"*Sixes Hit:* {six}")
            st.markdown(f"*Centuries:* {centuries}")
            st.markdown(f"*Half-Centuries:* {half_centuries}")

            # Display most wickets in a season and in a single match
            if highest_score > 0:
                st.subheader('Batting Achievements')
                st.markdown(f"*Highest Score:* {highest_score}")
            # Display player of the match count
            if player_of_match > 0:
                st.subheader('Player of the Match Count')
                st.markdown(f"*Player of the Match Awards:* {player_of_match}")

            orange_cap_count = 0
            for i in most_runs_by_season['striker']:
                if i == selected_player:
                    orange_cap_count += 1
            if orange_cap_count > 0:
                st.subheader('Orange cap winner')
                st.markdown(f"*Orange cap winner:* {orange_cap_count} ")

    with col7:
        selected_baller = st.selectbox('Select a Player', options=sorted(combined), key='baller')
        plot5 = st.button('Player bowling stats')
        if plot5:
            # bowler total maches
            bowler_total_mache_played = merged_df[(merged_df['bowler'] == selected_baller)].groupby(['id'])['id'].nunique().sum()
            # total wickets in all seasons
            total_wickets = merged_df[(merged_df['bowler'] == selected_baller)]['is_wicket_delivery'].sum()
            # bowling average all season
            total_runs_conceded = merged_df[(merged_df['bowler'] == selected_baller)]['total_run'].sum()
            total_wickets_taken = merged_df[(merged_df['bowler'] == selected_baller)]['is_wicket_delivery'].sum()
            bowling_avg = round(total_runs_conceded / total_wickets_taken, ndigits=2)
            # economy rate all season
            total_runs_conceded_all = merged_df[(merged_df['bowler'] == selected_baller)]['total_run'].sum()
            total_overs_bowled = len(merged_df[(merged_df['bowler'] == selected_baller)]['ball']) / 6
            economy = round(total_runs_conceded_all / total_overs_bowled, ndigits=2)
            # purple cap winner
            most_wickets = merged_df.groupby(['season', 'bowler'])['is_wicket_delivery'].sum().reset_index()
            most_wickets = most_wickets.sort_values(by=['season', 'is_wicket_delivery'], ascending=[True, False])
            most_wickets = most_wickets.groupby(['season'])[['bowler', 'is_wicket_delivery']].first()
            # most wickets in one match
            temp_wickets = merged_df[(merged_df['bowler'] == selected_baller)].groupby('id')[
                'is_wicket_delivery'].sum().reset_index()
            temp_wickets = temp_wickets['is_wicket_delivery'].max()
            # player of the match
            merged_df['player_of_match'] = merged_df['player_of_match'].fillna(0)
            player_of_match = merged_df[(merged_df['player_of_match'] == selected_baller)].groupby(['id'])['id'].nunique().sum()

            # Display the analysis using Streamlit
            st.subheader(f'{selected_baller} Bowling Stats')

            # Display total matches, wickets, bowling average, and economy rate
            st.subheader('General Bowling Statistics')
            st.markdown(f"*Total Matches Played:* {bowler_total_mache_played}")
            st.markdown(f"*Total Wickets Taken:* {total_wickets}")
            if bowling_avg > 0:
                st.markdown(f"*Bowling Average:* {bowling_avg}")
            if economy > 0:
                st.markdown(f"*Economy Rate:* {economy}")

            # Display most wickets in a season and in a single match
            if temp_wickets > 0:
                st.subheader('Wickets Achievements')
                st.markdown(f"*Most Wickets in a Single Match:* {temp_wickets}")

            # Display player of the match count
            if player_of_match > 0:
                st.subheader('Player of the Match Count')
                st.markdown(f"*Player of the Match Awards:* {player_of_match}")

            purple_cap_count = 0
            for i in most_wickets['bowler']:
                if i == selected_baller:
                    purple_cap_count += 1
            if purple_cap_count >0:
                st.markdown(f"*Purple cap winner:* {purple_cap_count}  ")

    st.write(
        '--------------------------------------------------------------------------------------------------------------------------')

    st.title('Player Stats in One Season')
    # all players
    combined = pd.concat([merged_df['striker'], merged_df['non_striker']], axis=0)
    combined = sorted(list(set(combined)))
    season = sorted(merged_df['season'].unique())

    col10, col11 = st.columns(2)
    with col10:
        selected_player1 = st.selectbox(f"Select player",options=combined, key='seasonal_anlaysis')
    with col11:
        selected_season = st.selectbox(f"Select Season",options=season)

    col12, col13 = st.columns(2)
    with col12:
        show = st.button('Batting Stats')
        show1 = st.button('Bowling Stats')

    if show:
        # player total matches
        total_match_in_season = merged_df[((merged_df['striker'] == selected_player1) | (merged_df['non_striker'] == selected_player1)) & (
                    merged_df['season'] == selected_season)].groupby(['id'])['id'].nunique().sum()
        # total runs
        total_runs_in_season = merged_df[(merged_df['striker'] == selected_player1) & (merged_df['season'] == selected_season)][
            'runs_off_bat'].sum()
        # highest score
        if merged_df[(merged_df['striker'] == selected_player1)& (merged_df['season'] == selected_season)]['runs_off_bat'].count() > 0:
            highest_score_in_season = merged_df[(merged_df['striker'] == selected_player1) & (merged_df['season'] == selected_season)].groupby('id')[
                'runs_off_bat'].sum().sort_values(ascending=False).values[0]
        # batting average
        total_dissmissals_in_season = len(merged_df[((merged_df['striker'] == selected_player1) & (
                    merged_df['is_wicket_delivery'] == 1)) & (merged_df['season'] == selected_season)].groupby('id')['id'])
        batting_avg_in_season = round((total_runs_in_season / total_dissmissals_in_season), ndigits=2)
        ## Strike rate in season
        total_balls_faced_in_season = len(
            merged_df[((merged_df['striker'] == selected_player1) & (merged_df['season'] == selected_season)) & ~(merged_df['wides'] > 0)])
        strike_rate_in_season = round((total_runs_in_season / total_balls_faced_in_season) * 100, ndigits=2)
        # centuries and half centuries
        temp_centuries = merged_df[(merged_df['striker'] == selected_player1) & (merged_df['season'] == selected_season)].groupby('id')[
            'runs_off_bat'].sum().reset_index()
        # half centuries
        half_centuries_in_season =  temp_centuries[(temp_centuries['runs_off_bat'] < 100) & (temp_centuries['runs_off_bat'] > 49)][
            'runs_off_bat'].count()
        # centuries
        centuries_in_season = temp_centuries[(temp_centuries['runs_off_bat'] > 99)]['runs_off_bat'].count()
        ## boundries count
        fours = len(merged_df[((merged_df['striker'] == selected_player1) & (merged_df['runs_off_bat'] == 4)) & (
                    merged_df['season'] == selected_season)])
        six = len(merged_df[((merged_df['striker'] == selected_player1) & (merged_df['runs_off_bat'] == 6)) & (
                    merged_df['season'] == selected_season)])
        # player of the match
        merged_df['player_of_match'] = merged_df['player_of_match'].fillna(0)
        player_of_match_in_season =  merged_df[(merged_df['player_of_match'] == selected_player1) & (merged_df['season'] == selected_season)].groupby(['id'])[
            'id'].nunique().sum()

        if total_match_in_season > 0:
            st.subheader(f'{selected_player1} Batting Stats for Season {selected_season}')

            # General Statistics Section
            st.subheader('General Statistics')
            st.markdown(f"*Total Matches Played in Season:* {total_match_in_season}")
            st.markdown(f"*Total Runs Scored in Season:* {total_runs_in_season}")
            st.markdown(f"*Highest Score in Season:* {highest_score_in_season}")
            st.markdown(f"*Batting Average in Season:* {batting_avg_in_season}")
            st.markdown(f"*Strike Rate in Season:* {strike_rate_in_season}")

            # Achievements Section
            st.subheader('Achievements in Season')
            st.markdown(f"*Half-Centuries in Season:* {half_centuries_in_season}")
            st.markdown(f"*Centuries in Season:* {centuries_in_season}")
            st.markdown(f"*Fours Hit in Season:* {fours}")
            st.markdown(f"*Sixes Hit in Season:* {six}")
            if player_of_match_in_season > 0:
                st.markdown(f"*Player of the Match Awards in Season:* {player_of_match_in_season}")
        else:
            st.write(f'• Not Batted in {selected_season} season')


    if show1:
        # bowler total maches
        bowler_total_match_in_season =  merged_df[(merged_df['bowler'] == selected_player1) & (merged_df['season'] == selected_season)].groupby(['id'])[
            'id'].nunique().sum()
        # total wickets in seasons
        total_wickets_in_season = merged_df[(merged_df['bowler'] == selected_player1) & (merged_df['season'] == selected_season)][
            'is_wicket_delivery'].sum()
        # bowling average in season
        total_runs_conceded_in_season = merged_df[(merged_df['bowler'] == selected_player1) & (merged_df['season'] == selected_season)][
            'total_run'].sum()
        total_wickets_taken_in_season = merged_df[(merged_df['bowler'] == selected_player1) & (merged_df['season'] == selected_season)][
            'is_wicket_delivery'].sum()
        bowling_avg_in_season = round(total_runs_conceded_in_season / total_wickets_taken_in_season, ndigits=2)
        # economy rate all season
        total_overs_bowled_in_season = len(
            merged_df[(merged_df['bowler'] == selected_player1) & (merged_df['season'] == selected_season)]['ball']) / 6
        economy_in_season = round(total_runs_conceded_in_season / total_overs_bowled_in_season, ndigits=2)
        # most wickets in one match
        temp_wickets_in_season =  merged_df[(merged_df['bowler'] == selected_player1) & (merged_df['season'] == selected_season)].groupby('id')[
            'is_wicket_delivery'].sum().reset_index()
        temp_wickets_in_season = temp_wickets_in_season['is_wicket_delivery'].max()
        # player of the match
        merged_df['player_of_match'] = merged_df['player_of_match'].fillna(0)
        player_of_match_in_season =  merged_df[(merged_df['player_of_match'] == selected_player1) & (merged_df['season'] == selected_season)].groupby(['id'])[
            'id'].nunique().sum()

        if bowler_total_match_in_season > 0:

            st.subheader(f'{selected_player1} Bowling Stats for Season {selected_season}')

            # General Statistics Section
            st.subheader('General Statistics')
            st.markdown(f"*Total Matches Played in Season:* {bowler_total_match_in_season}")
            st.markdown(f"*Total Wickets Taken in Season:* {total_wickets_in_season}")
            if total_wickets_in_season > 0:
                st.markdown(f"*Bowling Average in Season:* {bowling_avg_in_season}")
            if economy_in_season > 0:
                st.markdown(f"*Economy Rate in Season:* {economy_in_season}")

            # Achievements Section
            st.subheader('Achievements in Season')
            if temp_wickets_in_season > 0:
                st.markdown(f"*Most Wickets in One Match:* {temp_wickets_in_season}")
            if player_of_match_in_season > 0:
                st.markdown(f"*Player of the Match Awards in Season:* {player_of_match_in_season}")

        else:
            st.write(f'• Did Not bowl in {selected_season} season')

    st.write('--------------------------------------------------------------------------------------------------------------------------')


    st.title('Player Stats over all season')
    # all players
    combined = pd.concat([merged_df['striker'], merged_df['non_striker']], axis=0)
    combined = sorted(list(set(combined)))
    selected_player = st.selectbox(f"Select player, If you can't find the player's name, try searching by surname", options=combined, key='season_anlaysis')

    col8, col9 = st.columns(2)
    with col8:
        plot6 = st.button('Plot Batting Graph')
    with col9:
        plot7 = st.button('Plot Bowling Graph')

    # batter all season analysis
    runs_scored = merged_df[(merged_df['striker'] == selected_player)].groupby(['season'])['runs_off_bat'].sum().reset_index(
        name='Runs scored')
    dissmissed = merged_df[(merged_df['striker'] == selected_player) & (merged_df['player_dismissed'] == selected_player)].groupby(['season'])[
        'id'].count().reset_index(name='dissmissed')
    ball_faced = merged_df[(merged_df['striker'] == selected_player)& ~(merged_df['wides'] > 0)].groupby(['season'])['ball'].count().reset_index(
        name='ball_faced')
    temp_seasonal_performance = (
        runs_scored.merge(dissmissed, on='season', how='outer').merge(ball_faced, on='season', how='outer')).fillna(0)
    temp_seasonal_performance['Batting average'] = round(
        temp_seasonal_performance['Runs scored'] / temp_seasonal_performance['dissmissed'], ndigits=2)
    temp_seasonal_performance['Batting average'] = temp_seasonal_performance.apply(
        lambda row: row['Runs scored'] if np.isinf(row['Batting average']) else row['Batting average'], axis=1)
    temp_seasonal_performance['strike rate'] = round(
        (temp_seasonal_performance['Runs scored'] / temp_seasonal_performance['ball_faced']) * 100, ndigits=2)
    temp_seasonal_performance['season'] = temp_seasonal_performance['season'].astype(int)

    # bowler all season analysis

    wickets_take = merged_df[(merged_df['bowler'] == selected_player)].groupby(['season'])[
        'is_wicket_delivery'].sum().reset_index(name='Wickets taken')
    runs_conceded = merged_df[(merged_df['bowler'] == selected_player)].groupby(['season'])[
        'total_run'].sum().reset_index(name='Runs conceded')
    overs = (merged_df[(merged_df['bowler'] == selected_player)].groupby(['season'])[
                 'ball'].count() / 6).reset_index(name='Overs')
    temp2 = wickets_take.merge(runs_conceded, on='season', how='outer').merge(overs, on='season', how='outer')
    temp2['Bowling average'] = round(temp2['Runs conceded'] / temp2['Wickets taken'], ndigits=2)
    temp2['Economy rate'] = round(temp2['Runs conceded'] / temp2['Overs'], ndigits=2)


    if plot6:
        a = merged_df[(merged_df['striker'] == selected_player)]
        if a.empty:
            st.write('Not Batted')
        else:
            meaning = (f'Batting average :- The average number of runs scored by the batsman per dissmissal.<br>'
                        f'Strike rate :- The number of runs scored by the batsman per 100 balls.')
            st.write(meaning, unsafe_allow_html=True)
            # Create a line plot using Plotly Express
            fig = px.line(temp_seasonal_performance, x='season', y=['Runs scored', 'Batting average', 'strike rate'],
                          labels={'season': 'Season', 'value': 'value', 'variable': 'Metric'},
                          title=f"Season-wise Batting Performance of a {selected_player} ({temp_seasonal_performance['season'].min()}-{temp_seasonal_performance['season'].max()})",
                          width=800, height=600)

            # Customize the plot appearance
            fig.update_traces(mode='markers+lines', marker=dict(size=8),
                              line=dict(width=2))
            fig.update_layout(title_font_size=20, title_font_family='Arial',
                              legend=dict(title_font_size=14),
                              xaxis=dict(title_font_size=14, tickangle=45, dtick=1),
                              yaxis=dict(title_font_size=14),
                              plot_bgcolor='whitesmoke')

            # Display the plot
            st.plotly_chart(fig)

    if plot7:
        a = merged_df[(merged_df['bowler'] == selected_player)]
        if a.empty:
            st.write('Did Not Bowl')
        else:
            meaning1 = (f'Bowling average :- How many runs a bowler gives up on average to take one wicket.<br>'
                     f'Economy rate :- How many runs a bowler concedes on average per over')
            st.write(meaning1, unsafe_allow_html=True)
            # Create a line plot using Plotly Express
            fig1 = px.line(temp2, x='season', y=['Wickets taken', 'Bowling average', 'Economy rate'],
                          labels={'season': 'Season', 'value': 'value', 'variable': 'Metric'},
                          title=f"Season-wise Bowling Performance of a {selected_player} ({temp_seasonal_performance['season'].min()}-{temp_seasonal_performance['season'].max()})",
                          width=800, height=600)

            # Customize the plot appearance
            fig1.update_traces(mode='markers+lines', marker=dict(size=8),
                              line=dict(width=2))
            fig1.update_layout(title_font_size=20, title_font_family='Arial',
                              legend=dict(title_font_size=14),
                              xaxis=dict(title_font_size=14, tickangle=45, dtick=1),
                              yaxis=dict(title_font_size=14),
                              plot_bgcolor='whitesmoke')
            # Display the plot
            st.plotly_chart(fig1)


def show_page_4():
    add_name_to_header('@abhishek_rawool')
    st.subheader('Match Analysis')
    # match analysis

    teams = sorted(merged_df['batting_team'].unique())
    season = sorted(merged_df['season'].unique())
    col14, col15 = st.columns(2)
    with col14:
        selected_team1 = st.selectbox('Select Batting Team', options=teams)
    with col15:
        selected_team2 = st.selectbox('Select Bowling Team', options=sorted(teams,reverse=True))
    col16, col17 = st.columns(2)
    with col16:
        selected_season = st.selectbox('Select Season', options=season)
    with col17:

        date = sorted(merged_df[((merged_df['batting_team'] == selected_team1) & (
                merged_df['bowling_team'] == selected_team2)) & (merged_df['season'] == selected_season)][ 'start_date'].unique())
        selected_date = st.selectbox('Select Date', options=date)

    a = merged_df[(merged_df['batting_team'] == selected_team1) & (merged_df['bowling_team'] == selected_team2)& (merged_df['start_date'] == selected_date)]

    if a.empty:
        st.write('Not played a match')
    else:
        temp = a.copy()
        temp = temp.reset_index()

        runs_per_over = []
        runs = 0
        for i in range(len(temp['ball'])):
            total = runs + temp['total_run'][i]
            runs = total
            if str(temp['ball'][i]).endswith('1') or temp['ball'][i] == temp['ball'].max():
                runs = runs - temp['total_run'][i]
                runs_per_over.append(runs)
                runs = temp['total_run'][i]
        first_over = runs_per_over[0] + runs_per_over[1]
        last_over = runs_per_over[-1] + runs
        runs_per_over.pop(0)
        runs_per_over.pop(0)
        runs_per_over.pop(-1)
        runs_per_over.append(last_over)
        runs_per_over.insert(0, first_over)
        wickets = []
        out = 0
        for i in range(len(temp['ball'])):
            total = out + temp['is_wicket'][i]
            out = total
            if str(temp['ball'][i]).endswith('1') or temp['ball'][i] == temp['ball'].max():
                out = out - temp['is_wicket'][i]
                wickets.append(out)
                out = temp['is_wicket'][i]

        first_over_wicket = wickets[0] + wickets[1]
        last_over_wicket = wickets[-1] + out
        wickets.pop(0)
        wickets.pop(0)
        wickets.pop(-1)
        wickets.append(last_over_wicket)
        wickets.insert(0, first_over_wicket)
        overs = range(1, len(runs_per_over) + 1)  # Over numbers
        data = {'Over': overs, 'Runs': runs_per_over, 'Wickets': wickets}
        match_data = pd.DataFrame(data)
        show = st.button('Click here to see Scoring Comparison')
        if show:
            b = merged_df[
                (merged_df['batting_team'] == selected_team2) & (merged_df['bowling_team'] == selected_team1) & (
                        merged_df['start_date'] == selected_date)]

            if b.empty:
                st.write('Not played a match')
            else:
                temp = b.copy()
                temp = temp.reset_index()

                runs_per_over = []
                runs = 0
                for i in range(len(temp['ball'])):
                    total = runs + temp['total_run'][i]
                    runs = total
                    if str(temp['ball'][i]).endswith('1') or temp['ball'][i] == temp['ball'].max():
                        runs = runs - temp['total_run'][i]
                        runs_per_over.append(runs)
                        runs = temp['total_run'][i]
                first_over = runs_per_over[0] + runs_per_over[1]
                last_over = runs_per_over[-1] + runs
                runs_per_over.pop(0)
                runs_per_over.pop(0)
                runs_per_over.pop(-1)
                runs_per_over.append(last_over)
                runs_per_over.insert(0, first_over)
                wickets = []
                out = 0
                for i in range(len(temp['ball'])):
                    total = out + temp['is_wicket'][i]
                    out = total
                    if str(temp['ball'][i]).endswith('1') or temp['ball'][i] == temp['ball'].max():
                        out = out - temp['is_wicket'][i]
                        wickets.append(out)
                        out = temp['is_wicket'][i]

                first_over_wicket = wickets[0] + wickets[1]
                last_over_wicket = wickets[-1] + out
                wickets.pop(0)
                wickets.pop(0)
                wickets.pop(-1)
                wickets.append(last_over_wicket)
                wickets.insert(0, first_over_wicket)
                overs = range(1, len(runs_per_over) + 1)  # Over numbers
                data = {'Over': overs, 'Runs': runs_per_over, 'Wickets': wickets}
                match_data1 = pd.DataFrame(data)
                match_data['Runs'].cumsum().plot(kind='line')
                match_data1['Runs'].cumsum().plot(kind='line')

                # Create a Plotly figure
                fig = go.Figure()

                # Add trace for cumulative sum from match_data
                fig.add_trace(go.Scatter(x=match_data.index, y=match_data['Runs'].cumsum(), mode='lines',
                                         name=f'{selected_team1}'))

                # Add trace for cumulative sum from match_data1
                fig.add_trace(go.Scatter(x=match_data1.index, y=match_data1['Runs'].cumsum(), mode='lines',
                                         name=f'{selected_team2}'))

                # Update layout
                fig.update_layout(title=F"Runs scored Comparison<br>"
                                        f"Click or hover on line to see info.",
                                  xaxis_title='Index',
                                  yaxis_title='Runs')

                # Show the Plotly figure
                st.plotly_chart(fig)

        match_analysis = st.button('Show analysis')
        if match_analysis:

            match_veune = temp['venue'][0]
            st.info(f"• **Matched played at {match_veune}**")

            match_winner = temp['winner'][0]
            st.info(f"• **Winner of this match is {match_winner}**")

            total_run_by_team = temp['total_run'].sum()

            # Calculate run rate for each over
            match_data['Run Rate'] = match_data['Runs'].cumsum() / match_data['Over']
            # Create the figure
            fig = go.Figure()
            # Add bars for runs per over
            fig.add_trace(go.Bar(
                x=match_data['Over'],
                y=match_data['Runs'],
                name='Runs per Over',
                marker_color='skyblue'
            ))

            # Add scatter plot for wickets fallen
            dot_size = 10  # Adjust dot size
            dot_spacing = 0.7  # Adjust spacing between dots vertically
            for over, wickets in zip(match_data['Over'], match_data['Wickets']):
                for i in range(wickets):  # Add multiple scatter plot points for each wicket in the same over
                    y_position = match_data.loc[
                                     over - 1, 'Runs'] + i * dot_spacing  # Adjust y-position for each additional wicket
                    fig.add_trace(go.Scatter(
                        x=[over],
                        y=[y_position],
                        mode='markers',
                        name=f'Wicket in Over{over}',
                        showlegend=True,  # Exclude scatter plot points from legend
                        marker=dict(
                            color='red',
                            size=dot_size,
                            line=dict(width=2, color='DarkSlateGrey')
                        )
                    ))

            # Add run rate line plot
            fig.add_trace(go.Scatter(
                x=match_data['Over'],
                y=match_data['Run Rate'],
                mode='lines',
                name='Run Rate',
                line=dict(color='green', width=2)
            ))

            # Update layout
            fig.update_layout(
                title=f'{selected_team1} Runs per Over with Wickets Fallen and Run Rate<br>'
                      f'• Total runs {total_run_by_team}',
                xaxis_title='Over',
                yaxis_title='Runs',
                xaxis=dict(tickmode='linear', dtick=1),
                showlegend=True
            )

            st.plotly_chart(fig)

            a = 0
            partnership = []
            first_inning_last_ball = temp[(temp['innings'] == 1)]['ball'].max()
            second_inning_last_ball = temp[(temp['innings'] == 2)]['ball'].max()

            for i in range(len(temp)):
                total = a + temp['total_run'][i]
                a = total

                if (temp['is_wicket'][i] == 1) or (
                        temp['ball'][i] == first_inning_last_ball or temp['ball'][i] == second_inning_last_ball):
                    partnership.append(temp['striker'][i] + ' and ' + temp['non_striker'][i] + ' : ' + str(a))
                    a = 0

            partnership_data = {}
            for i in partnership:
                name, runs = i.split(':')
                partnership_data[name.strip()] = int(runs)

            # Sample data (replace these with your actual data)
            partnerships = list(partnership_data.keys())
            partnership_runs = list(partnership_data.values())

            # Create Funnel chart
            fig1 = go.Figure(go.Funnel(
                y=partnerships,
                x=partnership_runs,

            ))

            # Update layout
            fig1.update_layout(
                title='Partnership Runs',
                funnelmode="stack",
                yaxis_title='Partnerships',
                xaxis_title='Runs',
                showlegend=False, width=550, height=400
            )

            st.plotly_chart(fig1)

            # Perform the batting statistics analysis
            player_runs = temp.copy()

            run_score = player_runs.groupby(['striker'], sort=False)['runs_off_bat'].sum().reset_index(
                name='Runs Scored')
            run_score['length'] = np.arange(len(run_score))
            ball_face = player_runs[~(player_runs['wides'] > 0)].groupby(['striker'], sort=False)['ball'].count().reset_index(
                name='Balls Faced')
            fours = player_runs[(player_runs['runs_off_bat'] == 4)].groupby(['striker'], sort=False)[
                'runs_off_bat'].count().reset_index(name='Fours')
            sixes = player_runs[(player_runs['runs_off_bat'] == 6)].groupby(['striker'], sort=False)[
                'runs_off_bat'].count().reset_index(name='Sixes')
            team_bat_stat = (
                run_score.merge(ball_face, on='striker', how='outer').merge(fours, on='striker', how='outer').merge(
                    sixes, on='striker', how='outer')).fillna('0')
            team_bat_stat['Fours'] = team_bat_stat['Fours'].astype(int)
            team_bat_stat['Sixes'] = team_bat_stat['Sixes'].astype(int)
            team_bat_stat['Strike rate'] = round((team_bat_stat['Runs Scored'] / team_bat_stat['Balls Faced']) * 100,
                                                 ndigits=2)
            team_bat_stat = team_bat_stat.rename(columns={'striker': 'Batting'})
            out_by = (player_runs[player_runs['is_wicket'] == 1].groupby(['striker'], sort=False)['bowler'].first().reset_index(
                name='Out by'))
            team_bat_stat = (team_bat_stat.merge(out_by, left_on='Batting', right_on='striker', how='outer').drop(
                columns='striker')).fillna('Not Out')
            wicket_type = player_runs.groupby(['striker'], sort=False)['wicket_type'].first().reset_index(name='Wicket Type')
            team_bat_stat = (team_bat_stat.merge(wicket_type, left_on='Batting', right_on='striker', how='outer').drop(
                columns='striker')).fillna('Not Out')

            # Display overall batting statistics table
            st.subheader(f'{selected_team1} Batting')
            team_bat_stat = team_bat_stat.sort_values(by='length')
            team_bat_stat = team_bat_stat.drop(columns=['length'])
            st.dataframe(team_bat_stat)


            # Perform the bowling statistics analysis
            wicket_take = player_runs.groupby(['bowler'], sort=False)['is_wicket_delivery'].sum().reset_index(
                name='Wickets')
            balls = (player_runs[~((player_runs['wides'] > 0) | (player_runs['noballs'] > 0))]
                     .groupby(['bowler'], sort=False)['ball'].count().reset_index(name='Balls'))
            runs_conce = player_runs.groupby(['bowler'])['total_run'].sum().reset_index(name='Runs conceded')
            team_bowl_stat = balls.merge(wicket_take, on='bowler', how='outer').merge(runs_conce, on='bowler',
                                                                                      how='outer')
            team_bowl_stat['Economy rate'] = round(team_bowl_stat['Runs conceded'] / (team_bowl_stat['Balls'] / 6),
                                                   ndigits=2)


            # Display overall bowling statistics table
            st.subheader(f'{selected_team2} Bowling')
            st.dataframe(team_bowl_stat)






def show_page_5():
    add_name_to_header('@abhishek_rawool')
    st.header('Team Analysis')
    team = merged_df['batting_team'].unique()
    selected_team = st.selectbox('Select a Team', options=team)
    show = st.button('Show')
    if show:

        # total runs
        total_runs = merged_df[(merged_df['batting_team'] == selected_team)]['total_run'].sum()
        st.info(f"Total runs scored by {selected_team}: {total_runs}")

        # total matches
        total_matches = merged_df[(merged_df['batting_team'] == selected_team) | (merged_df['bowling_team'] == selected_team)][
            'id'].nunique()
        st.markdown(f"**Total Matches played by {selected_team}: {total_matches}**")

        col1, col2= st.columns(2)
        with col1:
            # win matches
            win_match = merged_df[((merged_df['batting_team'] == selected_team) | (merged_df['bowling_team'] == selected_team)) & (
                            merged_df['winner'] == selected_team)]['id'].nunique()
            st.info(f"•Win Matches: {win_match}")
        with col2:
            # loss matches
            loss_match = merged_df[((merged_df['batting_team'] == selected_team) | (merged_df['bowling_team'] == selected_team)) & (
                            merged_df['winner'] != selected_team)]['id'].nunique()
            st.info(f"•Loss Matches: {loss_match}")

        bat = ( merged_df[
            (merged_df['toss_winner'] == selected_team) & (merged_df['toss_decision'] == 'bat') & (
                        merged_df['winner'] == selected_team)].groupby('id')['id'].nunique().sum())

        field = (merged_df[
            (merged_df['toss_winner'] == selected_team) & (merged_df['toss_decision'] == 'field') & (
                        merged_df['winner'] == selected_team)].groupby('id')['id'].nunique().sum())
        st.markdown(f"• **Toss win choose to field and match win:- {field}**<br>"
                    f"• **Toss win choose to bat and match win:- {bat}**", unsafe_allow_html=True)


        # fielding
        run_out = len(merged_df[(merged_df['bowling_team'] == selected_team) & (merged_df['wicket_type'] == 'run out')])

        stumped = len(merged_df[(merged_df['bowling_team'] == selected_team) & (merged_df['wicket_type'] == 'stumped')])

        catch = len(merged_df[(merged_df['bowling_team'] == selected_team) & (merged_df['wicket_type'] == 'caught')])
        st.markdown(f"• **Total run out done by {selected_team}: {run_out}**<br>"
                    f"• **Total stumped out done by {selected_team}: {stumped}**<br>"
                    f"• **Total catches taken by {selected_team}: {catch}**", unsafe_allow_html=True)

        # while run chasing and defending
        chasing_wins = merged_df[((merged_df['batting_team'] == selected_team) & (merged_df['innings'] == 2)) & (
                    merged_df['winner'] == selected_team)]['id'].nunique()
        defending_wins = merged_df[((merged_df['bowling_team'] == selected_team) & (merged_df['innings'] == 2)) & (
                    merged_df['winner'] == selected_team)]['id'].nunique()
        chasing_total_match = \
        merged_df[((merged_df['batting_team'] == selected_team) & (merged_df['innings'] == 2))]['id'].nunique()
        defending_total_match = \
        merged_df[((merged_df['bowling_team'] == selected_team) & (merged_df['innings'] == 2))]['id'].nunique()

        a = (chasing_wins / chasing_total_match) * 100
        b = (defending_wins / defending_total_match) * 100


        import plotly.graph_objects as go
        # Create data for the pie chart
        labels = ['Chasing Win Percentage', 'Defending Win Percentage']
        values = [a,b]
        colors = ['#007be','#f5e0d3']
        # Create a Plotly pie chart
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=['#008080','#f5e0d3']))])

        # Update layout and title
        fig.update_layout(title=f'{selected_team} Wins Percentage While Chasing vs Defending')

        # Show the plot
        st.plotly_chart(fig)

        # Filter data for matches won by the selected team
        temp_match_win = merged_df[merged_df['winner'] == selected_team].groupby(['season', 'id'])[
            'winner'].nunique().reset_index()

        # Filter data for matches lost by the selected team
        temp_match_loss = merged_df[((merged_df['batting_team'].str.contains(selected_team)) | (
            merged_df['bowling_team'].str.contains(selected_team))) & (merged_df['winner'] != selected_team)]
        temp_match_loss_team = temp_match_loss.groupby(['season', 'id'])['winner'].nunique().reset_index()

        # Combine win and loss data
        combined_data = pd.concat([temp_match_win['season'].value_counts().sort_index(),
                                   temp_match_loss_team['season'].value_counts().sort_index()], axis=1)
        combined_data.columns = ['Wins', 'Losses']  # Rename columns for clarity

        # Reset index and rename columns
        combined_data = combined_data.reset_index()
        # Melt the DataFrame to have 'Wins' and 'Losses' as values under a new 'Match Outcome' column
        melted_data = combined_data.melt(id_vars='season', var_name='Match Outcome', value_name='Count')

        # Create a Plotly bar chart
        fig1 = px.bar(melted_data, x='season', y='Count', color='Match Outcome', barmode='group',
                      labels={'Count': 'Number of Matches', 'Season': 'Season', 'Match Outcome': 'Match Outcome'},
                      title=f'{selected_team} Win-Loss Distribution by Season')
        # Update x-axis tick labels
        fig1.update_xaxes(tickangle=-40, tickmode='linear', tickvals=melted_data['season'],
                          ticktext=melted_data['season'])
        # Show the plot
        st.plotly_chart(fig1)

    st.write(
        '--------------------------------------------------------------------------------------------------------------------------')

    st.header("Overall Analysis")

    # Assuming you have already calculated temp_runs dataframe
    temp_runs = merged_df[(merged_df['batting_team'].isin(['Royal Challengers Bangalore',
                                                           'Punjab Kings', 'Mumbai Indians',
                                                           'Kolkata Knight Riders', 'Rajasthan Royals',
                                                           'Chennai Super Kings', 'Sunrisers Hyderabad',
                                                           'Delhi Capitals',
                                                           'Lucknow Super Giants', 'Gujarat Titans']))].groupby(
        ['batting_team'])['total_run'].sum().reset_index(name='Runs')

    # Define the colors for each team
    colors = {
        'Chennai Super Kings': 'yellow',
        'Delhi Capitals': 'dodgerblue',
        'Gujarat Titans': 'navy',
        'Kolkata Knight Riders': 'purple',
        'Lucknow Super Giants': 'lightblue',
        'Mumbai Indians': 'blue',
        'Punjab Kings': 'lightcoral',
        'Rajasthan Royals': 'pink',
        'Royal Challengers Bangalore': 'red',
        'Sunrisers Hyderabad': 'orange'
    }

    plt.rcParams['figure.figsize'] = (12, 6)
    ax = temp_runs.plot(kind='bar', x='batting_team', y='Runs', color=temp_runs['batting_team'].map(colors))
    plt.title('Total Runs scored by each Team in IPL')
    plt.xlabel('Teams', weight='bold', size=12)
    xticks_labels = ['\n'.join(label.split()) for label in temp_runs['batting_team'].values]
    ax.set_xticklabels(xticks_labels, rotation=0, ha='center')
    st.pyplot(plt)


    # Assuming you have already calculated temp_runs dataframe
    temp_wickets = merged_df[(merged_df['bowling_team'].isin(['Royal Challengers Bangalore',
                                                              'Punjab Kings', 'Mumbai Indians', 'Kolkata Knight Riders',
                                                              'Rajasthan Royals',
                                                              'Chennai Super Kings', 'Sunrisers Hyderabad',
                                                              'Delhi Capitals',
                                                              'Lucknow Super Giants', 'Gujarat Titans']))].groupby(
        ['bowling_team'])['is_wicket'].sum().reset_index(name='Wickets')

    # Define the colors for each team
    colors1 = {
        'Chennai Super Kings': 'yellow',
        'Delhi Capitals': 'dodgerblue',
        'Gujarat Titans': 'navy',
        'Kolkata Knight Riders': 'purple',
        'Lucknow Super Giants': 'lightblue',
        'Mumbai Indians': 'blue',
        'Punjab Kings': 'lightcoral',
        'Rajasthan Royals': 'pink',
        'Royal Challengers Bangalore': 'red',
        'Sunrisers Hyderabad': 'orange'
    }

    plt.rcParams['figure.figsize'] = (12, 6)
    ax = temp_wickets.plot(kind='bar', x='bowling_team', y='Wickets', color=temp_wickets['bowling_team'].map(colors1))
    plt.title('Total Wickets taken by each Team in IPL')
    plt.xlabel('Teams', weight='bold', size=12)
    xticks_labels = ['\n'.join(label.split()) for label in temp_wickets['bowling_team'].values]
    ax.set_xticklabels(xticks_labels, rotation=0, ha='center')
    st.pyplot(plt)

    # boundary distribution all seasons

    temp_boundries = merged_df[
        (merged_df['runs_off_bat'].isin([4, 6])) & (merged_df['batting_team'].isin(['Royal Challengers Bangalore',
                                                                                    'Punjab Kings',
                                                                                    'Mumbai Indians',
                                                                                    'Kolkata Knight Riders',
                                                                                    'Rajasthan Royals',
                                                                                    'Chennai Super Kings',
                                                                                    'Sunrisers Hyderabad',
                                                                                    'Delhi Capitals',
                                                                                    'Lucknow Super Giants',
                                                                                    'Gujarat Titans']))].groupby(
        ['batting_team'])['runs_off_bat'].value_counts().unstack()

    plt.rcParams['figure.figsize'] = (12, 6)
    ax = temp_boundries.plot(kind='bar')
    plt.title('Total Four and Six Hit by each Team')
    plt.xlabel('Teams', weight='bold', size=12)
    plt.legend(title='Boundries')
    xticks_labels = ['\n'.join(label.split()) for label in temp_boundries.index]
    ax.set_xticklabels(xticks_labels, rotation=0, ha='center')
    st.pyplot(plt)



def show_page_6():
    add_name_to_header('@abhishek_rawool')
    st.subheader('Player vs Player')
    st.write("**• If you can't find the player's name, try searching by surname.**")

    combined = pd.concat([merged_df['striker'], merged_df['non_striker']], axis=0)
    players = sorted(list(set(combined)))

    col18, col19 = st.columns(2)
    with col18:
        selected_batsman = st.selectbox('Select a Batsman', options=players)
    with col19:
        selected_bowler = st.selectbox('Select a bowler', options=players)

    p_v_p_but = st.button('Show')
    if p_v_p_but:
        a = merged_df[(merged_df['striker'] == selected_batsman) & (merged_df['bowler'] == selected_bowler)]
        if a.empty:
            st.write("Not played against each other.")
        else:
            # player vs player
            p_v_p = merged_df[(merged_df['striker'] == selected_batsman) & (merged_df['bowler'] == selected_bowler)]
            total_matches = p_v_p['id'].nunique()
            total_runs_scored = p_v_p['runs_off_bat'].sum()
            balls_faced = len(p_v_p)
            strike_rate = round((total_runs_scored / balls_faced) * 100, ndigits=2)
            strike_r = None
            if strike_rate > 0:
                strike_r = strike_rate
            else:
                strike_r = 0
            dismissals_count = p_v_p['is_wicket_delivery'].sum()
            dismissals_type = p_v_p['wicket_type'].value_counts()
            overs_bowled = len(p_v_p) / 6
            economy = round(total_runs_scored / overs_bowled, ndigits=2)
            economy_r = None
            if economy > 0:
                economy_r = economy
            else:
                economy_r = 0
            boundaries_count = p_v_p[p_v_p['runs_off_bat'].isin([4, 6])].shape[0]

            data = {'Index': ['Total Matches', 'Total Runs Scored', 'Balls Faced', 'Strike Rate', 'Dismissals Count',
                            'Economy rate', 'Boundaries and Six Count'],
                  "values": [total_matches, total_runs_scored, balls_faced, strike_r, dismissals_count, economy_r,  boundaries_count]}

            p_data = pd.DataFrame(data)
            # Format 'Strike Rate' and 'Economy Rate' columns as percentages
            p_data.loc[p_data['Index'] == 'Strike Rate', 'values'] = p_data.loc[p_data['Index'] == 'Strike Rate', 'values'].astype(float)
            p_data.loc[p_data['Index'] == 'Economy rate', 'values'] = p_data.loc[p_data['Index'] == 'Economy rate', 'values'].astype(float)
            p_data.loc[p_data['Index'].isin(['Total Matches', 'Total Runs Scored', 'Boundaries and Six Count', 'Dismissals Count',
                                     'Balls Faced']), 'values'] = p_data.loc[p_data['Index'].isin(
                ['Total Matches', 'Total Runs Scored', 'Boundaries and Six Count', 'Dismissals Count',
                 'Balls Faced']), 'values'].astype(str).str.split('.').str[0]
            p_data = p_data.set_index('Index')

            st.write(f"{selected_batsman} Vs {selected_bowler}")

            col20, col21 = st.columns(2)
            with col20:
                st.dataframe(p_data, width=250)
            with col21:
                wicket_type = p_v_p['wicket_type'].value_counts().reset_index()
                wicket_type = wicket_type.rename(columns={'wicket_type': 'Dissmissal type'})
                wicket_type = wicket_type.set_index('Dissmissal type')
                if wicket_type.empty:
                    pass
                else:
                    st.dataframe(wicket_type, width=250)

            p_season_runs = merged_df[(merged_df['striker'] == selected_batsman) & (merged_df['bowler'] == selected_bowler)].groupby(['season'])[
                'runs_off_bat'].sum().reset_index(name='Runs')
            p_season_ball = merged_df[((merged_df['striker'] == selected_batsman) & (merged_df['bowler'] == selected_bowler)) & ~(
                        merged_df['wides'] > 0)].groupby(['season'])['ball'].count().reset_index(name='Balls')
            p_overs = (merged_df[((merged_df['striker'] == selected_batsman) & (merged_df['bowler'] == selected_bowler)) & ~(
                        merged_df['wides'] > 0)].groupby(['season'])['ball'].count() / 6).reset_index(name='Overs')
            p_v_p_season = (p_season_runs.merge(p_season_ball, on='season', how='outer').merge(p_overs, on='season',
                                                                                               how='outer')).fillna(0)
            p_v_p_season['Strike Rate'] = round((p_v_p_season['Runs'] / p_v_p_season['Balls']) * 100, ndigits=2)

            import plotly.graph_objects as go

            # Create a line plot using Plotly
            fig = go.Figure()

            # Add a line trace to the figure
            fig.add_trace(go.Scatter(x=p_v_p_season['season'], y=p_v_p_season['Strike Rate'],
                                     mode='lines+markers', name='Strike Rate'))

            # Update the layout of the graph
            fig.update_layout(title=f'{selected_batsman} Seasonal Strike Rate against {selected_bowler}',
                              xaxis_title='Season',
                              yaxis_title='Strike Rate',
                              plot_bgcolor='rgba(0,0,0,0)',
                              xaxis=dict(showgrid=False),
                              yaxis=dict(showgrid=False),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                              )

            # Display the plot
            st.plotly_chart(fig)

    st.subheader('Team vs Team')

    Team1=  ['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore', 'Kolkata Knight Riders','Lucknow Super Giants',
    'Punjab Kings','Chennai Super Kings','Rajasthan Royals','Delhi Capitals', 'Gujarat Titans']
    Team1.sort(reverse=True)
    Team2 =  ['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore', 'Kolkata Knight Riders','Lucknow Super Giants',
    'Punjab Kings','Chennai Super Kings','Rajasthan Royals','Delhi Capitals', 'Gujarat Titans']


    col22, col23 = st.columns(2)
    with col22:
        selected_team1 = st.selectbox('Select Team1', options=Team1)
    with col23:
        selected_team2 = st.selectbox('Select Team2', options=sorted(Team2))

    t_v_t = st.button('Plot')
    if t_v_t:
        # head to head match wins
        temp_head_to_head = merged_df[(merged_df['batting_team'].isin([selected_team1, selected_team2])) & (
            merged_df['bowling_team'].isin([selected_team2, selected_team1]))]
        matches_bet_two_team = temp_head_to_head.groupby('id')['id'].nunique().sum()
        matches_won_by_one_team = temp_head_to_head[temp_head_to_head['winner'] == selected_team1].groupby('id')[
            'id'].nunique().sum()
        matches_won_by_second_team = \
        temp_head_to_head[temp_head_to_head['winner'] == selected_team2].groupby('id')['id'].nunique().sum()
        head_to_head_one_win_percentage = round((matches_won_by_one_team / matches_bet_two_team) * 100, ndigits=2)
        head_to_head_second_win_percentage = round((matches_won_by_second_team / matches_bet_two_team) * 100, ndigits=1)

        st.subheader(f'{selected_team1} vs {selected_team2}')
        st.markdown(f"**Total matches played between {selected_team1} and {selected_team2}: {matches_bet_two_team}**")

        team_one_low = merged_df[(merged_df['batting_team'] == selected_team1) & (
                    merged_df['bowling_team'] == selected_team2)].groupby(['id'])['total_run'].sum().min()
        team_one_high = merged_df[(merged_df['batting_team'] == selected_team1) & (
                    merged_df['bowling_team'] == selected_team2)].groupby(['id'])['total_run'].sum().max()

        team_two_low = merged_df[(merged_df['batting_team'] == selected_team2) & (
                merged_df['bowling_team'] == selected_team1)].groupby(['id'])['total_run'].sum().min()
        team_two_high = merged_df[(merged_df['batting_team'] == selected_team2) & (
                merged_df['bowling_team'] == selected_team1)].groupby(['id'])['total_run'].sum().max()

        st.markdown(f"**• {selected_team1} Highest score against {selected_team2} is {team_one_high} and Lowest score is {team_one_low}**<br>"
                    f"**• {selected_team2} Highest score against {selected_team1} is {team_two_high} and Lowest score is {team_two_low}**", unsafe_allow_html=True)

        import plotly.graph_objects as go
        # Data for the bar plot
        x_values = [selected_team1, selected_team2]
        y_values = [matches_won_by_one_team, matches_won_by_second_team]

        # Create the bar trace
        bar_trace = go.Bar(x=x_values, y=y_values, marker=dict(color=['#4bdb35', '#f77707']), width=0.4)

        # Create layout for the graph
        layout = go.Layout(title=f'Match wins by {selected_team1} and {selected_team2}', xaxis=dict(title='Teams', tickfont=dict(size=14)),
                           yaxis=dict(title='Win Matches count'), plot_bgcolor='lightgrey', width=500)

        # Create the figure and add the bar trace to it
        fig = go.Figure(data=[bar_trace], layout=layout)

        # Display the graph
        st.plotly_chart(fig)

        # Create a DataFrame for pie chart data
        pie_data = pd.DataFrame({
            'Team': [selected_team1, selected_team2],
            'Win Percentage': [head_to_head_one_win_percentage, head_to_head_second_win_percentage]
        })
        # Create a pie chart using Plotly
        fig = px.pie(pie_data, values='Win Percentage', names='Team', title='Head-to-Head Win Percentage')
        fig.update_traces(textfont_size=15)
        st.plotly_chart(fig)

        win_by_venue = temp_head_to_head.groupby(['venue', 'id'])['winner'].first().reset_index(name='winner team')
        one_team = win_by_venue[win_by_venue['winner team'] == selected_team1].groupby(['venue'])[
            'winner team'].value_counts().reset_index(name='counts')
        second_team = win_by_venue[win_by_venue['winner team'] == selected_team2].groupby(['venue'])[
            'winner team'].value_counts().reset_index(name='counts')
        final = one_team.merge(second_team, on='venue', how='outer').fillna(0)

        # Sample data (replace this with your actual data)
        venues = final['venue']
        team_a_wins = final['counts_x']  # Number of wins for Team A at each venue
        team_b_wins = final['counts_y']  # Number of wins for Team B at each venue

        # Create traces for Team A and Team B
        trace_a = go.Bar(y=venues, x=team_a_wins, orientation='h', name=f'{selected_team1} Wins', text=team_a_wins,
                         textposition='inside', insidetextanchor='middle', hoverinfo='x', marker=dict(color='#315cf7'))
        trace_b = go.Bar(y=venues, x=team_b_wins, orientation='h', name=f'{selected_team2} Wins', text=team_b_wins,
                         textposition='inside', insidetextanchor='middle', hoverinfo='x', marker=dict(color='#4ef259'))

        # Create layout
        layout = go.Layout(title=f'{selected_team1} vs. {selected_team2} Wins by Venue in IPL Matches', barmode='stack',
                           yaxis=dict(title='Venues'),
                           xaxis=dict(title='Number of Wins'), hovermode='closest', width=800  )

        # Create figure
        fig1 = go.Figure(data=[trace_a, trace_b], layout=layout)

        # Show the plot
        st.plotly_chart(fig1)

        fours_powerplay = len(temp_head_to_head[(temp_head_to_head['ball'] < 6) & (
                    (temp_head_to_head['batting_team'] == selected_team1) & (
                        temp_head_to_head['runs_off_bat'] == 4))])
        six_powerplay = len(temp_head_to_head[(temp_head_to_head['ball'] < 6) & (
                    (temp_head_to_head['batting_team'] == selected_team1) & (
                        temp_head_to_head['runs_off_bat'] == 6))])
        st.info(f"• {selected_team1} Hits {fours_powerplay} fours and {six_powerplay} sixes in powerplay against {selected_team2}")

        fours_powerplay = len(temp_head_to_head[(temp_head_to_head['ball'] < 6) & (
                (temp_head_to_head['batting_team'] == selected_team2) & (
                temp_head_to_head['runs_off_bat'] == 4))])
        six_powerplay = len(temp_head_to_head[(temp_head_to_head['ball'] < 6) & (
                (temp_head_to_head['batting_team'] == selected_team2) & (
                temp_head_to_head['runs_off_bat'] == 6))])
        st.info(f"• {selected_team2} Hits {fours_powerplay} fours and {six_powerplay} sixes in powerplay against {selected_team1}")

        team_one_runs_powerplay = temp_head_to_head[
            (temp_head_to_head['ball'] < 6) & (temp_head_to_head['batting_team'] == selected_team1)].groupby(['id'])[
            'runs_off_bat'].sum().reset_index(name='team1 runs')
        team_two_runs_powerplay = temp_head_to_head[
            (temp_head_to_head['ball'] < 6) & (temp_head_to_head['batting_team'] == selected_team2)].groupby(
            ['id'])['runs_off_bat'].sum().reset_index(name='team2 runs')

        team_two_wickets_taken = temp_head_to_head[
            (temp_head_to_head['ball'] < 6) & (temp_head_to_head['batting_team'] == selected_team1)].groupby(['id'])[
            'is_wicket'].sum().reset_index(name='team1 wickets')
        team_one_wickets_taken = temp_head_to_head[
            (temp_head_to_head['ball'] < 6) & (temp_head_to_head['batting_team'] == selected_team2)].groupby(
            ['id'])['is_wicket'].sum().reset_index(name='team2 wickets')


        wickets_taken_team1 = team_one_wickets_taken['team2 wickets']
        wickets_taken_team2 = team_two_wickets_taken['team1 wickets']

        match_numbers = np.arange(len(team_one_runs_powerplay))  # Match numbers
        run_rates_team1 = round(team_one_runs_powerplay['team1 runs'] / 6, ndigits=2)  # Run rates for Team 1
        run_rates_team2 = round(team_two_runs_powerplay['team2 runs'] / 6, ndigits=2)  # Run rates for Team 2

        # Create traces for run rates of each team
        trace_team1_run_rate = go.Scatter(x=match_numbers, y=run_rates_team1, mode='lines+markers',
                                          name=f'{selected_team1} - Run Rate')
        trace_team2_run_rate = go.Scatter(x=match_numbers, y=run_rates_team2, mode='lines+markers',
                                          name=f'{selected_team2} - Run Rate')

        # Create traces for wickets taken by each team
        trace_team1_wickets = go.Bar(x=match_numbers, y=wickets_taken_team1, name=f'{selected_team1} - Wickets')
        trace_team2_wickets = go.Bar(x=match_numbers, y=wickets_taken_team2, name=f'{selected_team2} - Wickets')

        # Create layout for the graph
        layout = go.Layout(title='Run Rate and Wickets Taken During Powerplay',
                           xaxis=dict(title='Match Number'),
                           yaxis=dict(title='Run Rate', side='left', rangemode='tozero'),
                           yaxis2=dict(title='Wickets Taken', overlaying='y', side='right'),
                           plot_bgcolor='lightgrey', showlegend=True, width=900, height=550)

        # Create the figure and add traces to it
        fig2 = go.Figure(data=[trace_team1_run_rate, trace_team2_run_rate, trace_team1_wickets, trace_team2_wickets],
                        layout=layout)

        # Display the graph
        st.plotly_chart(fig2)

        fours_death = len(temp_head_to_head[(temp_head_to_head['ball'] > 15) & (
                (temp_head_to_head['batting_team'] == selected_team1) & (
                temp_head_to_head['runs_off_bat'] == 4))])
        six_death = len(temp_head_to_head[(temp_head_to_head['ball'] > 15) & (
                (temp_head_to_head['batting_team'] == selected_team1) & (
                temp_head_to_head['runs_off_bat'] == 6))])
        st.info(f"• {selected_team1} Hits {fours_death} fours and {six_death} sixes in death overs against {selected_team2}")

        fours_death = len(temp_head_to_head[(temp_head_to_head['ball'] > 15) & (
                (temp_head_to_head['batting_team'] == selected_team2) & (
                temp_head_to_head['runs_off_bat'] == 4))])
        six_death = len(temp_head_to_head[(temp_head_to_head['ball'] > 15) & (
                (temp_head_to_head['batting_team'] == selected_team2) & (
                temp_head_to_head['runs_off_bat'] == 6))])
        st.info(f"• {selected_team2} Hits {fours_death} fours and {six_death} sixes in death overs against {selected_team1}")

        team_one_runs_death = temp_head_to_head[
            (temp_head_to_head['ball'] > 15) & (temp_head_to_head['batting_team'] == selected_team1)].groupby(['id'])[
            'runs_off_bat'].sum().reset_index(name='team1 runs')
        team_two_runs_death = temp_head_to_head[
            (temp_head_to_head['ball'] > 15) & (temp_head_to_head['batting_team'] == selected_team2)].groupby(
            ['id'])['runs_off_bat'].sum().reset_index(name='team2 runs')

        team_two_wickets_death = temp_head_to_head[
            (temp_head_to_head['ball'] > 15) & (temp_head_to_head['batting_team'] == selected_team1)].groupby(['id'])[
            'is_wicket'].sum().reset_index(name='team1 wickets')
        team_one_wickets_death = temp_head_to_head[
            (temp_head_to_head['ball'] > 15) & (temp_head_to_head['batting_team'] == selected_team2)].groupby(
            ['id'])['is_wicket'].sum().reset_index(name='team2 wickets')

        death_over_run_wicket = team_one_runs_death.merge(team_two_runs_death, on='id', how='outer').merge(
            team_two_wickets_death, on='id', how='outer').merge(team_one_wickets_death, on='id', how='outer').fillna(0)

        wickets_taken_team1 = death_over_run_wicket['team1 wickets']
        wickets_taken_team2 = death_over_run_wicket['team2 wickets']

        match_numbers = np.arange(len(death_over_run_wicket))  # Match numbers or powerplay phases
        run_rates_team1 = death_over_run_wicket['team1 runs'] / 5  # Run rates for Team 1 in each phase
        run_rates_team2 = death_over_run_wicket['team2 runs'] / 5  # Run rates for Team 2 in each phase

        # Create traces for run rates of each team
        trace_team1_run_rate = go.Scatter(x=match_numbers, y=run_rates_team1, mode='lines+markers',
                                          name=f'{selected_team1} - Run Rate', line=dict(color='yellow'))
        trace_team2_run_rate = go.Scatter(x=match_numbers, y=run_rates_team2, mode='lines+markers',
                                          name=f'{selected_team2} - Run Rate', line=dict(color='red'))
        # Create traces for wickets taken by each team
        trace_team1_wickets = go.Bar(x=match_numbers, y=wickets_taken_team1, name=f'{selected_team1} - Wickets',
                                     marker=dict(color='#015485'))
        trace_team2_wickets = go.Bar(x=match_numbers, y=wickets_taken_team2, name=f'{selected_team2} - Wickets',
                                     marker=dict(color='#07ab22'))

        # Create layout for the graph
        layout = go.Layout(title='Run Rate and Wickets Taken During Death overs',
                           xaxis=dict(title='Match Number'),
                           yaxis=dict(title='Run Rate', side='left', rangemode='tozero'),
                           yaxis2=dict(title='Wickets Taken', overlaying='y', side='right'),
                           plot_bgcolor='lightgrey', showlegend=True, width=950, height=550)

        # Create the figure and add traces to it
        fig3 = go.Figure(data=[trace_team1_run_rate, trace_team2_run_rate, trace_team1_wickets, trace_team2_wickets],
                        layout=layout)

        # Display the graph
        st.plotly_chart(fig3)




def show_page_7():
    add_name_to_header('@abhishek_rawool')
    st.title('Seasonal Stats')
    season = merged_df['season'].unique()
    selected_season = st.selectbox('Select Season', options=season)
    show = st.button('Show')

    if show:
        # total season
        col0, col01 = st.columns(2)
        with col0:
            st.info(f"• **Season: {selected_season}**")

            # total matches
            total_matches = merged_df[merged_df['season']==selected_season]['id'].nunique()
            st.info(f"• **Total Matches played in IPL {selected_season}: {total_matches}**")

            # total balls
            total_balls = len(merged_df[merged_df['season']==selected_season])
            st.info(f"• **Total Balls bowl in IPL {selected_season}: {total_balls}**")

            # total runs
            total_runs = merged_df[merged_df['season']==selected_season]['total_run'].sum()
            st.info(f"• **Total Runs scored in IPL {selected_season}: {total_runs}**")

            # total wickets
            total_wickets = len(merged_df[(merged_df['is_wicket'] == 1) & (merged_df['season'] == selected_season)])
            st.info(f"• **Total Wickets taken in IPL {selected_season}: {total_wickets}**")

        with col01:
            # total 50s
            fifties = merged_df[merged_df['season'] == selected_season].groupby(['id', 'striker'])['runs_off_bat'].sum().reset_index()
            fifties = fifties[(fifties['runs_off_bat'] > 49) & (fifties['runs_off_bat'] < 100)]['striker'].count()
            st.info(f"• **Total Fifties score in IPL {selected_season}: {fifties}**")

            # total 100s
            hundreds = merged_df[merged_df['season'] == selected_season].groupby(['id', 'striker'])['runs_off_bat'].sum().reset_index()
            hundreds = hundreds[(hundreds['runs_off_bat'] > 99)]['striker'].count()
            st.info(f"• **Total Centuries score in IPL {selected_season}: {hundreds}**")

            # total fours
            total_fours = len(merged_df[(merged_df['runs_off_bat'] == 4) & (merged_df['season'] == selected_season)])
            st.info(f"• **Total Fours hit in IPL {selected_season}: {total_fours}**")

            # Total sixes
            total_sixes = len(merged_df[(merged_df['runs_off_bat'] == 6) & (merged_df['season'] == selected_season)])
            st.info(f"• **Total Sixes hit in IPL {selected_season}: {total_sixes}**")

        st.write(
            '--------------------------------------------------------------------------------------------------------------------------')

        ipl_winners_data = {
            'Season': [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            'Winner': ['Rajasthan Royals', 'Deccan Chargers', 'Chennai Super Kings', 'Chennai Super Kings',
                       'Kolkata Knight Riders', 'Mumbai Indians', 'Kolkata Knight Riders', 'Mumbai Indians',
                       'Sunrisers Hyderabad', 'Mumbai Indians', 'Chennai Super Kings', 'Mumbai Indians',
                       'Mumbai Indians', 'Chennai Super Kings', 'Gujarat Titans', 'Chennai Super Kings']
        }
        ipl_winners_df = pd.DataFrame(ipl_winners_data)
        ipl_winners_df = ipl_winners_df[ipl_winners_df['Season'] == selected_season].set_index('Season')
        st.markdown(f"• **{selected_season} Season winner**")
        st.dataframe(ipl_winners_df, width=250)


        st.write(
            '--------------------------------------------------------------------------------------------------------------------------')

        col1, col2 = st.columns(2)
        with col1:
            # orange cap winner
            most_runs_by_season = merged_df[merged_df['season'] == selected_season].groupby(['season', 'striker'])['runs_off_bat'].sum().reset_index()
            most_runs_by_season = most_runs_by_season.sort_values(by=['season', 'runs_off_bat'], ascending=[True, False])
            most_runs_by_season = most_runs_by_season.drop_duplicates(subset='season')
            most_runs_by_season = most_runs_by_season.set_index('season')
            most_runs_by_season = most_runs_by_season.rename(columns={'striker': 'Batsman', 'runs_off_bat': 'Runs'})
            st.markdown("• **Orange cap winner**")
            st.dataframe(most_runs_by_season, width=250)
        with col2:
            # purple cap winner
            most_wickets = merged_df[merged_df['season'] == selected_season].groupby(['season', 'bowler'])['is_wicket_delivery'].sum().reset_index()
            most_wickets = most_wickets.sort_values(by=['season', 'is_wicket_delivery'], ascending=[True, False])
            most_wickets = most_wickets.groupby(['season'])[['bowler', 'is_wicket_delivery']].first()
            most_wickets = most_wickets.rename(columns={'bowler': 'Bowler', 'is_wicket_delivery': 'Wickets'})
            st.markdown("• **Purple cap winner**")
            st.dataframe(most_wickets, width=250)
        st.write(
            '--------------------------------------------------------------------------------------------------------------------------')

        col3, col4, col5, col6 = st.columns(4)
        with col3:
            most_runs = merged_df[merged_df['season'] == selected_season].groupby(['striker'])['runs_off_bat'].sum().sort_values(ascending=False).head(
                3).reset_index(name='Runs').rename(columns={'striker': 'Batsman'}).set_index('Batsman')
            st.write(f"• **Top 3 most Runs scorer**")
            st.dataframe(most_runs)
        with col4:
            most_runs_powerplay = merged_df[(merged_df['ball'] < 6) & (merged_df['season'] == selected_season)].groupby(['striker'])['runs_off_bat'].sum().sort_values(
                ascending=False).head(3).reset_index(name='Runs').rename(columns={'striker': 'Batsman'}).set_index(
                'Batsman')
            st.write(f"• **Top 3 most Runs scorer in Powerplay**")
            st.dataframe(most_runs_powerplay)
        with col5:
            most_runs_middle_over = merged_df[((merged_df['ball'] > 6) & (merged_df['ball'] < 15)) & (merged_df['season'] == selected_season)].groupby(['striker'])[
                'runs_off_bat'].sum().sort_values(ascending=False).head(3).reset_index(name='Runs').rename(
                columns={'striker': 'Batsman'}).set_index('Batsman')
            st.write(f"• **Top 3 most Runs scorer in Middle overs**")
            st.dataframe(most_runs_middle_over)
        with col6:
            most_runs_death_over = merged_df[(merged_df['ball'] > 15) & (merged_df['season'] == selected_season)].groupby(['striker'])[
                'runs_off_bat'].sum().sort_values(ascending=False).head(3).reset_index(name='Runs').rename(
                columns={'striker': 'Batsman'}).set_index('Batsman')
            st.write(f"• **Top 3 most Runs scorer in Death overs**")
            st.dataframe(most_runs_death_over)
        st.write(
            '--------------------------------------------------------------------------------------------------------------------------')

        col7, col8, col9, col10 = st.columns(4)
        with col7:
            most_wickets = merged_df[merged_df['season'] == selected_season].groupby(['bowler'])['is_wicket_delivery'].sum().sort_values(ascending=False).head(
                3).reset_index(name='Wickets').rename(columns={'bowler': 'Bowler'}).set_index('Bowler')
            st.write(f"• **Top 3 most Wicket takers in {selected_season}**")
            st.dataframe(most_wickets)
        with col8:
            most_wickets_powerplay = merged_df[(merged_df['ball'] < 6) & (merged_df['season'] == selected_season)].groupby(['bowler'])[
                'is_wicket_delivery'].sum().sort_values(ascending=False).head(3).reset_index(name='Wickets').rename(
                columns={'bowler': 'Bowler'}).set_index('Bowler')
            st.write(f"• **Top 3 most Wicket takers in an Powerplay in {selected_season}**")
            st.dataframe(most_wickets_powerplay)
        with col9:
            most_wickets_middle_overs = merged_df[((merged_df['ball'] > 6) & (merged_df['ball'] < 15)) & (merged_df['season'] == selected_season)].groupby(['bowler'])[
                'is_wicket_delivery'].sum().sort_values(ascending=False).head(3).reset_index(name='Wickets').rename(
                columns={'bowler': 'Bowler'}).set_index('Bowler')
            st.write(f"• **Top 3 most Wicket takers in an Middle overs in {selected_season}**")
            st.dataframe(most_wickets_middle_overs)
        with col10:
            most_wickets_death_overs = merged_df[(merged_df['ball'] > 15) & (merged_df['season'] == selected_season)].groupby(['bowler'])[
                'is_wicket_delivery'].sum().sort_values(ascending=False).head(3).reset_index(name='Wickets').rename(
                columns={'bowler': 'Bowler'}).set_index('Bowler')
            st.write(f"• **Top 3 most Wicket takers in an Death overs in {selected_season}**")
            st.dataframe(most_wickets_death_overs)
        st.write(
            '--------------------------------------------------------------------------------------------------------------------------')

        col11, col12, = st.columns(2)
        with col11:
            # most win team
            most_win_team = merged_df[merged_df['season'] == selected_season].groupby(['winner'])['id'].nunique().sort_values(ascending=False).head(3).reset_index(
                name='Wins').rename(columns={'winner': 'Team'}).set_index('Team')
            st.write(f"• **Top 3 most Match win Teams in {selected_season}**")
            st.dataframe(most_win_team)
        with col12:
            # most match played team
            most_match_play_team = merged_df[merged_df['season'] == selected_season].groupby(['batting_team'])['id'].nunique().sort_values(ascending=False).head(
                3).reset_index(name='Total').rename(columns={'batting_team': 'Team'}).set_index('Team')
            st.write(f"• **Top 3 most Match played Teams in {selected_season}**")
            st.dataframe(most_match_play_team)
        st.write(
            '--------------------------------------------------------------------------------------------------------------------------')

        col13, col14, col15, col16 = st.columns(4)
        with col13:
            # highest run in ipl by batsman
            most_run_in_inning = merged_df[merged_df['season'] == selected_season].groupby(['id', 'striker'])['runs_off_bat'].sum().sort_values(
                ascending=False).reset_index().head(3).drop('id', axis=1).rename(
                columns={'striker': 'Batsman', 'runs_off_bat': 'Runs'}).set_index('Batsman')
            st.write(f"• **Top 3 most Runs scorer in an inning in {selected_season}**")
            st.dataframe(most_run_in_inning)
        with col14:
            # highest wicket in ipl by bowler
            most_wicket_in_inning = merged_df[merged_df['season'] == selected_season].groupby(['id', 'bowler'])['is_wicket_delivery'].sum().sort_values(
                ascending=False).reset_index().head(
                3).drop('id', axis=1).rename(columns={'bowler': 'Bowler', 'is_wicket_delivery': 'Runs'}).set_index('Bowler')
            st.write(f"• **Top 3 most Wicket takers in an inning in {selected_season}**")
            st.dataframe(most_wicket_in_inning)
        with col15:
            # most 50s
            fifties = merged_df[merged_df['season'] == selected_season].groupby(['id', 'striker'])['runs_off_bat'].sum().reset_index()
            fifties = fifties[(fifties['runs_off_bat'] > 49) & (fifties['runs_off_bat'] < 100)][
                'striker'].value_counts().head(3)
            st.write(f"• **Top 3 most Fifties scorer in {selected_season}**")
            st.dataframe(fifties)
        with col16:
            # most 100s
            hundreds = merged_df[merged_df['season'] == selected_season].groupby(['id', 'striker'])['runs_off_bat'].sum().reset_index()
            hundreds = hundreds[(hundreds['runs_off_bat'] > 99)]['striker'].value_counts().head(3)
            st.write(f"• **Top 3 most Centuries scorer in {selected_season}**")
            st.dataframe(hundreds)
        st.write(
            '--------------------------------------------------------------------------------------------------------------------------')

        col17, col18, col19 = st.columns(3)
        with col17:
            # most player of the match award winner
            p_of_match = merged_df[merged_df['season'] == selected_season].groupby(['player_of_match'])['id'].nunique().sort_values(ascending=False).head(
                3).reset_index(name='count').rename(columns={'player_of_match': 'Player of Match'}).set_index(
                'Player of Match')
            st.write(f"• **Top 3 most POM award winner in {selected_season}**")
            st.dataframe(p_of_match)
        with col18:
            # most fours hitter
            most_four_hitter = merged_df[(merged_df['runs_off_bat'] == 4) & (merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(
                ascending=False).head(3).reset_index(name='Fours').rename(columns={'striker': 'Player'}).set_index('Player')
            st.write(f"• **Top 3 Most Four Hitter in {selected_season}**")
            st.dataframe(most_four_hitter, width=150)
        with col19:
            # most six hitter
            most_six_hitter = merged_df[(merged_df['runs_off_bat'] == 6)& (merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(
                ascending=False).head(
                3).reset_index(name='Sixes').rename(columns={'striker': 'Player'}).set_index('Player')
            st.write(f"• **Top 3 Most Six Hitter in {selected_season}**")
            st.dataframe(most_six_hitter, width=150)
        st.write(
            '--------------------------------------------------------------------------------------------------------------------------')
        col21, col22, col23, col24, col25 = st.columns(5)
        with col21:
            # most run out
            most_run_out = \
            merged_df[(merged_df['wicket_type'] == 'run out') & (merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(ascending=False).index[
                0]
            count = \
            merged_df[(merged_df['wicket_type'] == 'run out')& (merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(ascending=False)[0]
            st.info(f"• **Most ot the time Run out Player**:- "
                    f"{most_run_out} was run out {count} times in IPL {selected_season} ")
        with col22:
            # most lbw
            most_lbw = \
            merged_df[(merged_df['wicket_type'] == 'lbw') & (merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(ascending=False).index[0]
            count = \
                merged_df[(merged_df['wicket_type'] == 'lbw')& (merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(ascending=False)[0]
            st.info("• **Most of the time lbw out Player**:- "
                    f"{most_lbw} was lbw {count} times in IPL {selected_season} ")
        with col23:
            # most bowled
            most_bowled = \
            merged_df[(merged_df['wicket_type'] == 'bowled')&(merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(ascending=False).index[
                0]
            count = \
            merged_df[(merged_df['wicket_type'] == 'bowled')& (merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(ascending=False)[0]
            st.info("• **Most of time the bowled out Player**:- "
                    f"{most_bowled} was bowled {count} times in IPL {selected_season} ")
        with col24:
            # most stumped
            most_stumped = \
            merged_df[(merged_df['wicket_type'] == 'stumped')&(merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(ascending=False).index[
                0]
            count = \
            merged_df[(merged_df['wicket_type'] == 'stumped')& (merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(ascending=False)[0]
            st.info("• **Most of the time stumped out Player**:- "
                    f"{most_stumped} was stumped {count} times in IPL {selected_season} ")
        with col25:
            # most caught
            most_caught = \
            merged_df[(merged_df['wicket_type'] == 'caught')&(merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(ascending=False).index[
                0]
            count = \
            merged_df[(merged_df['wicket_type'] == 'caught')& (merged_df['season'] == selected_season)]['striker'].value_counts().sort_values(ascending=False)[0]
            st.info(f"• **Most of the time catch out Player**:- "
                    f"{most_caught} was catch out {count} times in IPL {selected_season} ")

    st.write(
        '--------------------------------------------------------------------------------------------------------------------------')

    st.subheader(f"**Click on Start button to see the animation**")
    most_runs_player_race.drop(columns=['Unnamed: 0'], inplace=True)
    # Melt the data for Plotly animation
    df_melt = most_runs_player_race.melt(var_name='Player', value_name='Runs',
                                         ignore_index=False).reset_index()

    # Create animated bar chart using Plotly
    fig = px.bar(df_melt, x='Player', y='Runs', animation_frame='index',
                 title='Most Runs Scored by Players in IPL from first season till now',
                 labels={'index': '', 'Runs': 'Runs'}, barmode='group')

    # Increase animation speed by reducing the animation duration (e.g., 200 milliseconds)
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 200

    fig.update_yaxes(range=[0, df_melt.sort_values(by='Runs')[
        'Runs'] + 100])  # Add some buffer to the y-axis range

    # Update layout for animation
    fig.update_layout(xaxis_title='Player Name', yaxis_title=' Runs Scored', showlegend=False)

    # Show the animated bar chart
    st.plotly_chart(fig)



    most_wickets_player_race.drop(columns=['Unnamed: 0'], inplace=True)
    # Melt the data for Plotly animation
    df_melt1 = most_wickets_player_race.melt(var_name='Player', value_name='Wickets', ignore_index=False).reset_index()

    # Create animated bar chart using Plotly
    fig1 = px.bar(df_melt1, x='Player', y='Wickets', animation_frame='index',
                 title='Most Wickets taken by Players in IPL from first season till now',
                 labels={'index': '', 'Wickets': 'Wickets'}, barmode='group')

    # Increase animation speed by reducing the animation duration (e.g., 200 milliseconds)
    fig1.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 200

    fig1.update_yaxes(
        range=[0, df_melt1.sort_values(by='Wickets')['Wickets'] + 10])  # Add some buffer to the y-axis range

    # Update layout for animation
    fig1.update_layout(xaxis_title='Player Name', yaxis_title='Wickets', showlegend=False)

    # Show the animated bar chart
    st.plotly_chart(fig1)



def show_page_8():
    add_name_to_header('@abhishek_rawool')
    st.title('Overall')
    # total season
    col0, col01 = st.columns(2)
    with col0:
        total_season = merged_df['season'].nunique()
        st.info(f"• **Total Season in IPL: {total_season}**")

        # total matches
        total_matches = merged_df['id'].nunique()
        st.info(f"• **Total Matches played in IPL: {total_matches}**")

        # total balls
        total_balls = len(merged_df['ball'])
        st.info(f"• **Total Balls bowl in IPL: {total_balls}**")

        # total runs
        total_runs = merged_df['total_run'].sum()
        st.info(f"• **Total Runs scored in IPL: {total_runs}**")

        # total wickets
        total_wickets = len(merged_df[(merged_df['is_wicket'] == 1)])
        st.info(f"• **Total Wickets taken in IPL: {total_wickets}**")


    with col01:
        # total 50s
        fifties = merged_df.groupby(['id', 'striker'])['runs_off_bat'].sum().reset_index()
        fifties = fifties[(fifties['runs_off_bat'] > 49) & (fifties['runs_off_bat'] < 100)]['striker'].count()
        st.info(f"• **Total Fifties score in IPL: {fifties}**")

        # total 100s
        hundreds = merged_df.groupby(['id', 'striker'])['runs_off_bat'].sum().reset_index()
        hundreds = hundreds[(hundreds['runs_off_bat'] > 99)]['striker'].count()
        st.info(f"• **Total Centuries score in IPL: {hundreds}**")

        # total fours
        total_fours = len(merged_df[(merged_df['runs_off_bat'] == 4)])
        st.info(f"• **Total Fours hit in IPL: {total_fours}**")

        # Total sixes
        total_sixes = len(merged_df[(merged_df['runs_off_bat'] == 6)])
        st.info(f"• **Total Sixes hit in IPL: {total_sixes}**")

    st.write(
        '--------------------------------------------------------------------------------------------------------------------------')

    ipl_winners_data = {
        'Season': [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        'Winner': ['Rajasthan Royals', 'Deccan Chargers', 'Chennai Super Kings', 'Chennai Super Kings',
                   'Kolkata Knight Riders', 'Mumbai Indians', 'Kolkata Knight Riders', 'Mumbai Indians',
                   'Sunrisers Hyderabad', 'Mumbai Indians', 'Chennai Super Kings', 'Mumbai Indians',
                   'Mumbai Indians', 'Chennai Super Kings', 'Gujarat Titans', 'Chennai Super Kings']
    }
    st.subheader("• **All IPL Season winners**")
    st.dataframe(ipl_winners_data, width=400, height=600)

    st.write('--------------------------------------------------------------------------------------------------------------------------')

    col1,col2 = st.columns(2)
    with col1:
        # orange cap winner
        most_runs_by_season = merged_df.groupby(['season', 'striker'])['runs_off_bat'].sum().reset_index()
        most_runs_by_season = most_runs_by_season.sort_values(by=['season', 'runs_off_bat'], ascending=[True, False])
        most_runs_by_season = most_runs_by_season.drop_duplicates(subset='season')
        most_runs_by_season = most_runs_by_season.set_index('season')
        most_runs_by_season = most_runs_by_season.rename(columns={'striker': 'Batsman', 'runs_off_bat': 'Runs'})
        st.subheader("• **Orange cap winners**")
        st.dataframe(most_runs_by_season, width=250)
    with col2:
        # purple cap winner
        most_wickets = merged_df.groupby(['season', 'bowler'])['is_wicket_delivery'].sum().reset_index()
        most_wickets = most_wickets.sort_values(by=['season', 'is_wicket_delivery'], ascending=[True, False])
        most_wickets = most_wickets.groupby(['season'])[['bowler', 'is_wicket_delivery']].first()
        most_wickets = most_wickets.rename(columns={'bowler': 'Bowler', 'is_wicket_delivery': 'Wickets'})
        st.subheader("• **Purple cap winners**")
        st.dataframe(most_wickets, width=250)
    st.write(
        '--------------------------------------------------------------------------------------------------------------------------')

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        most_runs = merged_df.groupby(['striker'])['runs_off_bat'].sum().sort_values(ascending=False).head(
            3).reset_index(name='Runs').rename(columns={'striker': 'Batsman'}).set_index('Batsman')
        st.write("• **Top 3 most Runs scorer**")
        st.dataframe(most_runs)
    with col4:
        most_runs_powerplay = merged_df[(merged_df['ball'] < 6)].groupby(['striker'])['runs_off_bat'].sum().sort_values(
            ascending=False).head(3).reset_index(name='Runs').rename(columns={'striker': 'Batsman'}).set_index('Batsman')
        st.write("• **Top 3 most Runs scorer in Powerplay**")
        st.dataframe(most_runs_powerplay)
    with col5:
        most_runs_middle_over = merged_df[(merged_df['ball'] > 6 ) & (merged_df['ball'] < 15 )].groupby(['striker'])['runs_off_bat'].sum().sort_values(ascending=False).head(3).reset_index(name='Runs').rename(columns={'striker':'Batsman'}).set_index('Batsman')
        st.write("• **Top 3 most Runs scorer in Middle overs**")
        st.dataframe(most_runs_middle_over)
    with col6:
        most_runs_death_over = merged_df[(merged_df['ball'] > 15 )].groupby(['striker'])['runs_off_bat'].sum().sort_values(ascending=False).head(3).reset_index(name='Runs').rename(columns={'striker':'Batsman'}).set_index('Batsman')
        st.write("• **Top 3 most Runs scorer in Death overs**")
        st.dataframe(most_runs_death_over)
    st.write(
        '--------------------------------------------------------------------------------------------------------------------------')

    col7, col8, col9, col10 = st.columns(4)
    with col7:
        most_wickets = merged_df.groupby(['bowler'])['is_wicket_delivery'].sum().sort_values(ascending=False).head(3).reset_index(name='Wickets').rename(columns={'bowler':'Bowler'}).set_index('Bowler')
        st.write("• **Top 3 most Wicket takers**")
        st.dataframe(most_wickets)
    with col8:
        most_wickets_powerplay = merged_df[(merged_df['ball'] < 6 )].groupby(['bowler'])['is_wicket_delivery'].sum().sort_values(ascending=False).head(3).reset_index(name='Wickets').rename(columns={'bowler':'Bowler'}).set_index('Bowler')
        st.write("• **Top 3 most Wicket takers in Powerplay**")
        st.dataframe(most_wickets_powerplay)
    with col9:
        most_wickets_middle_overs = merged_df[(merged_df['ball'] > 6 ) & (merged_df['ball'] < 15 )].groupby(['bowler'])['is_wicket_delivery'].sum().sort_values(ascending=False).head(3).reset_index(name='Wickets').rename(columns={'bowler':'Bowler'}).set_index('Bowler')
        st.write("• **Top 3 most Wicket takers in Middle overs**")
        st.dataframe(most_wickets_middle_overs)
    with col10:
        most_wickets_death_overs = merged_df[(merged_df['ball'] > 15 )].groupby(['bowler'])['is_wicket_delivery'].sum().sort_values(ascending=False).head(3).reset_index(name='Wickets').rename(columns={'bowler':'Bowler'}).set_index('Bowler')
        st.write("• **Top 3 most Wicket takers in Death overs**")
        st.dataframe(most_wickets_death_overs)
    st.write(
        '--------------------------------------------------------------------------------------------------------------------------')

    col11, col12, = st.columns(2)
    with col11:
        # most win team
        most_win_team = merged_df.groupby(['winner'])['id'].nunique().sort_values(ascending=False).head(3).reset_index(name='Wins').rename(columns={'winner': 'Team'}).set_index('Team')
        st.write("• **Top 3 most Match win Teams**")
        st.dataframe(most_win_team)
    with col12:
        # most match played team
        most_match_play_team = merged_df.groupby(['batting_team'])['id'].nunique().sort_values(ascending=False).head(3).reset_index(name='Total').rename(columns={'batting_team':'Team'}).set_index('Team')
        st.write("• **Top 3 most Match played Teams**")
        st.dataframe(most_match_play_team)

    st.write('--------------------------------------------------------------------------------------------------------------------------')

    col13, col14, col15, col16 = st.columns(4)
    with col13:
        # highest run in ipl by batsman
        most_run_in_inning = merged_df.groupby(['id', 'striker'])['runs_off_bat'].sum().sort_values(
            ascending=False).reset_index().head(3).drop('id', axis=1).rename(
            columns={'striker': 'Batsman', 'runs_off_bat': 'Runs'}).set_index('Batsman')
        st.write("• **Top 3 most Runs scorer in inning**")
        st.dataframe(most_run_in_inning)
    with col14:
        # highest wicket in ipl by bowler
        most_wicket_in_inning = merged_df.groupby(['id', 'bowler'])['is_wicket_delivery'].sum().sort_values(ascending=False).reset_index().head(
            3).drop('id', axis=1).rename(columns={'bowler': 'Bowler', 'is_wicket_delivery': 'Runs'}).set_index('Bowler')
        st.write("• **Top 3 most Wicket takers in inning**")
        st.dataframe(most_wicket_in_inning)
    with col15:
        # most 50s
        fifties = merged_df.groupby(['id', 'striker'])['runs_off_bat'].sum().reset_index()
        fifties = fifties[(fifties['runs_off_bat'] > 49) & (fifties['runs_off_bat'] < 100)]['striker'].value_counts().head(3)
        st.write("• **Top 3 most Fifties scorer**")
        st.dataframe(fifties)
    with col16:
        # most 100s
        hundreds = merged_df.groupby(['id', 'striker'])['runs_off_bat'].sum().reset_index()
        hundreds = hundreds[(hundreds['runs_off_bat'] > 99)]['striker'].value_counts().head(3)
        st.write("• **Top 3 most Centuries scorer**")
        st.dataframe(hundreds)
    st.write(
        '--------------------------------------------------------------------------------------------------------------------------')

    col17, col18, col19= st.columns(3)
    with col17:
        # most player of the match award winner
        p_of_match = merged_df.groupby(['player_of_match'])['id'].nunique().sort_values(ascending=False).head(
            3).reset_index(name='count').rename(columns={'player_of_match': 'Player of Match'}).set_index('Player of Match')
        st.write("• **Top 3 most POM award winner**")
        st.dataframe(p_of_match)
    with col18:
        # most fours hitter
        most_four_hitter = merged_df[(merged_df['runs_off_bat'] == 4)]['striker'].value_counts().sort_values(ascending=False).head(3).reset_index(name='Fours').rename(columns={'striker': 'Player'}).set_index('Player')
        st.write("• **Top 3 Most Four Hitter**")
        st.dataframe(most_four_hitter, width=150)
    with col19:
        # most six hitter
        most_six_hitter = merged_df[(merged_df['runs_off_bat'] == 6)]['striker'].value_counts().sort_values(ascending=False).head(
            3).reset_index(name='Sixes').rename(columns={'striker': 'Player'}).set_index('Player')
        st.write("• **Top 3 Most Six Hitter**")
        st.dataframe(most_six_hitter, width=150)
    st.write(
        '--------------------------------------------------------------------------------------------------------------------------')
    col21, col22, col23, col24, col25 = st.columns(5)
    with col21:
        # most run out
        most_run_out = merged_df[(merged_df['wicket_type'] == 'run out')]['striker'].value_counts().sort_values(ascending=False).index[0]
        count = merged_df[(merged_df['wicket_type'] == 'run out')]['striker'].value_counts().sort_values(ascending=False)[0]
        st.info("• **Most ot the time Run out Player**:- "
                f"{most_run_out} was run out {count} times in IPL ")
    with col22:
        # most lbw
        most_lbw = merged_df[(merged_df['wicket_type'] == 'lbw')]['striker'].value_counts().sort_values(ascending=False).index[0]
        count = \
        merged_df[(merged_df['wicket_type'] == 'lbw')]['striker'].value_counts().sort_values(ascending=False)[0]
        st.info("• **Most of the time lbw out Player**:- "
                f"{most_lbw} was lbw {count} times in IPL ")
    with col23:
        # most bowled
        most_bowled = merged_df[(merged_df['wicket_type'] == 'bowled')]['striker'].value_counts().sort_values(ascending=False).index[0]
        count = merged_df[(merged_df['wicket_type'] == 'bowled')]['striker'].value_counts().sort_values(ascending=False)[0]
        st.info("• **Most of time the bowled out Player**:- "
                f"{most_bowled} was bowled {count} times in IPL ")
    with col24:
        # most stumped
        most_stumped = merged_df[(merged_df['wicket_type'] == 'stumped')]['striker'].value_counts().sort_values(ascending=False).index[0]
        count = merged_df[(merged_df['wicket_type'] == 'stumped')]['striker'].value_counts().sort_values(ascending=False)[0]
        st.info("• **Most of the time stumped out Player**:- "
                f"{most_stumped} was stumped {count} times in IPL ")
    with col25:
        # most caught
        most_caught = merged_df[(merged_df['wicket_type'] == 'caught')]['striker'].value_counts().sort_values(ascending=False).index[0]
        count = merged_df[(merged_df['wicket_type'] == 'caught')]['striker'].value_counts().sort_values(ascending=False)[0]
        st.info(f"• **Most of the time catch out Player**:- "
                f"{most_caught} was catch out {count} times in IPL ")



def add_name_to_header(name):
    st.markdown(
        f"""
        <style>
        .header-name {{
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 15px;
            font-weight: bold;
            color: #848587;
        }}
        </style>
        <div class="header-name">{name}</div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()

