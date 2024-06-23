import pandas as pd  # Pandas for data manipulation
import dash  # Dash library for creating web applications
from dash import dcc, html, dash_table  # Components for building layout
from dash.dependencies import Input, Output  # Callbacks to update layout based on user input
import plotly.express as px  # Plotly Express for creating interactive visualizations
import plotly.graph_objects as go  # Plotly Graph Objects for more control over visualizations

# Load the dataset
data_path = "Solar_Orbiter_with_anomalies.csv"  # Path to dataset file
data_path2 = "Solar_Orbiter_with_anomalies2.csv"
shap_values_path = "shap_values.csv"  # Path to SHAP values file

solar_data = pd.read_csv(data_path)  # Read dataset into DataFrame
solar_data2 = pd.read_csv(data_path2)  # Read second dataset into DataFrame
shap_values = pd.read_csv(shap_values_path)  # Read SHAP values into DataFrame

"""
    Run the dashboard to visualize solar orbiter instrument data.

    It consists of several components:
  
    1. Instrument Checklist: Allows users to select instruments from a checklist.
    2. Date Range Picker: Enables users to select a range of dates for analysis.
    3. Time Series Chart: Displays the time series data for selected instruments.
    4. Correlation Heatmap: Shows the correlation between selected instruments.
    5. Anomaly Score Chart: Visualizes anomaly scores over time.
    6. SHAP Values: Provides Feature importance using SHAP Values.
"""

# Initialize the Dash app
app = dash.Dash(__name__, title="Solar Orbiter Data Visualization")  # Title of the Dash app which is showed in the browser tab
server = app.server

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Solar Orbiter Instrument Data Visualization", style={'text-align': 'center'}),  # Title
    # Checklist to select instruments
    dcc.Checklist(
        id='instrument-checklist',  # Component ID
        options=[{'label': col, 'value': col} for col in solar_data.columns[1:-2]],  # Options for checklist
        # I have removed the last two columns as they contain anomaly scores, which are not required for visualization.
        value=[solar_data.columns[1]],  # Default selected value (first instrument)
        inline=True
    ),
    # Date range picker
    dcc.DatePickerRange(
        id='date-picker-range',
        min_date_allowed=solar_data['Date'].min(),  # Minimum date allowed
        max_date_allowed=solar_data['Date'].max(),  # Maximum date allowed
        start_date=solar_data['Date'].min(),  # Default start date
        end_date=solar_data['Date'].max()  # Default end date
    ),
    # Two rows, each containing two graphs
    html.Div([
        html.Div([dcc.Graph(id='time-series-chart')], className="six columns"),  # Time Series Chart
        html.Div([dcc.Graph(id='correlation-heatmap')], className="six columns"),  # Correlation Heatmap
    ], className="row"),
    html.Div([
        html.Div([dcc.Graph(id='anomaly-score-chart')], className="six columns"),  # Anomaly Score Chart
    ], className="row"),
    html.Div([
        html.Div([dcc.Graph(id='shap-feature-importance-chart')], className="six columns"),  # SHAP Feature Importance Chart
    ], className="row"),
    html.Div(id='anomaly-stats', style={'margin-top': '20px', 'text-align': 'center'}),  # Anomaly Stats
    html.Iframe(
        srcDoc=open("shap_values_plot.html").read(),
        style={"height": "500px", "width": "100%"}
    )
])

# Callbacks to update graphs
@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('anomaly-score-chart', 'figure'),
     Output('shap-feature-importance-chart', 'figure')],
    [Input('instrument-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graphs(selected_instruments, start_date, end_date):
    """
    Callback function to update graphs based on user input.
    Args:
    selected_instruments (list): List of selected instruments.
    start_date (str): Start date selected by the user.
    end_date (str): End date selected by the user.
    Returns:
    figs (list): List of figures for each graph.
    """
    filtered_data = solar_data[(solar_data['Date'] >= start_date) & (solar_data['Date'] <= end_date)]  # Filtering data based on selected date range
    filtered_data2 = solar_data2[(solar_data2['Date'] >= start_date) & (solar_data2['Date'] <= end_date)]  # Filtering data based on selected date range
    filtered_shap_values = shap_values[(shap_values['Date'] >= start_date) & (shap_values['Date'] <= end_date)]  # Filtering SHAP values based on selected date range

    # Time Series Chart
    time_series_fig = go.Figure()  # Creating a new figure for time series chart
    for instrument in selected_instruments:
        time_series_fig.add_trace(
            go.Scatter(
                x=filtered_data['Date'],  # X-axis data
                y=filtered_data[instrument],  # Y-axis data
                mode='lines+markers',  # Display mode
                name=instrument  # Instrument name
            )
        )
    time_series_fig.update_layout(title="Time Series of Selected Instruments")  # Updating layout of time series chart

    # Correlation Heatmap
    correlation_fig = go.Figure(
        go.Heatmap(
            z=filtered_data[selected_instruments].corr(),  # Calculating correlation matrix
            x=selected_instruments,  # X-axis labels
            y=selected_instruments,  # Y-axis labels
            colorscale='Viridis'  # Color scale
        )
    )
    correlation_fig.update_layout(title="Correlation Heatmap")  # Updating layout of correlation heatmap

    # Anomaly Score Chart
    anomaly_score_fig = go.Figure()  # Create a new figure for the anomaly score chart
    anomaly_score_fig.add_trace(go.Scatter(
        x=filtered_data2['Date'],  # Set the x-axis as the Date column of the filtered data
        y=filtered_data2['anomaly_score'],  # Set the y-axis as the anomaly_score column of the filtered data
        mode='lines+markers',  # Display both lines and markers on the graph
        name='Anomaly Score',  # Name the trace, which will appear in the legend
        marker=dict(
            color=['red' if val < 0 else 'blue' for val in filtered_data2['anomaly_score']],  # Use list comprehension to assign colors conditionally
            # Markers will be red if the anomaly score is below 0, otherwise blue
            size=5,  # Set the size of the markers
            line=dict(
                color='DarkSlateGrey',  # Color of the line around each marker
                width=2  # Width of the line around each marker
            )
        )
    ))
    anomaly_score_fig.update_layout(
        title="Anomaly Scores Over Time (Lower the scores, higher chances of anomaly, negative score means definitely anomaly)",  # Main title of the chart
        xaxis_title='Date',  # Title for the x-axis
        yaxis_title='Anomaly Score'  # Title for the y-axis
    )

    # SHAP Feature Importance Chart
    shap_feature_importance_fig = go.Figure()
    for feature in selected_instruments:
        shap_feature_importance_fig.add_trace(
            go.Scatter(
                x=filtered_shap_values['Date'],  # X-axis data
                y=filtered_shap_values[feature],  # Y-axis data
                mode='lines',  # Display mode
                stackgroup='one',  # Stack the lines
                name=feature  # Feature name
            )
        )
    shap_feature_importance_fig.update_layout(
        title="SHAP Feature Importance Over Time",
        xaxis_title='Date',
        yaxis_title='SHAP Value'
    )

    return time_series_fig, correlation_fig, anomaly_score_fig, shap_feature_importance_fig  # Return updated figures

"""References:
1. https://dash.plotly.com/ - Dash Documentation
2. https://dash.plotly.com/layout - Dash Layout (HTML Components)
3. https://dash.plotly.com/dash-core-components - Dash Core Components ( DatePickerRange, Checklist)
4. https://dash.plotly.com/dash-html-components - Dash HTML Components (Div, H1 , Iframe)
5. https://plotly.com/python/plotly-express/ - Plotly Express ( px.line, px.scatter, px.bar)
6. https://plotly.com/python/graph-objects/ - Plotly Graph Objects ( go.Scatter, go.Heatmap, go.Figure)
7. https://www.coursera.org/projects/interactive-dashboards-plotly-dash?tab=guided-projects - Coursera Project
"""

if __name__ == "__main__":
    app.run_server(debug=True)  # Start the Dash server in debug mode
