from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource

# setting epochs
number_epochs = 100

# loading the CSV file based on number of epochs into a DataFrame
df = pd.read_csv(f'prediction_results_with_{number_epochs}_epochs.csv')

# Create a Bokeh figure
p = figure(x_axis_label='Date', y_axis_label='Price', title='Actual vs Predicted Prices', x_range=df['Date'].astype(str).tolist(), height=500, width=900)

# Create a ColumnDataSource
source = ColumnDataSource(df)

# Plot actual prices as bars
p.vbar(x='Date', top='Actual', width=0.5, source=source, legend_label='Actual Price', color='blue')

# Plot predicted prices as a line
p.line(x='Date', y='Predicted', source=source, legend_label='Predicted Price', line_width=2, line_color='red')

# Add hover tool
hover = HoverTool()
hover.tooltips = [('Date', '@Date{%F}'), ('Actual Price', '@Actual'), ('Predicted Price', '@Predicted')]
hover.formatters = {'@Date': 'datetime'}
p.add_tools(hover)

# Customize the plot layout
p.legend.location = 'top_left'
p.legend.click_policy = 'hide'

# Output to notebook
output_notebook()

# Display the plot in the notebook
show(p, notebook_handle=True)