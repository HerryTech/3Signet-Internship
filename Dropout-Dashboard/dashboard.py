import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="School Dropout Dashboard", page_icon=":bar_chart", layout="wide")
st.title(":bar_chart: School Dropout Dashboard")
st.markdown("<style>div.block-container{padding-top:1.7rem;}</style>", unsafe_allow_html=True)

# Replace with your GitHub repository details
repo_owner = "HerryTech"  # Replace with your GitHub username
repo_name = "3Signet-Internship"  # Replace with your repository name
file_name = "Dropout-Dashboard/cleaned_transformed_dataset.csv"  # Update path

# Construct the raw content URL for the CSV file
url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{file_name}"

# Read the CSV data from GitHub
try:
    school_data = pd.read_csv(url)
    st.success(f"Successfully loaded data from {url}")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()  # Stop execution if data loading fails


st.sidebar.header("Choose your filter:")

#Create for Target
target = st.sidebar.multiselect("Target", school_data["Target"].unique())
if not target:
   df = school_data.copy()
else:
    df = school_data[school_data["Target"].isin(target)]

#Create for Scholarship holder
scholarship_holder = st.sidebar.multiselect("Scholarship holder", df["Scholarship holder"].unique())
if not target:
   df2 = df.copy()
else:
    df2 = df[df["Scholarship holder"].isin(scholarship_holder)]

#Create for Gender
gender = st.sidebar.multiselect("Gender", df2["Gender"].unique())

#filter the data based on the filters
if not target and not scholarship_holder and not gender:
    filter_data = school_data
elif not scholarship_holder and not gender:
    filter_data = school_data[school_data["Target"].isin(target)]
elif not target and not gender:
    filter_data = school_data[school_data["Scholarship holder"].isin(scholarship_holder)]
elif scholarship_holder and gender:
    filter_data = df2[df2["Gender"].isin(gender) & df2["Scholarship holder"].isin(scholarship_holder)]
elif target and gender:
    filter_data = df2[df2["Target"].isin(target) & df2["Gender"].isin(gender)]
elif target and scholarship_holder:
    filter_data = df2[df2["Target"].isin(target) & df2["Scholarship holder"].isin(scholarship_holder)]
elif gender:
    filter_data = df2[df2["Gender"].isin(gender)]
else:
    filter_data = df2[df2["Target"].isin(target) & df2["Scholarship holder"].isin(scholarship_holder) & df2["Gender"].isin(gender)]

col1, col2 = st.columns((2))

# Aggregate data to get counts
count_data = filter_data.groupby(['Target', 'Gender']).size().reset_index(name='Count')

fig = px.bar(count_data, 
             x="Gender", 
             y="Count", 
             color="Target", 
             title="Distribution of Target by Gender",
             labels={"Gender": "Gender Value", "Count": "Count", "Target":"Target (0=Dropout, 1=Enrolled, 2=Graduate)"},
             category_orders={"Gender": [0, 1]},  # Ensure correct order for Gender
             template="seaborn")
# Adjust layout to increase space between x-axis categories
fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': [0, 1]},  # Maintain category order
                      bargap=0.5  # Adjust the gap between bars (higher value increases the space)
                      )
# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)


with col1:
# Create a mapping for the Target values
    target_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}

# Apply the mapping to create a new column with descriptive labels
    filter_data['Target_Label'] = filter_data['Target'].map(target_map)

# Create the pie chart using the new descriptive labels
    fig = px.pie(filter_data, names='Target_Label', title='Target Status Distribution', hole=0.5, template="seaborn")

# Update traces to show the original 'Target' values as text labels outside the pie chart
#fig.update_traces(text=filter_data["Target_Label"], textposition="outside")

# Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


with col2:
    fig = px.histogram(filter_data, x='Age at enrollment', nbins=30, title='Distribution of Age at enrollment', height=500,
                  color_discrete_sequence=px.colors.qualitative.Set2) 
    st.plotly_chart(fig,use_container_width=True)


with col1:
    fig = px.scatter(filter_data, x='Admission grade', y='GPA', color='Target',
                 title="Admission Grade vs GPA by Target",
                 labels={'Admission grade': 'Admission Grade', 'GPA': 'Final GPA'})
    st.plotly_chart(fig,use_container_width=True)


with col2:
    fig = px.pie(filter_data, names='Debtor', title='Percentage of Students with Debt', template="seaborn")
    #fig.update_traces(text = filter_data["Target"], textposition = "outside")
    st.plotly_chart(fig,use_container_width=True)


with col1:
    fig = px.box(filter_data,
    x="Target",
    y="Admission grade",
    title='Boxplot of Target by Admission grade',
    labels={'Admission grade': 'Admission grade', 'Target': 'Target'},
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.pie(filter_data, names='Scholarship holder', title='Scholarship holder Distribution', hole = 0.5, template="seaborn")
    fig.update_traces(text = filter_data["Scholarship holder"], textposition = "outside")
    st.plotly_chart(fig,use_container_width=True)
    
fig = px.scatter(filter_data, x='Total Credits Earned', y='GPA', 
                 size='Total Units Enrolled', color='Target', 
                 title='Total Credits Earned vs. GPA (Bubble Size: Total Units Enrolled)')
st.plotly_chart(fig,use_container_width=True)


count_data1 = filter_data.groupby(['Target', 'Tuition fees up to date']).size().reset_index(name='Count')
fig = px.bar(count_data1, 
             x="Tuition fees up to date", 
             y="Count", 
             color="Target", 
             title="Distribution of Target by Tuition fees up to date",
             labels={"Tuition fees up to date": "Tuition fees up to date", "Count": "Count", "Target":"Target (0=Dropout, 1=Enrolled, 2=Graduate)"},
             category_orders={"Tuition fees up to date": [0, 1]},  # Ensure correct order for Gender
             template="seaborn")
# Adjust layout to increase space between x-axis categories
fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': [0, 1]},  # Maintain category order
                      bargap=0.5  # Adjust the gap between bars (higher value increases the space)
                      )
# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)
