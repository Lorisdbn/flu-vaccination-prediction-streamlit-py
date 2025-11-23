import os
import pickle
import streamlit as st
import pandas as pd
import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.stats as stats
import plotly.express as px
from sklearn.metrics import roc_curve, auc

# ⚡ Première commande obligatoire
st.set_page_config(
    page_title="H1N1 and Seasonal Flu Vaccines",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🎨 CSS Styling : Fond jaune, texte noir, contour noir des expanders
st.markdown(
    """
    <style>
    html, body, .stApp {
        background-color: #fff2cc !important;
        color: black !important;
    }
    [data-testid="stSidebar"] {
        background-color: #cfe2f3 !important;
        color: black !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
        font-weight: bold !important;
    }
    p, div, label, span {
        color: black !important;
    }
    /* --- Correctif Expander --- */
    [data-testid="stExpander"] {
        border: 2px solid black !important;
        border-radius: 8px !important;
        background-color: #fff2cc !important;
        padding: 10px !important;
        margin-bottom: 20px;
    }
    [data-testid="stExpander"] > div {
        background-color: #fff2cc !important;
        color: black !important;
    }
    [data-testid="stExpanderDetails"] {
        background-color: #fff2cc !important;
    }
    .st-expanderHeader {
        color: black !important;
    }
    .st-expanderHeader:hover {
        color: #3A1078 !important;
    }
     /* Barre du haut transparente */
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ===== Functions =====

@st.cache_resource
def load_model(model_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_filename)
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        st.error(f"Model {model_filename} not found.")
        return None

@st.cache_data
def load_csv(file_name):
    try:
        return pd.read_csv(file_name)
    except FileNotFoundError:
        st.error(f"File {file_name} not found.")
        return None

def apply_plotly_style(fig):
    fig.update_layout(
        font=dict(color="black"),
        legend=dict(font=dict(color="black")),
        xaxis=dict(titlefont=dict(color="black"), tickfont=dict(color="black")),
        yaxis=dict(titlefont=dict(color="black"), tickfont=dict(color="black")),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    return fig

# ===== Load models =====
h1n1_model = load_model('h1n1_vaccine_neuronalmodel.pkl')
seasonal_model = load_model('seasonal_vaccine_neuronalmodel.pkl')

# ===== Load datasets =====
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets = {
    "test_features_processed": os.path.join(script_dir, "test_features_processed.csv"),
    "test_set_features": os.path.join(script_dir, "test_set_features.csv"),
    "test_set_labels": os.path.join(script_dir, "test_set_labels.csv"),
    "training_set": os.path.join(script_dir, "training_set.csv"),
    "training_set_features": os.path.join(script_dir, "training_set_features.csv"),
    "training_set_labels": os.path.join(script_dir, "training_set_labels.csv"),
    "training_set_processed": os.path.join(script_dir, "training_set_processed.csv")
}

test_features_processed = load_csv(datasets["test_features_processed"])
test_set_features = load_csv(datasets["test_set_features"])
test_set_labels = load_csv(datasets["test_set_labels"])
training_set = load_csv(datasets["training_set"])
training_set_features = load_csv(datasets["training_set_features"])
training_set_labels = load_csv(datasets["training_set_labels"])
training_set_processed = load_csv(datasets["training_set_processed"])

# Set title of the app
st.title("H1N1 / Seasonal flu vaccines learning and predictions")

# Sidebar title
st.sidebar.title("Summary")

# List of pages
pages = ["📄 Project description", "🔍 Data exploration", "📊 Data visualization", "🛠️ Data preparation", "🔮 Modelling & predictions"]

# Radio button to navigate pages
page = st.sidebar.radio("Select a page:", pages, label_visibility="collapsed")

# Adding authors' names in the sidebar
st.sidebar.markdown("### Author")
st.sidebar.markdown("""
Loris Durbano  <a href="https://www.linkedin.com/in/lorisdurbano/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="20" height="20"></a>
""", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("""
**Data source:** [drivendata.org](https://www.drivendata.org/competitions/66/flu-shot-learning/data/)
""", unsafe_allow_html=True)
st.sidebar.markdown("""DrivenData. (2020). Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines.""")

# Page content logic
if page == pages[0]:  
    # Check if the models were loaded successfully
    if h1n1_model is None or seasonal_model is None:
        st.error("One or both models could not be loaded.")
    else:
        st.success("Models loaded successfully.")

    # Check if all datasets are loaded successfully
    if any(df is None for df in [test_features_processed, test_set_features, test_set_labels, training_set, training_set_features, training_set_labels, training_set_processed]):
        st.error("One or more datasets failed to load.")
    else:
        st.success("All datasets loaded successfully.")

    st.markdown("<h2 style='text-align: center;'>Project description</h2>", unsafe_allow_html=True)
    st.text(" ")
    st.markdown("<p style='text-align: center; font-weight: bold;'>Can you predict whether individuals received the H1N1 and seasonal flu vaccines using information about their backgrounds, opinions, and health behaviors?</p>",unsafe_allow_html=True,)
    st.text(" ")
    st.write("In this project, we explore vaccination, a crucial public health strategy in the fight against infectious diseases. Vaccines not only protect individuals by providing immunization but also contribute to herd immunity, which helps curb the spread of diseases within a community.")
    # Display the image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'vaccin1.jpg')
    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.error(f"Image {image_path} not found.")
    st.text(" ")
    st.write("At the time of this competition's launch, vaccines for the COVID-19 virus were still under development and not yet available. Therefore, this challenge revisits the public health response to a previous major respiratory disease pandemic. In the spring of 2009, the H1N1 influenza virus, commonly known as swine flu, triggered a global pandemic. Researchers estimate that the virus was responsible for between 151,000 to 575,000 deaths worldwide in its first year.")
    st.text(" ")
    st.write("A vaccine for the H1N1 virus became available to the public in October 2009. To understand vaccination patterns, the United States conducted the National 2009 H1N1 Flu Survey in late 2009 and early 2010. This phone survey asked respondents whether they had received the H1N1 and seasonal flu vaccines, along with questions about their social, economic, and demographic background, opinions on the risks of illness and vaccine effectiveness, and behaviors toward mitigating transmission. Analyzing these characteristics can provide valuable insights for guiding future public health efforts.")
    st.text(" ")
    st.markdown("<h2 style='text-align: center;'>Project overview</h2>", unsafe_allow_html=True)
    st.text(" ")
    st.markdown(""" To tackle this challenge, we followed a structured process:

1. **Data exploration:** we began by thoroughly exploring the dataset to understand its structure, identify key variables, and detect any anomalies or missing values. This step was crucial for gaining initial insights and setting the foundation for further analysis.
2. **Data visualization:** next, we visualized the data to uncover patterns and relationships between different variables. This involved generating various plots and charts to better understand the distribution of features and their potential impact on vaccination outcomes.
3. **Data processing for machine learning:** after gaining insights from exploration and visualization, we preprocessed the data for machine learning. This step included handling missing values, encoding categorical variables, scaling features, and selecting relevant features for model training.
4. **Model training and prediction:** with the data prepared, we trained machine learning models to predict whether individuals received the H1N1 and seasonal flu vaccines. We experimented with different algorithms, fine-tuned hyperparameters, and validated the models' performance using appropriate metrics.

Finally, we applied the trained models to a test set of patients to generate predictions on their vaccination status. These predictions were evaluated to assess the model's accuracy and its potential utility in guiding public health decisions.

This systematic approach allowed us to effectively analyze the dataset and make informed predictions about vaccination patterns.
""")

elif page == pages[1]:
    st.markdown("<h2 style='text-align: center;'>Dataset overview</h2>", unsafe_allow_html=True)
    # Proceed with data exploration and analysis

    # Display the first few rows of the datasets
    st.markdown("""**Let's start our exploration by taking a look at the training dataset**""", unsafe_allow_html=True)
    st.text(" ")
    with st.expander("Training set dataframe"):
        st.dataframe(training_set.head())
        st.markdown("""The two last columns <u>h1n1_vaccine</u> and <u>seasonal_vaccine</u> are the target variable, the ones we want to predict - on another dataset. We will train our model on this training data in order to be able to predict the vaccination rate on another unknown panel based on the 35 features we currently have.""",unsafe_allow_html=True)

    with st.expander("Variables description"):
        st.markdown("""
    **Labels**  
    For this project, there are two target variables:
    
    - **h1n1_vaccine**: Whether the respondent received the H1N1 flu vaccine.  
      *Binary variable*: 0 = No; 1 = Yes.
    
    - **seasonal_vaccine**: Whether the respondent received the seasonal flu vaccine.  
      *Binary variable*: 0 = No; 1 = Yes.
    
    Both are binary variables: 0 = No; 1 = Yes. Some respondents didn't get either vaccine, others got only one, and some got both. This is formulated as a multilabel (and not multiclass) problem.
    
    **The features in this dataset**  
    We are provided with a dataset containing 36 columns. The first column, **respondent_id**, is a unique and random identifier. The remaining 35 features are described below:
    
    For all binary variables: 0 = No; 1 = Yes.
    
    - **h1n1_concern**: Level of concern about the H1N1 flu.  
      *0 = Not at all concerned; 1 = Not very concerned; 2 = Somewhat concerned; 3 = Very concerned.*
    
    - **h1n1_knowledge**: Level of knowledge about H1N1 flu.  
      *0 = No knowledge; 1 = A little knowledge; 2 = A lot of knowledge.*
    
    - **behavioral_antiviral_meds**: Has taken antiviral medications. (binary)
    
    - **behavioral_avoidance**: Has avoided close contact with others with flu-like symptoms. (binary)
    
    - **behavioral_face_mask**: Has bought a face mask. (binary)
    
    - **behavioral_wash_hands**: Has frequently washed hands or used hand sanitizer. (binary)
    
    - **behavioral_large_gatherings**: Has reduced time at large gatherings. (binary)
    
    - **behavioral_outside_home**: Has reduced contact with people outside of their own household. (binary)
    
    - **behavioral_touch_face**: Has avoided touching eyes, nose, or mouth. (binary)
    
    - **doctor_recc_h1n1**: H1N1 flu vaccine was recommended by a doctor. (binary)
    
    - **doctor_recc_seasonal**: Seasonal flu vaccine was recommended by a doctor. (binary)
    
    - **chronic_med_condition**: Has any of the following chronic medical conditions: asthma or another lung condition, diabetes, a heart condition, a kidney condition, sickle cell anemia or other anemia, a neurological or neuromuscular condition, a liver condition, or a weakened immune system caused by a chronic illness or by medicines taken for a chronic illness. (binary)
    
    - **child_under_6_months**: Has regular close contact with a child under the age of six months. (binary)
    
    - **health_worker**: Is a healthcare worker. (binary)
    
    - **health_insurance**: Has health insurance. (binary)
    
    - **opinion_h1n1_vacc_effective**: Respondent's opinion about H1N1 vaccine effectiveness.  
      *1 = Not at all effective; 2 = Not very effective; 3 = Don't know; 4 = Somewhat effective; 5 = Very effective.*
    
    - **opinion_h1n1_risk**: Respondent's opinion about the risk of getting sick with H1N1 flu without a vaccine.  
      *1 = Very Low; 2 = Somewhat low; 3 = Don't know; 4 = Somewhat high; 5 = Very high.*
    
    - **opinion_h1n1_sick_from_vacc**: Respondent's worry of getting sick from taking the H1N1 vaccine.  
      *1 = Not at all worried; 2 = Not very worried; 3 = Don't know; 4 = Somewhat worried; 5 = Very worried.*
    
    - **opinion_seas_vacc_effective**: Respondent's opinion about seasonal flu vaccine effectiveness.  
      *1 = Not at all effective; 2 = Not very effective; 3 = Don't know; 4 = Somewhat effective; 5 = Very effective.*
    
    - **opinion_seas_risk**: Respondent's opinion about the risk of getting sick with seasonal flu without a vaccine.  
      *1 = Very Low; 2 = Somewhat low; 3 = Don't know; 4 = Somewhat high; 5 = Very high.*
    
    - **opinion_seas_sick_from_vacc**: Respondent's worry of getting sick from taking the seasonal flu vaccine.  
      *1 = Not at all worried; 2 = Not very worried; 3 = Don't know; 4 = Somewhat worried; 5 = Very worried.*
    
    - **age_group**: Age group of the respondent.
    
    - **education**: Self-reported education level.
    
    - **race**: Ethnicity of the respondent.
    
    - **sex**: Sex of the respondent.
    
    - **income_poverty**: Household annual income of the respondent with respect to 2008 Census poverty thresholds.
    
    - **marital_status**: Marital status of the respondent.
    
    - **rent_or_own**: Housing situation of the respondent.
    
    - **employment_status**: Employment status of the respondent.
    
    - **hhs_geo_region**: Respondent's residence using a 10-region geographic classification defined by the U.S. Dept. of Health and Human Services. Values are represented as short random character strings.
    
    - **census_msa**: Respondent's residence within metropolitan statistical areas (MSA) as defined by the U.S. Census.
    
    - **household_adults**: Number of other adults in the household, top-coded to 3.
    
    - **household_children**: Number of children in the household, top-coded to 3.
    
    - **employment_industry**: Type of industry the respondent is employed in. Values are represented as short random character strings.
    
    - **employment_occupation**: Type of occupation of the respondent. Values are represented as short random character strings.
    """)

    # Display data types
    with st.expander("Training set dataframe types:"):
        st.text(training_set.dtypes)
        st.text("")
        st.markdown("""Based on this information we can observe some categorical variables that will **need to be converted in order to train our future machine learning models** (since machine learning models cannot work with other type of data than numerical).""")

    # Display shape of the datasets
    with st.expander("Number of entries & columns"):
        st.text({training_set.shape})
        st.markdown("""As mentionned above we have **35 features, 1 index column and 2 target variables** we want to predict. In this training dataset we have more than 26 700 patients to train our model""")

    # Display basic statistics
    with st.expander("Training set dataframe description (basic statistics)"):
        st.dataframe(training_set.describe())
        st.markdown("""We can observe large variations on data quality throughout the dataset with some missing values. Let's take a deeper analysis of these missing values 👇 """)

    # Calculate the percentage of missing values per column
    missing_values_percent = training_set.isnull().mean() * 100

    # Round the values to 1 decimal place and add a "%" symbol
    missing_values_percent = missing_values_percent.round(1).astype(str) + '%'

    # Convert to a DataFrame for better display
    missing_values_df = missing_values_percent.reset_index()
    missing_values_df.columns = ['Column', 'Percentage of missing values']

    # Display the DataFrame in Streamlit
    with st.expander("Percentage of missing values per column:"):
        st.dataframe(missing_values_df)
        st.markdown("""Overall, the dataset seems to be of good quality, with most columns being nearly complete. This should facilitate robust statistical analysis and model building, as missing data handling can be kept to a minimum or dealt with straightforward methods like mean/median imputation.""")

    # Check for duplicates
    duplicates = training_set.duplicated()

    # Count the number of duplicate rows
    num_duplicates = duplicates.sum()

    # Display the result in Streamlit
    if num_duplicates > 0:
        st.write(f"### Number of Duplicate Rows: {num_duplicates}")
        st.write("### Duplicate Rows:")
        st.dataframe(training_set[duplicates])
    else:
        st.markdown("""*Nota bene : There is no duplicate rows found in the dataFrame.*""")

    st.text(" ")
    st.markdown("""**We will now check if there is a correlation between the two target variables <u>h1n1_vaccine</u> and <u>seasonal_vaccine</u>.**""", unsafe_allow_html=True)
    st.text(" ")
    st.markdown("""**The Chi-square** test is used to determine whether there is a significant association between two categorical variables.""")
    st.markdown("""**A higher Chi-square value** indicates a greater deviation from what would be expected if there were no association between the variables. This suggests a stronger association between the variables. **A lower Chi-square value** indicates less deviation from expected counts, suggesting a weaker or no association.""")

    # Chi-square test
    with st.expander("## Chi-Square Test for H1N1 and Seasonal Vaccines"):
        contingency_table = pd.crosstab(training_set_processed['h1n1_vaccine'], training_set_processed['seasonal_vaccine'])
        chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
        st.write(f"Chi2: {chi2}")
        st.write(f"p-value: {p}")
        st.markdown("""The combination of a very large Chi-square statistic and a p-value of 0 <u>strongly indicates that there is a statistically significant association between the two categorical variables</u> being tested. This association is unlikely to be due to random chance and warrants further investigation to understand its nature and implications.""", unsafe_allow_html=True)

    st.text(" ")

    st.markdown("""
Let's start another test to measure the strength of the correlation between the two variables. **Kendall's Tau correlation test** is a non-parametric measure of the strength and direction of association between two variables. It assesses how the rankings of the data are related.

The Kendall's Tau coefficient ranges from **-1 to 1**:
- 1 indicates a perfect positive association.
- -1 indicates a perfect negative association.
- 0 indicates no association.
""")

    # Kendall's tau correlation
    with st.expander("Kendall's Tau Correlation"):
        kendall_corr = training_set_processed[['h1n1_vaccine', 'seasonal_vaccine']].corr(method='kendall')
        st.write(kendall_corr)
        st.write("The Kendall coefficient is positive, indicating a moderate correlation between the two variables.")

    st.text(" ")

    st.markdown("""**Finally, let's check  the distribution of the two target variables (h1n1_vaccine and seasonal_vaccine) among our training set.**""", unsafe_allow_html=True)

    st.text(" ")

    # Pie charts for vaccine distribution
    with st.expander(" Vaccine distribution pie charts"):

        n_obs = training_set.shape[0]
        h1n1_proportions = training_set['h1n1_vaccine'].value_counts().div(n_obs)
        seasonal_proportions = training_set['seasonal_vaccine'].value_counts().div(n_obs)

        fig_h1n1_pie = go.Figure(data=[go.Pie(labels=h1n1_proportions.index.map({0: 'No', 1: 'Yes'}),
                                  values=h1n1_proportions,
                                  marker=dict(colors=['orange', 'green']),
                                  textinfo='label+percent',
                                  textfont=dict(size=14, color='white', family='Arial Black'))])

        fig_h1n1_pie.update_layout(
        title='Proportion of H1N1 Vaccine',
        title_x=0.5, plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)'
        )

        fig_seasonal_pie = go.Figure(data=[go.Pie(labels=seasonal_proportions.index.map({0: 'No', 1: 'Yes'}),
                                      values=seasonal_proportions,
                                      marker=dict(colors=['orange', 'green']),
                                      textinfo='label+percent',
                                      textfont=dict(size=14, color='black', family='Arial Black'))])

        fig_seasonal_pie.update_layout(
        title='Proportion of Seasonal Vaccine',
        title_x=0.5, plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)'
        )

        st.plotly_chart(fig_h1n1_pie)
        st.plotly_chart(fig_seasonal_pie)

        st.text(" ")

        st.markdown("""The dataset appears to be **unbalanced**, particularly with regard to the h1n1_vaccine variable, which could impact modeling efforts.
**Vaccine uptake is low for H1N1 and somewhat more balanced for the seasonal flu**, highlighting potential areas for further investigation or intervention.""")

    st.markdown("<h2 style='text-align: center;'>Conclusion </h2>", unsafe_allow_html=True)
    st.markdown("""
**Robustness and readiness:**  
The dataset is well-suited for analysis, with a solid foundation in terms of data quality. However, special attention must be given to the imbalance in target variables and the handling of missing data.

**Analytical potential:**  
The dataset offers valuable insights into vaccination behavior, with the potential to inform public health strategies. The significant associations and correlations within the data can be leveraged to build predictive models that help anticipate vaccine uptake and address barriers to vaccination.

**Future work:**  
Further exploration could involve more detailed feature engineering, investigating potential biases in the data, and applying advanced modeling techniques that account for the imbalances and correlations identified.
""")

elif page == pages[2]:
    st.markdown("<h2 style='text-align: center;'>Data visualization</h2>", unsafe_allow_html=True)

    st.markdown("""**Let's start our visualization to get insightful information about our data**""", unsafe_allow_html=True)
    st.text(" ")

# Plotly charts for various data explorations

# Plot the respondents demographic pyramid  
    # Count the number of respondents by age group and sex
    age_sex_distribution = training_set.groupby(['age_group', 'sex']).size().unstack().fillna(0)
    # Create the demographic pyramid
    fig = go.Figure()
    # Add bars for males (Sex 'Male')
    fig.add_trace(go.Bar(
        y=age_sex_distribution.index,
        x=-age_sex_distribution['Male'],  
        name='Male',
        orientation='h',
        marker=dict(color='green')))
# Add bars for females (Sex 'Female')
    fig.add_trace(go.Bar(
        y=age_sex_distribution.index,
        x=age_sex_distribution['Female'],  
        name='Female',
        orientation='h',
        marker=dict(color='orange')))
    # Update the layout of the chart
    fig.update_layout(
        barmode='overlay',
        xaxis=dict(
            title='Number of respondents',
            titlefont=dict(color='black'),
            tickfont=dict(color='black'),
            tickvals=[-500, -250, 0, 250, 500],
            ticktext=[500, 250, 0, 250, 500]
        ),
        yaxis=dict(
            title='Age group',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),
        legend=dict(font=dict(color='black')), # Texte global en noir
        bargap=0.1,
        plot_bgcolor='#fff2cc',      
        paper_bgcolor='#fff2cc'     
    )

    # Display the chart in Streamlit
    with st.expander("Respondents demographic pyramid"):
        st.plotly_chart(fig)
        st.markdown("""This plot reveals that the dataset is **skewed toward older age groups and has a higher proportion of female respondents**. These factors could influence the outcomes of any analysis, making it important to consider age and gender when interpreting the results or building predictive models.""")
 
# Plot the respondents ethnical pyramid   
    # Count the number of respondents by race and sex
    race_sex_distribution = training_set.groupby(['race', 'sex']).size().unstack().fillna(0)
    # Create the demographic pyramid
    fig = go.Figure()
    # Add bars for males (Sex 'Male')
    fig.add_trace(go.Bar(
        y=race_sex_distribution.index,
        x=-race_sex_distribution['Male'],  
        name='Male',
        orientation='h',
        marker=dict(color='green')))
    # Add bars for females (Sex 'Female')
    fig.add_trace(go.Bar(
        y=race_sex_distribution.index,
        x=race_sex_distribution['Female'],  
        name='Female',
        orientation='h',
        marker=dict(color='orange')))
    # Update the layout of the chart
    fig.update_layout(
        title=dict(
            text="Respondents race pyramid",
            font=dict(color='black')
        ),
        barmode='overlay',
        xaxis=dict(
            title='Number of respondents',
            titlefont=dict(color='black'),
            tickfont=dict(color='black'),
            tickvals=[-500, -250, 0, 250, 500],
            ticktext=[500, 250, 0, 250, 500]
        ),
        yaxis=dict(
            title='Race',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),  # Texte général noir
        legend=dict(font=dict(color='black')),  # Légende en noir
        bargap=0.1,
        plot_bgcolor='#fff2cc',      # Fond du graphe en jaune clair
        paper_bgcolor='#fff2cc'      # Fond du papier en jaune clair
    )

    # Display the chart in Streamlit
    with st.expander("Respondents demographic pyramid based on ethnicity"):
        st.plotly_chart(fig)
        st.markdown("""This plot highlights a significant ethnical imbalance in the dataset, with **the White population being overrepresented compared to other ethnic groups**. The findings and models derived from this dataset should be interpreted with caution, taking into account the potential biases introduced by this demographic skew.""")


# Correlation heatmaps with Plotly
    with st.expander("Correlation of features with H1N1 and seasonal vaccines"):
        correlations_h1n1 = training_set_processed.corr()[['h1n1_vaccine']].drop('h1n1_vaccine')
        correlations_seasonal = training_set_processed.corr()[['seasonal_vaccine']].drop('seasonal_vaccine')
        # Heatmap for H1N1 Vaccine
        fig_h1n1 = go.Figure(data=go.Heatmap(
        z=correlations_h1n1.values.T,
        x=correlations_h1n1.index,
        y=['H1N1 Vaccine'],
        colorscale='Blues',
        zmin=-1, zmax=1,
        hoverongaps=False
        ))
        fig_h1n1.update_layout(
        title=dict(
            text='Correlation of features with H1N1 vaccine',
            font=dict(color='black')
        ),
        xaxis=dict(
            title='Features',
            titlefont=dict(color='black'),
            tickfont=dict(color='black'),
            tickangle=-45
        ),
        yaxis=dict(
            title='H1N1 vaccine',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),
        legend=dict(font=dict(color='black')),  # Même si pas de legend ici, bon pour standardiser
        plot_bgcolor='#fff2cc',
        paper_bgcolor='#fff2cc'
    )

        # Heatmap for Seasonal Vaccine
        fig_seasonal = go.Figure(data=go.Heatmap(
        z=correlations_seasonal.values.T,
        x=correlations_seasonal.index,
        y=['Seasonal Vaccine'],
        colorscale='Greens',
        zmin=-1, zmax=1,
        hoverongaps=False
        ))
    fig_seasonal.update_layout(
        title=dict(
            text='Correlation of features with seasonal vaccine',
            font=dict(color='black')
        ),
        xaxis=dict(
            title='Features',
            titlefont=dict(color='black'),
            tickfont=dict(color='black'),
            tickangle=-45
        ),
        yaxis=dict(
            title='Seasonal vaccine',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),
        legend=dict(font=dict(color='black')),  # Même si pas de légende ici, on garde pour uniformiser
        plot_bgcolor='#fff2cc',  # Fond du graphique jaune clair
        paper_bgcolor='#fff2cc'  # Fond du "papier" jaune clair
    )

    st.plotly_chart(fig_h1n1)
    st.plotly_chart(fig_seasonal)
    st.markdown("""
    **The strongest correlations with both vaccines are observed with variables related to doctor recommendations and personal opinions about the vaccine’s effectiveness.**  
    This highlights the importance of medical advice and public perception in vaccination decisions. Features like **education, employment status, and income tend to have weaker correlations with both vaccines**, suggesting that these demographic factors may play a less direct role in the decision to vaccinate (although they influence the opinion a patient may have about vaccination). **Public health strategies** aiming to increase vaccine uptake might focus on enhancing the role of healthcare providers in recommending vaccines and addressing public concerns or misconceptions about vaccine efficacy.""")

# Plot the vaccination rate by level of concern about H1N1 virus
    concern_labels = {
    0: 'Not at all concerned',
    1: 'Not very concerned',
    2: 'Somewhat concerned',
    3: 'Very concerned'
    }
    # Group by h1n1_concern and h1n1_vaccine, then calculate the counts
    counts = training_set.groupby(['h1n1_concern', 'h1n1_vaccine']).size().unstack(fill_value=0)
    # Convert the raw counts to percentages
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    # Remap the concern levels
    percentages.index = percentages.index.map(concern_labels)
    # Create the bar chart with Plotly
    fig = go.Figure()
    # Add bars for each vaccine status (0 = No, 1 = Yes)
    for column in percentages.columns:
        color = '#387F39' if column == 1 else '#ff5733'
        fig.add_trace(go.Bar(
            y=percentages.index,
            x=percentages[column],
            name=f"Vaccine Status: {'Yes' if column == 1 else 'No'}",
            orientation='h',
            text=percentages[column].round(1).astype(str) + '%',  
            marker_color=color,
            textposition='auto'
        ))
    # Customize the appearance of the chart
    fig.update_layout(
        title=dict(
            text='Concern levels about H1N1 vaccine',
            font=dict(color='black')
        ),
        xaxis=dict(
            title='Percentage of respondents',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title='Level of concern',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),  # Texte général en noir
        legend=dict(
            title=dict(text='H1N1 vaccine status', font=dict(color='black')),
            font=dict(color='black')
        ),
        barmode='stack',  # Utilisation de stack comme prévu
        plot_bgcolor='#fff2cc',  # Fond du graphique en jaune clair
        paper_bgcolor='#fff2cc'  # Fond du "papier" en jaune clair
    )

    # Display the chart in Streamlit
    with st.expander("Vaccination rate by level of concern about H1N1 virus"):
        st.plotly_chart(fig)
        st.markdown("""The plot likely shows that **as the level of concern increases, the proportion of respondents who received the H1N1 vaccine increases**. This suggests that people who are more concerned about the H1N1 virus are more likely to get vaccinated.
                For those who are not at all concerned or not very concerned, the vaccination rate is probably much lower, indicating that **concern is a significant motivator for getting vaccinated**.
                These findings underscore **the importance of addressing public concerns in vaccination campaigns**. By increasing awareness and concern about the risks of a virus, public health initiatives may be able to boost vaccination rates.""")

# Plot the vaccination rate by level of knowledge about H1N1 virus
    # Define knowledge labels
    knowledge_labels = {
        0: 'No knowledge',
        1: 'A little knowledge',
        2: 'A lot of knowledge'}
    # Group by h1n1_knowledge and h1n1_vaccine, then calculate the counts
    counts = training_set.groupby(['h1n1_knowledge', 'h1n1_vaccine']).size().unstack(fill_value=0)
    # Convert the raw counts to percentages
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    # Remap the knowledge levels
    percentages.index = percentages.index.map(knowledge_labels)
    # Create the bar chart with Plotly
    fig = go.Figure()
    # Add bars for each vaccine status (0 = No, 1 = Yes)
    for column in percentages.columns:
        color = '#387F39' if column == 1 else '#ff5733'
        fig.add_trace(go.Bar(
            y=percentages.index,
            x=percentages[column],
            name=f"Vaccine Status: {'Yes' if column == 1 else 'No'}",
            orientation='h',
            marker_color=color,
            text=percentages[column].round(1).astype(str) + '%',  # Add percentage labels
            textposition='auto'))
    # Customize the appearance of the chart
    fig.update_layout(
        title=dict(
            text='Knowledge levels about H1N1 vaccine',
            font=dict(color='black')
        ),
        xaxis=dict(
            title='Percentage of respondents',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title='Level of knowledge',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),  # Texte global noir
        legend=dict(
            title=dict(text='H1N1 vaccine status', font=dict(color='black')),
            font=dict(color='black')
        ),
        barmode='stack',  # Empilement des barres
        plot_bgcolor='#fff2cc',  # Fond jaune clair
        paper_bgcolor='#fff2cc'  # Fond jaune clair aussi
    )

    # Display the chart in Streamlit
    with st.expander("Vaccination rate by level of knowledge about H1N1 virus"):
        st.plotly_chart(fig)
        st.markdown("""The plot likely reveals that **individuals with a lot of knowledge about the H1N1 virus are more likely to get vaccinated**. This indicates that knowledge and awareness of the virus play a crucial role in vaccination decisions.
                    This highlights **the critical role of education and information dissemination in public health campaigns**. Ensuring that the public has access to accurate and comprehensive information about vaccines and viruses can significantly influence vaccination rates.""")

# Plot the concern level by age group
    # Calculate the distribution of the number of people for each level of concern, by age group
    concern_distribution = training_set.groupby(['age_group', 'h1n1_concern']).size().unstack().fillna(0)
    # Convert the raw counts to percentages
    concern_percentages = concern_distribution.div(concern_distribution.sum(axis=1), axis=0) * 100
    # # Colors for each level of concern
    colors = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'}
    # Create the figure
    fig = go.Figure()
    # Add bars for each level of concern with specific colors
    for concern_level in concern_percentages.columns:
        fig.add_trace(go.Bar(
            x=concern_percentages.index,
            y=concern_percentages[concern_level],
            name=f'Concern Level {concern_level}',
            text=concern_percentages[concern_level].round(1).astype(str) + '%', 
            textposition='auto',
            marker_color=colors[concern_level]))
    # Update the layout of the chart
    fig.update_layout(
        title=dict(
            text='Knowledge levels by age group',
            font=dict(color='black')
        ),
        xaxis=dict(
            title='Age group',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title='Percentage of respondents',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),  # Couleur générale noire
        legend=dict(font=dict(color='black')),  # Légende noire (même si pas précisé ici)
        barmode='stack',  # Empilage
        plot_bgcolor='#fff2cc',  # Fond jaune clair
        paper_bgcolor='#fff2cc'  # Fond jaune clair
    )

    # Display the chart in Streamlit
    with st.expander("Distribution of H1N1 concern by age group"):
        st.plotly_chart(fig)
        st.markdown("This plot reveals that **concern about H1N1 generally increases with age**, with older respondents showing higher levels of concern compared to younger ones. This trend underscores the need for age-specific public health strategies **to ensure that information and interventions are appropriately targeted to different segments of the population**.")

# Plot the vaccination rates by age group
    # Calculate the proportion of people vaccinated by age group and sex
    vaccination_distribution = training_set.groupby(['age_group', 'sex'])['h1n1_vaccine'].mean().unstack().fillna(0)
    # Create the figure
    fig = go.Figure()
    # Add bars for males
    fig.add_trace(go.Bar(
        x=vaccination_distribution.index,
        y=vaccination_distribution['Male'],
        name='Male',
        text=(vaccination_distribution['Male'] * 100).round(1).astype(str) + '%',
        textposition='auto',
        marker_color='green'))
    # Add bars for females
    fig.add_trace(go.Bar(
        x=vaccination_distribution.index,
        y=vaccination_distribution['Female'],
        name='Female',
        text=(vaccination_distribution['Female'] * 100).round(1).astype(str) + '%',
        textposition='auto',
        marker_color='orange'))
    # Update the layout of the chart
    fig.update_layout(
        title=dict(
            text='Vaccination rate by age group',
            font=dict(color='black')
        ),
        xaxis=dict(
            title='Age group',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title='Vaccination rate',
            titlefont=dict(color='black'),
            tickfont=dict(color='black'),
            tickformat='%'
        ),
        font=dict(color='black'),  # Couleur générale noire
        legend=dict(font=dict(color='black')),  # Légende noire si présente
        barmode='group',  # Barres côte à côte
        plot_bgcolor='#fff2cc',  # Fond du graphe jaune clair
        paper_bgcolor='#fff2cc'  # Fond du papier jaune clair
    )

    # Display the chart in Streamlit
    with st.expander("Comparison of H1N1 Vaccination rates by sex and age group"):
        st.plotly_chart(fig)
        st.markdown("""This plot highlights **significant differences in H1N1 vaccination rates across age groups and between genders**. Older males tend to have higher vaccination rates, while younger age groups and older females show relatively lower rates. Public health strategies should consider these trends when designing targeted interventions to increase vaccination coverage, ensuring that both younger individuals and older women are effectively reached.""")

# Plot the vaccination rates based on doctor recommendation
    # Calculate the vaccination rates based on doctor recommendation for H1N1 vaccine
    vaccination_by_doctor_recommendation = training_set.groupby('doctor_recc_h1n1')['h1n1_vaccine'].mean()
    # Convert the rates to percentages
    vaccination_by_doctor_recommendation *= 100
    # Define labels for the recommendation status
    recommendation_labels = {0: 'No recommendation', 1: 'Doctor recommended'}
    # Create the figure
    fig = go.Figure()
    # Add bar for vaccination rates
    fig.add_trace(go.Bar(
        x=[recommendation_labels[0], recommendation_labels[1]],
        y=vaccination_by_doctor_recommendation,
        text=vaccination_by_doctor_recommendation.round(1).astype(str) + '%',
        textposition='auto',
        marker_color=['red', 'green']))
    # Update the layout of the chart
    fig.update_layout(
        title=dict(
            text='Impact of Doctor Recommendation on H1N1 Vaccination',
            font=dict(color='black')
        ),
        xaxis=dict(
            title='Doctor recommendation for H1N1 vaccine',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title='Vaccination Rate (%)',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),  # Texte général noir
        legend=dict(font=dict(color='black')),  # Légende noire si présente
        plot_bgcolor='#fff2cc',  # Fond jaune clair
        paper_bgcolor='#fff2cc'  # Fond jaune clair
    )

    # Display the chart in Streamlit
    with st.expander("H1N1 vaccination rates based on doctor recommendation"):
        st.plotly_chart(fig)
        st.markdown("""The plot underscores **the critical role of doctor recommendations in promoting H1N1 vaccination**. With over half of the individuals getting vaccinated when their doctor recommended it, compared to a small fraction who did so without a recommendation, it’s clear that doctor-patient interactions are vital in driving public health outcomes. Public health strategies should continue to leverage the trust and authority of healthcare providers to encourage vaccinations, especially in populations that might otherwise be hesitant.""")

# Plot the vaccination rates based on opinion
    # Calculate the vaccination rates based on opinions about H1N1 vaccine effectiveness
    vaccination_by_opinion = training_set.groupby('opinion_h1n1_vacc_effective')['h1n1_vaccine'].mean()
    # Convert the rates to percentages
    vaccination_by_opinion *= 100
    # Define labels for the opinion levels
    opinion_labels = {
        1: 'Not at all effective',
        2: 'Not very effective',
        3: 'Don\'t know',
        4: 'Somewhat effective',
        5: 'Very effective'}
    # Create the figure
    fig = go.Figure()
    # Add bar for vaccination rates by opinion
    fig.add_trace(go.Bar(
        x=[opinion_labels[1], opinion_labels[2], opinion_labels[3], opinion_labels[4], opinion_labels[5]],
        y=vaccination_by_opinion,
        text=vaccination_by_opinion.round(1).astype(str) + '%',
        textposition='auto',
        marker_color='lightblue'))
    # Update the layout of the chart
    fig.update_layout(
        title=dict(
            text='Impact of Perceived H1N1 Vaccine Effectiveness on Vaccination',
            font=dict(color='black')
        ),
        xaxis=dict(
            title='Perceived H1N1 vaccine effectiveness',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title='Vaccination rate (%)',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),  # Texte général noir
        legend=dict(font=dict(color='black')),  # Légende noire (par sécurité même si absente)
        plot_bgcolor='#fff2cc',  # Fond du graphe jaune clair
        paper_bgcolor='#fff2cc'  # Fond du papier jaune clair
    )

    # Display the chart in Streamlit
    with st.expander("H1N1 vaccination rates based on perceived vaccine effectiveness"):
        st.plotly_chart(fig)
        st.markdown("""This plot demonstrates **a clear and strong relationship between perceived vaccine effectiveness and vaccination rates**. Individuals who believe the H1N1 vaccine is effective are significantly more likely to get vaccinated, while those who doubt its effectiveness or are unsure are much less likely to do so. Public health strategies aimed at increasing vaccination rates should prioritize improving public confidence in vaccine effectiveness through targeted communication and education efforts.""")

    st.markdown("<h2 style='text-align: center;'>Conclusion </h2>", unsafe_allow_html=True)
    st.markdown("""The visualization step has provided valuable insights into the factors that influence vaccination behavior. 
                The <u>strong impact of medical recommendations and public perception on vaccine uptake, 
                along with the observed demographic disparities, underline the importance of a targeted and informed public health approach</u>. 
                Moving forward, these findings can guide the development of predictive models and the design of interventions aimed at 
                increasing vaccination rates and improving public health outcomes.""", unsafe_allow_html=True)

elif page == pages[3]:
    st.markdown("<h2 style='text-align: center;'>Data processing</h2>", unsafe_allow_html=True)

    st.markdown("""**After having explored the training dataset, we will now process the data to make it ready for machine learning training**""", unsafe_allow_html=True)
    st.text(" ")
    st.markdown("""Based on the previous insights, the main steps will be handling missing values, encoding numerical data and standardize the data.
                Machine learning algorithms typically work with numerical data. Encoding converts categorical data into a numerical format that the algorithms can process.
                Standardizing ensures that all features have the same scale, particularly important for algorithms that rely on distance metrics and it usually involves rescaling the data so that it has a mean of 0 and a standard deviation of 1.
                We will do the same work on the test dataset to be sure the 2 dataframes are similarly processed""")
    st.text(" ")
    st.markdown("""Let's start by managing the missing values""")

# Managing missing values               
    nan_counts = training_set.isna().sum()
    # Convert the Series to a DataFrame to add column labels
    nan_counts_df = nan_counts.reset_index()
    nan_counts_df.columns = ["Column", "Missing values"]
    st.text("")
    st.markdown("""#### Synthesis of missing value handling""")

        # Display the missing values per column under an expander
    with st.expander("Missing value per column"):
        st.write(nan_counts_df)

    st.markdown("""In handling missing values, we used a combination of median imputation, mode imputation, and specific constant filling based on the nature of the variables:""")
    
    st.markdown("""
    - **Median imputation**: for continuous or ordinal variables, the median was used because it is less affected by outliers and preserves the data's central tendency.
    - **Mode imputation**: for binary and categorical variables, the mode (most frequent value) was used to maintain the original distribution of the data.
    - **Specific constant filling**: in cases where missing data might represent a distinct category (e.g., lack of recommendation), a specific constant (e.g., 'Unknown') was used to retain this information.
    - **Median for top-coded variables**: for variables with top-coding, the median was used to ensure that imputation did not exceed the top-coded value.
    """, unsafe_allow_html=True)

    nan_counts = training_set_processed.isna().sum()
    # Convert the Series to a DataFrame to add column labels
    nan_counts_df = nan_counts.reset_index()
    nan_counts_df.columns = ["Column", "Missing values"]
    # Display the missing values per column under an expander
    with st.expander("Missing value per column after processing"):
        st.write(nan_counts_df)
    st.text(' ')       
    st.markdown("""#### Synthesis of value encoding""")
    st.markdown(""" Let's now convert the categorical data using the factorize method""")
    with st.expander ("Variable types before encoding"):
        training_set.dtypes
    with st.expander("Variable types after encoding"):
        training_set_processed.dtypes
    
    st.text(" ")
    st.markdown(""" A last step to prepare our dataset is the standardization.""")
    st.markdown(""" As explained above, standardizing the values of the encoded dataframe is important because it ensures that all features have the same scale. This is particularly crucial for algorithms like logistic regression or support vector machines which rely on the distances or gradients in the feature space. If features are on different scales, those with larger ranges could disproportionately influence the model, leading to biased results. Therefore standardization helps ensure that each feature contributes equally to the model's learning process.""")
    st.text("")
    with st.expander("Encoded & standardized dataframe, ready for model's training"):
        st.dataframe(training_set_processed.head(5))
    st.text(" ")

    st.markdown("""#### Conclusion of data processing
The data processing steps, including handling missing values, encoding categorical variables, and standardizing the dataset, have effectively prepared the training data for machine learning. These steps ensure that the data is clean, consistent, and suitable for the algorithms to accurately learn and make predictions.

The standardization process, in particular, ensures that all features contribute equally to the learning process, avoiding biases that could arise from varying scales among features. 

The same preprocessing steps have also been applied to the test dataset to guarantee that the model trained on the training set can be effectively applied to the test set, ensuring consistent performance across both datasets.
""")


elif page == pages[4]:
    st.markdown("<h2 style='text-align: center;'>Data modelling</h2>", unsafe_allow_html=True)
    st.markdown(""" **Various models have been trained on the processed training set to get the most relevant predictions on the test set, aiming to predict whether a specific patient will get vaccinated for the H1N1 and/or the seasonal flu virus**""", unsafe_allow_html=True)
    with st.expander("We trained the following models as part of our analysis:"):
        st.markdown("""
    1. **Logistic Regression**: a baseline model to understand the linear separability of the data.
    2. **Random Forest**: an ensemble method that combines multiple decision trees to improve prediction accuracy and control over-fitting.
    3. **Random Forest (GridSearchCV)**: a tuned version of the Random Forest using GridSearchCV for hyperparameter optimization.
    4. **Random Forest (StratifiedKFold)**: Random Forest trained with Stratified K-Folds cross-validation to ensure balanced distribution across training and validation sets.
    5. **Light GBM**: a highly efficient gradient boosting model that is optimized for speed and performance, particularly with large datasets.
    6. **GBM (Gradient Boosting Machine)**: a powerful boosting algorithm that iteratively improves weak learners.
    7. **Stacking (RF + GBM + LGBM)**: a stacked ensemble of Random Forest, Gradient Boosting, and Light GBM to leverage the strengths of multiple models.
    8. **Voting Classifier**: a simple ensemble model that combines the predictions of multiple models through majority voting.
    9. **SMOTETomek**: a combination of oversampling (SMOTE) and undersampling (Tomek links) techniques used to balance the data before training.
    10. **XGBoost (XGB)**: an efficient and scalable implementation of gradient boosting that is widely used in machine learning competitions.
    11. **AdaBoost**: an adaptive boosting technique that adjusts the weights of incorrectly classified instances to improve performance.
    12. **Neural Network**: a deep learning approach that can capture complex patterns in the data through multiple layers of abstraction.

    These models were selected based on their ability to handle class imbalance, their performance in similar tasks, and their potential to capture complex patterns within the dataset.""")
        
    st.markdown("""The performance metrics and detailed analysis of the models are available in a separate Jupyter notebook. You can view it [here](https://github.com/Lorisdbn/Fluvaccine/blob/main/Flu_data_modelling.ipynb).""")
    
    st.markdown("""Through this modeling process, we've identified the most effective machine learning models for predicting outcomes on our unbalanced dataset. The <u>neuronal network model</u>, in particular, demonstrated strong performance in capturing the complexities of the minority class, making it our top choice for further optimization and testing.
    To ensure the robustness of our predictions, we will apply the same modeling techniques to the test dataset, allowing us to validate the generalizability of our final model.
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center;'>Data predictions</h2>", unsafe_allow_html=True)
    st.markdown(""" **The most performant model has been selected therefore we can switch to the final part of our work, the predictions part.**""", unsafe_allow_html=True)
    st.markdown("""*Nota bene : the respondent id has been anonimized for privacy concerns*""")
    # Remove the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in test_features_processed.columns:
        test_features_processed = test_features_processed.drop(columns=['Unnamed: 0'])
# Predict probabilities using Keras models
    h1n1_probs = h1n1_model.predict(test_features_processed).flatten()
    seasonal_probs = seasonal_model.predict(test_features_processed).flatten()
    # Create the test_set_labels dataframe
    test_set_labels = pd.DataFrame(index=test_features_processed.index)
    test_set_labels['h1n1_vaccine'] = (h1n1_probs > 0.5).astype(int)
    test_set_labels['seasonal_vaccine'] = (seasonal_probs > 0.5).astype(int)
    test_set_labels.to_csv("predictions.csv")
# Display the predictions
    with st.expander("Predictions for test set"):
        st.markdown("""*1 = vaccinated ; 0 = not vaccinated*""")
        st.dataframe(test_set_labels)

    # Calculate the ROC curve and AUC for H1N1
    fpr_h1n1, tpr_h1n1, _ = roc_curve(test_set_labels['h1n1_vaccine'], h1n1_probs)
    roc_auc_h1n1 = auc(fpr_h1n1, tpr_h1n1)
    # Calculate the ROC curve and AUC for Seasonal Flu
    fpr_seasonal, tpr_seasonal, _ = roc_curve(test_set_labels['seasonal_vaccine'], seasonal_probs)
    roc_auc_seasonal = auc(fpr_seasonal, tpr_seasonal)
# Plotting ROC curves using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_h1n1, y=tpr_h1n1,
                     mode='lines',
                     name=f'H1N1 Vaccine (AUC = {roc_auc_h1n1:.2f})',
                     line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=fpr_seasonal, y=tpr_seasonal,
                     mode='lines',
                     name=f'Seasonal Vaccine (AUC = {roc_auc_seasonal:.2f})',
                     line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                     mode='lines', 
                     line=dict(color='gray', dash='dash'),
                     showlegend=False))
    fig.update_layout(
        title=dict(
            text='ROC Curve',
            font=dict(color='black')
        ),
        xaxis=dict(
            title='False Positive Rate (FPR)',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title='True Positive Rate (TPR)',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),  # Texte général en noir
        legend=dict(font=dict(color='black')),  # Légende en noir
        plot_bgcolor='#fff2cc',  # Fond jaune clair
        paper_bgcolor='#fff2cc',  # Fond jaune clair
        showlegend=True  # Légende activée
    )

    # Display the chart in Streamlit
    with st.expander('Receiver Operating Characteristic (ROC) Curve'):
        st.markdown("""A ROC curve (Receiver Operating Characteristic) is a graph used to evaluate the performance of a binary classification model. It shows how well the model can distinguish between two classes, such as "vaccinated" and "not vaccinated.""")
        st.plotly_chart(fig)

    # Calculate the counts for vaccinated and not vaccinated
    h1n1_vaccinated_count = (test_set_labels['h1n1_vaccine'] == 1).sum()
    h1n1_not_vaccinated_count = (test_set_labels['h1n1_vaccine'] == 0).sum()
    seasonal_vaccinated_count = (test_set_labels['seasonal_vaccine'] == 1).sum()
    seasonal_not_vaccinated_count = (test_set_labels['seasonal_vaccine'] == 0).sum()
    # Create the pie chart for H1N1 vaccine
    fig_h1n1 = go.Figure(data=[go.Pie(labels=['Vaccinated', 'Not vaccinated'], 
                                  values=[h1n1_vaccinated_count, h1n1_not_vaccinated_count],
                                  hole=.3, marker=dict(colors=['#c6f848', '#f4c536']))])
    fig_h1n1.update_layout(
        annotations=[dict(
            text='H1N1',
            x=0.5,
            y=0.5,
            font=dict(size=18, color='black'),  # Annotation noire
            showarrow=False
        )],
        title=dict(
            text='H1N1 Vaccine Outcome',  # (Optionnel si tu veux un titre principal)
            font=dict(color='black')
        ),
        xaxis=dict(
            title='',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title='',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),  # Texte global noir
        legend=dict(
            font=dict(color='black')
        ),
        plot_bgcolor='#fff2cc',  # Fond graphique jaune clair
        paper_bgcolor='#fff2cc'  # Fond papier jaune clair
    )


    # Create the pie chart for Seasonal vaccine
    fig_seasonal = go.Figure(data=[go.Pie(labels=['Vaccinated', 'Not Vaccinated'], 
                                      values=[seasonal_vaccinated_count, seasonal_not_vaccinated_count],
                                      hole=.3, marker=dict(colors=['#c6f848', '#f4c536']))])
    fig_seasonal.update_layout(
        annotations=[dict(
            text='Seasonal',
            x=0.5,
            y=0.5,
            font=dict(size=18, color='black'),  # Annotation noire
            showarrow=False
        )],
        title=dict(
            text='Seasonal Vaccine Outcome',  # Facultatif, tu peux adapter
            font=dict(color='black')
        ),
        xaxis=dict(
            title='',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title='',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        font=dict(color='black'),  # Texte général noir
        legend=dict(
            font=dict(color='black')
        ),
        plot_bgcolor='#fff2cc',  # Fond jaune clair
        paper_bgcolor='#fff2cc'  # Fond jaune clair
    )


# Display the pie charts
    with st.expander("H1N1 vaccine distribution predictions"):
        st.plotly_chart(fig_h1n1)
    with st.expander("Seasonal vaccine distribution predictions"):
        st.plotly_chart(fig_seasonal)

# General conclusion & recommendations
    ### General conclusion:
    st.markdown("""The project successfully built and evaluated predictive models for the **H1N1** and **seasonal flu vaccination uptake** using the provided dataset.  The insights gathered from the analysis have the potential to inform public health strategies, with particular emphasis on factors that influence vaccination decisions.

### Main recommendations:
1. **Targeted interventions:**
   - **Healthcare provider influence:** the strong correlation between **doctor recommendations** and vaccination uptake suggests that enhancing the role of healthcare providers in vaccine advocacy could significantly boost vaccination rates.          
    - **Public health campaigns** should involve doctors and other healthcare professionals in promoting vaccinations.""")
    st.write("")               
    st.markdown("""2. **Public perception management:**
   - **Vaccine effectiveness communication:** the models highlighted that individuals' perception of **vaccine effectiveness** is a key driver of vaccination behavior. 
    - Efforts should be made to improve public understanding of vaccine efficacy, potentially through educational campaigns that address misconceptions and provide clear, evidence-based information.""")
    st.write("")
    st.markdown("""3. **Demographic-specific strategies:**
   - **Age and demographic sensitivity:** the analysis revealed varying levels of concern and knowledge across different age groups and demographics. 
    - **Tailored strategies** that address the specific concerns of different demographic segments could improve vaccine uptake. For instance, older populations may benefit from targeted messages that directly address their higher levels of concern.""")
    st.write("")
    st.markdown("""4. **Addressing the data imbalance:**
   - Given the **imbalance observed in the dataset**, particularly the lower rates of **H1N1 vaccination**, future models could benefit from techniques like **SMOTE** to address class imbalance. This could lead to even more robust predictive performance in underrepresented classes.""")
    st.write("")
    st.markdown("""5. **Continuous monitoring and adaptation:**
   - As public perceptions and behaviors evolve, it will be crucial to continuously monitor the performance of these models. **Regular updates** to the models with new data will ensure they remain accurate and relevant, allowing for timely adjustments to public health strategies.

#### Final note:
The steps and methodologies used in this project were meticulously applied to both the training and test datasets to ensure consistency and reliability of the model predictions. 

These findings, when implemented, could contribute to more effective vaccination campaigns, ultimately leading to **higher vaccination rates** and better **public health outcomes**.""")
    # Display the image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'vaccin2.jpg')
    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.error(f"Image {image_path} not found.")
