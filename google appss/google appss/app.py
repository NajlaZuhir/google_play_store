
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the dataset for reference
df = pd.read_csv("new_df_copy.csv")


# '''PREDICTION'''
# Load trained models
random_forest_model = pickle.load(open("random_forest.pkl", "rb"))
decision_tree_model = pickle.load(open("decision_tree.pkl", "rb"))

# Load saved feature names
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Define prediction function
def predict_app_success(model, input_data, feature_names):
    # Convert input to DataFrame and ensure one-hot encoding matches training
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Align with training feature names
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(input_df)
    return prediction[0]


# '''Recommednation'''
# Normalize numerical features# Normalize numerical features
numerical_features = ['Rating', 'Sentiment_Polarity', 'Discrepancy', 'User_Engagement']
scaler = MinMaxScaler()

# Handle missing values and scale numerical features
df[numerical_features] = df[numerical_features].fillna(0)
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Handle missing categorical values
df['Category'] = df['Category'].fillna('Unknown')
df['Success Category'] = df['Success Category'].fillna('Unknown')

# Apply Feature Weighting (Excluding Price)
df['Weighted_Rating'] = df['Rating'] * 0.5
df['Weighted_Sentiment_Polarity'] = df['Sentiment_Polarity'] * 0.3
df['Weighted_Reviews_Installs'] = df['User_Engagement'] * 0.1
df['Weighted_Discrepancy'] = df['Discrepancy'] * -0.2

# Combine numerical and categorical features
features = ['Category', 'Success Category', 'Weighted_Rating', 'Weighted_Sentiment_Polarity',
            'Weighted_Reviews_Installs', 'Weighted_Discrepancy']
feature_matrix = pd.get_dummies(df[features])

# Compute Similarity
similarity_matrix = cosine_similarity(feature_matrix)

# Map DataFrame indices to the feature matrix indices
df_to_feature_matrix_indices = {index: idx for idx, index in enumerate(df.index)}

# Recommendation Function with Hybrid Approach
def content_based_recommendations_by_category(category, n_recommendations=5, discrepancy_threshold=0.2):
    # Step 1: Filter apps in the specified category
    category_apps = df[(df['Category'] == category) &
                       (df['Discrepancy'] <= discrepancy_threshold)]
    if category_apps.empty:
        return f"No apps found in the category '{category}' with discrepancy <= {discrepancy_threshold}."

    # Step 2: Map the indices of category apps to the feature matrix
    category_indices = [df_to_feature_matrix_indices[idx] for idx in category_apps.index]

    # Step 3: Compute mean similarity scores for apps in the category
    scores = similarity_matrix[category_indices].mean(axis=0)

    # Step 4: Sort apps by similarity score
    top_indices = scores.argsort()[::-1][:n_recommendations]  # Indices of top scores
    top_apps = df.iloc[top_indices][['App', 'Category', 'Discrepancy']]  # Relevant columns

    # Update the index to start from 1
    top_apps.index = range(1, len(top_apps) + 1)

    return top_apps


# '''Graph'''
def barPlotDrawer(x, y):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x=x, y=y, palette="viridis", ax=ax)
    ax.set_title(f"Bar Plot of {y} by {x}", fontsize=16, weight='bold')
    ax.set_xlabel(x, fontsize=14)
    ax.set_ylabel(y, fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Adding annotations
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=10)
    
    st.pyplot(fig)


def scatterPlotDrawer(x, y, hue):
    fig = plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.xticks(rotation=90)
    plt.title(f"Scatter Plot of {y} vs {x}")
    st.pyplot(fig)

def lineChartDrawer(x, y):
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=x, y=y)
    plt.xticks(rotation=90)
    plt.title(f"Line Chart of {y} by {x}")
    st.pyplot(fig)

def linePlotDrawer(x_column, y_column, y_axis_metric):
    # Group data by selected x_column and calculate the chosen metric for the y_column
    if y_axis_metric == "sum":
        grouped_data = df.groupby(x_column)[y_column].sum()
    elif y_axis_metric == "mean":
        grouped_data = df.groupby(x_column)[y_column].mean()

    # Create a line plot
    fig = plt.figure(figsize=(12, 6))
    plt.plot(grouped_data.index, grouped_data.values, linestyle='--', color='r', marker='o')
    plt.title(f"{y_column} by {x_column} ({y_axis_metric.capitalize()})", fontsize=14)
    plt.xlabel(x_column, fontsize=12)
    plt.ylabel(y_column, fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

def countPlotDrawer(x):
    fig = plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x=x, order=df[x].value_counts().index, palette='Set2')
    plt.xticks(rotation=90)
    plt.title(f"Count Plot of {x}")
    plt.xlabel(x, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    st.pyplot(fig)


def main():
    # Themed Headers for pages
    html_temp_pred = """
    <div style="background-color:#34A853;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;font-family:Arial, sans-serif;"> Predict App Success </h2>
    </div>
    """

    html_temp_recommend = """
    <div style="background-color:#4285F4;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;font-family:Arial, sans-serif;"> App Recommendation </h2>
    </div>
    """

    html_temp_vis = """
    <div style="background-color:#E8A800;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;font-family:Arial, sans-serif;"> Visualize Google App Properties </h2>
    </div>
    """

    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #F4F4F4;
        }
        </style>
    """, unsafe_allow_html=True)
    select_type = st.sidebar.radio("Select a Page", ['Graph', 'Predict App Success', 'Recommendation'])

    if select_type == 'Graph':
        st.markdown(html_temp_vis, unsafe_allow_html=True)
        st.subheader("Relation between features: ")
        plot_type = st.selectbox("Select plot type", ["Bar Plot", "Scatter Plot",  "Count Plot", "Line Plot"])


        if plot_type == "Bar Plot":
            x = st.selectbox("Select a feature for x-axis", ("Content_Rating"))
            y = st.selectbox("Select a feature for y-axis", ("Reviews", "Installs", "Price", "Sentiment_Polarity"))
            if st.button("Visualize", key="bar_plot"):
                barPlotDrawer(x, y)

        elif plot_type == "Scatter Plot":
            variables = ["Rating", "Reviews", "Sentiment_Polarity", "Sentiment_Subjectivity", "Discrepancy"]
            x = st.selectbox("Select a feature for x-axis", variables)
            y = st.selectbox("Select a feature for y-axis", variables)
            hue = st.selectbox("Select a feature for color coding (optional)", ["Type", "Content_Rating", "Success Category"], index=0)
            if st.button("Visualize", key="scatter_plot"):
                scatterPlotDrawer(x, y, hue)

        elif plot_type == "Line Plot":
            x_column = st.selectbox("Select a feature for the x-axis (categorical)", ["Updated_Month", "Updated_Year", "Category"])
            y_column = st.selectbox("Select a feature for the y-axis (numerical)", ["Price", "Reviews", "Rating"])
            y_axis_metric = st.selectbox("Choose a metric for the line plot", ["sum", "mean"], help="Select whether to sum or average the y-axis values.")
            if st.button("Visualize", key="line_plot"):
                linePlotDrawer(x_column, y_column, y_axis_metric)

        elif plot_type == "Count Plot":
            x = st.selectbox("Select a feature for x-axis", ["Category", "Type", "Content_Rating", "Genres", "Success Category"])
            if st.button("Visualize", key="count_plot"):
                countPlotDrawer(x)

    elif select_type == 'Predict App Success':
        st.markdown(html_temp_pred, unsafe_allow_html=True)
        st.subheader("Drag to provide input values:")

        # Feature sliders
        input_data = {
            "Rating": st.slider('Rating', 0.0, 5.0, step=0.1),
            "Reviews": st.slider('Reviews', 0, int(df['Reviews'].max()), step=1000),
            "Installs": st.slider('Installs', 0, int(df['Installs'].max()), step=1000),
            "Price": st.slider('Price', 0.0, float(df['Price'].max()), step=0.01),
            "Sentiment_Polarity": st.slider('Sentiment Polarity', -1.0, 1.0, step=0.1),
            "Category": st.selectbox("Category", df["Category"].unique()),
        }

        # Model selection
        model_choice = st.selectbox("Select a Model", ["Random Forest", "Decision Tree"])

        # Predict button
        if st.button("Predict"):
            # Choose the model
            model = random_forest_model if model_choice == "Random Forest" else decision_tree_model

            # Perform prediction
            prediction = predict_app_success(model, input_data, feature_names)

            # Map prediction to labels
            prediction_label = {
                2: "Unsuccessful",
                1: "Successful",
                0: "Moderately Successful"
            }.get(prediction, "Unknown")  # Default to "Unknown" if prediction is out of bounds

            # Display prediction
            st.success(f"The predicted success category for the app is: {prediction_label}")

    elif select_type == 'Recommendation':
            st.markdown(html_temp_recommend, unsafe_allow_html=True)
            st.subheader("Get recommendations for Successful apps:")

            # Inputs for recommendation
            selected_category = st.selectbox("Select a Category", df['Category'].unique())
            num_recommendations = st.number_input("Number of Recommendations", min_value=1, max_value=10, value=5)
            discrepancy_threshold = st.slider("Set Maximum Discrepancy Threshold", 0.0, 1.0, 0.2)

            # Recommendation Button
            if st.button("Get Recommendations"):
                recommendations = content_based_recommendations_by_category(
                    selected_category, int(num_recommendations), discrepancy_threshold
                )
                
                if isinstance(recommendations, str):
                    st.warning(recommendations)
                else:
                    st.write("Here are your recommended apps:")
                    st.dataframe(recommendations)


if __name__ == "__main__":
    main()
