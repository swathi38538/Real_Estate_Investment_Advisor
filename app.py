# ============================================================
# REAL ESTATE INVESTMENT ADVISOR
# Predicting Property Profitability & Future Value
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# 2. LOAD DATASET AND MODELS
# ============================================================

df = pd.read_csv("data/india_housing_prices.csv")

# ============================================================
# CREATE ENGINEERED FEATURES USED BY THE APP
# ============================================================

# Number of amenities for each property
df["Amenity_Density_Score"] = (
    df["Amenities"]
    .fillna("")
    .str.split(",")
    .str.len()
)

# Good Investment feature used by Q20.
# This follows the same feature-engineering logic used during ML development.
median_price = df["Price_in_Lakhs"].median()
median_price_per_sqft = df["Price_per_SqFt"].median()
median_amenities = df["Amenity_Density_Score"].median()

price_condition = df["Price_in_Lakhs"] <= median_price
price_per_sqft_condition = (
    df["Price_per_SqFt"] <= median_price_per_sqft
)
amenity_condition = (
    df["Amenity_Density_Score"] >= median_amenities
)

investment_score = (
    price_condition.astype(int)
    + price_per_sqft_condition.astype(int)
    + amenity_condition.astype(int)
)

df["Good_Investment"] = (investment_score >= 2).astype(int)

# Selected deployment models
classification_model = joblib.load(
    "models/random_forest_classifier.pkl"
)

regression_model = joblib.load(
    "models/linear_regression_model.pkl"
)

# Preprocessing objects
classification_preprocessor = joblib.load(
    "models/classification_preprocessor.pkl"
)

regression_preprocessor = joblib.load(
    "models/regression_preprocessor.pkl"
)


# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🏠 Real Estate Investment Advisor")

page = st.sidebar.radio(
    "Select Page",
    [
        "Introduction",
        "EDA Visualizations",
        "Prediction"
    ]
)


# ============================================================
# 4. INTRODUCTION PAGE
# ============================================================

if page == "Introduction":

    st.title("🏠 Real Estate Investment Advisor")

    st.subheader(
        "Predicting Property Profitability & Future Value"
    )

    st.write(
        """
        This application uses Machine Learning to analyze real estate
        properties and provide investment insights.
        """
    )

    st.markdown("---")

    st.header("🎯 Project Overview")

    st.write(
        """
        The Real Estate Investment Advisor is a Machine Learning
        application designed to help analyze residential properties.
        """
    )

    st.write(
        """
        The application performs two main predictions:
        """
    )

    st.markdown(
        """
        **1. Classification**

        Predict whether a property is a **Good Investment** or
        **Not a Good Investment**.

        **2. Regression**

        Estimate the property's **Future Price After 5 Years**.
        """
    )

    st.markdown("---")

    st.header("📌 Project Objectives")

    st.markdown(
        """
        - Analyze real estate property data.
        - Perform exploratory data analysis.
        - Create useful features for Machine Learning.
        - Predict property investment potential.
        - Estimate future property value.
        - Provide visual insights through an interactive dashboard.
        """
    )

    st.markdown("---")

    st.header("🛠️ Technologies Used")

    st.markdown(
        """
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - XGBoost
        - MLflow
        - Streamlit
        - Plotly
        """
    )

    st.markdown("---")

    st.info(
        "Use the sidebar to explore the EDA visualizations "
        "or make a property prediction."
    )
    # ============================================================
# EDA VISUALIZATIONS
# ============================================================

elif page == "EDA Visualizations":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "Explore the major patterns, relationships, and investment-related "
        "factors identified during the EDA."
    )

    # --------------------------------------------------------
    # Create four EDA divisions
    # --------------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Property & Price Analysis",
        "2️⃣ Location & Property Characteristics",
        "3️⃣ Relationships & Correlation",
        "4️⃣ Investment Factors"
    ])

    # ========================================================
    # DIVISION 1
    # QUESTIONS 1–5
    # ========================================================

    with tab1:

        st.header("Property & Price Analysis")

        # ----------------------------------------------------
        # Question 1
        # Distribution of Property Prices
        # ----------------------------------------------------

        st.subheader("Q1. Distribution of Property Prices")

        fig1 = px.histogram(
            df,
            x="Price_in_Lakhs",
            nbins=50,
            title="Distribution of Property Prices",
            labels={
                "Price_in_Lakhs": "Property Price (Lakhs)"
            }
        )

        fig1.update_layout(
            xaxis_title="Property Price (Lakhs)",
            yaxis_title="Number of Properties"
        )

        st.plotly_chart(fig1, use_container_width=True)

        # ----------------------------------------------------
        # Question 2
        # Distribution of Property Sizes
        # ----------------------------------------------------

        st.subheader("Q2. Distribution of Property Sizes")

        fig2 = px.histogram(
            df,
            x="Size_in_SqFt",
            nbins=50,
            title="Distribution of Property Sizes",
            labels={
                "Size_in_SqFt": "Property Size (Sq Ft)"
            }
        )

        fig2.update_layout(
            xaxis_title="Property Size (Sq Ft)",
            yaxis_title="Number of Properties"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ----------------------------------------------------
        # Question 3
        # Price per Sq Ft by Property Type
        # ----------------------------------------------------

        st.subheader("Q3. Price per Sq Ft by Property Type")

        property_type_price = (
            df.groupby("Property_Type")["Price_per_SqFt"]
            .mean()
            .reset_index()
            .sort_values("Price_per_SqFt", ascending=False)
        )

        fig3 = px.bar(
            property_type_price,
            x="Property_Type",
            y="Price_per_SqFt",
            title="Average Price per Sq Ft by Property Type",
            labels={
                "Property_Type": "Property Type",
                "Price_per_SqFt": "Average Price per Sq Ft (Lakhs)"
            }
        )

        st.plotly_chart(fig3, use_container_width=True)

        # ----------------------------------------------------
        # Question 4
        # Property Size vs Price
        # ----------------------------------------------------

        st.subheader("Q4. Relationship Between Property Size and Price")

        fig4 = px.scatter(
            df.sample(min(10000, len(df)), random_state=42),
            x="Size_in_SqFt",
            y="Price_in_Lakhs",
            title="Property Size vs Property Price",
            labels={
                "Size_in_SqFt": "Property Size (Sq Ft)",
                "Price_in_Lakhs": "Price (Lakhs)"
            },
            opacity=0.5
        )

        st.plotly_chart(fig4, use_container_width=True)

        # ----------------------------------------------------
        # Question 5
        # Outliers
        # ----------------------------------------------------

        st.subheader("Q5. Outliers in Property Size and Price per Sq Ft")

        col1, col2 = st.columns(2)

        with col1:

            fig5a = px.box(
                df,
                y="Size_in_SqFt",
                title="Property Size Outliers",
                labels={
                    "Size_in_SqFt": "Size (Sq Ft)"
                }
            )

            st.plotly_chart(fig5a, use_container_width=True)

        with col2:

            fig5b = px.box(
                df,
                y="Price_per_SqFt",
                title="Price per Sq Ft Outliers",
                labels={
                    "Price_per_SqFt": "Price per Sq Ft (Lakhs)"
                }
            )

            st.plotly_chart(fig5b, use_container_width=True)


    # ========================================================
    # DIVISION 2
    # QUESTIONS 6–10
    # ========================================================

    with tab2:

        st.header("Location & Property Characteristics")

        # ----------------------------------------------------
        # Question 6
        # Average Price per Sq Ft by State
        # ----------------------------------------------------

        st.subheader("Q6. Average Price per Sq Ft by State")

        state_price = (
            df.groupby("State")["Price_per_SqFt"]
            .mean()
            .reset_index()
            .sort_values("Price_per_SqFt", ascending=False)
        )

        fig6 = px.bar(
            state_price,
            x="State",
            y="Price_per_SqFt",
            title="Average Price per Sq Ft by State",
            labels={
                "State": "State",
                "Price_per_SqFt": "Average Price per Sq Ft (Lakhs)"
            }
        )

        st.plotly_chart(fig6, use_container_width=True)

        # ----------------------------------------------------
        # Question 7
        # Average Property Price by City
        # ----------------------------------------------------

        st.subheader("Q7. Average Property Price by City")

        city_price = (
            df.groupby("City")["Price_in_Lakhs"]
            .mean()
            .reset_index()
            .sort_values("Price_in_Lakhs", ascending=False)
        )

        fig7 = px.bar(
            city_price.head(15),
            x="City",
            y="Price_in_Lakhs",
            title="Average Property Price by City (Top 15)",
            labels={
                "City": "City",
                "Price_in_Lakhs": "Average Price (Lakhs)"
            }
        )

        st.plotly_chart(fig7, use_container_width=True)

        # ----------------------------------------------------
        # Question 8
        # Median Age by Locality
        # ----------------------------------------------------

        st.subheader("Q8. Median Age of Properties by Locality")

        locality_age = (
            df.groupby("Locality")["Age_of_Property"]
            .median()
            .reset_index()
            .sort_values("Age_of_Property", ascending=False)
        )

        fig8 = px.bar(
            locality_age.head(15),
            x="Locality",
            y="Age_of_Property",
            title="Median Property Age by Locality (Top 15)",
            labels={
                "Locality": "Locality",
                "Age_of_Property": "Median Age (Years)"
            }
        )

        st.plotly_chart(fig8, use_container_width=True)

        # ----------------------------------------------------
        # Question 9
        # BHK Distribution Across Cities
        # ----------------------------------------------------

        st.subheader("Q9. BHK Distribution Across Cities")

        bhk_city = (
            df.groupby(["City", "BHK"])
            .size()
            .reset_index(name="Property_Count")
        )

        fig9 = px.bar(
            bhk_city,
            x="City",
            y="Property_Count",
            color="BHK",
            title="BHK Distribution Across Cities",
            labels={
                "City": "City",
                "Property_Count": "Number of Properties",
                "BHK": "BHK"
            },
            barmode="group"
        )

        st.plotly_chart(fig9, use_container_width=True)

        # ----------------------------------------------------
        # Question 10
        # Top 5 Most Expensive Localities
        # ----------------------------------------------------

        st.subheader("Q10. Top 5 Most Expensive Localities")

        locality_price = (
            df.groupby("Locality")["Price_in_Lakhs"]
            .mean()
            .reset_index()
            .sort_values("Price_in_Lakhs", ascending=False)
        )

        top5_localities = locality_price.head(5)

        fig10 = px.bar(
            top5_localities,
            x="Locality",
            y="Price_in_Lakhs",
            title="Top 5 Most Expensive Localities",
            labels={
                "Locality": "Locality",
                "Price_in_Lakhs": "Average Price (Lakhs)"
            }
        )

        st.plotly_chart(fig10, use_container_width=True)

        st.info(
            "The dataset does not contain a property listing/sale date column, "
            "so a true time-based price trend cannot be plotted. "
            "This visualization shows the top 5 localities by average property price."
        )


    # ========================================================
    # DIVISION 3
    # QUESTIONS 11–15
    # ========================================================

    with tab3:

        st.header("Relationships & Correlation")

        # ----------------------------------------------------
        # Question 11
        # Numeric Feature Correlation
        # ----------------------------------------------------

        st.subheader("Q11. Correlation Between Numeric Features")

        numeric_columns = [
            "BHK",
            "Size_in_SqFt",
            "Price_in_Lakhs",
            "Price_per_SqFt",
            "Year_Built",
            "Floor_No",
            "Total_Floors",
            "Age_of_Property",
            "Nearby_Schools",
            "Nearby_Hospitals",
            "Amenity_Density_Score"
        ]

        correlation_matrix = df[numeric_columns].corr()

        fig11 = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix of Numeric Features"
        )

        st.plotly_chart(fig11, use_container_width=True)

        # ----------------------------------------------------
        # Question 12
        # Nearby Schools vs Price per Sq Ft
        # ----------------------------------------------------

        st.subheader("Q12. Nearby Schools vs Price per Sq Ft")

        schools_price = (
            df.groupby("Nearby_Schools")["Price_per_SqFt"]
            .mean()
            .reset_index()
        )

        fig12 = px.line(
            schools_price,
            x="Nearby_Schools",
            y="Price_per_SqFt",
            markers=True,
            title="Nearby Schools vs Average Price per Sq Ft",
            labels={
                "Nearby_Schools": "Nearby Schools",
                "Price_per_SqFt": "Average Price per Sq Ft (Lakhs)"
            }
        )

        st.plotly_chart(fig12, use_container_width=True)

        # ----------------------------------------------------
        # Question 13
        # Nearby Hospitals vs Price per Sq Ft
        # ----------------------------------------------------

        st.subheader("Q13. Nearby Hospitals vs Price per Sq Ft")

        hospitals_price = (
            df.groupby("Nearby_Hospitals")["Price_per_SqFt"]
            .mean()
            .reset_index()
        )

        fig13 = px.line(
            hospitals_price,
            x="Nearby_Hospitals",
            y="Price_per_SqFt",
            markers=True,
            title="Nearby Hospitals vs Average Price per Sq Ft",
            labels={
                "Nearby_Hospitals": "Nearby Hospitals",
                "Price_per_SqFt": "Average Price per Sq Ft (Lakhs)"
            }
        )

        st.plotly_chart(fig13, use_container_width=True)

        # ----------------------------------------------------
        # Question 14
        # Furnished Status vs Price
        # ----------------------------------------------------

        st.subheader("Q14. Price by Furnished Status")

        furnished_price = (
            df.groupby("Furnished_Status")["Price_in_Lakhs"]
            .mean()
            .reset_index()
            .sort_values("Price_in_Lakhs", ascending=False)
        )

        fig14 = px.bar(
            furnished_price,
            x="Furnished_Status",
            y="Price_in_Lakhs",
            title="Average Property Price by Furnished Status",
            labels={
                "Furnished_Status": "Furnished Status",
                "Price_in_Lakhs": "Average Price (Lakhs)"
            }
        )

        st.plotly_chart(fig14, use_container_width=True)

        # ----------------------------------------------------
        # Question 15
        # Facing Direction vs Price per Sq Ft
        # ----------------------------------------------------

        st.subheader("Q15. Price per Sq Ft by Facing Direction")

        facing_price = (
            df.groupby("Facing")["Price_per_SqFt"]
            .mean()
            .reset_index()
            .sort_values("Price_per_SqFt", ascending=False)
        )

        fig15 = px.bar(
            facing_price,
            x="Facing",
            y="Price_per_SqFt",
            title="Average Price per Sq Ft by Facing Direction",
            labels={
                "Facing": "Facing Direction",
                "Price_per_SqFt": "Average Price per Sq Ft (Lakhs)"
            }
        )

        st.plotly_chart(fig15, use_container_width=True)


    # ========================================================
    # DIVISION 4
    # QUESTIONS 16–20
    # ========================================================

    with tab4:

        st.header("Investment Factors")

        # ----------------------------------------------------
        # Question 16
        # Owner Type
        # ----------------------------------------------------

        st.subheader("Q16. Properties by Owner Type")

        owner_count = (
            df["Owner_Type"]
            .value_counts()
            .reset_index()
        )

        owner_count.columns = ["Owner_Type", "Property_Count"]

        fig16 = px.bar(
            owner_count,
            x="Owner_Type",
            y="Property_Count",
            title="Number of Properties by Owner Type",
            labels={
                "Owner_Type": "Owner Type",
                "Property_Count": "Number of Properties"
            }
        )

        st.plotly_chart(fig16, use_container_width=True)

        # ----------------------------------------------------
        # Question 17
        # Availability Status
        # ----------------------------------------------------

        st.subheader("Q17. Properties by Availability Status")

        availability_count = (
            df["Availability_Status"]
            .value_counts()
            .reset_index()
        )

        availability_count.columns = [
            "Availability_Status",
            "Property_Count"
        ]

        fig17 = px.bar(
            availability_count,
            x="Availability_Status",
            y="Property_Count",
            title="Number of Properties by Availability Status",
            labels={
                "Availability_Status": "Availability Status",
                "Property_Count": "Number of Properties"
            }
        )

        st.plotly_chart(fig17, use_container_width=True)

        # ----------------------------------------------------
        # Question 18
        # Parking Space vs Property Price
        # ----------------------------------------------------

        st.subheader("Q18. Does Parking Space Affect Property Price?")

        parking_price = (
            df.groupby("Parking_Space")["Price_in_Lakhs"]
            .mean()
            .reset_index()
        )

        fig18 = px.bar(
            parking_price,
            x="Parking_Space",
            y="Price_in_Lakhs",
            title="Average Property Price by Parking Availability",
            labels={
                "Parking_Space": "Parking Space",
                "Price_in_Lakhs": "Average Price (Lakhs)"
            }
        )

        st.plotly_chart(fig18, use_container_width=True)

        # ----------------------------------------------------
        # Question 19
        # Amenities vs Price per Sq Ft
        # ----------------------------------------------------

        st.subheader("Q19. Amenities vs Price per Sq Ft")

        # Amenity_Density_Score was created above from the Amenities column
        amenity_price = (
            df.groupby("Amenity_Density_Score")["Price_per_SqFt"]
            .mean()
            .reset_index()
        )

        fig19 = px.bar(
            amenity_price,
            x="Amenity_Density_Score",
            y="Price_per_SqFt",
            title="Amenity Density vs Average Price per Sq Ft",
            labels={
                "Amenity_Density_Score": "Amenity Density Score",
                "Price_per_SqFt": "Average Price per Sq Ft (Lakhs)"
            }
        )

        st.plotly_chart(fig19, use_container_width=True)

        # ----------------------------------------------------
        # Question 20
        # Public Transport Accessibility
        # ----------------------------------------------------

        st.subheader(
            "Q20. Public Transport Accessibility vs Price per Sq Ft"
        )

        transport_price = (
            df.groupby("Public_Transport_Accessibility")
            .agg(
                Average_Price_per_SqFt=(
                    "Price_per_SqFt",
                    "mean"
                ),
                Investment_Rate=(
                    "Good_Investment",
                    "mean"
                )
            )
            .reset_index()
        )

        transport_price["Investment_Rate"] = (
            transport_price["Investment_Rate"] * 100
        )

        fig20 = px.bar(
            transport_price,
            x="Public_Transport_Accessibility",
            y="Average_Price_per_SqFt",
            title="Public Transport Accessibility vs Average Price per Sq Ft",
            labels={
                "Public_Transport_Accessibility":
                    "Public Transport Accessibility",
                "Average_Price_per_SqFt":
                    "Average Price per Sq Ft (Lakhs)"
            }
        )

        st.plotly_chart(fig20, use_container_width=True)

        # Show investment rate as a small supporting table
        st.write("Investment Rate by Public Transport Accessibility")

        investment_rate_display = transport_price[
            [
                "Public_Transport_Accessibility",
                "Investment_Rate"
            ]
        ].copy()

        investment_rate_display["Investment_Rate"] = (
            investment_rate_display["Investment_Rate"]
            .round(2)
        )

        st.dataframe(
            investment_rate_display,
            use_container_width=True,
            hide_index=True
        )
# ============================================================
# 6. PREDICTION PAGE
# ============================================================

elif page == "Prediction":

    st.title("🔮 Property Investment Prediction")

    st.write(
        """
        Enter the property details below and click the
        Predict button to get both investment and future
        price predictions.
        """
    )

    st.markdown("---")


    # ========================================================
    # PROPERTY INPUT FORM
    # ========================================================

    with st.form("property_prediction_form"):

        st.header("🏠 Property Details")

        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # COLUMN 1
        # ----------------------------------------------------

        with col1:

            state = st.selectbox(
                "State",
                sorted(
                    df["State"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            city = st.selectbox(
                "City",
                sorted(
                    df["City"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            locality = st.selectbox(
                "Locality",
                sorted(
                    df["Locality"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            property_type = st.selectbox(
                "Property Type",
                sorted(
                    df["Property_Type"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            bhk = st.number_input(
                "BHK",
                min_value=1,
                max_value=10,
                value=2,
                step=1
            )

            size_sqft = st.number_input(
                "Size (SqFt)",
                min_value=100.0,
                max_value=10000.0,
                value=1000.0,
                step=50.0
            )


        # ----------------------------------------------------
        # COLUMN 2
        # ----------------------------------------------------

        with col2:

            price_lakhs = st.number_input(
                "Current Price (Lakhs)",
                min_value=1.0,
                max_value=1000.0,
                value=50.0,
                step=1.0
            )

            year_built = st.number_input(
                "Year Built",
                min_value=1950,
                max_value=2026,
                value=2015,
                step=1
            )

            furnished_status = st.selectbox(
                "Furnished Status",
                sorted(
                    df["Furnished_Status"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            floor_no = st.number_input(
                "Floor Number",
                min_value=0,
                max_value=50,
                value=2,
                step=1
            )

            total_floors = st.number_input(
                "Total Floors",
                min_value=1,
                max_value=100,
                value=5,
                step=1
            )

            nearby_schools = st.number_input(
                "Nearby Schools",
                min_value=0,
                max_value=20,
                value=3,
                step=1
            )


        # ----------------------------------------------------
        # COLUMN 3
        # ----------------------------------------------------

        with col3:

            nearby_hospitals = st.number_input(
                "Nearby Hospitals",
                min_value=0,
                max_value=20,
                value=2,
                step=1
            )

            public_transport = st.selectbox(
                "Public Transport Accessibility",
                sorted(
                    df["Public_Transport_Accessibility"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            parking_space = st.selectbox(
                "Parking Space",
                sorted(
                    df["Parking_Space"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            security = st.selectbox(
                "Security",
                sorted(
                    df["Security"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            amenities = st.selectbox(
                "Amenities",
                sorted(
                    df["Amenities"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            facing = st.selectbox(
                "Facing",
                sorted(
                    df["Facing"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            owner_type = st.selectbox(
                "Owner Type",
                sorted(
                    df["Owner_Type"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            availability_status = st.selectbox(
                "Availability Status",
                sorted(
                    df["Availability_Status"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )


        # ----------------------------------------------------
        # PREDICT BUTTON
        # ----------------------------------------------------

        submit_prediction = st.form_submit_button(
            "🔮 Predict"
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if submit_prediction:

        # ----------------------------------------------------
        # Derived Features
        # ----------------------------------------------------

        # Price per square foot
        price_per_sqft = (
            price_lakhs / size_sqft
        )

        # Age of property
        current_year = 2026

        calculated_age = (
            current_year - year_built
        )

        # Number of amenities
        amenity_density = len(
            [
                x
                for x in amenities.split(",")
                if x.strip()
            ]
        )


        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "BHK": [bhk],

            "Size_in_SqFt": [size_sqft],

            "Price_in_Lakhs": [price_lakhs],

            "Price_per_SqFt": [price_per_sqft],

            "State": [state],

            "City": [city],

            "Locality": [locality],

            "Property_Type": [property_type],

            "Furnished_Status": [furnished_status],

            "Year_Built": [year_built],

            "Floor_No": [floor_no],

            "Total_Floors": [total_floors],

            "Age_of_Property": [calculated_age],

            "Nearby_Schools": [nearby_schools],

            "Nearby_Hospitals": [nearby_hospitals],

            "Public_Transport_Accessibility": [
                public_transport
            ],

            "Parking_Space": [parking_space],

            "Security": [security],

            "Amenities": [amenities],

            "Facing": [facing],

            "Owner_Type": [owner_type],

            "Availability_Status": [
                availability_status
            ],

            "Amenity_Density_Score": [
                amenity_density
            ]
        })


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        classification_input = (
            classification_preprocessor.transform(
                input_data
            )
        )

        classification_prediction = (
            classification_model.predict(
                classification_input
            )[0]
        )

        classification_probability = (
            classification_model.predict_proba(
                classification_input
            )[0]
        )

        confidence = (
            classification_probability[
                classification_prediction
            ] * 100
        )


        # ====================================================
        # REGRESSION
        # ====================================================

        regression_input = (
            regression_preprocessor.transform(
                input_data
            )
        )

        future_price_prediction = (
            regression_model.predict(
                regression_input
            )[0]
        )


        # ====================================================
        # PRICE GROWTH
        # ====================================================

        price_growth = (
            (
                future_price_prediction
                - price_lakhs
            )
            / price_lakhs
        ) * 100


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.markdown("---")

        st.header("📊 Prediction Results")


        # ----------------------------------------------------
        # INVESTMENT DECISION
        # ----------------------------------------------------

        if classification_prediction == 1:

            st.success(
                "✅ Investment Decision: "
                "GOOD INVESTMENT"
            )

        else:

            st.warning(
                "⚠️ Investment Decision: "
                "NOT A GOOD INVESTMENT"
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        # ----------------------------------------------------
        # FUTURE PRICE
        # ----------------------------------------------------

        st.subheader(
            "💰 Estimated Future Price After 5 Years"
        )

        st.metric(
            "Future Price",
            f"₹ {future_price_prediction:.2f} Lakhs"
        )


        # ----------------------------------------------------
        # PRICE GROWTH
        # ----------------------------------------------------

        st.metric(
            "Price Growth",
            f"{price_growth:.2f}%"
        )


        # ----------------------------------------------------
        # INPUT SUMMARY
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader("📋 Property Information")

        summary_col1, summary_col2, summary_col3 = (
            st.columns(3)
        )

        with summary_col1:

            st.write(
                f"**State:** {state}"
            )

            st.write(
                f"**City:** {city}"
            )

            st.write(
                f"**Locality:** {locality}"
            )

            st.write(
                f"**Property Type:** {property_type}"
            )

        with summary_col2:

            st.write(
                f"**BHK:** {bhk}"
            )

            st.write(
                f"**Size:** {size_sqft:.0f} SqFt"
            )

            st.write(
                f"**Current Price:** "
                f"₹ {price_lakhs:.2f} Lakhs"
            )

            st.write(
                f"**Price/SqFt:** "
                f"{price_per_sqft:.4f} Lakhs"
            )

        with summary_col3:

            st.write(
                f"**Year Built:** {year_built}"
            )

            st.write(
                f"**Age:** {calculated_age} years"
            )

            st.write(
                f"**BHK:** {bhk}"
            )

            st.write(
                f"**Amenities:** {amenity_density}"
            )