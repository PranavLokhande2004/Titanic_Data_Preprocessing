\# Titanic Dataset — Data Cleaning \& Preprocessing



\## Project Overview



This project focuses on acquiring, exploring, cleaning, and preprocessing the Titanic dataset using Python.



The main objective is to prepare a real-world dataset for further data analysis and machine learning by identifying data-quality issues, handling missing values, examining potential outliers, creating useful features, and converting categorical data into numerical form.



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn

\- VS Code



\## Dataset



The project uses the public Titanic dataset containing passenger information.



Original dataset:



\- Rows: 891

\- Columns: 12



The dataset contains information such as passenger class, age, gender, fare, cabin, port of embarkation, and survival status.



\## Project Structure



```text

Titanic\_Data\_Preprocessing/

│

├── data/

│   ├── titanic\_raw.csv

│   ├── titanic\_cleaned.csv

│   └── titanic\_preprocessed.csv

│

├── notebooks/

│

├── outputs/

│   ├── missing\_values\_before\_cleaning.png

│   ├── age\_distribution.png

│   ├── fare\_outliers.png

│   ├── survival\_distribution.png

│   ├── survival\_by\_gender.png

│   ├── survival\_by\_class.png

│   ├── correlation\_heatmap.png

│   └── family\_size\_distribution.png

│

└── src/

&#x20;   ├── data\_exploration.py

&#x20;   ├── data\_cleaning.py

&#x20;   ├── verify\_cleaned\_data.py

&#x20;   ├── data\_preprocessing.py

&#x20;   ├── verify\_preprocessed\_data.py

&#x20;   └── data\_visualization.py


## Week 2: Exploratory Data Analysis & Visualization

### Objective

The objective of Week 2 was to perform Exploratory Data Analysis (EDA) on the cleaned Titanic dataset and identify important patterns, relationships, and anomalies using statistical analysis and data visualization.

### Tools and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

### EDA Performed

The following analyses were performed:

1. Statistical summary of numerical features
2. Passenger class distribution
3. Gender distribution
4. Survival distribution
5. Age distribution
6. Survival by gender
7. Survival by passenger class
8. Survival rate by age group
9. Family size distribution
10. Survival rate by family size
11. Fare distribution and outlier analysis
12. Correlation analysis using a heatmap

### Key Findings

- The overall survival rate was **38.38%**.
- Female passengers had a survival rate of **74.20%**, compared with **18.89%** for male passengers.
- First-class passengers had a survival rate of **62.96%**, compared with **47.28%** for second class and **24.24%** for third class.
- Children had a survival rate of **57.97%**, while senior passengers had a survival rate of **22.73%**.
- Family sizes of 2–4 members generally showed higher survival rates than passengers travelling alone or in larger families.
- The correlation between `Pclass` and `Survived` was **-0.3385**.
- The correlation between `Fare` and `Survived` was **0.2573**.
- The analysis identified **116 potential Fare outliers** using the IQR method.

### Visualizations

The following visualizations were generated and saved in the `week2_outputs` directory:

- Survival Distribution
- Survival by Gender
- Survival by Passenger Class
- Age Distribution
- Fare Distribution and Outliers
- Correlation Heatmap
- Family Size Distribution
- Survival Rate by Age Group

### Conclusion

The EDA showed that survival outcomes varied across several passenger characteristics. Gender and passenger class showed substantial differences in survival rates, while age and family size also showed observable patterns. Correlation analysis provided additional information about relationships between numerical features and survival. These findings can be useful for understanding the Titanic dataset and for further machine learning analysis.

### Week 2 Output Files

- `week2_eda/eda_analysis.py`
- `week2_outputs/eda_summary.csv`
- `week2_outputs/eda_findings.csv`
- `week2_outputs/age_distribution.png`
- `week2_outputs/correlation_heatmap.png`
- `week2_outputs/family_size_distribution.png`
- `week2_outputs/fare_outliers.png`
- `week2_outputs/survival_by_age_group.png`
- `week2_outputs/survival_by_class.png`
- `week2_outputs/survival_by_gender.png`
- `week2_outputs/survival_distribution.png`


