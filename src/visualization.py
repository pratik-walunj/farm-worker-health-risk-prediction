import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_visualizations(df):
    os.makedirs("outputs/figures", exist_ok=True)

    # 1. Heart Rate vs Temperature
    plt.figure()
    sns.scatterplot(
        x='Temperature', y='Heart_Rate',
        hue='Risk_Level', data=df
    )
    plt.title("Heart Rate vs Temperature with Risk Levels")
    plt.savefig("outputs/figures/fig1_hr_temp.png", dpi=300)
    plt.close()

    # 2. Heat Stress Distribution
    plt.figure()
    sns.histplot(df['Heat_Stress_Index'], kde=True)
    plt.title("Distribution of Heat Stress Index")
    plt.savefig("outputs/figures/fig2_hsi.png", dpi=300)
    plt.close()

    # 3. Risk Level Count
    plt.figure()
    sns.countplot(x='Risk_Level', data=df)
    plt.title("Risk Level Distribution")
    plt.savefig("outputs/figures/fig3_risk.png", dpi=300)
    plt.close()

    

def plot_health_risk_distribution(df):
    plt.figure()
    plt.hist(df["Health_Risk_Index"], bins=20)
    plt.xlabel("Health Risk Index")
    plt.ylabel("Number of Workers")
    plt.title("Distribution of Farm Worker Health Risk Index")
    plt.tight_layout()
    plt.show()


def plot_temperature_vs_risk(df):
    plt.figure()
    plt.scatter(df["Temperature"], df["Health_Risk_Index"])
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Health Risk Index")
    plt.title("Temperature vs Health Risk Index")
    plt.tight_layout()
    plt.show()

