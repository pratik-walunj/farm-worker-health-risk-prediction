from src.data_loader import load_data
from src.analysis import (
    compute_health_risk_index,
    correlation_analysis,
    add_risk_category,
    train_ml_model,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    generate_pdf_report,
    live_sensor_simulation,
    save_model,
    load_saved_model,
    simulate_real_time_prediction
)
from src.visualization import (
    plot_health_risk_distribution,
    plot_temperature_vs_risk
)


def main():

    print("\n--- Smart Farm Worker Health Monitoring System ---\n")

    # 1️⃣ Load dataset
    df = load_data("data/farm_worker_health.csv")

    # 2️⃣ Compute Health Risk Index
    df = compute_health_risk_index(df)

    # 3️⃣ Correlation Analysis
    correlation_analysis(df)

    # 4️⃣ Add Risk Category
    df = add_risk_category(df)

    # 5️⃣ Visualization
    plot_health_risk_distribution(df)
    plot_temperature_vs_risk(df)

    # 6️⃣ Train Machine Learning Model
    model, feature_names, X_test, y_test, accuracy = train_ml_model(df)

    # 7️⃣ Feature Importance Graph
    plot_feature_importance(model, feature_names)

    # 8️⃣ Confusion Matrix
    plot_confusion_matrix(model, X_test, y_test)

    # 9️⃣ ROC Curve
    plot_roc_curve(model, X_test, y_test)

    # 🔟 Generate PDF Report
    generate_pdf_report(accuracy)

    # 1️⃣1️⃣ Save Model
    save_model(model)

    # 1️⃣2️⃣ Load Model (Verification)
    loaded_model = load_saved_model()

    # 1️⃣3️⃣ Real-Time CLI Prediction
    simulate_real_time_prediction(loaded_model)

    # 1️⃣4️⃣ Live Sensor Simulation
    live_sensor_simulation(loaded_model)


if __name__ == "__main__":
    main()
