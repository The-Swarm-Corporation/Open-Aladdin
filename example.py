from open_aladdin.main import fetch_stock_data, AdvancedRealTimeRiskAssessment
import time

if __name__ == "__main__":
    # Example usage
    # from data_integration import fetch_stock_data

    tickers = [
        "AAPL",
        "GOOGL",
        "MSFT",
        "AMZN",
        "^GSPC",
    ]  # Including S&P 500 for market returns
    historical_data = {
        ticker: fetch_stock_data(ticker) for ticker in tickers
    }

    risk_assessor = AdvancedRealTimeRiskAssessment(historical_data)
    risk_assessor.start_continuous_training()

    try:
        # Run for a while to allow some training iterations
        time.sleep(60)

        # Perform risk assessment
        risk_results = risk_assessor.run_risk_assessment(
            forecast_horizon=4
        )  # 1-year forecast

        # Output results
        risk_assessor.output_results(risk_results, "json")
        risk_assessor.output_results(risk_results, "csv")

        # Print some results
        for ticker, measures in risk_results.items():
            print(f"\nRisk Assessment for {ticker}:")
            for measure, value in measures.items():
                if isinstance(value, list):
                    print(
                        f"{measure}: [showing first 5 values] {value[:5]}"
                    )
                else:
                    print(f"{measure}: {value:.4f}")

    finally:
        # Ensure we stop the continuous training when done
        risk_assessor.stop_continuous_training()
