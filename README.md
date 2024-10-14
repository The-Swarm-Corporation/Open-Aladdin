

# Open-Aladdin

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


## Example

```python
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

```