# file: advanced_risk_assessment.py

import csv
import json
import time
from datetime import datetime
from threading import Thread
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: Historical stock data.
    """
    logger.info(f"Fetching data for {ticker}")
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")
    return data


def store_historical_data(ticker: str, data: pd.DataFrame) -> None:
    """
    Store historical data for a given ticker.

    Args:
        ticker (str): Stock ticker symbol.
        data (pd.DataFrame): Historical stock data.
    """
    logger.info(f"Storing historical data for {ticker}")
    data.to_csv(f"{ticker}_historical_data.csv")


logger.add("advanced_risk_assessment.log", rotation="10 MB")


class AdvancedRealTimeRiskAssessment:
    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame],
        update_interval: int = 10,
    ):
        """
        Initialize the AdvancedRealTimeRiskAssessment class.

        Args:
            historical_data (Dict[str, pd.DataFrame]): Historical data for multiple stocks.
            update_interval (int): Interval for model updates in seconds.
        """
        self.historical_data = historical_data
        self.update_interval = update_interval
        self.risk_model: XGBRegressor = XGBRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        )
        self.scaler: StandardScaler = StandardScaler()
        self.is_training: bool = False
        self.stop_training: bool = False
        logger.info("AdvancedRealTimeRiskAssessment initialized")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the risk model.

        Args:
            data (pd.DataFrame): Historical stock data.

        Returns:
            pd.DataFrame: Prepared features.
        """
        features = data[
            ["Open", "High", "Low", "Close", "Volume"]
        ].copy()
        features["Returns"] = features["Close"].pct_change()
        features["Log_Returns"] = np.log(
            features["Close"] / features["Close"].shift(1)
        )
        features["Volatility"] = (
            features["Returns"].rolling(window=20).std()
        )
        features["MACD"] = self.calculate_macd(features["Close"])
        features["RSI"] = self.calculate_rsi(features["Close"])
        features["Beta"] = self.calculate_beta(features["Returns"])

        # Add more advanced features
        features["EMA_50"] = (
            features["Close"].ewm(span=50, adjust=False).mean()
        )
        features["SMA_200"] = (
            features["Close"].rolling(window=200).mean()
        )
        features["Bollinger_Upper"], features["Bollinger_Lower"] = (
            self.calculate_bollinger_bands(features["Close"])
        )

        features["Day_of_Week"] = pd.to_datetime(data.index).dayofweek
        features["Month"] = pd.to_datetime(data.index).month

        features.dropna(inplace=True)
        return features

    def calculate_macd(
        self,
        prices: pd.Series,
        short_window: int = 12,
        long_window: int = 26,
    ) -> pd.Series:
        """Calculate the Moving Average Convergence Divergence (MACD)."""
        short_ema = prices.ewm(span=short_window, adjust=False).mean()
        long_ema = prices.ewm(span=long_window, adjust=False).mean()
        return short_ema - long_ema

    def calculate_rsi(
        self, prices: pd.Series, window: int = 14
    ) -> pd.Series:
        """Calculate the Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (
            (delta.where(delta > 0, 0)).rolling(window=window).mean()
        )
        loss = (
            (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        )
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_beta(
        self, returns: pd.Series, market_returns: pd.Series = None
    ) -> float:
        """Calculate the beta of a stock."""
        if market_returns is None:
            sp500 = self.historical_data.get("^GSPC")
            if sp500 is None:
                logger.warning(
                    "S&P 500 data not available, using stock's own returns as market proxy"
                )
                market_returns = returns
            else:
                market_returns = sp500["Close"].pct_change()

        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance

    def calculate_bollinger_bands(
        self, prices: pd.Series, window: int = 20, num_std: float = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    def train_model(self) -> None:
        """Train the XGBoost model for risk prediction."""
        logger.info("Training risk model")
        all_features = pd.concat(
            [
                self.prepare_features(data)
                for data in self.historical_data.values()
            ]
        )

        X = all_features.drop(
            ["Returns", "Log_Returns", "Volatility"], axis=1
        )
        y = all_features["Volatility"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.risk_model.fit(X_train_scaled, y_train)

        logger.info(
            f"Model training completed. Test score: {self.risk_model.score(X_test_scaled, y_test)}"
        )

    def start_continuous_training(self) -> None:
        """Start continuous training in a separate thread."""
        if not self.is_training:
            self.stop_training = False
            self.is_training = True
            Thread(target=self._continuous_training_loop).start()
            logger.info("Continuous training started")
        else:
            logger.warning("Continuous training is already running")

    def stop_continuous_training(self) -> None:
        """Stop the continuous training loop."""
        self.stop_training = True
        logger.info("Continuous training stopped")

    def _continuous_training_loop(self) -> None:
        """Continuous training loop to be run in a separate thread."""
        while not self.stop_training:
            self.train_model()
            time.sleep(self.update_interval)
        self.is_training = False

    def predict_risk(
        self, data: pd.DataFrame, forecast_horizon: int = 30
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Predict risk measures for a given stock using the trained model and additional risk metrics.

        Args:
            data (pd.DataFrame): Historical stock data.
            forecast_horizon (int): Number of days to forecast risk.

        Returns:
            Dict[str, Union[float, List[float]]]: Dictionary of predicted risk measures.
        """
        features = self.prepare_features(data)
        model_input = features.drop(
            ["Returns", "Log_Returns", "Volatility"], axis=1
        )
        model_input_scaled = self.scaler.transform(model_input)

        predicted_volatility = self.risk_model.predict(
            model_input_scaled
        )

        returns = features["Returns"]
        log_returns = features["Log_Returns"]

        var = self.calculate_value_at_risk(returns)
        es = self.calculate_expected_shortfall(returns)
        beta = self.calculate_beta(returns)

        sharpe_ratio = (
            returns.mean() - 0.02
        ) / returns.std()  # Assuming 2% risk-free rate
        sortino_ratio = (returns.mean() - 0.02) / returns[
            returns < 0
        ].std()

        max_drawdown = (
            returns.cumsum() - returns.cumsum().cummax()
        ).min()

        # Long-term risk forecast
        long_term_forecast = self._forecast_risk(
            model_input_scaled, forecast_horizon
        )

        risk_measures = {
            "Current_Volatility": float(predicted_volatility[-1]),
            "Value_at_Risk": var,
            "Expected_Shortfall": es,
            "Beta": beta,
            "Sharpe_Ratio": sharpe_ratio,
            "Sortino_Ratio": sortino_ratio,
            "Max_Drawdown": max_drawdown,
            "Long_Term_Volatility_Forecast": long_term_forecast,
        }

        logger.info(f"Risk measures calculated: {risk_measures}")
        return risk_measures

    def _forecast_risk(
        self, current_features: np.ndarray, horizon: int
    ) -> List[float]:
        """
        Forecast risk for a given horizon.

        Args:
            current_features (np.ndarray): Current feature set.
            horizon (int): Number of days to forecast.

        Returns:
            List[float]: Forecasted risk values.
        """
        forecasted_risk = []
        current_prediction = current_features[-1].reshape(1, -1)

        for _ in range(horizon):
            risk = self.risk_model.predict(current_prediction)[0]
            forecasted_risk.append(float(risk))

            # Update prediction for next iteration (simplified approach)
            current_prediction = np.roll(current_prediction, -1)
            current_prediction[0, -1] = risk

        return forecasted_risk

    def calculate_value_at_risk(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculate Value at Risk (VaR)."""
        return float(
            np.percentile(returns, 100 * (1 - confidence_level))
        )

    def calculate_expected_shortfall(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculate Expected Shortfall (ES) or Conditional VaR."""
        var = self.calculate_value_at_risk(returns, confidence_level)
        return float(returns[returns <= var].mean())

    def run_risk_assessment(
        self, forecast_horizon: int = 30
    ) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Run the complete risk assessment for all stocks in the historical data.

        Args:
            forecast_horizon (int): Number of days to forecast risk.

        Returns:
            Dict[str, Dict[str, Union[float, List[float]]]]: Dictionary of risk measures for each stock.
        """
        logger.info("Starting complete risk assessment")
        self.train_model()

        all_risk_measures = {}
        for ticker, data in self.historical_data.items():
            logger.info(f"Assessing risk for {ticker}")
            risk_measures = self.predict_risk(data, forecast_horizon)
            all_risk_measures[ticker] = risk_measures

        logger.info("Risk assessment completed for all stocks")
        return all_risk_measures

    def output_results(
        self,
        results: Dict[str, Dict[str, Union[float, List[float]]]],
        output_format: str = "json",
    ) -> None:
        """
        Output the risk assessment results to a file.

        Args:
            results (Dict[str, Dict[str, Union[float, List[float]]]]): Risk assessment results.
            output_format (str): Output format ('json' or 'csv').
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"risk_assessment_results_{timestamp}.{output_format}"
        )

        if output_format == "json":
            with open(filename, "w") as f:
                json.dump(results, f, indent=4)
        elif output_format == "csv":
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Ticker"]
                    + list(next(iter(results.values())).keys())
                )
                for ticker, measures in results.items():
                    writer.writerow(
                        [ticker] + [str(v) for v in measures.values()]
                    )
        else:
            logger.error(
                f"Unsupported output format: {output_format}"
            )
            return

        logger.info(f"Results output to {filename}")


# if __name__ == "__main__":
#     # Example usage
#     # from data_integration import fetch_stock_data

#     tickers = [
#         "AAPL",
#         "GOOGL",
#         "MSFT",
#         "AMZN",
#         "^GSPC",
#     ]  # Including S&P 500 for market returns
#     historical_data = {
#         ticker: fetch_stock_data(ticker) for ticker in tickers
#     }

#     risk_assessor = AdvancedRealTimeRiskAssessment(historical_data)
#     risk_assessor.start_continuous_training()

#     try:
#         # Run for a while to allow some training iterations
#         time.sleep(60)

#         # Perform risk assessment
#         risk_results = risk_assessor.run_risk_assessment(
#             forecast_horizon=4
#         )  # 1-year forecast

#         # Output results
#         risk_assessor.output_results(risk_results, "json")
#         risk_assessor.output_results(risk_results, "csv")

#         # Print some results
#         for ticker, measures in risk_results.items():
#             print(f"\nRisk Assessment for {ticker}:")
#             for measure, value in measures.items():
#                 if isinstance(value, list):
#                     print(
#                         f"{measure}: [showing first 5 values] {value[:5]}"
#                     )
#                 else:
#                     print(f"{measure}: {value:.4f}")

#     finally:
#         # Ensure we stop the continuous training when done
#         risk_assessor.stop_continuous_training()
