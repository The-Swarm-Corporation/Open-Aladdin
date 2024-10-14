# file: ensemble_risk_assessment.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
from loguru import logger
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import yfinance as yf
from fredapi import Fred
import time
import json
import csv
from datetime import datetime
from threading import Thread

logger.add("ensemble_risk_assessment.log", rotation="10 MB")


class EnsembleRiskAssessment:
    def __init__(
        self,
        tickers: List[str],
        fred_api_key: str,
        update_interval: int = 3600,
    ):
        """
        Initialize the EnsembleRiskAssessment class.

        Args:
            tickers (List[str]): List of stock tickers to analyze.
            fred_api_key (str): API key for accessing FRED (Federal Reserve Economic Data).
            update_interval (int): Interval for model updates in seconds.
        """
        self.tickers = tickers
        self.fred = Fred(api_key=fred_api_key)
        self.update_interval = update_interval
        self.models = {
            "xgboost": XGBRegressor(random_state=42),
            # "lightgbm": LGBMRegressor(random_state=42),
            "random_forest": RandomForestRegressor(random_state=42),
        }
        self.scaler = StandardScaler()
        self.is_training = False
        self.stop_training = False
        self.historical_data = self._fetch_data()
        logger.info("EnsembleRiskAssessment initialized")

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical stock data and fundamental/macroeconomic indicators."""
        data = {}
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            data[ticker] = stock.history(period="5y")

            # Ensure stock data index is timezone-naive
            data[ticker].index = data[ticker].index.tz_localize(None)

            # Fetch fundamental data
            info = stock.info
            data[ticker]["P/E Ratio"] = info.get("forwardPE", np.nan)
            data[ticker]["Debt to Equity"] = info.get(
                "debtToEquity", np.nan
            )

        # Fetch macroeconomic indicators
        gdp_growth = self.fred.get_series("A191RL1Q225SBEA")
        interest_rate = self.fred.get_series("FEDFUNDS")

        # Convert FRED data to timezone-naive
        gdp_growth.index = gdp_growth.index.tz_localize(None)
        interest_rate.index = interest_rate.index.tz_localize(None)

        # Align macroeconomic data with stock data
        for ticker in self.tickers:
            data[ticker]["GDP Growth"] = gdp_growth.reindex(
                data[ticker].index, method="ffill"
            )
            data[ticker]["Interest Rate"] = interest_rate.reindex(
                data[ticker].index, method="ffill"
            )

        return data

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features including technical indicators, fundamental data, and macroeconomic indicators."""
        features = data[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "P/E Ratio",
                "Debt to Equity",
                "GDP Growth",
                "Interest Rate",
            ]
        ].copy()

        # Technical indicators
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
        features["EMA_50"] = (
            features["Close"].ewm(span=50, adjust=False).mean()
        )
        features["SMA_200"] = (
            features["Close"].rolling(window=200).mean()
        )
        features["Bollinger_Upper"], features["Bollinger_Lower"] = (
            self.calculate_bollinger_bands(features["Close"])
        )

        # Interaction terms
        features["Vol_PE"] = (
            features["Volatility"] * features["P/E Ratio"]
        )
        features["Returns_DebtEquity"] = (
            features["Returns"] * features["Debt to Equity"]
        )
        features["MACD_InterestRate"] = (
            features["MACD"] * features["Interest Rate"]
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

    def train_models(self) -> None:
        """Train all models using grid search for hyperparameter tuning."""
        logger.info("Training risk models")
        all_features = pd.concat(
            [
                self.prepare_features(data)
                for data in self.historical_data.values()
            ]
        )

        X = all_features.drop(
            ["Returns", "Log_Returns", "Volatility"], axis=1
        )
        y = all_features[
            ["Volatility", "Returns", "Beta"]
        ]  # Multi-task learning

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # XGBoost parameters (unchanged)
        xgb_params = {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5, 7],
        }
        self.models["xgboost"] = GridSearchCV(
            XGBRegressor(random_state=42), xgb_params, cv=3
        )
        self.models["xgboost"].fit(
            X_train_scaled, y_train["Volatility"]
        )

        # # Updated LightGBM parameters
        # lgb_params = {
        #     'n_estimators': [100, 200],
        #     'learning_rate': [0.01, 0.1],
        #     'num_leaves': [31, 63],
        #     'min_child_samples': [20, 50],
        #     'max_depth': [5, 10],
        #     'lambda_l1': [0, 0.5, 1],
        #     'lambda_l2': [0, 0.5, 1],
        #     'min_split_gain': [0, 0.1, 0.2],
        #     'feature_fraction': [0.8, 1.0],
        #     'force_col_wise': [True]
        # }
        # self.models['lightgbm'] = GridSearchCV(LGBMRegressor(random_state=42), lgb_params, cv=3)
        # self.models['lightgbm'].fit(X_train_scaled, y_train['Volatility'])

        # Random Forest parameters (unchanged)
        rf_params = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
        self.models["random_forest"] = GridSearchCV(
            RandomForestRegressor(random_state=42), rf_params, cv=3
        )
        self.models["random_forest"].fit(
            X_train_scaled, y_train["Volatility"]
        )

        logger.info("Model training completed")
        self._evaluate_models(X_test_scaled, y_test)

    def _evaluate_models(
        self, X_test: np.ndarray, y_test: pd.DataFrame
    ) -> None:
        """Evaluate all models and log their performance."""
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test["Volatility"], y_pred)
            logger.info(f"{name.upper()} Model - MSE: {mse}")

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
            self.historical_data = self._fetch_data()  # Refresh data
            self.train_models()
            time.sleep(self.update_interval)
        self.is_training = False

    def predict_risk(
        self, data: pd.DataFrame, forecast_horizon: int = 30
    ) -> Dict[str, Union[float, List[float]]]:
        """Predict risk measures using ensemble of models."""
        features = self.prepare_features(data)
        model_input = features.drop(
            ["Returns", "Log_Returns", "Volatility"], axis=1
        )
        model_input_scaled = self.scaler.transform(model_input)

        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(model_input_scaled)
            predictions[name] = pred

        # Ensemble prediction (simple average)
        ensemble_pred = np.mean(
            [pred for pred in predictions.values()], axis=0
        )

        volatility = ensemble_pred
        returns = features["Returns"].values
        beta = features["Beta"].values

        var = self.calculate_value_at_risk(returns)
        es = self.calculate_expected_shortfall(returns)

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
            model_input_scaled[-1], forecast_horizon
        )

        risk_measures = {
            "Current_Volatility": float(volatility[-1]),
            "Predicted_Returns": float(returns[-1]),
            "Predicted_Beta": float(beta[-1]),
            "Value_at_Risk": var,
            "Expected_Shortfall": es,
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
        """Forecast risk for a given horizon using ensemble of models."""
        forecasted_risk = []
        current_prediction = current_features.reshape(1, -1)

        for _ in range(horizon):
            model_predictions = [
                model.predict(current_prediction)[0]
                for model in self.models.values()
            ]
            risk = np.mean(model_predictions)
            forecasted_risk.append(float(risk))

            # Update prediction for next iteration (simplified approach)
            current_prediction = np.roll(current_prediction, -1)
            current_prediction[0, -1] = risk

        return forecasted_risk

    def calculate_value_at_risk(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> float:
        """Calculate Value at Risk (VaR)."""
        return float(
            np.percentile(returns, 100 * (1 - confidence_level))
        )

    def calculate_expected_shortfall(
        self, returns: np.ndarray, confidence_level: float = 0.95
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
        self.train_models()

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
                json.dump(
                    results,
                    f,
                    indent=4,
                    default=lambda x: (
                        float(x) if isinstance(x, np.float32) else x
                    ),
                )
        elif output_format == "csv":
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                headers = ["Ticker"] + list(
                    next(iter(results.values())).keys()
                )
                writer.writerow(headers)
                for ticker, measures in results.items():
                    row = [ticker] + [
                        (
                            str(v)
                            if not isinstance(v, list)
                            else ",".join(map(str, v))
                        )
                        for v in measures.values()
                    ]
                    writer.writerow(row)
        else:
            logger.error(
                f"Unsupported output format: {output_format}"
            )
            return

        logger.info(f"Results output to {filename}")


if __name__ == "__main__":
    # Example usage
    tickers = [
        "AAPL",
        "GOOGL",
        "MSFT",
        "AMZN",
        "^GSPC",
    ]  # Including S&P 500 for market returns
    fred_api_key = "7b91965cec4e139790a774d2d5bea476 "  # Replace with your actual FRED API key

    risk_assessor = EnsembleRiskAssessment(
        tickers, fred_api_key, update_interval=1
    )
    risk_assessor.start_continuous_training()

    try:
        # Run for a while to allow some training iterations
        time.sleep(3)  # Wait for 5 minutes

        # Perform risk assessment
        risk_results = risk_assessor.run_risk_assessment(
            forecast_horizon=365
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
