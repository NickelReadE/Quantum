import yfinance as yf


def sample_r_and_Sigma():
    sample_tickers = ["2222.SR",  # Aramco
                      "CVX",  # Chevron
                      "601857.SS",  # PetroChina
                      "PBR",  # Petrobras
                      "EQNR",  # Equinor ASA
                      "SU",  # Suncor Energy Inc
                      "ENI.MI",  # Eni S.p.A.
                      "IMO",  # Imperial Oil Limited
                      "CVE",  # Cenovus Energy Inc
                      "BP"  # BP
                      ]
    for ticker in sample_tickers:
        stock_data = yf.download(ticker, start='2020-01-01', end='2023-09-27')

        # Calculate daily returns
        stock_data['Daily Return'] = stock_data['Close'].pct_change()

        # Calculate the average return (expected return)
        expected_return = stock_data['Daily Return'].mean()

        # Convert to annualized expected return (for daily returns, multiply by 252 trading days)
        annualized_return = (1 + expected_return) ** 252 - 1

        print(f"{ticker} Expected Daily Return: {expected_return:.4%}")
        print(f"{ticker} Expected Annualized Return: {annualized_return:.4%}")


def get_r_and_Sigma(tickers):
    data = yf.download(tickers, period='1y')['Adj Close']

    # Calculate daily returns
    returns = data.pct_change(fill_method=None).dropna()

    # Calculate the return vector (mean daily returns)
    return_vector = returns.mean()

    # Calculate the asset price correlation matrix
    correlation_matrix = returns.corr()

    # Print for visualization
    print("Return Vector:\n", return_vector)
    print("\nCorrelation Matrix:\n", correlation_matrix)

    return return_vector.values, correlation_matrix.values


if __name__ == '__main__':
    get_r_and_Sigma(["2222.SR",  # Aramco
                      "CVX",  # Chevron
                      "601857.SS",  # PetroChina
                      "PBR",  # Petrobras
                      "EQNR",  # Equinor ASA
                      "SU",  # Suncor Energy Inc
                      "ENI.MI",  # Eni S.p.A.
                      "IMO",  # Imperial Oil Limited
                      "CVE",  # Cenovus Energy Inc
                      "BP"  # BP
                      ])
