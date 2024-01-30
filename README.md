# Overview
OptionStrategy is a package, tailored to simplify the implementation of various common option trading strategies. It aims to provide users with an intuitive and user-friendly interface for executing common option trading strategies without the need for intricate calculations. OptionStrategy offers an effortless experience for traders, investors, and decision-makers.
It leverages several fundamental libraries like NumPy, Pandas, Matplotlib, Math, SciPy, and yfinance to handle mathematical computations, data manipulation, visualization, and fetching financial data from Yahoo Finance.

# Package Contents and Functionality
The OptionStrategy class encapsulates a range of user-friendly functions for implementing popular option trading strategies. 
The OptionStrategy class enables users to effortlessly execute the following option trading strategies:

*Long Condor using Calls*

*Short Condor using Calls*

*Bull Call Spread*

*Bear Call Spread*

*Long Butterfly using Calls*

*Short Butterfly using Calls*

*Long Strangle*

*Short Strangle*

*Long Straddle*

*Short Straddle*

*Synthetic Long*

*Synthetic Short*

For each strategy, the class generates visual representations of the profit and loss profile as a function of the future stock price at the expiration date of the option. It also provides information on the cost/credit of the strategy (given the current stock price and current RF rate), the stock price intervals for loss and profit, and prompts the user for a future stock price to determine the strategy's output.

# Usage and Instructions
To utilize the OptionStrategy class, users can instantiate it and then call the specific functions corresponding to the desired trading strategy.
Users can initialize the OptionStrategy class with specific parameters (current stock price, risk-free rate, and time to maturity) and leverage functions.


# Conclusion
The OptionStrategy class provides an user-friendly interface for implementing and visualizing popular option trading strategies. With its intuitive functions and insightful visualizations, it empowers users to make informed decisions and explore the potential outcomes of different option strategies in the ever-evolving financial landscape.
