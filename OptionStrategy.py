# Import packages needed
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.stats import norm

class OptionStrategy:
    # Initialize the option strategy with stock price (S), call strike price, put strike price (K), risk-free rate (r), and time to maturity (tau)
    def __init__(self, S, r, tau):
        self.S = S
        self.r = r
        self.tau = tau
        
    # Implement the long straddle strategy by calculating option prices and plotting profit/loss
    # Calculate total cost, generate stock price range, compute profit/loss, and plot the graph
    # Identify breakeven points and display information about profit/loss
    def long_straddle(self, sigma, K_call, K_put):
        if K_call == K_put:
            d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
            d1 = d2 + sigma * math.sqrt(self.tau)
            call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
            d2 = (math.log(self.S / K_put) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
            d1 = d2 + sigma * math.sqrt(self.tau)
            put_price = K_put * math.exp(-self.r * self.tau) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
            total_cost = call_price + put_price

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace(K_call/2, K_call*1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses = -(call_price - np.maximum(stock_prices - K_call, 0)) - (put_price - np.maximum(K_put - stock_prices, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1, x2 = stock_prices[indices[0]], stock_prices[indices[-1]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.axvline(x=x2, color='g', linestyle='--')
            plt.title('Long Straddle Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_cost}")
            print(f"Profit if price will be lower than {x1} or higher than {x2}")
            print(f"Loss if price will be between {x1} and {x2}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", - (call_price - np.maximum(price - K_call, 0)) - (put_price - np.maximum(K_put - price, 0)))
        else:
            print("Can implement Long Straddle option trading strategy \nonly if Put/Call strike prices are equal")
        
    # Implement the short straddle strategy by calculating option prices and plotting profit/loss
    # Calculate total cost, generate stock price range, compute profit/loss, and plot the graph
    # Identify breakeven points and display information about profit/loss    
    def short_straddle(self, sigma, K_call, K_put):
        if K_call == K_put:
            d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
            d1 = d2 + sigma * math.sqrt(self.tau)
            call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
            d2 = (math.log(self.S / K_put) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
            d1 = d2 + sigma * math.sqrt(self.tau)
            put_price = K_put * math.exp(-self.r * self.tau) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
            # Since it's a short straddle, we are selling both call and put
            total_credit = call_price + put_price

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace(K_call/2, K_call*1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses = (call_price - np.maximum(stock_prices - K_call, 0)) + (put_price - np.maximum(K_put - stock_prices, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1, x2 = stock_prices[indices[0]], stock_prices[indices[-1]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.axvline(x=x2, color='g', linestyle='--')
            plt.title('Short Straddle Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_credit}")
            print(f"Profit if price will be between {x1} and {x2}")
            print(f"Loss if price will be lower than {x1} or higher than {x2} ")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", (call_price - np.maximum(price - K_call, 0)) + (put_price - np.maximum(K_put - price, 0)))
        else:
            print("Can implement Short Straddle option trading strategy \nonly if Put/Call strike prices are equal")

    # Implement the bull call spread strategy by calculating option prices and plotting profit/loss
    # Calculate total cost, generate stock price range, compute profit/loss, and plot the graph
    # Identify breakeven points and display information about profit/loss    
    def bull_call_spread(self, sigma, K_call, K_call2):
        d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call2) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price2 = self.S * norm.cdf(d1) - K_call2 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        
        if K_call < K_call2:
            # Since it's a bull call spread, we are buying the first and selling the second call
            total_credit = -call_price + call_price2

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace((K_call2 + K_call)/4, (K_call2 + K_call)/2*1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses = (call_price2 - np.maximum(stock_prices - K_call2, 0)) - (call_price - np.maximum(stock_prices - K_call, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1 = stock_prices[indices[0]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.title('Bull Call Spread Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_credit}")
            print(f"Profit if price will be higher than {x1} ")
            print(f"Loss if price will be lower than {x1}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", (call_price2 - np.maximum(price - K_call2, 0)) - (call_price - np.maximum(price - K_call, 0)))
        elif K_call > K_call2 :
            # Since it's a bull call spread, we are buying the first and selling the second call
            total_credit = + call_price - call_price2

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace((K_call + K_call2)/4, (K_call + K_call2)/2*1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses = - (call_price2 - np.maximum(stock_prices - K_call2, 0)) + (call_price - np.maximum(stock_prices - K_call, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1 = stock_prices[indices[0]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.title('Bull Call Spread Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_credit}")
            print(f"Profit if price will be higher than {x1} ")
            print(f"Loss if price will be lower than {x1}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", - (call_price2 - np.maximum(price - K_call2, 0)) + (call_price - np.maximum(price - K_call, 0)))
        else:
            print("Can implement a Bull Call srategy \nonly if Call strike prices are different")

    # Implement the bear call spread strategy by calculating option prices and plotting profit/loss
    # Calculate total cost, generate stock price range, compute profit/loss, and plot the graph
    # Identify breakeven points and display information about profit/loss    
    def bear_call_spread(self, sigma, K_call, K_call2):
        d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call2) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price2 = self.S * norm.cdf(d1) - K_call2 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        
        if K_call < K_call2:
            total_credit = call_price - call_price2

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace((K_call2 + K_call)/4, (K_call2 + K_call)/2*1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses = - (call_price2 - np.maximum(stock_prices - K_call2, 0)) + (call_price - np.maximum(stock_prices - K_call, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1 = stock_prices[indices[0]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.title('Bear Call Spread Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_credit}")
            print(f"Profit if price will be higher than {x1} ")
            print(f"Loss if price will be lower than {x1}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", - (call_price2 - np.maximum(price - K_call2, 0)) + (call_price - np.maximum(price - K_call, 0)))
        elif K_call > K_call2 :
            total_credit = - call_price + call_price2

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace((K_call + K_call2)/4, (K_call + K_call2)/2*1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses =  (call_price2 - np.maximum(stock_prices - K_call2, 0)) - (call_price - np.maximum(stock_prices - K_call, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1 = stock_prices[indices[0]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.title('Bear Call Spread Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_credit}")
            print(f"Profit if price will be higher than {x1} ")
            print(f"Loss if price will be lower than {x1}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", (call_price2 - np.maximum(price - K_call2, 0)) - (call_price - np.maximum(price - K_call, 0)))
        else:
            print("Can implement a Bull Call srategy \nonly if Call strike prices are different")

    # Implement the long butterfly call strategy by calculating option prices and plotting profit/loss
    # Calculate total cost, generate stock price range, compute profit/loss, and plot the graph
    # Identify breakeven points and display information about profit/loss    
    def long_butterfly_call(self, sigma, K_call, K_call2, K_call3):
        d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call2) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price2 = self.S * norm.cdf(d1) - K_call2 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call3) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price3 = self.S * norm.cdf(d1) - K_call3 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        
        if K_call < K_call2 < K_call3:
            total_credit = - call_price + 2 * call_price2 - call_price3

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace(K_call2 / 2 , K_call2 * 1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses = (call_price2 - np.maximum(stock_prices - K_call2, 0)) * 2 - (call_price - np.maximum(stock_prices - K_call, 0)) - (call_price3 - np.maximum(stock_prices - K_call3, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1, x2 = stock_prices[indices[0]], stock_prices[indices[-1]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.axvline(x=x2, color='g', linestyle='--')
            plt.title('Long Batterfly Call Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_credit}")
            print(f"Profit if price will be between {x1} and {x2}")
            print(f"Loss if price will be lower than {x1} or higher than {x2}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", (call_price2 - np.maximum(price - K_call2, 0)) * 2 - (call_price - np.maximum(price - K_call, 0)) - (call_price3 - np.maximum(price - K_call3, 0)))
        else:
            print("K_call must be lower than K_call2 \nand K_call2 must be lower than K_call3")
    
    # Implement the short butterfly call strategy by calculating option prices and plotting profit/loss
    # Calculate total cost, generate stock price range, compute profit/loss, and plot the graph
    # Identify breakeven points and display information about profit/loss    
    def short_butterfly_call(self, sigma, K_call, K_call2, K_call3):
        d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call2) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price2 = self.S * norm.cdf(d1) - K_call2 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call3) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price3 = self.S * norm.cdf(d1) - K_call3 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        
        if K_call < K_call2 < K_call3:
            total_credit = call_price - 2 * call_price2 + call_price3

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace(K_call2 / 2 , K_call2 * 1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses = - (call_price2 - np.maximum(stock_prices - K_call2, 0)) * 2 + (call_price - np.maximum(stock_prices - K_call, 0)) + (call_price3 - np.maximum(stock_prices - K_call3, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1, x2 = stock_prices[indices[0]], stock_prices[indices[-1]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.axvline(x=x2, color='g', linestyle='--')
            plt.title('Short Batterfly Call Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_credit}")
            print(f"Loss if price will be between {x1} and {x2}")
            print(f"Profit if price will be lower than {x1} or higher than {x2}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", - (call_price2 - np.maximum(price - K_call2, 0)) * 2 + (call_price - np.maximum(price - K_call, 0)) + (call_price3 - np.maximum(price - K_call3, 0)))
        else:
            print("K_call must be lower than K_call2 \nand K_call2 must be lower than K_call3")
            
    # Implement long call and short put strategy (synthetic long position)
    def synthetic_long_position(self, sigma, K_call, K_put):
        if K_call == K_put:
            d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
            d1 = d2 + sigma * math.sqrt(self.tau)
            call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
            d2 = (math.log(self.S / K_put) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
            d1 = d2 + sigma * math.sqrt(self.tau)
            put_price = K_put * math.exp(-self.r * self.tau) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
            
            # Net cost or credit of the position
            net_cost_credit = call_price - put_price

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace(K_call/2, K_call * 1.5, 1000000)
            
            # Calculate profit/loss for each stock price
            profits_losses = (call_price - np.maximum(stock_prices - K_call, 0)) - (put_price - np.maximum(K_put - stock_prices, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            plt.axvline(x=K_call+net_cost_credit, color='g', linestyle='--', label='Breakeven')
            plt.title('Synthetic Long Position Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {net_cost_credit}")
            print(f"Profit if price will higher than {K_call+net_cost_credit}")
            print(f"Loss if price will be lower than {K_call+net_cost_credit}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", (call_price - np.maximum(price - K_call, 0)) - (put_price - np.maximum(K_put - price, 0)))
        else:
            print("Can replicate a Synthetic Long Position in the stock \nonly if Put/Call strike prices are equal")
        
    # Implement short call and long put strategy (synthetic short position)
    def synthetic_short_position(self, sigma, K_call, K_put):
        if K_call == K_put:
            d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
            d1 = d2 + sigma * math.sqrt(self.tau)
            call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
            d2 = (math.log(self.S / K_put) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
            d1 = d2 + sigma * math.sqrt(self.tau)
            put_price = K_put * math.exp(-self.r * self.tau) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
            
            # Net cost or credit of the position
            net_cost_credit = - call_price + put_price

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace(K_call/2, K_call * 1.5, 1000000)
            
            # Calculate profit/loss for each stock price
            profits_losses = - (call_price - np.maximum(stock_prices - K_call, 0)) + (put_price - np.maximum(K_put - stock_prices, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            plt.axvline(x=K_call - net_cost_credit, color='g', linestyle='--', label='Breakeven')
            plt.title('Synthetic Short Position Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {net_cost_credit}")
            print(f"Loss if price will higher than {K_call - net_cost_credit}")
            print(f"Profit if price will be lower than {K_call - net_cost_credit}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", - (call_price - np.maximum(price - K_call, 0)) + (put_price - np.maximum(K_put - price, 0)))
        else:
            print("Can replicate a Synthetic Short Position in the stock \nonly if Put/Call strike prices are equal")

    # Implement the long strangle strategy by calculating option prices and plotting profit/loss
    # Calculate total cost, generate stock price range, compute profit/loss, and plot the graph
    # Identify breakeven points and display information about profit/loss    
    def long_strangle(self, sigma, K_call, K_put):
        d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_put) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        put_price = K_put * math.exp(-self.r * self.tau) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        
        if K_call > K_put:
            total_credit = - call_price - put_price

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace((K_call + K_put)/4 , (K_call + K_put)/2 * 1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses = - (call_price - np.maximum(stock_prices - K_call, 0)) - (put_price - np.maximum(K_put - stock_prices, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1, x2 = stock_prices[indices[0]], stock_prices[indices[-1]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.axvline(x=x2, color='g', linestyle='--')
            plt.title('Long Strangle Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_credit}")
            print(f"Profit if price will be lower than {x1} or higher than {x2}")
            print(f"Loss if price will be between {x1} and {x2}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", - (call_price - np.maximum(price - K_call, 0)) - (put_price - np.maximum(K_put - price, 0)))
        else:
            print("K_call must be higher than K_put")
    
    # Implement the short strangle strategy by calculating option prices and plotting profit/loss
    # Calculate total cost, generate stock price range, compute profit/loss, and plot the graph
    # Identify breakeven points and display information about profit/loss    
    def short_strangle(self, sigma, K_call, K_put):
        d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_put) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        put_price = K_put * math.exp(-self.r * self.tau) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        
        if K_call > K_put:
            total_credit = call_price + put_price

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace((K_call + K_put)/4 , (K_call + K_put)/2 * 1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses = (call_price - np.maximum(stock_prices - K_call, 0)) + (put_price - np.maximum(K_put - stock_prices, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1, x2 = stock_prices[indices[0]], stock_prices[indices[-1]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.axvline(x=x2, color='g', linestyle='--')
            plt.title('Short Strangle Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_credit}")
            print(f"Loss if price will be lower than {x1} or higher than {x2}")
            print(f"Profit if price will be between {x1} and {x2}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", (call_price - np.maximum(price - K_call, 0)) + (put_price - np.maximum(K_put - price, 0)))
        else:
            print("K_call must be higher than K_put")

    # Implement the long condor call strategy by calculating option prices and plotting profit/loss
    # Calculate total cost, generate stock price range, compute profit/loss, and plot the graph
    # Identify breakeven points and display information about profit/loss    
    def long_condor_call(self, sigma, K_call, K_call2, K_call3, K_call4):
        d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call2) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price2 = self.S * norm.cdf(d1) - K_call2 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call3) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price3 = self.S * norm.cdf(d1) - K_call3 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call4) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price4 = self.S * norm.cdf(d1) - K_call4 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        
        if K_call < K_call2 < K_call3 < K_call4:
            total_credit = - call_price + call_price2 + call_price3 - call_price4

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace((K_call + K_call4)/4 , (K_call + K_call4)/2 * 1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses = - (call_price - np.maximum(stock_prices - K_call, 0)) + (call_price2 - np.maximum(stock_prices - K_call2, 0)) + (call_price3 - np.maximum(stock_prices - K_call3, 0)) - (call_price4 - np.maximum(stock_prices - K_call4, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1, x2 = stock_prices[indices[0]], stock_prices[indices[-1]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.axvline(x=x2, color='g', linestyle='--')
            plt.title('Long Condor Call Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_credit}")
            print(f"Loss if price will be lower than {x1} or higher than {x2}")
            print(f"Profit if price will be between {x1} and {x2}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", - (call_price - np.maximum(price - K_call, 0)) + (call_price2 - np.maximum(price - K_call2, 0)) + (call_price3 - np.maximum(price - K_call3, 0)) - (call_price4 - np.maximum(price - K_call4, 0)))
        else:
            print("K_call must be lower than K_call2, \nK_call2 must be lower than K_call3 and \nK_call3 must be lower than K_call4")
    
    def short_condor_call(self, sigma, K_call, K_call2, K_call3, K_call4):
        d2 = (math.log(self.S / K_call) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price = self.S * norm.cdf(d1) - K_call * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call2) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price2 = self.S * norm.cdf(d1) - K_call2 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call3) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price3 = self.S * norm.cdf(d1) - K_call3 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        d2 = (math.log(self.S / K_call4) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
        d1 = d2 + sigma * math.sqrt(self.tau)
        call_price4 = self.S * norm.cdf(d1) - K_call4 * math.exp(-self.r * self.tau) * norm.cdf(d2)
        
        if K_call < K_call2 < K_call3 < K_call4:
            total_credit = call_price - call_price2 - call_price3 + call_price4

            # Generate a range of stock prices for the graph
            stock_prices = np.linspace((K_call + K_call4)/4 , (K_call + K_call4)/2 * 1.5, 1000000)
        
            # Calculate profit/loss for each stock price
            profits_losses = (call_price - np.maximum(stock_prices - K_call, 0)) - (call_price2 - np.maximum(stock_prices - K_call2, 0)) - (call_price3 - np.maximum(stock_prices - K_call3, 0)) + (call_price4 - np.maximum(stock_prices - K_call4, 0))

            # Plot the profit/loss graph
            sns.set_style("darkgrid", { "axes.facecolor": "#001F3F"})
            plt.figure(figsize=(7, 4))
            plt.plot(stock_prices, profits_losses)
            condition = (-0.001 < profits_losses) & (profits_losses < 0.001)
            indices = np.where(condition)[0]
            if indices.size > 0:
                x1, x2 = stock_prices[indices[0]], stock_prices[indices[-1]]
            plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
            plt.axvline(x=x2, color='g', linestyle='--')
            plt.title('Short Condor Call Strategy Profit/Loss')
            plt.xlabel('Stock Price')
            plt.ylabel('Profit/Loss')
            plt.legend(labelcolor='white')
            plt.grid(True, alpha=0.1)
            plt.ylim(-max(abs(profits_losses))*2, max(abs(profits_losses))*2)
            plt.xlim(min(stock_prices), max(stock_prices))
            plt.show()
            print(f"Net Cost/Credit: {total_credit}")
            print(f"Loss if price will be lower than {x1} or higher than {x2}")
            print(f"Profit if price will be between {x1} and {x2}")
            user_input = input("Profit/Loss for Stock Price equal to: ")
            price = int(user_input)
            print(f"Profit/Loss for Stock Price equal to {price}: ", (call_price - np.maximum(price - K_call, 0)) - (call_price2 - np.maximum(price - K_call2, 0)) - (call_price3 - np.maximum(price - K_call3, 0)) + (call_price4 - np.maximum(price - K_call4, 0)))
        else:
            print("K_call must be lower than K_call2, \nK_call2 must be lower than K_call3 and \nK_call3 must be lower than K_call4")