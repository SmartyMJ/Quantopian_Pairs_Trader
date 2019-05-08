import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import pandas as pd

"""
Purpose:
    Econ 493-Spring2019
Authors:
    3:30pm Group 1 - Malik Jabati, Lelia Burkart, James Gresset, James Yang, Julia Friou — 7May2019
    UNC Honor Pledge: I certify that no unauthorized assistance has been received or given in the completion of this work.
Note:
    This .py file was adapted from Bart Chrzaszcz at the University of Waterloo.
    https://github.com/bartchr808/Quantopian_Pairs_Trader
Assumptions:
    Database = Quantopian US Tradable Stocks
    Assets = 
    Frequency = Weekly
"""

# ~~~~~~~~~~~~~~~~~~~~~~ TESTS FOR FINDING PAIR TO TRADE ON ~~~~~~~~~~~~~~~~~~~~~~
class ADF(object):
    """
    Augmented Dickey–Fuller (ADF) unit root test
    Source: http://www.pythonforfinance.net/2016/05/09/python-backtesting-mean-reversion-part-2/
    
    The null hypothesis of the ADF test is that there is a unit root present in a time series. While the alternative hypothesis is that there is no unit root and the time series is mean reverting.

    The ADF test determines whether our time series is stationary by looking for the absence of unit roots.
    """
    def __init__(self):
        self.p_value = None
        self.five_perc_stat = None
        self.perc_stat = None
        self.p_min = .0

        # Smaller values indicate increased confidence that the null hypothesis is false
        self.p_max = .05
        self.look_back = 63

    def apply_adf(self, time_series):
        model = ts.adfuller(time_series, 1)
        self.p_value = model[1]
        self.five_perc_stat = model[4]['5%']
        self.perc_stat = model[0]

    # Reject null hypothesis if the p-value is less than 5%
    def use_P(self):
        return (self.p_value > self.p_min) and (self.p_value < self.p_max)
    
    def use_critical(self):
        return abs(self.perc_stat) > abs(self.five_perc_stat)


class Half_Life(object):
    """
    Half Life test from the Ornstein-Uhlenbeck process 
    Source: http://www.pythonforfinance.net/2016/05/09/python-backtesting-mean-reversion-part-2/
    
    Helps us determine whether that series will experience mean reverting properties we can exploit within a reasonable time frame. For example, we don’t want to open an order on the prediction that the pair will revert back to the mean a year from now.

    By looking at the original time series and a (time) lagged version of itself, we can run linear regression against it to get a beta value (i.e. the slope/coefficient of the regression). And then we can pass that to the Ornstein-Uhlenbeck process.

    This can be used to calculate the average time it will take to get half way back to the mean (i.e., the half life).
    """

    def __init__(self):

        # The minimum and maximum acceptable value of the half life. Changing these helps change how long you think your algorithm will want to keep orders open for (think long term vs short term)
        self.hl_min = 1.0
        self.hl_max = 42.0

        # Describes how many days must have passed before you can start using this test, and is used to pass the last n days of info about the pair
        self.look_back = 43
        self.half_life = None

    def apply_half_life(self, time_series):
        lag = np.roll(time_series, 1)
        lag[0] = 0
        ret = time_series - lag
        ret[0] = 0

        # Adds intercept terms to X variable for regression
        lag2 = sm.add_constant(lag)

        model = sm.OLS(ret, lag2)
        res = model.fit()

        self.half_life = -np.log(2) / res.params[1]

    def use(self):
        return (self.half_life < self.hl_max) and (self.half_life > self.hl_min)


class Hurst():
    """
    If Hurst Exponent is under the 0.5 value of a random walk, then the series is mean reverting
    Source: https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing

    The hurst exponent helps us determine whether a time series is mean reverting or not. The outputted value H from the hurst formula is some value between 0 and 1.

    If H=0.5: then the time series experiences Geometric Brownian Motion (continous time random walk).

    If 0<H<0.5: then the time series is mean reverting. The closer to 0, the more “mean reverting” it is.

    If 0.5<H<1: then the time series experiences positive or negative correlation (i.e. autocorrelation) over a long period of time.
    """

    def __init__(self):
        # The H value must be between 0 and 0.4 to be considered (reasonably) mean reverting.
        self.h_min = 0.0
        self.h_max = 0.4

        # Describes how many days must have passed before you can start using this test, and is used to pass the last n days of info about the pair
        self.look_back = 126
        
        # How large the time lags (i.e., delay between time series) will get when calculating the Hurst exponent. It can be tricky to figure out what value you want and that depends on the look_back window size the Hurst function looks at. See here: https://robotwealth.com/demystifying-the-hurst-exponent-part-1/
        self.lag_max = 100
        self.h_value = None
    
    def apply_hurst(self, time_series):
        """Returns the Hurst Exponent of the time series vector ts"""
        # Create the range of lag values
        lags = range(2, self.lag_max)

        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log10(lags), np.log10(tau), 1)

        # Return the Hurst exponent from the polyfit output
        self.h_value = poly[0]*2.0 

    def use(self):
        return (self.h_value < self.h_max) and (self.h_value > self.h_min)

# ~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS FOR FILING AN ORDER ~~~~~~~~~~~~~~~~~~~~~~
def hedge_ratio(Y, X):
    """
    The benefit of using OLS to calculate the hedge ratio is that it can take into account the past prices
    """
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model.params[1]
    # The formula for getting the spread will be Price A — hedge * Price B.

def softmax_order(stock_1_shares, stock_2_shares, stock_1_price, stock_2_price):
    """
    We use the softmax function to calculate the percentage of how much of each security in the pair we should order by passing in the hedge ratio and the relative prices.

    Softmax is very popular in deep learning in the final layers where you want your model to tell you what it thinks your input is. It does this through normalizing k amount of values to be between 0 and 1, where the k values add up to 1.

    Here, this is helpful because it tells us the relative percentage of how much of each security in the pair we should get based on how much each would cost.

    CAUTION: It does not normalize values linearly!
    """
    stock_1_cost = stock_1_shares * stock_1_price
    stock_2_cost = stock_2_shares * stock_2_price
    costs = np.array([stock_1_cost, stock_2_cost])
    return np.exp(costs) / np.sum(np.exp(costs), axis=0)

def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    
    context.asset_pairs = [[symbol('PX'), symbol('BHP'), {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([])}], 
                           [symbol('CTL'), symbol('DISH'), {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([])}],
                           [symbol('HMC'), symbol('DIS'), {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([])}],
                           [symbol('CLX'), symbol('TAP'), {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([])}],
                           [symbol('IMO'), symbol('TOT'), {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([])}],
                           [symbol('AXP'), symbol('AIG'), {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([])}],
                           [symbol('AET'), symbol('CAH'), {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([])}],
                           [symbol('ADP'), symbol('CMI'), {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([])}],
                           [symbol('EQR'), symbol('BAM'), {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([])}],
                           [symbol('CERN'), symbol('AAPL'), {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([])}],
                           [symbol('PCG'), symbol('ED'), {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([])}]
                           ]
    
    # Should change in accordance with min/max of half life. The larger it is, the more values you can look at to calculate your mean.
    context.z_back = 20
    context.hedge_lag = 2

    # Larger values mean the spread's pair has to deviate even more from the mean in order to open an order
    context.entry_z = 0.5
    
    schedule_function(my_handle_data, date_rules.every_day(),
                      time_rules.market_open(hours=1))
    # Typical slippage I have seen others use and default slippage used in templates by Quantopian
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))


def my_handle_data(context, data):
    """
    Called every day.

    First, determine whether there are any open orders. If not, continue and start looking at pairs in process_pair().
    """
    
    if get_open_orders():
        return
    
    for i in range(len(context.asset_pairs)):
        pair = context.asset_pairs[i]      
        new_pair = process_pair(pair, context, data)
        context.asset_pairs[i] = new_pair

def process_pair(pair, context, data):
    """
    Main function that will execute an order for every pair.
    """
    
    # Get stock data
    stock_1 = pair[0]
    stock_2 = pair[1]
    # Get daily price data from the last 300 days
    prices = data.history([stock_1, stock_2], "price", 300, "1d")
    stock_1_P = prices[stock_1]
    stock_2_P = prices[stock_2]
    in_short = pair[2]['in_short']
    in_long = pair[2]['in_long']
    spread = pair[2]['spread']
    hedge_history = pair[2]['hedge_history']

    # Get hedge ratio and store it for later
    try:
        hedge = hedge_ratio(stock_1_P, stock_2_P)
    except ValueError as e:
        log.error(e)
        return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]
    
    hedge_history = np.append(hedge_history, hedge)
    
    # Check whether context.hedge_lag number of days has passed 
    if hedge_history.size < context.hedge_lag:
        log.debug("Hedge history too short!")
        return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]

    # Calculate the spread
    hedge = hedge_history[-context.hedge_lag]
    spread = np.append(
        spread, stock_1_P[-1] - hedge * stock_2_P[-1])
    spread_length = spread.size

    adf = ADF()
    half_life = Half_Life()
    hurst = Hurst()

    # Check if current window size is large enough for adf, half life, and hurst exponent
    if (spread_length < adf.look_back) or (spread_length < half_life.look_back) or (spread_length < hurst.look_back):
        return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]

    
    # Possible "SVD did not converge" error because of OLS
    try:
        adf.apply_adf(spread[-adf.look_back:])
        half_life.apply_half_life(spread[-half_life.look_back:])
        hurst.apply_hurst(spread[-hurst.look_back:])
    except:
        return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]

    # Check if they are in fact a stationary (or possibly trend stationary...need to avoid this [KPSS test helps]) time series
    # * Only cancel if all measures believe it isn't stationary
    if not adf.use_P() and not adf.use_critical() and not half_life.use() and not hurst.use():
        """
        If all of all tells indicate that this pair creates a mean reverting stationary time series, then skip this if block and move on to possibly executing a trade. However, if one of the tests tell us it doesn’t, we go ahead and close our position if we have an open order on the pair (as we can’t determine whether they will revert back to the mean anymore) or just skip this pair entirely for the day and try again the next day.
        """
        if in_short or in_long:

            log.info('Tests have failed. Exiting open positions')
            order_target(stock_1, 0)    # Quantopian syntax to close order
            order_target(stock_2, 0)
            in_short = in_long = False
            return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]
            
        log.debug("Not Stationary!")
        return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]
    
    # Check if current window size is large enough for Z score
    if spread_length < context.z_back:
        return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]

    """
    Calculate the Z-score for the given spread. The Z-score in this case tells us for the given spread, how many standard deviations the current price is away from the mean price over some given look back window (i.e., z_back).
    """
    spreads = spread[-context.z_back:]
    z_score = (spreads[-1] - spreads.mean()) / spreads.std()
                                 
    
    # Close order logic
    if in_short and z_score < 0.0:
        """
        If in_short is True, it means that we already opened a position where the Z-score was positive and we thought that the current spread is too high and will go back to the mean. And if the Z-score becomes negative, it means that it has started to revert back to the mean and it is time to close the order.
        """
        order_target(stock_1, 0)
        order_target(stock_2, 0)
        in_short = False
        in_long = False
        return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]
    elif in_long and z_score > 0.0:
        """
        If in_long is True, it means that we already opened a position where the Z-score was negative and we thought that the current spread is too low and will go back to the mean. And if the Z-score becomes positive, it means that it has started to revert back to the mean and it is time to close the order.
        """
        order_target(stock_1, 0)
        order_target(stock_2, 0)
        in_short = False
        in_long = False
        return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]
    
    # Open order 
    """
    Same logic with Z-score and being long/short applies here when we want to open an order. If the Z-score is less (greater) than our threshold and we don’t have a long (short) position already open, create an order.

    We divide order target portfolio percentages by 11 to normalize values according to total pairs.
    """
    if (z_score < -context.entry_z) and (not in_long):
        stock_1_perc = 1 / float(11)  #long top
        stock_2_perc = -hedge / float(11)  #short bottom  
        in_long = True
        in_short = False
        order_target_percent(stock_1, stock_1_perc)
        order_target_percent(stock_2, stock_2_perc)
        return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]
    elif z_score > context.entry_z and (not in_short):
        stock_1_perc = -1 / float(11)   #short top
        stock_2_perc = hedge / float(11) #long bottom
        in_short = True
        in_long = False
        order_target_percent(stock_1, stock_1_perc)
        order_target_percent(stock_2, stock_2_perc)
        print("GOING SHORT!")
        print(hedge)
        print(stock_1_perc)
        print(stock_2_perc)
        return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]

    return [stock_1, stock_2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history}]
