# Quantopian_Pairs_Trader
Read bartchr808's Medium article about this project [here](https://medium.com/@bart.chr/pairs-trading-for-algorithmic-trading-breakdown-d8b709f59372)!

This is my implementation of a Pairs Trading Algorithm on the algorithmic trading research/competition platform [Quantopian](https://www.quantopian.com/home) so I can dive deeper and learn more about [Pairs Trading](http://www.investopedia.com/university/guide-pairs-trading/) and implementing trading algorithms. Some tests/measures I'm currently learning about and using include:

* Augmented Dickey–Fuller ([ADF](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)) unit root test
* [Hedge ratio](http://www.investopedia.com/terms/h/hedgeratio.asp)
* Half life of [mean-reversion](http://www.investopedia.com/terms/m/meanreversion.asp) from the [Ornstein-Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
* [Hurst exponent](https://en.wikipedia.org/wiki/Hurst_exponent)
* [Softmax function](https://en.wikipedia.org/wiki/Softmax_function) for calculating percent of each security on open order

## Algorithms
* 'Systematic Pairs Trading Algo.py' is my updated pairs trading algorithm
* 'Find Cointegrated Pairs.ipynb' is a Jupyter notebook with code to select the 11 optimal pairs
* 'algo.py' is bartchr808's original algorithm. (There were some errors and bugs within this algorithm, but it provided a great starting point from which to build.)

## Current Results
Currently, my implementation will be able to run on an arbitrary number of possible pairs that a user could provide. I ran regressions to identify one cointegrated pair per each of the eleven sectors. The trading algorithm is iniatilized with these eleven pairs.

## Issues/Next Steps
* Reduce the drawdown and beta and get the leveraging under control.
* Increasing alpha
* Look into cleaning up how I'm currently returning a completely new pair object in `process_pair` and replacing the old pair in the for-loop in `my_handle_data`.
* Haven't looked at using Kalman filters for determining hedge ratios. Not sure if I need to or if the way I did it sufficient.
* Need to look into how Quantopian's `order_target_percent` function works when I have several different pairs and not one or two (e.g. will the first opening order take up my entire portfolio?).
* Implement KPSS stationarity test
