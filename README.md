This isn't the cleanest collection of code I've written, sorry about that.

Python 2.7 is required.
Run the run.py file to get the results
On line 1091 of the run.py file you can change the company name to get results for different companies.
Company names - 'Macys', 'McDonalds', 'Apple', 'Nike', 'HomeDepot', 'Gap'.
McDonalds_prediction_plot_only_stock.png is the plot of the prediction for McDonalds stock price 9 periods (27 months into the future) using only its past stock price values.
McDonalds_prediction_plot.png is the plot of the prediction for McDonalds stock price 9 periods (27 months into the future) using its past stock price values, number of yelp 5 star reviews and number of glassdoor 5 star reviews.
The code might take a minute to run and the RMSE and classification accuracy is printed out at the bottom. Each run might give you a slightly different result because of the stochastic nature of the LSTM model.
To change the features used, change line 1121 from features = ['stock', 'glass_5', 'yelp_5'] to features = ['stock'], to use only past stock data.

If you want to run the Granger Causality test for the company, uncomment line 1118 and line 1119.
You will have to scroll up a bit to see the test results.

McDonalds_plot_glass_5.png is the correlation plot of McDonalds stock price with its number of 5 star Glassdoor ratings.
Similarily, McDonalds_plot_yelp_5.png is the correlation plot of McDonalds stock price with its number of 5 star Yelp ratings, etc.

You can run the pre processing file pre_process.py too if needed. It creates the files Final_Data_McDonalds.csv, etc.

yelp_scraper.py - Used to scrape Yelp for customer ratings.
glassdoor_scraper.py - Used to scrape Glassdoor for employee ratings.
