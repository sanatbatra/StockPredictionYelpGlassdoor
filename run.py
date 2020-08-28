import matplotlib
import numpy
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
import math
import statsmodels.api as sm
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import copy
from sklearn.metrics import mean_squared_error
import math
from statsmodels.tsa.seasonal import seasonal_decompose


def plot(data, company_name, features):
    data_2 = data[company_name]
    df = pd.DataFrame(data_2)
    plt.xlabel('Period')
    plt.xticks(rotation='vertical')
    plt.rc('xtick', labelsize='6')

    for feature in features:
        plt.plot('period', feature, data=df)

    plt.legend()
    plt.savefig('results/%s_plot_%s.png' % (company_name, features[1]))
    plt.clf()


def do_granger_causality(data, company_name, features, max_lag):
    df = pd.DataFrame(data)
    print '\nGranger causality test for %s:' % company_name
    print '\nTest to see if %s causes %s' % (features[1], features[0])
    print '\n'
    print(sm.tsa.stattools.grangercausalitytests(df[[features[0], features[1]]].dropna(), max_lag))


def do_lstm(data1, company, no_of_steps, which_features, test_percentage, no_of_epochs, min_stock, max_stock, loss_function):
    test_size = int((len(data1['stock'])*test_percentage)/100)
    negative_test_size = -1 * test_size
    predictions = []
    actual_values = data1['stock'][negative_test_size:]
    train_actual_values = data1['stock'][:negative_test_size]
    print train_actual_values
    print "\n\n%s" % which_features
    # first_period = {}
    first_period = data1['period'][0]
    check_price = data1['stock'][len(data1['stock'])-1]
    last_period = data1['period'][len(data1['period'])-2]
    y1 = int(last_period.split('_')[0])
    m1 = int(last_period.split('_')[1])
    m1 = m1 + 3
    if m1 > 12:
        m1 = m1 % 12
        y1 += 1
    if m1 < 10:
        date1 = '%s-0%s-01' % (y1, m1)
    else:
        date1 = '%s-%s-01' % (y1, m1)

    y = int(first_period.split('_')[0])
    m = int(first_period.split('_')[1])
    m = m + 1
    if m == 13:
        m = 1
        y += 1
    if m < 10:
        date = '%s-0%s-01' % (y, m)
    else:
        date = '%s-%s-01' % (y, m)

    first_stock_price = None
    last_stock_price = None
    with open('Stock_%s.csv' % company, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            if row[0] == date:
                first_stock_price = float(row[4])
            elif row[0] == date1:
                last_stock_price = float(row[4])

    first_stock_price = (first_stock_price - min_stock) / (max_stock - min_stock)
    last_stock_price = (last_stock_price - min_stock) / (max_stock - min_stock)
    for i1 in range(0, test_size):
        data = copy.deepcopy(data1)
        stock = data['stock']
        yelp = data['yelp']
        glass = data['glass']
        for k in range(0, test_size - (i1+1)):
            stock.pop()
            yelp.pop()
            glass.pop()

        j = no_of_steps
        input_list = []
        size = len(stock)
        for i in range(0, no_of_steps):
            if which_features == "Only Yelp":
                input_list.append([yelp[size - j], stock[size - (j + 1)]])
            elif which_features == "Only Glassdoor":
                input_list.append([glass[size - j], stock[size - (j + 1)]])
            elif which_features == "Only Stock":
                input_list.append([stock[size - (j + 1)]])
            else:
                input_list.append([yelp[size - j], glass[size - j], stock[size - (j + 1)]])
            j -= 1

        print(input_list)
        previous = stock[size - 2]
        expected_output = stock[size - 1]
        if expected_output != actual_values[i1]:
            print 'Error'
            exit()
        stock.pop()
        yelp.pop()
        glass.pop()
        current_stock = [first_stock_price]
        current_stock.extend(stock)
        current_stock.pop()
        in_seq1 = array(yelp)
        in_seq2 = array(glass)
        in_seq3 = array(current_stock)
        out_seq = array(stock)
        # convert to [rows, columns] structure
        in_seq1 = in_seq1.reshape((len(in_seq1), 1))
        in_seq2 = in_seq2.reshape((len(in_seq2), 1))
        in_seq3 = in_seq3.reshape((len(in_seq3), 1))
        out_seq = out_seq.reshape((len(out_seq), 1))
        # horizontally stack columns

        if which_features == "Only Yelp":
            dataset = hstack((in_seq1, in_seq3, out_seq))
        elif which_features == "Only Glassdoor":
            dataset = hstack((in_seq2, in_seq3, out_seq))
        elif which_features == "Only Stock":
            dataset = hstack((in_seq3, out_seq))
        else:
            dataset = hstack((in_seq1, in_seq2, in_seq3, out_seq))
        X, y = split_sequences(dataset, no_of_steps)
        print(X.shape, y.shape)
        # summarize the data
        # for i in range(len(X)):
        #     print(X[i], y[i])
        n_features = X.shape[2]
        model = Sequential()

        # model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(no_of_steps, n_features)))
        # model.add(LSTM(50, activation='relu'))

        model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(no_of_steps, n_features)))
        model.add(LSTM(50, activation='tanh'))

        model.add(Dense(1))
        model.compile(optimizer='adam', loss=loss_function)
        model.fit(X, y, epochs=no_of_epochs, verbose=0)
        x_input = array(input_list)
        x_input = x_input.reshape((1, no_of_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat[0,0])
        print('Previous: %s' % previous)
        print('Predicted: %s' % yhat[0,0])
        print('Actual: %s' % expected_output)

    print 'Actual'
    print actual_values
    print 'Predicted'
    print predictions
    correct_predictions = 0
    total_predictions = 0
    no_of_times_increased = 0
    for i in range(1,test_size):
        total_predictions += 1
        if (actual_values[i] - actual_values[i - 1]) > 0:
            no_of_times_increased += 1
        if (actual_values[i] - actual_values[i-1]) >= 0 and (predictions[i] - actual_values[i-1]) >= 0:
            correct_predictions += 1
        elif (actual_values[i] - actual_values[i-1]) <= 0 and (predictions[i] - actual_values[i-1]) <= 0:
            correct_predictions += 1

    rmse = math.sqrt(mean_squared_error(predictions, actual_values))
    pred = train_actual_values
    for i in range(0, test_size):
        pred.append(predictions[i])
    print pred
    d = {'period': data1['period'], 'actual_price': data1['stock'], 'predicted_price': pred}
    df = pd.DataFrame(d)
    plt.plot('period', 'predicted_price', data=df)
    plt.plot('period', 'actual_price', data=df)

    plt.rc('xtick', labelsize=2)
    plt.legend()
    plt.savefig('%s_prediction_plot.png' % company)
    print 'RMSE: %s' %rmse
    print 'Prediction accuracy: %s' %(correct_predictions*100.0/total_predictions)
    print 'Percentage of time price increases: %s' %(no_of_times_increased*100.0/total_predictions)


def read_data(company):
    with open('final_data/Final_Data_%s.csv' % company, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        data = {}
        d = {'period': [], 'yelp_avg': [], 'yelp_no_of_reviews': [], 'yelp_5': [], 'percentage_yelp_5': [], 'yelp_4': [], 'yelp_3': [],
             'yelp_2': [], 'yelp_1': [], 'glass_avg': [], 'glass_no_of_reviews': [], 'glass_5': [], 'percentage_glass_5': [],
             'glass_4': [], 'glass_3': [], 'glass_2': [],
             'glass_1': [],
             'stock': []}
        for row in reader:
            data[row[0]] = {'yelp_avg': float(row[3]), 'yelp_no_of_reviews': float(row[4]), 'yelp_5': float(row[5]),
                            'percentage_yelp_5': float(row[6]),
                            'yelp_4': float(row[7]), 'yelp_3': float(row[8]), 'yelp_2': float(row[9]),
                            'yelp_1': float(row[10]), 'glass_avg': float(row[12]), 'glass_no_of_reviews': float(row[13]),
                            'glass_5': float(row[14]), 'percentage_glass_5': float(row[15]),
                            'glass_4': float(row[16]), 'glass_3': float(row[17]),
                            'glass_2': float(row[18]), 'glass_1': float(row[19]), 'stock': float(row[20])}
        min_year = 2019
        min_month = 12
        max_year = 0
        max_month = 0

        for key in data:
            y = int(key.split('_')[0])
            m = int(key.split('_')[1])
            if y < min_year:
                min_year = y
                min_month = m
            if y == min_year and m < min_month:
                min_month = m
            if y > max_year:
                max_year = y
                max_month = m
            if y == max_year and m > max_month:
                max_month = m

        year = min_year
        month = min_month
        min_stock = 100000.0
        max_stock = 0.0
        while year != max_year or not (month > max_month):

            period = str(year) + '_' + str(month)

            next_month = month + 1
            next_year = year
            if next_month == 13:
                next_month = 1
                next_year = year+1
            if ((data.get(period) is None) or (company == 'Nike' and period.startswith('2009'))
                    or (company == 'Nike' and period.startswith('2010')) or
                    (company == 'Nike' and period.startswith('2008')) or (company == 'Adidas' and period.startswith('2008'))
                or (company == 'Adidas' and period.startswith('2009')) or
                                              (company == 'Adidas' and period.startswith('2010')) or
                                               (company == 'Adidas' and period.startswith('2011')) or
                                                (company == 'Adidas' and period.startswith('2012'))):
                print 'Missing period: %s for company: %s' % (period, company)
                year = next_year
                month = next_month
                continue

            d['period'].append(period)
            d['yelp_avg'].append(data[period]['yelp_avg'])
            d['yelp_no_of_reviews'].append(data[period]['yelp_no_of_reviews'])
            d['yelp_5'].append(data[period]['yelp_5'])
            d['percentage_yelp_5'].append(data[period]['percentage_yelp_5'])
            d['yelp_4'].append(data[period]['yelp_4'])
            d['yelp_3'].append(data[period]['yelp_3'])
            d['yelp_2'].append(data[period]['yelp_2'])
            d['yelp_1'].append(data[period]['yelp_1'])
            d['glass_avg'].append(data[period]['glass_avg'])
            d['glass_no_of_reviews'].append(data[period]['glass_no_of_reviews'])
            d['glass_5'].append(data[period]['glass_5'])
            d['percentage_glass_5'].append(data[period]['percentage_glass_5'])
            d['glass_4'].append(data[period]['glass_4'])
            d['glass_3'].append(data[period]['glass_3'])
            d['glass_2'].append(data[period]['glass_2'])
            d['glass_1'].append(data[period]['glass_1'])
            d['stock'].append(float(data[period]['stock']))
            if float(data[period]['stock']) < float(min_stock):
                min_stock = data[period]['stock']
            if float(data[period]['stock']) > float(max_stock):
                max_stock = data[period]['stock']

            year = next_year
            month = next_month
        new_stock_list = []
        count = 0
        for val in d['stock']:
            val = (val - min_stock) / (max_stock - min_stock)
            new_stock_list.append(val)
            count += 1
        d['stock'] = new_stock_list
        return d, min_stock, max_stock


def do_lstm_stationary(data1, company, no_of_steps, which_features, test_percentage):
    test_size = int((len(data1['stock'])* test_percentage)/100)
    negative_test_size = -1 * test_size
    predictions = []
    actual_values = data1['stock'][negative_test_size:]
    train_actual_values = data1['stock'][:negative_test_size]
    # ss_decomposition = seasonal_decompose(x=data1['stock'], model='additive', freq=6)
    # estimated_trend = ss_decomposition.trend
    # estimated_seasonal = ss_decomposition.seasonal
    # estimated_residual = ss_decomposition.resid
    # plt.plot(estimated_trend)
    # plt.savefig('trend_plot.png')
    # plt.clf()
    # plt.plot(estimated_seasonal)
    # plt.savefig('seasonal_plot.png')
    # plt.clf()
    # plt.plot(estimated_residual)
    # plt.savefig('residual_plot.png')
    # plt.clf()
    # print len(estimated_residual)
    # print len(data1['stock'])
    # print estimated_residual
    # exit()
    data1_copy = copy.deepcopy(data1)
    stock_copy = data1_copy['stock']
    stock_differenced = [0.0]
    for i in range(1,len(stock_copy)):
        stock_differenced.append(stock_copy[i] - stock_copy[i-1])

    plt.plot(stock_differenced)
    plt.savefig('residual_plot_diff.png')
    plt.clf()

    print train_actual_values
    print "\n\n%s" % which_features
    first_period = data1['period'][0]
    y = int(first_period.split('_')[0])
    m = int(first_period.split('_')[1])
    m = m + 1
    if m == 13:
        m = 1
        y += 1
    if m < 10:
        date = '%s-0%s-01' % (y, m)
    else:
        date = '%s-%s-01' % (y, m)

    first_stock_price = None
    with open('Stock_%s.csv' % company, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            if row[0] == date:
                first_stock_price = row[4]

    for i1 in range(0, test_size):

        data = copy.deepcopy(data1)
        stock = data['stock']
        yelp = data['yelp']
        glass = data['glass']
        stock_differenced_copy = copy.deepcopy(stock_differenced)

        for k in range(0, test_size - (i1+1)):
            stock.pop()
            yelp.pop()
            glass.pop()
            stock_differenced_copy.pop()

        j = no_of_steps
        input_list = []
        size = len(stock)
        for i in range(0, no_of_steps):
            if which_features == "Only Yelp":
                input_list.append([yelp[size - j], stock[size - (j + 1)]])
            elif which_features == "Only Glassdoor":
                input_list.append([glass[size - j], stock[size - (j + 1)]])
            else:
                input_list.append([yelp[size - j], glass[size - j], stock[size - (j + 1)]])
            j -= 1

        print(input_list)
        expected_output = stock[size - 1]
        if expected_output != actual_values[i1]:
            print 'Error'
            exit()
        stock.pop()
        yelp.pop()
        glass.pop()
        current_stock = [float(first_stock_price)]
        current_stock.extend(stock)
        current_stock.pop()
        in_seq1 = array(yelp)
        in_seq2 = array(glass)
        in_seq3 = array(current_stock)
        out_seq = array(stock)
        # convert to [rows, columns] structure
        in_seq1 = in_seq1.reshape((len(in_seq1), 1))
        in_seq2 = in_seq2.reshape((len(in_seq2), 1))
        in_seq3 = in_seq3.reshape((len(in_seq3), 1))
        out_seq = out_seq.reshape((len(out_seq), 1))
        # horizontally stack columns

        if which_features == "Only Yelp":
            dataset = hstack((in_seq1, in_seq3, out_seq))
        elif which_features == "Only Glassdoor":
            dataset = hstack((in_seq2, in_seq3, out_seq))
        else:
            dataset = hstack((in_seq1, in_seq2, in_seq3, out_seq))
        X, y = split_sequences(dataset, no_of_steps)
        print(X.shape, y.shape)
        # summarize the data
        # for i in range(len(X)):
        #     print(X[i], y[i])
        n_features = X.shape[2]
        model = Sequential()

        # model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(no_of_steps, n_features)))
        # model.add(LSTM(50, activation='relu'))

        model.add(LSTM(50, activation='relu', input_shape=(no_of_steps, n_features)))

        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=75, verbose=0)
        x_input = array(input_list)
        x_input = x_input.reshape((1, no_of_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat[0,0])
        print('Predicted: %s' % yhat[0,0])
        print('Actual: %s' % expected_output)

    print 'Actual'
    print actual_values
    print 'Predicted'
    print predictions
    correct_predictions = 0
    for i in range(1,test_size):
        if (actual_values[i] - actual_values[i-1]) >= 0 and (predictions[i] - actual_values[i-1]) >= 0:
            correct_predictions += 1
        elif (actual_values[i] - actual_values[i-1]) <= 0 and (predictions[i] - actual_values[i-1]) <= 0:
            correct_predictions += 1

    rmse = math.sqrt(mean_squared_error(predictions, actual_values))
    pred = train_actual_values
    for i in range(0, test_size):
        pred.append(predictions[i])
    print pred
    d = {'period': data1['period'], 'actual_price': data1['stock'], 'predicted_price': pred}
    df = pd.DataFrame(d)
    plt.plot('period', 'predicted_price', data=df)
    plt.plot('period', 'actual_price', data=df)

    plt.rc('xtick', labelsize=2)
    plt.legend()
    plt.savefig('%s_prediction_plot.png' % company)
    print 'RMSE: %s' %rmse
    print 'Prediction accuracy: %s' %(correct_predictions*100.0/(test_size-1))


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def plot_all_graphs(lag, normalized_or_actual_or_detrended, predict_with_detrended):
    company = "Nike"
    with open('Final_Data_%s.csv' % company, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        data = {}
        d = {'period': [], 'yelp': [], 'glass': [], 'stock': []}
        for row in reader:
            data[row[0]] = {'yelp': float(row[3]), 'glass': float(row[6]), 'stock': float(row[8])}
        min_year = 2019
        min_month = 12
        max_year = 0
        max_month = 0

        for key in data:
            y = int(key.split('_')[0])
            m = int(key.split('_')[1])
            if y < min_year:
                min_year = y
                min_month = m
            if y == min_year and m < min_month:
                min_month = m
            if y > max_year:
                max_year = y
                max_month = m
            if y == max_year and m > max_month:
                max_month = m


        year = min_year
        month = min_month
        min_stock = 100000.0
        max_stock = 0.0
        while year != max_year or month != max_month:

            period = str(year) + '_' + str(month)

            next_month = month + 1
            next_year = year
            if next_month == 13:
                next_month = 1
                next_year = year+1
            if data.get(period) is None:
                print 'Missing period: %s for company: %s' % (period, company)
                year = next_year
                month = next_month
                continue

            d['period'].append(period)
            d['yelp'].append(data[period]['yelp'])
            d['glass'].append(data[period]['glass'])
            d['stock'].append(float(data[period]['stock']))
            if float(data[period]['stock']) < float(min_stock):
                min_stock = data[period]['stock']
            if float(data[period]['stock']) > float(max_stock):
                max_stock = data[period]['stock']


            year = next_year
            month = next_month
        new_stock_list = []
        count = 0
        for val in d['stock']:
            val = (val - min_stock) / (max_stock - min_stock)
            new_stock_list.append(val)
            count += 1
        d['stock'] = new_stock_list
        if normalized_or_actual_or_detrended == "detrended" or normalized_or_actual_or_detrended == "detrended_differenced":
            X = [i for i in range(0, len(new_stock_list))]
            X = numpy.reshape(X, (len(X), 1))
            y = new_stock_list
            model = LinearRegression()
            model.fit(X, y)
            # calculate trend
            trend = model.predict(X)
            # plot trend
            plt.plot(y)
            plt.plot(trend)
            plt.savefig('%s_plot_trend.png'% company)
            plt.clf()
            # detrend
            detrended = [y[i] - trend[i] for i in range(0, len(new_stock_list))]
            max_detrended_val = max(detrended)
            min_detrended_val = min(detrended)
            for i in range(0, len(detrended)):
                detrended[i] = (detrended[i] - min_detrended_val)/(max_detrended_val-min_detrended_val)


            # plot detrended
            plt.plot(detrended)
            plt.savefig('%s_plot_stock_detrended.png'% company)
            d['stock'] = detrended
            plt.clf()
            if normalized_or_actual_or_detrended == "detrended_differenced":
                detrended_differenced = []
                for i in range(1, len(detrended)):
                    detrended_differenced.append(detrended[i] - detrended[i - 1])

                max_detrended_val = max(detrended_differenced)
                min_detrended_val = min(detrended_differenced)
                for i in range(0, len(detrended_differenced)):
                    detrended_differenced[i] = (detrended_differenced[i] - min_detrended_val) / (max_detrended_val - min_detrended_val)

                result = adfuller(detrended_differenced)
                print('ADF Statistic: %f' % result[0])
                print('p-value: %f' % result[1])
                print('Critical Values:')
                for key, value in result[4].items():
                    print('\t%s: %.3f' % (key, value))
                if predict_with_detrended:
                    d['stock'] = detrended
                    detrended_differenced = detrended
                else:
                    d['stock'] = detrended_differenced
                    d['yelp'].pop(0)
                    d['period'].pop(0)
                    d['glass'].pop(0)
                    plt.plot(detrended_differenced)
                    plt.savefig('%s_plot_stock_detrended_differenced.png' % company)
                    plt.clf()
                df = pd.DataFrame(d)
                df['period'] = pd.to_datetime(df.period, format='%Y_%m')
                data = df.drop(['period'], axis=1)
                data.index = df.period
                train = data[:int(0.8 * (len(data)))]
                valid = data[int(0.8 * (len(data))):]
                model = VAR(endog=train)
                model_fit = model.fit()
                prediction = model_fit.forecast(model_fit.y, steps=len(valid))
                print(prediction)
                dd_validate = detrended_differenced[int(0.8 * (len(data))):]
                print dd_validate

        if not predict_with_detrended:
            df = pd.DataFrame(d)
            plt.plot('period','yelp',data = df)

            plt.plot('period','glass',data = df)
            plt.plot('period','stock',data = df)
            plt.rc('xtick', labelsize=2)
            plt.savefig('%s_plot_%s.png'% (company,normalized_or_actual_or_detrended))
            plt.clf()


def split_sequences_multistep(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def do_lstm_multi_step(data1, company, no_of_steps, which_features, test_percentage, no_of_epochs, min_stock, max_stock, loss_function, no_of_steps_out):
    correct_predictions = 0
    total_predictions = 0
    no_of_times_increased = 0
    test_size = int((len(data1['stock'])*test_percentage)/100)
    negative_test_size = -1 * test_size
    predictions = []
    actual_values = data1['stock'][negative_test_size:]
    train_actual_values = data1['stock'][:negative_test_size]
    print train_actual_values
    print "\n\n%s" % which_features
    # first_period = {}
    first_period = data1['period'][0]
    check_price = data1['stock'][len(data1['stock'])-1]
    last_period = data1['period'][len(data1['period'])-2]
    y1 = int(last_period.split('_')[0])
    m1 = int(last_period.split('_')[1])
    m1 = m1 + 3
    if m1 > 12:
        m1 = m1 % 12
        y1 += 1
    if m1 < 10:
        date1 = '%s-0%s-01' % (y1, m1)
    else:
        date1 = '%s-%s-01' % (y1, m1)

    y = int(first_period.split('_')[0])
    m = int(first_period.split('_')[1])
    m = m + 1
    if m == 13:
        m = 1
        y += 1
    if m < 10:
        date = '%s-0%s-01' % (y, m)
    else:
        date = '%s-%s-01' % (y, m)

    first_stock_price = None
    last_stock_price = None
    with open('Stock_%s.csv' % company, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            if row[0] == date:
                first_stock_price = float(row[4])
            elif row[0] == date1:
                last_stock_price = float(row[4])

    first_stock_price = (first_stock_price - min_stock) / (max_stock - min_stock)
    last_stock_price = (last_stock_price - min_stock) / (max_stock - min_stock)
    print last_stock_price
    output_expected_list = []
    for i1 in range(0, test_size):
        data = copy.deepcopy(data1)
        stock = data['stock']
        yelp = data['yelp']
        glass = data['glass']
        output_expected = []

        print 'Stock: '
        print stock
        for k in range(0, test_size-(i1+1)):
            stock.pop()
            glass.pop()
            yelp.pop()

        for k in range(0, no_of_steps_out):
            output_expected.append(stock[len(stock) - (no_of_steps_out-k)])

        for k in range(0, no_of_steps_out):
            if k!=0:
                yelp.pop()
                glass.pop()
                stock.pop()

        print 'Output expected: %s ' % output_expected

        j = no_of_steps
        input_list = []
        size = len(stock)
        for i in range(0, no_of_steps):
            if which_features == "Only Yelp":
                input_list.append([yelp[size - j], stock[size - (j + 1)]])
            elif which_features == "Only Glassdoor":
                input_list.append([glass[size - j], stock[size - (j + 1)]])
            elif which_features == "Only Stock":
                input_list.append([stock[size - (j + 1)]])
            else:
                input_list.append([yelp[size - j], glass[size - j], stock[size - (j + 1)]])
            j -= 1

        print(input_list)
        previous = stock[size - 2]
        stock.pop()
        yelp.pop()
        glass.pop()
        current_stock = [first_stock_price]
        current_stock.extend(stock)
        current_stock.pop()
        in_seq1 = array(yelp)
        in_seq2 = array(glass)
        in_seq3 = array(current_stock)
        out_seq = array(stock)
        # convert to [rows, columns] structure
        in_seq1 = in_seq1.reshape((len(in_seq1), 1))
        in_seq2 = in_seq2.reshape((len(in_seq2), 1))
        in_seq3 = in_seq3.reshape((len(in_seq3), 1))
        out_seq = out_seq.reshape((len(out_seq), 1))
        # horizontally stack columns

        if which_features == "Only Yelp":
            dataset = hstack((in_seq1, in_seq3, out_seq))
        elif which_features == "Only Glassdoor":
            dataset = hstack((in_seq2, in_seq3, out_seq))
        elif which_features == "Only Stock":
            dataset = hstack((in_seq3, out_seq))
        else:
            dataset = hstack((in_seq1, in_seq2, in_seq3, out_seq))
        X, y = split_sequences_multistep(dataset, no_of_steps, no_of_steps_out)
        print(X.shape, y.shape)
        # summarize the data
        # for i in range(len(X)):
        #     print(X[i], y[i])
        n_features = X.shape[2]
        model = Sequential()

        # model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(no_of_steps, n_features)))
        # model.add(LSTM(50, activation='relu'))

        model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(no_of_steps, n_features)))
        model.add(LSTM(50, activation='tanh'))

        model.add(Dense(no_of_steps_out))
        model.compile(optimizer='adam', loss=loss_function)
        model.fit(X, y, epochs=no_of_epochs, verbose=0)
        x_input = array(input_list)
        x_input = x_input.reshape((1, no_of_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat[0,2])
        print('Previous: %s' % previous)
        print('Predicted: %s' % yhat[0,2])
        print('Actual: ' )
        print output_expected
        total_predictions += 1
        output_expected_list.append(output_expected[no_of_steps_out-1])
        if (output_expected[no_of_steps_out-1] - previous) > 0:
            no_of_times_increased += 1
        if (output_expected[no_of_steps_out-1] - previous) >= 0 and (yhat[0,2] - previous) >= 0:
            correct_predictions += 1
        elif (output_expected[no_of_steps_out-1] - previous) <= 0 and (yhat[0,2] - previous) <= 0:
            correct_predictions += 1


    # print 'Actual'
    # print actual_values
    # print 'Predicted'
    # print predictions
    # for i in range(1,test_size):
    #     total_predictions += 1
    #     if (actual_values[i] - actual_values[i - 1]) > 0:
    #         no_of_times_increased += 1
    #     if (actual_values[i] - actual_values[i-1]) >= 0 and (predictions[i] - actual_values[i-1]) >= 0:
    #         correct_predictions += 1
    #     elif (actual_values[i] - actual_values[i-1]) <= 0 and (predictions[i] - actual_values[i-1]) <= 0:
    #         correct_predictions += 1

    rmse = math.sqrt(mean_squared_error(predictions, output_expected_list))
    pred = train_actual_values
    for i in range(0, test_size):
        pred.append(predictions[i])
    print pred
    d = {'period': data1['period'], 'actual_price': data1['stock'], 'predicted_price': pred}
    df = pd.DataFrame(d)
    plt.plot('period', 'predicted_price', data=df)
    plt.plot('period', 'actual_price', data=df)

    plt.rc('xtick', labelsize=2)
    plt.legend()
    plt.savefig('%s_prediction_plot.png' % company)
    print 'RMSE: %s' %rmse
    print 'Prediction accuracy: %s' %(correct_predictions*100.0/total_predictions)
    print 'Percentage of time price increases: %s' %(no_of_times_increased*100.0/total_predictions)


def do_lstm_new(data, company_name, no_of_steps, features, test_percentage, lag, loss_function, no_of_epochs):
    data1 = copy.deepcopy(data[company_name])

    test_size = int((len(data1['stock']) * test_percentage) / 100)
    print 'Test size: %s' % test_size
    print 'Period list:'
    print data1['period']
    print 'Stock list:'
    print data1['stock']
    print 'Current stock: '
    current_stock = data1['stock'][:(-1*lag)]
    future_stock = data1['stock'][lag:]
    print current_stock
    print 'Future stock:'
    print future_stock
    dictionary = {}
    for ele in features:
        if ele == 'stock':
            continue
        dictionary[ele] = data1[ele][:(-1*lag)]
        dictionary['train_'+ele] = dictionary[ele][:(-1*lag)]

    train_current_stock = current_stock[:(-1*lag)]
    train_future_stock = future_stock[:(-1*lag)]

    print 'Train current stock:'
    print train_current_stock
    print 'Train future stock:'
    print train_future_stock
    expected_stock_outputs = future_stock[(-1*test_size):]
    current_stock_to_test_output_diff_with = current_stock[(-1*test_size):]
    print 'Expected stock outputs: '
    print expected_stock_outputs

    predicted_stock_prices = []

    for i1 in range(0, test_size):
        no_of_items_to_remove = test_size - (i1 + 1)
        if no_of_items_to_remove == 0:
            for ele in features:
                if ele == 'stock':
                    continue
                dictionary['this_iteration_train_' + ele] = dictionary['train_' + ele][:]
                dictionary['this_iteration_input_' + ele] = dictionary[ele][:]

            this_iteration_train_current_stock = train_current_stock[:]
            this_iteration_train_future_stock = train_future_stock[:]
            this_iteration_input_current_stock = current_stock[:]
        else:
            for ele in features:
                if ele == 'stock':
                    continue
                dictionary['this_iteration_train_' + ele] = dictionary['train_' +ele][:(-1*no_of_items_to_remove)]
                dictionary['this_iteration_input_' + ele] = dictionary[ele][:(-1*no_of_items_to_remove)]
            this_iteration_train_current_stock = train_current_stock[:(-1*no_of_items_to_remove)]
            this_iteration_train_future_stock = train_future_stock[:(-1*no_of_items_to_remove)]
            this_iteration_input_current_stock = current_stock[:(-1 * no_of_items_to_remove)]
        print '\n\nThis iteration train current stock: '
        print this_iteration_train_current_stock
        print '\nThis iteration input current stock: '
        print this_iteration_input_current_stock
        print '\nThis iteration train future stock: '
        print this_iteration_train_future_stock

        j = (-1*no_of_steps)
        input_list = []
        for i in range(0, no_of_steps):
            input_list_append_list = []
            for ele in features:
                if ele == 'stock':
                    continue
                input_list_append_list.append(dictionary['this_iteration_input_' +ele][j])

            input_list_append_list.append(this_iteration_input_current_stock[j])
            input_list.append(input_list_append_list)

            j += 1

        print 'Input list:'
        print input_list

        cnt = 0
        for ele in features:
            if ele == 'stock':
                continue
            cnt += 1
            dictionary['in_seq%s' % cnt] = array(dictionary['this_iteration_train_' + ele])
            dictionary['in_seq%s' % cnt] = dictionary['in_seq%s' % cnt].reshape((len(dictionary['in_seq%s' % cnt]), 1))

        cnt += 1
        dictionary['in_seq%s' % cnt] = array(this_iteration_train_current_stock)
        dictionary['in_seq%s' % cnt] = dictionary['in_seq%s' % cnt].reshape((len(dictionary['in_seq%s' % cnt]), 1))
        out_seq = array(this_iteration_train_future_stock)
        out_seq = out_seq.reshape((len(out_seq), 1))

        length = len(features)
        if length == 17:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], dictionary['in_seq6'], dictionary['in_seq7'], dictionary['in_seq8'],
                 dictionary['in_seq9'], dictionary['in_seq10'], dictionary['in_seq11'], dictionary['in_seq12'],
                 dictionary['in_seq13'], dictionary['in_seq14'],  dictionary['in_seq15'],
                 dictionary['in_seq16'], dictionary['in_seq17'], out_seq))

        elif length == 16:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], dictionary['in_seq6'], dictionary['in_seq7'], dictionary['in_seq8'],
                 dictionary['in_seq9'], dictionary['in_seq10'], dictionary['in_seq11'], dictionary['in_seq12'],
                 dictionary['in_seq13'], dictionary['in_seq14'],  dictionary['in_seq15'],
                 dictionary['in_seq16'], out_seq))

        elif length == 15:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], dictionary['in_seq6'], dictionary['in_seq7'], dictionary['in_seq8'],
                 dictionary['in_seq9'], dictionary['in_seq10'], dictionary['in_seq11'], dictionary['in_seq12'],
                 dictionary['in_seq13'], dictionary['in_seq14'],  dictionary['in_seq15'],
                 out_seq))

        elif length == 14:
            dataset = hstack((dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                              dictionary['in_seq5'], dictionary['in_seq6'], dictionary['in_seq7'], dictionary['in_seq8'],
                              dictionary['in_seq9'], dictionary['in_seq10'], dictionary['in_seq11'], dictionary['in_seq12'],
                              dictionary['in_seq13'], dictionary['in_seq14'], out_seq))

        elif length == 13:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], dictionary['in_seq6'], dictionary['in_seq7'], dictionary['in_seq8'],
                 dictionary['in_seq9'], dictionary['in_seq10'], dictionary['in_seq11'], dictionary['in_seq12'],
                 dictionary['in_seq13'], out_seq))

        elif length == 12:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], dictionary['in_seq6'], dictionary['in_seq7'], dictionary['in_seq8'],
                 dictionary['in_seq9'], dictionary['in_seq10'], dictionary['in_seq11'], dictionary['in_seq12'], out_seq))

        elif length == 11:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], dictionary['in_seq6'], dictionary['in_seq7'], dictionary['in_seq8'],
                 dictionary['in_seq9'], dictionary['in_seq10'], dictionary['in_seq11'], out_seq))

        elif length == 10:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], dictionary['in_seq6'], dictionary['in_seq7'], dictionary['in_seq8'],
                 dictionary['in_seq9'], dictionary['in_seq10'], out_seq))

        elif length == 9:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], dictionary['in_seq6'], dictionary['in_seq7'], dictionary['in_seq8'],
                 dictionary['in_seq9'], out_seq))

        elif length == 8:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], dictionary['in_seq6'], dictionary['in_seq7'], dictionary['in_seq8'], out_seq))

        elif length == 7:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], dictionary['in_seq6'], dictionary['in_seq7'], out_seq))

        elif length == 6:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], dictionary['in_seq6'], out_seq))

        elif length == 5:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'],
                 dictionary['in_seq5'], out_seq))

        elif length == 4:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], dictionary['in_seq4'], out_seq))

        elif length == 3:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'], dictionary['in_seq3'], out_seq))

        elif length == 2:
            dataset = hstack(
                (dictionary['in_seq1'], dictionary['in_seq2'],  out_seq))

        elif length == 1:
            dataset = hstack((dictionary['in_seq1'], out_seq))

        X, y = split_sequences(dataset, no_of_steps)
        print 'X:'
        print X
        print 'Y:'
        print y
        print(X.shape, y.shape)
        # summarize the data
        # for i in range(len(X)):
        #     print(X[i], y[i])
        n_features = X.shape[2]
        model = Sequential()

        # model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(no_of_steps, n_features)))
        # model.add(LSTM(50, activation='relu'))

        model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(no_of_steps, n_features)))
        model.add(LSTM(50, activation='tanh'))

        model.add(Dense(1))
        model.compile(optimizer='adam', loss=loss_function)
        model.fit(X, y, epochs=no_of_epochs, verbose=0)
        x_input = array(input_list)
        x_input = x_input.reshape((1, no_of_steps, n_features))
        print '\nInput:'
        print x_input
        yhat = model.predict(x_input, verbose=0)
        predicted_stock_prices.append(yhat[0, 0])
        print('\nCurrent: %s' % this_iteration_input_current_stock[-1])
        print('Future Predicted: %s' % yhat[0, 0])
        print('Future Actual: %s' % expected_stock_outputs[i1])

    print 'Expected future stock prices:'
    print expected_stock_outputs
    print 'Predicted future stock prices:'
    print predicted_stock_prices
    correct_predictions = 0
    total_predictions = 0
    no_of_times_increased = 0

    print 'current_stock_to_test_output_diff_with'
    print current_stock_to_test_output_diff_with
    for i in range(0, test_size):
        total_predictions += 1
        if (expected_stock_outputs[i] - current_stock_to_test_output_diff_with[i]) > 0:
            no_of_times_increased += 1
        if (predicted_stock_prices[i] - current_stock_to_test_output_diff_with[i]) >= 0 \
                and (expected_stock_outputs[i] - current_stock_to_test_output_diff_with[i]) >= 0:
            correct_predictions += 1
        elif (predicted_stock_prices[i] - current_stock_to_test_output_diff_with[i]) <= 0 and\
                (expected_stock_outputs[i] - current_stock_to_test_output_diff_with[i]) <= 0:
            correct_predictions += 1

    rmse = math.sqrt(mean_squared_error(predicted_stock_prices, expected_stock_outputs))
    all_stock_prices_except_predicted = data1['stock'][:(-1 * test_size)]
    for i in range(0, test_size):
        all_stock_prices_except_predicted.append(predicted_stock_prices[i])
    print all_stock_prices_except_predicted
    d = {'period': data1['period'], 'Actual Price': data1['stock'], 'Predicted Price': all_stock_prices_except_predicted}
    df = pd.DataFrame(d)
    plt.plot('period', 'Predicted Price', data=df)
    plt.plot('period', 'Actual Price', data=df)

    plt.rc('xtick', labelsize=2)
    plt.legend()
    if features == ['stock']:
        plt.savefig('results/%s_prediction_plot_only_stock.png' % company_name)
    else:
        plt.savefig('results/%s_prediction_plot.png' % company_name)

    print '\nRMSE: %s' % rmse
    print 'Prediction accuracy: %s' % (correct_predictions * 100.0 / total_predictions)
    # print 'Percentage of time price increases: %s' % (no_of_times_increased * 100.0 / total_predictions)


if __name__ == '__main__':
    test_percentage = 10
    company_name = 'Nike'

    company_list = ['Macys', 'McDonalds', 'Apple', 'Nike', 'HomeDepot', 'Gap']

    data = {}
    min_stock = {}
    max_stock = {}
    for c in company_list:
        data[c], min_stock[c], max_stock[c] = read_data(c)

    plot(data, company_name, ['stock', 'percentage_glass_5'])
    plot(data, company_name, ['stock', 'glass_4'])
    plot(data, company_name, ['stock', 'glass_3'])
    plot(data, company_name, ['stock', 'glass_2'])
    plot(data, company_name, ['stock', 'glass_1'])
    plot(data, company_name, ['stock', 'glass_5'])
    plot(data, company_name, ['stock', 'glass_avg'])
    plot(data, company_name, ['stock', 'glass_no_of_reviews'])
    plot(data, company_name, ['stock', 'yelp_5'])
    plot(data, company_name, ['stock', 'percentage_yelp_5'])
    plot(data, company_name, ['stock', 'yelp_4'])
    plot(data, company_name, ['stock', 'yelp_3'])
    plot(data, company_name, ['stock', 'yelp_2'])
    plot(data, company_name, ['stock', 'yelp_1'])
    plot(data, company_name, ['stock', 'yelp_avg'])
    plot(data, company_name, ['stock', 'yelp_no_of_reviews'])

    # do_granger_causality(data[company_name], company_name, ['stock', 'glass_5'], 11)
    # exit()

    features = ['stock', 'glass_5', 'yelp_5']
    # features = ['stock']

    lag = 9
    no_of_steps = 5

    do_lstm_new(data, company_name, no_of_steps, features, test_percentage, lag, 'mae', 100)
