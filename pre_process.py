import csv
from datetime import datetime


class DataPreProcessing:
    def __init__(self, company_names, rating_avg_interval, months_after, no_of_periods_as_features, useful_count_factor,
                 base, overall_rating_factor, work_life_factor, culture_factor, career_opp_factor,
                 compensation_benefits_factor, senior_management_factor):
        self.company_names = company_names
        self.rating_avg_interval = rating_avg_interval
        self.months_after = months_after
        self.useful_count_factor = useful_count_factor
        self.base = base
        self.overall_rating_factor = overall_rating_factor
        self.work_life_factor = work_life_factor
        self.culture_factor = culture_factor
        self.career_opp_factor = career_opp_factor
        self.compensation_benefits_factor = compensation_benefits_factor
        self.senior_management_factor = senior_management_factor

    def GetStockData(self, company_name):
        file_name = "stock/Stock_%s.csv" % company_name
        with open(file_name, 'rU') as csv_file2:
            reader2 = csv.reader(csv_file2)
            next(reader2)
            stock_data = dict()
            for row in reader2:
                date_split = row[0].split('-')
                month = int(date_split[1])
                year = date_split[0]

                stock_data[year + "_%s" % month] = {'open_price': row[1], 'close_price': row[4], 'volume': row[6]}
            return stock_data

    def GetYelpData(self, company_name):
        file_name = "yelp/Yelp_%s.csv" % company_name
        with open(file_name, 'rU') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            yelp_data = []
            for row in reader:
                # yelp_data.append({'company_name': company_name, 'date': row[0], 'rating': row[1], 'friends': row[2],
                #                   'reviews': row[3], 'photos': row[4], 'check_ins': row[5], 'useful_count': row[6]})
                yelp_data.append({'company_name': company_name, 'date': row[0], 'rating': row[1]})
            return yelp_data

    def GetGlassdoorData(self, company_name):
        file_name = "glassdoor/Glassdoor_%s.csv" % company_name
        glassdoor_data = []
        with open(file_name, 'rU') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                if not row[1]:
                    continue

                emp_type = row[9].split(' ')[0]
                glassdoor_data.append({'date': row[1], 'overall_rating': row[3], 'work_life': row[4],
                                       'culture': row[5], 'career_opp': row[6], 'compensation_benefits': row[7],
                                       'senior_management': row[8], 'emp_type': emp_type, 'if_recommends': row[11],
                                       'outlook': row[12], 'ceo_approval': row[13]})
        return glassdoor_data

    def GetAvgRatings(self, company_name, avg_ratings, five_rating, four_rating, three_rating, two_rating, one_rating,
                      factor_sum, stock_data, yelp_or_glassdoor_str):
        avg_ratings_temp = []
        ratings = []
        for key in avg_ratings:
            avg_ratings[key] = avg_ratings[key] / factor_sum[key]
            avg_ratings_temp.append(avg_ratings[key])
            year = int(key.split('_')[0])
            year_current = year
            month_no_current = int(key.split('_')[1]) * self.rating_avg_interval + 1
            month_no = month_no_current + self.months_after
            while float(month_no) / 12.0 > 1.0:
                year += 1
                month_no = month_no % 12

            while float(month_no_current) / 12.0 > 1.0:
                year_current += 1
                month_no_current = month_no_current % 12

            stock_key_current = "%s_%s" % (str(year_current), str(month_no_current))
            stock_key = "%s_%s" % (str(year), str(month_no))
            months_after_stock_price = stock_data.get(stock_key)
            if not months_after_stock_price:
                print 'Cant find stock price for period: %s for company : %s' % (stock_key, company_name)
                continue
            ratings.append({'company': company_name, 'period': key, yelp_or_glassdoor_str+'_avg_rating':
                            avg_ratings[key], yelp_or_glassdoor_str+'_5': five_rating[key],
                            'percentage_' + yelp_or_glassdoor_str + '_5': float(five_rating[key]) / float(factor_sum[key]),
                            yelp_or_glassdoor_str+'_4': four_rating[key], yelp_or_glassdoor_str+'_3': three_rating[key],
                            yelp_or_glassdoor_str+'_2': two_rating[key], yelp_or_glassdoor_str+'_1': one_rating[key],
                            yelp_or_glassdoor_str + '_no_of_reviews': factor_sum[key],
                            'months_after_stock_price': stock_data[stock_key]['close_price'],
                            'stock_price_change': float(stock_data[stock_key]['close_price']) /
                                                  float(stock_data[stock_key_current]['close_price'])})
        return avg_ratings_temp,ratings

    def GetPeriodList(self, ratings):
        period_list = []
        for row in ratings:
            period_list.append(row['period'])
        return period_list

    def NormalizeRatings(self, avg_ratings_val, ratings, yelp_or_glassdoor_str):
        rating_5 = []
        rating_4 = []
        rating_3 = []
        rating_2 = []
        rating_1 = []
        percentage_rating_5 = []
        no_of_reviews = []
        for row in ratings:
            no_of_reviews.append(row[yelp_or_glassdoor_str+'_no_of_reviews'])
            rating_5.append(row[yelp_or_glassdoor_str+'_5'])
            rating_4.append(row[yelp_or_glassdoor_str+'_4'])
            rating_3.append(row[yelp_or_glassdoor_str+'_3'])
            rating_2.append(row[yelp_or_glassdoor_str+'_2'])
            rating_1.append(row[yelp_or_glassdoor_str+'_1'])
            percentage_rating_5.append(row['percentage_'+yelp_or_glassdoor_str+'_5'])

        max_no_of_reviews = max(no_of_reviews)
        min_no_of_reviews = min(no_of_reviews)
        max_avg_rating = max(avg_ratings_val)
        min_avg_rating = min(avg_ratings_val)
        max_5 = max(rating_5)
        min_5 = min(rating_5)
        max_4 = max(rating_4)
        min_4 = min(rating_4)
        max_3 = max(rating_3)
        min_3 = min(rating_3)
        max_2 = max(rating_2)
        min_2 = min(rating_2)
        max_1 = max(rating_1)
        min_1 = min(rating_1)
        max_p_5 = max(percentage_rating_5)
        min_p_5 = min(percentage_rating_5)
        ratings2 = ratings[:]
        for row in ratings2:
            temp_avg_rating = row[yelp_or_glassdoor_str+'_avg_rating']
            normalized_avg_rating = (temp_avg_rating - min_avg_rating) / (max_avg_rating - min_avg_rating)
            row['normalized_'+yelp_or_glassdoor_str+'_avg_rating'] = normalized_avg_rating
            temp_5 = (float(row[yelp_or_glassdoor_str + '_5'] - min_5) / float(max_5 - min_5))
            temp_4 = (float(row[yelp_or_glassdoor_str + '_4'] - min_4) / float(max_4 - min_4))
            temp_3 = (float(row[yelp_or_glassdoor_str + '_3'] - min_3) / float(max_3 - min_3))
            temp_2 = (float(row[yelp_or_glassdoor_str + '_2'] - min_2) / float(max_2 - min_2))
            temp_1 = (float(row[yelp_or_glassdoor_str + '_1'] - min_1) / float(max_1 - min_1))
            temp_p_5 = (float(row['percentage_'+yelp_or_glassdoor_str+'_5'] - min_p_5) / float(max_p_5 - min_p_5))
            row[yelp_or_glassdoor_str + '_5'] = temp_5
            row[yelp_or_glassdoor_str + '_4'] = temp_4
            row[yelp_or_glassdoor_str + '_3'] = temp_3
            row[yelp_or_glassdoor_str + '_2'] = temp_2
            row[yelp_or_glassdoor_str + '_1'] = temp_1
            row['percentage_' + yelp_or_glassdoor_str + '_5'] = temp_p_5

            row[yelp_or_glassdoor_str + '_no_of_reviews'] = \
                float(row[yelp_or_glassdoor_str + '_no_of_reviews'] - min_no_of_reviews) / float(max_no_of_reviews - min_no_of_reviews)

            if row['stock_price_change'] > 1.0:
                row['price_increase_or_decrease'] = 1
            else:
                row['price_increase_or_decrease'] = 0

        return ratings2

    def GetFinalData(self, ratings, period_list, yelp_or_glassdoor_str):
        final_data_to_store = []
        for row in ratings:
            period = row['period']
            period_split = period.split('_')
            period_year = int(period_split[0])
            period_interval_no = int(period_split[1])
            if period_interval_no == 1:
                prev_period_year = period_year - 1
                prev_period_interval_no = int(12 / self.rating_avg_interval)
            else:
                prev_period_year = period_year
                prev_period_interval_no = period_interval_no - 1
            prev_period = "%s_%s" % (prev_period_year, prev_period_interval_no)
            if prev_period not in period_list:
                continue

            exists = 0
            for row1 in ratings:
                if row1['period'] == prev_period:
                    exists = 1
                    prev_normalized_avg_rating = row1['normalized_'+yelp_or_glassdoor_str+'_avg_rating']
                    normalized_rating_diff_from_prev_period = row[
                                                                  'normalized_'+yelp_or_glassdoor_str+'_avg_rating'] - prev_normalized_avg_rating
                    row['normalized_rating_diff_from_prev_period'] = normalized_rating_diff_from_prev_period

            if exists == 1:
                final_data_to_store.append(row)
        return final_data_to_store

    def process_glassdoor_data(self):
        final_data_companywise = {}
        final_data_to_store = []

        for company_name in self.company_names:
            glassdoor_data = self.GetGlassdoorData(company_name)
            stock_data = self.GetStockData(company_name)
            avg_ratings = dict()
            factor_sum = dict()
            one_rating = dict()
            two_rating = dict()
            three_rating = dict()
            four_rating = dict()
            five_rating = dict()

            for row in glassdoor_data:
                if row['date'] == 'None':
                    continue
                date = (datetime.strptime(row['date'], "%b %d, %Y")).strftime("%m %Y")
                date_split = date.split(' ')
                month = int(date_split[0])
                year = date_split[1]
                month_interval_number = int((month - 1) / self.rating_avg_interval) + 1
                date_key = year + "_%s" % month_interval_number
                if date_key not in avg_ratings:
                    avg_ratings[date_key] = 0
                    factor_sum[date_key] = 0
                    one_rating[date_key] = 0
                    two_rating[date_key] = 0
                    three_rating[date_key] = 0
                    four_rating[date_key] = 0
                    five_rating[date_key] = 0

                # Factor formula
                overall_rating_factor = self.overall_rating_factor
                work_life_factor = self.work_life_factor
                culture_factor = self.culture_factor
                career_opp_factor = self.career_opp_factor
                compensation_benefits_factor = self.compensation_benefits_factor
                senior_management_factor = self.senior_management_factor
                if float(row['work_life']) == -1.0:
                    work_life_factor = 0
                if float(row['culture']) == -1.0:
                    culture_factor = 0
                if float(row['career_opp']) == -1.0:
                    career_opp_factor = 0
                if float(row['compensation_benefits']) == -1.0:
                    compensation_benefits_factor = 0
                if float(row['senior_management']) == -1.0:
                    senior_management_factor = 0

                av_rating = (((float(row['overall_rating']) * overall_rating_factor) + (float(row['work_life']) * work_life_factor) + (float(row['culture']) * culture_factor) + (float(row['career_opp']) * career_opp_factor)
                + (float(row['compensation_benefits']) * compensation_benefits_factor) + (float(row['senior_management']) * senior_management_factor))/
                                          (overall_rating_factor + work_life_factor + culture_factor + career_opp_factor + compensation_benefits_factor + senior_management_factor))

                if 4.5 < av_rating <= 5.0:
                    five_rating[date_key] += 1
                elif av_rating > 3.5:
                    four_rating[date_key] += 1
                elif av_rating > 2.5:
                    three_rating[date_key] += 1
                elif av_rating > 1.5:
                    two_rating[date_key] += 1
                elif av_rating > 0.5:
                    one_rating[date_key] += 1
                else:
                    print 'ERROR'
                    exit()

                avg_ratings[date_key] += av_rating
                factor_sum[date_key] += 1

            avg_ratings_val, glassdoor_ratings = \
                self.GetAvgRatings(company_name, avg_ratings, five_rating, four_rating, three_rating, two_rating,
                                   one_rating, factor_sum, stock_data, "glassdoor")

            period_list = self.GetPeriodList(glassdoor_ratings)
            glassdoor_ratings = self.NormalizeRatings(avg_ratings_val, glassdoor_ratings, "glassdoor")

            # final_data_to_store.extend(self.GetFinalData(glassdoor_ratings,period_list,"glassdoor"))
            final_data_to_store.extend(glassdoor_ratings)

            final_data_companywise[company_name] = glassdoor_ratings[:]

        return final_data_to_store, final_data_companywise

    def process_yelp_data(self):
        final_data_companywise = {}
        final_data_to_store = []
        for company_name in self.company_names:
            avg_ratings = dict()
            factor_sum = dict()
            one_rating = dict()
            two_rating = dict()
            three_rating = dict()
            four_rating = dict()
            five_rating = dict()
            yelp_data = self.GetYelpData(company_name)
            stock_data = self.GetStockData(company_name)

            for row in yelp_data:
                date = row['date']
                date_split = date.split('-')
                month = int(date_split[1])
                month_interval_number = int((month - 1) / self.rating_avg_interval) + 1
                year = date_split[0]
                date_key = year + "_%s" % month_interval_number
                if date_key not in avg_ratings:
                    avg_ratings[date_key] = 0
                    factor_sum[date_key] = 0
                    one_rating[date_key] = 0
                    two_rating[date_key] = 0
                    three_rating[date_key] = 0
                    four_rating[date_key] = 0
                    five_rating[date_key] = 0


                # useful_count = row['useful_count']
                # check_ins = row['check_ins']
                #
                # if useful_count > 30:
                #     factor = self.base + (self.useful_count_factor * 35)
                #
                # elif useful_count > 20:
                #     factor = self.base + (self.useful_count_factor * 20) + ((self.useful_count_factor)* (useful_count - 20))
                #
                # elif useful_count > 10:
                #     factor = self.base + (self.useful_count_factor * 10) + ((self.useful_count_factor)* (useful_count - 10))
                #
                # else:
                #     factor = self.base + (self.useful_count_factor * useful_count)
                #
                # if check_ins > 0:
                #     factor = factor*2
                factor = 1
                rating = float(row['rating'])
                avg_ratings[date_key] += (rating * factor)
                factor_sum[date_key] += factor
                if rating == 5.0:
                    five_rating[date_key] += 1
                elif rating == 4.0:
                    four_rating[date_key] += 1
                elif rating == 3.0:
                    three_rating[date_key] += 1
                elif rating == 2.0:
                    two_rating[date_key] += 1
                elif rating == 1.0:
                    one_rating[date_key] += 1
                else:
                    print 'ERROR'
                    exit()

            avg_ratings_val,yelp_ratings = \
                self.GetAvgRatings(company_name, avg_ratings, five_rating, four_rating, three_rating,
                                   two_rating, one_rating, factor_sum, stock_data, "yelp")

            period_list = self.GetPeriodList(yelp_ratings)
            yelp_ratings = self.NormalizeRatings(avg_ratings_val, yelp_ratings, "yelp")

            # final_data_to_store.extend(self.GetFinalData(yelp_ratings, period_list, "yelp"))
            final_data_to_store.extend(yelp_ratings)

            final_data_companywise[company_name] = yelp_ratings[:]

        return final_data_to_store, final_data_companywise

    def combine_yelp_glassdoor_final(self, final_data_yelp, final_data_glassdoor):
        final_data = []
        for row_yelp in final_data_yelp:
            for row_glassdoor in final_data_glassdoor:
                if(row_yelp['period'] == row_glassdoor['period'] and
                        row_yelp['company'] == row_glassdoor['company']):
                    # final_data.append({
                    #     "period": row_yelp['period'],
                    #     "company": row_yelp['company'],
                    #     "yelp_avg_rating": row_yelp['yelp_avg_rating'],
                    #     "normalized_yelp_avg_rating": row_yelp['normalized_yelp_avg_rating'],
                    #     "normalized_rating_diff_from_prev_period_yelp": row_yelp['normalized_rating_diff_from_prev_period'],
                    #     "glassdoor_avg_rating": row_glassdoor['glassdoor_avg_rating'],
                    #     "normalized_glassdoor_avg_rating": row_glassdoor['normalized_glassdoor_avg_rating'],
                    #     "normalized_rating_diff_from_prev_period_glassdoor": row_glassdoor["normalized_rating_diff_from_prev_period"],
                    #     "months_after_stock_price": row_yelp['months_after_stock_price'],
                    #     "stock_price_change": row_yelp['stock_price_change'],
                    #     "price_increase_or_decrease": row_yelp['price_increase_or_decrease']
                    # })
                    final_data.append({
                        "period": row_yelp['period'],
                        "company": row_yelp['company'],
                        "yelp_avg_rating": row_yelp['yelp_avg_rating'],
                        "normalized_yelp_avg_rating": row_yelp['normalized_yelp_avg_rating'],
                        "yelp_no_of_reviews": row_yelp['yelp_no_of_reviews'],
                        "yelp_5": row_yelp['yelp_5'], "percentage_yelp_5": row_yelp['percentage_yelp_5'], "yelp_4": row_yelp['yelp_4'], "yelp_3": row_yelp['yelp_3'],
                        "yelp_2": row_yelp['yelp_2'], "yelp_1": row_yelp['yelp_1'],
                        "glassdoor_avg_rating": row_glassdoor['glassdoor_avg_rating'],
                        "normalized_glassdoor_avg_rating": row_glassdoor['normalized_glassdoor_avg_rating'],
                        "glassdoor_no_of_reviews": row_glassdoor['glassdoor_no_of_reviews'],
                        "glassdoor_5": row_glassdoor['glassdoor_5'], "percentage_glassdoor_5": row_glassdoor['percentage_glassdoor_5'],
                        "glassdoor_4": row_glassdoor['glassdoor_4'],
                        "glassdoor_3": row_glassdoor['glassdoor_3'], "glassdoor_2": row_glassdoor['glassdoor_2'],
                        "glassdoor_1": row_glassdoor['glassdoor_1'],
                        "months_after_stock_price": row_yelp['months_after_stock_price'],
                        "stock_price_change": row_yelp['stock_price_change'],
                        "price_increase_or_decrease": row_yelp['price_increase_or_decrease']
                    })
                    break
        return final_data

    def combine_yelp_glassdoor_final_companywise(self, final_data_companywise_yelp, final_data_companywise_glassdoor):
        final_data_companywise = {}
        for key in final_data_companywise_yelp.keys():
            final_data_companywise[key] = []
            for row_yelp in final_data_companywise_yelp[key]:
                for row_glassdoor in final_data_companywise_glassdoor[key]:
                    if (row_yelp['period'] == row_glassdoor['period'] and
                            row_yelp['company'] == row_glassdoor['company']):
                        final_data_companywise[key].append({
                            "period": row_yelp['period'],
                            "company": row_yelp['company'],
                            "yelp_avg_rating": row_yelp['yelp_avg_rating'],
                            "normalized_yelp_avg_rating": row_yelp['normalized_yelp_avg_rating'],
                            "yelp_no_of_reviews": row_yelp['yelp_no_of_reviews'],
                            "yelp_5": row_yelp['yelp_5'], "percentage_yelp_5": row_yelp['percentage_yelp_5'],
                            "yelp_4": row_yelp['yelp_4'], "yelp_3": row_yelp['yelp_3'],
                            "yelp_2": row_yelp['yelp_2'], "yelp_1": row_yelp['yelp_1'],
                            "glassdoor_avg_rating": row_glassdoor['glassdoor_avg_rating'],
                            "normalized_glassdoor_avg_rating": row_glassdoor['normalized_glassdoor_avg_rating'],
                            "glassdoor_no_of_reviews": row_glassdoor['glassdoor_no_of_reviews'],
                            "glassdoor_5": row_glassdoor['glassdoor_5'], "percentage_glassdoor_5": row_glassdoor['percentage_glassdoor_5'],
                            "glassdoor_4": row_glassdoor['glassdoor_4'],
                            "glassdoor_3": row_glassdoor['glassdoor_3'], "glassdoor_2": row_glassdoor['glassdoor_2'],
                            "glassdoor_1": row_glassdoor['glassdoor_1'],
                            "months_after_stock_price": row_yelp['months_after_stock_price'],
                            "stock_price_change": row_yelp['stock_price_change'],
                            "price_increase_or_decrease": row_yelp['price_increase_or_decrease']
                        })
                        break
        return final_data_companywise

    def WriteToCSV(self, final_data):
        csv_columns = ["period", "company", "yelp_avg_rating", "normalized_yelp_avg_rating", "yelp_no_of_reviews",
                       "yelp_5", "percentage_yelp_5", "yelp_4", "yelp_3", "yelp_2", "yelp_1", "glassdoor_avg_rating",
                       "normalized_glassdoor_avg_rating", "glassdoor_no_of_reviews", "glassdoor_5", "percentage_glassdoor_5",
                       "glassdoor_4", "glassdoor_3", "glassdoor_2", "glassdoor_1", "months_after_stock_price",
                       "stock_price_change", "price_increase_or_decrease"]

        csv_file = "final_data/All_Final_Data.csv"

        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in final_data:
                    writer.writerow(data)
        except IOError:
            print("I/O error")

    def WriteToCSVCompanywise(self, final_data_companywise):
        csv_columns = ["period", "company", "yelp_avg_rating", "normalized_yelp_avg_rating", "yelp_no_of_reviews",
                       "yelp_5", "percentage_yelp_5", "yelp_4", "yelp_3", "yelp_2", "yelp_1", "glassdoor_avg_rating",
                       "normalized_glassdoor_avg_rating", "glassdoor_no_of_reviews", "glassdoor_5", "percentage_glassdoor_5",
                       "glassdoor_4",
                       "glassdoor_3", "glassdoor_2", "glassdoor_1", "months_after_stock_price", "stock_price_change",
                       "price_increase_or_decrease"]

        for company in final_data_companywise.keys():
            csv_file = "final_data/Final_Data_"+company+".csv"
            try:
                with open(csv_file, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in final_data_companywise[company]:
                        writer.writerow(data)
            except IOError:
                print("I/O error")

if __name__ == '__main__':
    data_pre_processing = DataPreProcessing([
                                            'Apple',
                                            'Gap',
                                            'HomeDepot',
                                            'Macys',
                                            'McDonalds',
                                            'Nike',
                                            ], 3, 0, 4, 5, 5, 12.0, 1.5, 1.5, 1.5, 1.5, 1.5)
    # company_names, rating_avg_interval, months_after, no_of_periods_as_features, useful_count_factor, base, overall_rating_factor

    final_data_yelp, final_data_companywise_yelp = data_pre_processing.process_yelp_data()
    final_data_glassdoor, final_data_companywise_glassdoor = data_pre_processing.process_glassdoor_data()

    final_data = data_pre_processing.combine_yelp_glassdoor_final(final_data_yelp,final_data_glassdoor)
    final_data_companywise = data_pre_processing.combine_yelp_glassdoor_final_companywise(final_data_companywise_yelp,
                                                                                          final_data_companywise_glassdoor)

    data_pre_processing.WriteToCSV(final_data)
    data_pre_processing.WriteToCSVCompanywise(final_data_companywise)
