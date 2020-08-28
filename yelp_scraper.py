from bs4 import BeautifulSoup
import requests
from time import sleep
import csv

location_links = {'NewYork': 'New%20York%2C%20NY', 'Chicago': 'Chicago%2C+IL', 'LosAngeles': 'Los%20Angeles%2C%20CA',
                  'Boston': 'Boston%2C%20MA', 'SanFransisco': 'San%20Francisco%2C%20CA', 'Houston': 'Houston%2C%20TX',
                  'Phoenix': 'Phoenix%2C%20AZ', 'Dallas': 'Dallas%2C%20TX', 'SanJose': 'San%20Jose%2C%20CA',
                  'Austin': 'Austin%2C%20TX', 'SanDiego': 'San%20Diego%2C%20CA'}
# company_desc = {'Gap': 'gap', 'Nike': 'nike', 'Adidas': 'adidas', 'Target': 'Target',
#                 'Macys': 'Macys%20Department%20Store', 'Walmart': 'Walmart%20Super%20Store',
#                 'Apple': 'Apple%20Store', 'HomeDepot': 'The+Home+Depot', 'McDonalds': 'McDonald%27s',
#                 'Starbucks': 'Starbucks', 'Chipotle': 'Chipotle+Mexican+Grill', 'BestBuy': 'Best+Buy'}
company_desc = {'Chipotle': 'Chipotle+Mexican+Grill', 'BestBuy': 'Best+Buy'}

biz_link = {'Nike': 'nike', 'Walmart': 'walmart', 'Apple': 'apple', 'Adidas': 'adidas',
            'HomeDepot': 'the-home-depot', 'McDonalds': 'mcdonalds', 'Target': 'target', 'Macys': 'macys',
            'Starbucks': 'starbucks', 'Chipotle': 'chipotle', 'BestBuy': 'best-buy', 'Gap': 'gap'}
for company in company_desc:
    reviews = []
    store_links = []
    for location_name in location_links:
        r = requests.get('https://www.yelp.com/search?find_desc=%s&find_loc=%s&sortby=review_count'
                         % (company_desc[company], location_links[location_name]))

        soup = BeautifulSoup(r.text, 'html.parser')

        for store in soup.find_all("a"):
            link = store.get('href')
            if link and link.startswith('/biz/%s' % biz_link[company]) and link not in store_links\
                    and link.find('?hrid=') == -1:
                store_links.append(link)

    print store_links
    for link in store_links:
        req = requests.get('https://www.yelp.com%s' % link)
        soup_2 = BeautifulSoup(req.text, 'html.parser')
        start = 0
        while True:
            sleep(0.25)
            print 'Iteration'
            for review in soup_2.find_all('div', itemprop='review'):
                start += 1
                rating = float(review.find('meta', itemprop='ratingValue')['content'])
                date = review.find('meta', itemprop='datePublished')['content']
                reviews.append({'date': date, 'rating': rating, 'link': link})

                # friend_count =\
                #     int(review.find('li', 'friend-count responsive-small-display-inline-block').find('b').text)
                # review_count =\
                #     int(review.find('li', 'review-count responsive-small-display-inline-block').find('b').text)
                # photo = review.find('li', 'photo-count responsive-small-display-inline-block')
                # photo_count = 0
                # check_in_count = 0
                # if photo:
                #     photo_count = int(photo.find('b').text)
                # check_in_tag = review.find('li', 'review-tags_item')
                # if check_in_tag:
                #     check_in_text = check_in_tag.find('span').text.strip()
                #     if check_in_text:
                #         if check_in_text.find('Review'):
                #             check_in_count = 0
                #         else:
                #             check_in_count = int(check_in_text.split(' ')[0])
                #     else:
                #         check_in_count = 0
                # useful_count_text_temp = review.find('a', 'ybtn ybtn--small ybtn--secondary useful'
                #                                      ' js-analytics-click')
                # if useful_count_text_temp:
                #     useful_count_text = useful_count_text_temp.find('span', 'count').text.strip()
                #     if useful_count_text:
                #         if useful_count_text == '':
                #             useful_count = 0
                #         else:
                #             useful_count = int(useful_count_text)
                #     else:
                #         useful_count = 0
                # else:
                #     useful_count = 0

                # reviews.append({'date': date.text.strip().split('\n')[0], 'rating': rating, 'friends': friend_count,
                #                 'reviews': review_count, 'photos': photo_count, 'check_ins': check_in_count,
                #                 'useful_count': useful_count, 'link': link})

            req = requests.get('https://www.yelp.com%s&start=%s' % (link, start))
            soup_2 = BeautifulSoup(req.text, 'html.parser')
            if not soup_2.find_all('div', itemprop='review'):
                break

    csv_columns = ['date', 'rating', 'link']

    csv_file = "Yelp_%s.csv" % company

    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in reviews:
                writer.writerow(data)
    except IOError:
        print("I/O error")
