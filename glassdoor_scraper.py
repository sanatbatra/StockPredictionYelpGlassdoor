
# import sys
# sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')

import random
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait

user_agents = [
'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0',
]

header = {}
def randomUserAgents():
    head = random.choice(user_agents)
    head = head.split(") ")
    agent = head[0]+')'
    accept = head[1]
    header["User-Agent"] = agent
    header["Accept"] = accept
    return header

# url = 'https://www.glassdoor.com/Reviews/NIKE-Reviews-E1699.htm'
head = randomUserAgents()
start = time.time()


def soup(url, headers):
    ''' url = full glassdoor.com/reviews url'''
    session = requests.Session()
    req = session.get(url, headers=headers)
    bs = BeautifulSoup(req.text, 'lxml')
    return bs


startTime = start - time.time()

a = []
date = []
revNo = []
employee = []
position = []
summ = []
pro = []
con = []
advice = []
review = []
subReviews = []
workLife = []
culture = []
helpful = []
careerOpp = []
compBenefits = []
srManagement = []
authorlocation = []
recommend = []
outlook = []
ceo = []
link = []
# "NIKE-Reviews-E1699", "adidas-Reviews-E10692"
url_list = {'Nike':"NIKE-Reviews-E1699", 'Adidas':"adidas-Reviews-E10692", 'Walmart':"Walmart-Reviews-E715", 'Macys':"Macy-s-Reviews-E1079", 'Target':"Target-Reviews-E194", 'Gap':"Gap-Reviews-E114118",'HomeDepot':"The-Home-Depot-Reviews-E655",'Apple':"Apple-Reviews-E1138",'McDonalds':"McDonald-s-Reviews-E432",'Starbucks':"Starbucks-Reviews-E2202",'Chipotle':"Chipotle-Reviews-E15228",'BestBuy':"Best-Buy-Reviews-E97"}
# url_list = ["Target-Reviews-E194"]
# for url_str_key,url_str_value in url_list.items():
# m=2
# while m == 2:
print("walmart")
count = 1
    # url_prefix = "https://www.glassdoor.com/Reviews/" + url_str_value
    # bs = soup(url_prefix + ".htm", head)
i = 1
k = 1
while k == 1:
        # print('hello')
        # z = bs.find('li',{'class','pagination__PaginationStyle__next'}).find('a')['class'][0].find('disabled')
        # print(z)
        # chrome_options = webdriver.ChromeOptions()
        # chrome_options.add_argument('--window-size=1420,1080')
        # chrome_options.add_argument('--headless')
        # chrome_options.add_argument('--no-sandbox')
        # chrome_options.add_argument('--disable-dev-shm-usage')
        # driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
        #Here Search for the company url and write it here, we have to do this beacause for every company there's a random number associated to it
        url = "https://www.glassdoor.com/Reviews/adidas-Reviews-E10692_P" + str(i) + ".htm"
        # driver.get(url)
        # driver = webdriver.Firefox()
        # driver.get(url)
        print(url)
        i = i + 1
        # content = driver.page_source
        # time.sleep(1)
        # element = WebDriverWait(driver, 10)
        # bs = BeautifulSoup(content)
        bs = soup(url, head)
        print(" ")
        # bs.find('li', {'class', 'pagination__PaginationStyle__next'}) == None or 'disabled' in ('\t'.join(bs.find('li', {'class', 'pagination__PaginationStyle__next'}).find('a')['class'])):
        if bs.find('li', {'class', 'pagination__PaginationStyle__next'}) == None:
          print('None in this page')
        elif i == 211:
            # here change the value to the last page number of the particular company
            # this is being done because glassdoor is changing the identification of last page for a company frequently
            # print(bs)
            k = 2
            exit()
        else:
            print(i)
            for x in bs.findAll('li', {'class', 'empReview cf'}):
                a.append(count)
                count += 1

                ## Rev Number
                try:
                    revNo.append(x['id'])
                except:
                    revNo.append('None')

                ## overall rating
                try:
                    review.append(x.find('span', {'class': 'rating'}).find('span', {'class': 'value-title'})['title'])
                except:
                    review.append('None')

                ## subRatings list
                try:
                    subclasspresent = {}
                    subratingarrays = [workLife, culture, careerOpp, compBenefits, srManagement]
                    subratingclasses = ['Work/Life Balance', 'Culture & Values', 'Career Opportunities',
                                        'Compensation and Benefits', 'Senior Management']
                    z = 0
                    for subclass in range(len(subratingclasses)):
                        if subratingclasses[subclass] in (x.find('ul', {'class': 'undecorated'}).text):
                            z = 1
                        else:
                            subratingarrays[subclass].append(-1)
                    for rate in x.find('ul', {'class': 'undecorated'}).findAll('li'):
                        subratingarrays[subratingclasses.index(rate.find('div', {'class': 'minor'}).text)].append(
                            rate.find('span', {'class': 'gdBars gdRatings med'})['title'])
                except:
                    for subratingarr in subratingarrays:
                        subratingarr.append(-1)
                    # print('Error in subratings list')

                ## Date
                try:
                    # print(x.find('time', {'class': 'date subtle small'}).text)
                    date.append(x.find('time', {'class': 'date subtle small'}).text)
                except:
                    date.append('None')

                ## Employee Type
                try:
                    employee.append(x.find('span', {'class': "authorJobTitle"}).text)
                    # print(x.find('span',{'class':"authorJobTitle"}).text)
                except:
                    employee.append('None')

                ##Location
                try:
                    position.append(x.find('span', {'class': 'authorLocation'}).text)
                except:
                    position.append('None')

                ##Recommendoutlookceo
                try:
                    subarraying = [recommend, outlook, ceo]
                    counti = [0, 0, 0]
                    indices = {'Recommends':5,'Doesn\'t Recommend':0,'None':-1,'Positive Outlook':5,'Neutral Outlook':2.5,'Negative Outlook':0,'Approves of CEO':5,'No opinion of CEO':2.5,'Disapproves of CEO':0}
                    for subreview in x.find('div', {'class': 'row reviewBodyCell recommends'}).findAll('div', {
                        'class': 'col-sm-4'}):
                        if 'Recommend' in subreview.find('span').text:
                            counti[0] = 1
                            recommend.append(indices[subreview.find('span').text])
                        elif 'Outlook' in subreview.find('span').text:
                            counti[1] = 1
                            outlook.append(indices[subreview.find('span').text])
                        elif 'CEO' in subreview.find('span').text:
                            counti[2] = 1
                            ceo.append(indices[subreview.find('span').text])
                    for indi in range(len(counti)):
                        if counti[indi] == 0:
                            subarraying[indi].append('None')
                except:
                    recommend.append('None')
                    outlook.append('None')
                    ceo.append('None')
                    # print('Error in recommendoutlook try block')

                ## Helpful votes
                try:
                    helpful.append(
                        re.findall('\d+', x.find('div', {'class': 'helpfulReviews helpfulCount small subtle'}).text)[0])
                except:
                    # print('dfs')
                    helpful.append('None')
        # driver.quit()
        # time.sleep(1)
df = pd.DataFrame(index=a)
df['date'] = date
df['reviewNo'] = revNo
df['overallStar'] = review
df['workLifeStar'] = workLife
df['cultureStar'] = culture
df['careerOppStar'] = careerOpp
df['comBenefitsStar'] = compBenefits
df['srManagementStar'] = srManagement
df['employeeType'] = employee
df['location'] = position
df['recommend'] = recommend
df['outlook'] = outlook
df['ceo'] = ceo
df['helpful'] = helpful
print(df)
df.tail()
df.to_csv('Glassdor_Adidas.csv', sep=',')