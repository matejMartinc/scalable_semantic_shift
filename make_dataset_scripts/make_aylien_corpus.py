import json
from nltk import sent_tokenize, RegexpTokenizer
from collections import defaultdict

tokenizer = RegexpTokenizer('[\(]|[\w-]+|\$[\d\.]+|\S+')

def build_lm_train_test():

    train = open('train_lm.txt', 'w', encoding='utf8')
    test = open('test_lm.txt', 'w', encoding='utf8')

    months = defaultdict(int)

    with open('aylien-covid-news.jsonl', 'r', encoding='utf8') as f:
        counter = 0
        for line in f:
            #print(line)
            doc = json.loads(line)
            #print(doc['author'])
            text = doc['body']
            text = " ".join(text.split())
            text = text.replace('“','').replace('”','').replace('"','')
            for sent in sent_tokenize(text):
                if counter % 10 == 0:
                    test.write(sent + '\n')
                else:
                    train.write(sent + '\n')
            time = doc["published_at"]
            month = time.split()[0].split('-')[1]
            months[month] += 1
            country = doc['locations']
            counter += 1
        print(counter)
        print('months: ', months)

def build_time_dataset():

    january = open('aylien_january.txt', 'w', encoding='utf8')
    february = open('aylien_february.txt', 'w', encoding='utf8')
    march = open('aylien_march.txt', 'w', encoding='utf8')
    april = open('aylien_april.txt', 'w', encoding='utf8')

    months = defaultdict(int)

    with open('aylien-covid-news.jsonl', 'r', encoding='utf8') as f:
        counter = 0
        for line in f:
            # print(line)
            doc = json.loads(line)
            # print(doc['author'])
            text = doc['body']
            text = " ".join(text.split())
            text = text.replace('“', '').replace('”', '').replace('"', '').strip()
            time = doc["published_at"]
            month = time.split()[0].split('-')[1]
            #country = doc['locations']
            counter += 1
            text = {'content': text}
            line = json.dumps(text) + '\n'
            if str(month) == '01':
                january.write(line)
                months['jan'] += 1
            if str(month) == '02':
                february.write(line)
                months['feb'] += 1
            if str(month) == '03':
                march.write(line)
                months['mar'] += 1
            if str(month) == '04':
                april.write(line)
                months['apr'] += 1
        print("Num all papers: ", counter)
        print('months: ', months)
    january.close()
    february.close()
    march.close()
    april.close()

#build_time_dataset()

def balanced_time_dataset_stats():

    months = {}
    months['jan'] = [0,0]
    months['feb'] = [0,0]
    months['mar'] = [0,0]
    months['apr'] = [0,0]

    with open('aylien-covid-news.jsonl', 'r', encoding='utf8') as f:
        counter = 0
        token_counter = 0
        for line in f:
            # print(line)
            doc = json.loads(line)
            # print(doc['author'])
            text = doc['body']
            text = " ".join(text.split())
            text = text.replace('“', '').replace('”', '').replace('"', '').strip()
            time = doc["published_at"]
            #print(time)
            month = time.split()[0].split('-')[1]
            #country = doc['locations']
            counter += 1
            text = {'content': text}
            num_tokens = len(json.dumps(text).split())
            token_counter += num_tokens
            if str(month) == '01':
                months['jan'][0] += 1
                months['jan'][1] += num_tokens
            if str(month) == '02':
                if counter % 4 == 0:
                    months['feb'][0] += 1
                    months['feb'][1] += num_tokens
            if str(month) == '03':
                if counter % 18 == 0:
                    months['mar'][0] += 1
                    months['mar'][1] += num_tokens
            if str(month) == '04':
                if counter % 4 == 0:
                    months['apr'][0] += 1
                    months['apr'][1] += num_tokens
        print("Num all papers: ", counter)
        print("Num tokens: ", token_counter)
        print('months (num papers, num_tokens): ', months)

#balanced_time_dataset_stats()


def build_balanced_time_dataset():

    january = open('aylien_january_balanced.txt', 'w', encoding='utf8')
    february = open('aylien_february_balanced.txt', 'w', encoding='utf8')
    march = open('aylien_march_balanced.txt', 'w', encoding='utf8')
    april = open('aylien_april_balanced.txt', 'w', encoding='utf8')

    months = defaultdict(int)

    with open('aylien-covid-news.jsonl', 'r', encoding='utf8') as f:
        counter = 0
        for line in f:
            # print(line)
            doc = json.loads(line)
            # print(doc['author'])
            text = doc['body']
            text = " ".join(text.split())
            text = text.replace('“', '').replace('”', '').replace('"', '').strip()
            time = doc["published_at"]
            #print(time)
            month = time.split()[0].split('-')[1]
            #country = doc['locations']
            counter += 1
            text = {'content': text}
            line = json.dumps(text) + '\n'
            if str(month) == '01':
                january.write(line)
                months['jan'] += 1
            if str(month) == '02':
                if counter % 4 == 0:
                    february.write(line)
                    months['feb'] += 1
            if str(month) == '03':
                if counter % 18 == 0:
                    march.write(line)
                    months['mar'] += 1
            if str(month) == '04':
                if counter % 4 == 0:
                    april.write(line)
                    months['apr'] += 1
        print("Num all papers: ", counter)
        print('months: ', months)
    january.close()
    february.close()
    march.close()
    april.close()

def get_target_words():


    freqs = defaultdict(int)
    punctuation = "!#$%&'()*+,.:;<=>?@[\]^_`{|}~"

    with open('aylien-covid-news.jsonl', 'r', encoding='utf8') as f:
        counter = 0
        for line in f:
            # print(line)
            doc = json.loads(line)
            # print(doc['author'])
            text = doc['body']
            text = " ".join(text.split())
            text = text.replace('“', '').replace('”', '').replace('"', '').lower()
            for sent in sent_tokenize(text):
                for word in tokenizer.tokenize(sent):
                    for p in punctuation:
                        if p in word:
                            break
                    else:
                        freqs[word] += 1

            counter += 1
        print(counter)

    print('All vocab size: ', len(freqs))

    freqs = list(freqs.items())
    freqs = sorted(freqs, key=lambda x:x[1], reverse=True)
    with open('vocab.csv', 'w', encoding='utf8') as f:
        f.write('word,mean\n')
        for w, freq in freqs[:10000]:
            f.write(w + ',' + str(freq) + '\n')


def get_stats():


    freqs = defaultdict(int)
    punctuation = "!#$%&'()*+,.:;<=>?@[\]^_`{|}~"

    with open('aylien-covid-news.jsonl', 'r', encoding='utf8') as f:
        counter = 0
        for line in f:
            # print(line)
            doc = json.loads(line)
            # print(doc['author'])
            text = doc['body']
            text = " ".join(text.split())
            text = text.replace('“', '').replace('”', '').replace('"', '').lower()
            for sent in sent_tokenize(text):
                for word in tokenizer.tokenize(sent):
                    for p in punctuation:
                        if p in word:
                            break
                    else:
                        freqs[word] += 1

            counter += 1
        print(counter)

    print('All vocab size: ', len(freqs))

    print("All tokens: ", sum(freqs.values()))
    freqs = list(freqs.items())
    freqs = sorted(freqs, key=lambda x:x[1], reverse=True)
    freqs = [x[1] for x in freqs][:10000]
    print("Vocab tokens: ", sum(freqs))



def build_country_dataset():

    usa = open('aylien_usa.txt', 'w', encoding='utf8')
    uk = open('aylien_uk.txt', 'w', encoding='utf8')

    countries = defaultdict(int)

    with open('aylien-covid-news.jsonl', 'r', encoding='utf8') as f:
        counter = 0
        for line in f:
            # print(line)
            doc = json.loads(line)
            # print(doc['author'])
            text = doc['body']
            text = " ".join(text.split())
            text = text.replace('“', '').replace('”', '').replace('"', '').strip()

            country = doc['source']['locations']
            #print(country)
            if country:
                country = country[0]['country']
                #print(country)
                counter += 1
                text = {'content': text}
                line = json.dumps(text) + '\n'
                countries[country] += 1
                '''if str(country) == 'usa':
                    usa.write(line)
                    countries['usa'] += 1
                if str(country) == 'uk':
                    usa.write(line)
                    countries['uk'] += 1'''

        print("Num all papers: ", counter)
        print('countries: ', countries)
    usa.close()
    uk.close()


def build_source_dataset():
    fox = open('aylien_fox.txt', 'w', encoding='utf8')
    cnn = open('aylien_cnn.txt', 'w', encoding='utf8')

    sources = defaultdict(int)

    with open('aylien-covid-news.jsonl', 'r', encoding='utf8') as f:
        counter = 0
        for line in f:
            # print(line)
            doc = json.loads(line)
            # print(doc['author'])
            text = doc['body']
            text = " ".join(text.split())
            text = text.replace('“', '').replace('”', '').replace('"', '').strip()

            source = doc['source']['domain']
            #print(source)
            counter += 1
            text = {'content': text}
            line = json.dumps(text) + '\n'
            sources[source] += 1
            if str(source) == 'foxnews.com':
                fox.write(line)
            if str(source) == 'cnn.com':
                cnn.write(line)


        print("Num all papers: ", counter)
        print('source: ', sorted(sources.items(), key=lambda x: x[1], reverse=True))
    fox.close()
    cnn.close()

def source_dataset_stats():

    sources = {}


    with open('aylien-covid-news.jsonl', 'r', encoding='utf8') as f:
        counter = 0
        for line in f:
            # print(line)
            doc = json.loads(line)
            # print(doc['author'])
            text = doc['body']
            text = " ".join(text.split())
            text = text.replace('“', '').replace('”', '').replace('"', '').strip()

            source = doc['source']['domain']
            #print(source)
            counter += 1
            text = {'content': text}
            num_tokens = len(json.dumps(text).split())
            if source in sources:
                sources[source][0] += 1
                sources[source][1] += num_tokens
            else:
                sources[source] = [1, num_tokens]




        print("Num all papers: ", counter)
        print('source: ', sorted(sources.items(), key=lambda x: x[1], reverse=True))

source_dataset_stats()

#get_stats()

#build_country_dataset()
#build_source_dataset()
#get_target_words()
#build_balanced_time_dataset()

#528848 articles
#balanced months:  defaultdict(<class 'int'>, {'apr': 19672, 'mar': 19833, 'feb': 18014, 'jan': 21102})

#months:  defaultdict(<class 'int'>, {'04': 78690, '03': 356983, '02': 72057, '01': 21102, '12': 8, '11': 8})
#countries:  defaultdict(<class 'int'>, {'US': 134231, 'AU': 14619, 'IN': 63087, 'NZ': 4296, 'PH': 7505, 'GB': 74829, 'CA': 20099, 'IT': 2653, 'RU': 4733, 'IE': 5634, 'KR': 946, 'DE': 715, 'CN': 4560, 'ES': 714, 'JP': 1447, 'GH': 2593, 'HK': 3136, 'KE': 2839, 'NG': 2143, 'FR': 765, 'ZA': 1181, 'MX': 169, 'UA': 220, 'IL': 68, 'PK': 7847, 'IR': 344, 'FI': 194, 'BE': 182, None: 477, 'MT': 74, 'SV': 1, 'MM': 234, 'DO': 1, 'NL': 31, 'PT': 2, 'LU': 57, 'ID': 769, 'AR': 2, 'MA': 2, 'EC': 20, 'BR': 4, 'TW': 6, 'CH': 1, 'TN': 1})
#sources::  [('reuters.com', 35938), ('dailymail.co.uk', 31378), ('yahoo.com', 26279), ('business-standard.com', 15469), ('urdupoint.com', 13925), ('bizjournals.com', 12603), ('indiatimes.com', 11939), ('cbslocal.com', 9423), ('seekingalpha.com', 8644), ('thehindu.com', 8262), ('thesun.co.uk', 7713), ('inquirer.net', 7504), ('news.com.au', 7195), ('investing.com', 7058), ('straitstimes.com', 6867), ('google.ca', 6819), ('forbes.com', 6578), ('businessinsider.com', 6433), ('thenews.com.pk', 6288), ('cnn.com', 6107), ('news18.com', 6090), ('metro.co.uk', 5985), ('mirror.co.uk', 5834), ('foxnews.com', 5792), ('stuff.co.nz', 5563), ('indianexpress.com', 5406), ('ndtv.com', 5373), ('fool.com', 4999), ('ctvnews.ca', 4919), ('washingtonpost.com', 4869), ('express.co.uk', 4808), ('thehill.com', 4780), ('google.com', 4582), ('theguardian.com', 4453), ('cnbc.com', 4441), ('sfgate.com', 4337), ('nypost.com', 4229), ('bbc.co.uk', 4227), ('globalnews.ca', 3999), ('usatoday.com', 3955), ('nzherald.co.nz', 3943), ('nytimes.com', 3724), ('hindustantimes.com', 3661), ('marketwatch.com', 3603), ('cbc.ca', 3426), ('nydailynews.com', 3185), ('scmp.com', 3126), ('abc.net.au', 3072), ('independent.ie', 2911), ('moneycontrol.com', 2805), ('smh.com.au', 2730), ('irishtimes.com', 2723), ('nbcnews.com', 2670), ('theepochtimes.com', 2663), ('breitbart.com', 2650), ('ghanaweb.com', 2593), ('latimes.com', 2590), ('rappler.com', 2583), ('vanguardngr.com', 2556), ('cbsnews.com', 2513), ('sputniknews.com', 2502), ('abs-cbn.com', 2449), ('intoday.in', 2430), ('npr.org', 2395), ('go.com', 2271), ('rt.com', 2059), ('thetimes.co.uk', 2035), ('google.it', 1972), ('dailycaller.com', 1934), ('chicagotribune.com', 1914), ('people.com', 1891), ('people.com.cn', 1884), ('zerohedge.com', 1864), ('wsj.com', 1854), ('cnet.com', 1797), ('standardmedia.co.ke', 1708), ('huffingtonpost.com', 1705), ('newsweek.com', 1693), ('firstpost.com', 1680), ('msnbc.com', 1624), ('variety.com', 1595), ('sbs.com.au', 1578), ('rediff.com', 1473), ('thedailybeast.com', 1434), ('kbs.co.kr', 1431), ('dailypost.ng', 1430), ('skysports.com', 1421), ('comicbook.com', 1389), ('nbcsports.com', 1359), ('chron.com', 1341), ('usnews.com', 1328), ('sky.com', 1260), ('deadline.com', 1217), ('goal.com', 1212), ('indiatoday.in', 1204), ('dailystar.co.uk', 1181), ('news24.com', 1181), ('sportskeeda.com', 1139), ('nation.co.ke', 1131), ('aljazeera.com', 1124), ('nikkei.com', 1102), ('fortune.com', 1078), ('aol.com', 1061), ('thehansindia.com', 1055), ('billboard.com', 1043), ('msn.com', 1015), ('itv.com', 984), ('techcrunch.com', 974), ('denverpost.com', 911), ('time.com', 891), ('hotnewhiphop.com', 872), ('politico.com', 865), ('cbssports.com', 848), ('dunyanews.tv', 834), ('buzzfeed.com', 833), ('vice.com', 820), ('telegraph.co.uk', 781), ('mainichi.jp', 773), ('psychologytoday.com', 770), ('tempo.co', 769), ('realclearpolitics.com', 749), ('india.com', 731), ('mashable.com', 730), ('dawn.com', 725), ('premiumtimesng.com', 713), ('marca.com', 702), ('theverge.com', 702), ('techradar.com', 691), ('foxbusiness.com', 683), ('dw.com', 676), ('jiji.com', 674), ('theadvocate.com', 665), ('www.gov.uk', 661), ('ansa.it', 651), ('espn.com', 613), ('alaraby.co.uk', 613), ('vox.com', 606), ('rollingstone.com', 600), ('bgr.com', 600), ('slate.com', 557), ('qz.com', 547), ('complex.com', 537), ('zdnet.com', 537), ('asahi.com', 535), ('vnexpress.net', 533), ('chosun.com', 531), ('fastcompany.com', 530), ('pbs.org', 526), ('eonline.com', 523), ('ycombinator.com', 502), ('mehrnews.com', 501), ('france24.com', 500), ('digitalspy.com', 489), ('euronews.com', 488), ('bustle.com', 483), ('ensonhaber.com', 472), ('ign.com', 460), ('digitaltrends.com', 455), ('lindaikejisblog.com', 444), ('gizmodo.com', 442), ('theatlantic.com', 384), ('tmz.com', 382), ('vulture.com', 375), ('psu.edu', 375), ('tradingeconomics.com', 373), ('cointelegraph.com', 363), ('radiotimes.com', 359), ('brobible.com', 354), ('vanityfair.com', 352), ('gamespot.com', 352), ('webmd.com', 351), ('medicinenet.com', 341), ('pcmag.com', 333), ('wired.com', 323), ('beinsports.com', 323), ('timeout.com', 322), ('popsugar.com', 319), ('9to5mac.com', 300), ('nymag.com', 299), ('vogue.com', 294), ('irna.ir', 293), ('heavy.com', 292), ('channel4.com', 274), ('inc.com', 273), ('motorsport.com', 260), ('europa.eu', 258), ('menshealth.com', 255), ('rfi.fr', 255), ('livescience.com', 251), ('phys.org', 246), ('lifehacker.com', 240), ('androidcentral.com', 238), ('irrawaddy.com', 234), ('avclub.com', 233), ('medium.com', 231), ('techspot.com', 229), ('jezebel.com', 211), ('movieweb.com', 210), ('investopedia.com', 206), ('yle.fi', 194), ('thomsonreuters.com', 192), ('hitc.com', 187), ('arstechnica.com', 183), ('economist.com', 180), ('newyorker.com', 169), ('macrumors.com', 162), ('hypebeast.com', 160), ('polygon.com', 157), ('entrepreneur.com', 157), ('meduza.io', 156), ('indiewire.com', 152), ('techrepublic.com', 152), ('hpe.com', 151), ('refinery29.com', 148), ('ew.com', 147), ('censor.net.ua', 146), ('engadget.com', 143), ('ibtimes.co.in', 140), ('cosmopolitan.com', 137), ('pcgamer.com', 136), ('jalopnik.com', 133), ('bmj.com', 133), ('eluniversal.com.mx', 132), ('elle.com', 125), ('scientificamerican.com', 123), ('kotaku.com', 121), ('nature.com', 121), ('sbnation.com', 121), ('sciencemag.org', 119), ('un.org', 117), ('gq.com', 114), ('motor1.com', 110), ('sciencedaily.com', 109), ('sme.sk', 108), ('pagesix.com', 108), ('gsmarena.com', 107), ('producthunt.com', 107), ('pitchfork.com', 101), ('eurogamer.net', 99), ('esquire.com', 97), ('harvard.edu', 96), ('legacy.com', 95), ('hackernoon.com', 89), ('healthline.com', 89), ('bbc.com', 86), ('bloomberg.com', 80), ('nba.com', 76), ('space.com', 75), ('fivethirtyeight.com', 74), ('timesofmalta.com', 74), ('cdc.gov', 72), ('snopes.com', 72), ('delta.com', 71), ('ynet.co.il', 68), ('theglobeandmail.com', 66), ('yelp.com', 62), ('knowyourmeme.com', 62), ('virginia.edu', 57), ('uol.com.br', 56), ('techtarget.com', 55), ('mk.co.kr', 55), ('tasnimnews.com', 51), ('deadspin.com', 51), ('worldbank.org', 48), ('ucla.edu', 48), ('parliament.uk', 44), ('hbr.org', 43), ('nih.gov', 40), ('howtogeek.com', 40), ('acm.org', 40), ('pcworld.com', 39), ('thekitchn.com', 38), ('xe.com', 38), ('mit.edu', 38), ('moneysavingexpert.com', 37), ('medscape.com', 36), ('spiegel.de', 36), ('milenio.com', 35), ('www.nhs.uk', 33), ('archdaily.com', 31), ('ieee.org', 31), ('ucdavis.edu', 30), ('archive.org', 30), ('elsevier.com', 29), ('umich.edu', 28), ('ca.gov', 27), ('canada.ca', 24), ('makeuseof.com', 23), ('arizona.edu', 23), ('anandtech.com', 22), ('elcomercio.com', 20), ('northwestern.edu', 19), ('slideshare.net', 18), ('kbb.com', 18), ('uchicago.edu', 18), ('plos.org', 18), ('prnewswire.com', 17), ('ox.ac.uk', 16), ('sec.gov', 16), ('ft.com', 15), ('epochtimes.com', 15), ('ufl.edu', 14), ('dzone.com', 14), ('usda.gov', 14), ('edmunds.com', 13), ('researchgate.net', 11), ('who.int', 11), ('nfl.com', 10), ('ted.com', 10), ('lapatilla.com', 9), ('fao.org', 9), ('lefigaro.fr', 8), ('repubblica.it', 8), ('ed.gov', 7), ('sciencenet.cn', 7), ('cool3c.com', 7), ('msu.edu', 7), ('lastampa.it', 7), ('nasdaq.com', 7), ('newsru.com', 6), ('linkedin.com', 5), ('derstandard.at', 5), ('20minutos.es', 5), ('film.ru', 5), ('ftchinese.com', 5), ('acs.org', 4), ('giantbomb.com', 4), ('nj.com', 4), ('fitbit.com', 4), ('frontiersin.org', 4), ('bodybuilding.com', 4), ('haberler.com', 4), ('orf.at', 4), ('appledaily.com', 4), ('udn.com', 4), ('xinhuanet.com', 4), ('infobae.com', 3), ('cisco.com', 3), ('scitation.org', 3), ('cuny.edu', 3), ('hootsuite.com', 3), ('nasa.gov', 3), ('abril.com.br', 3), ('intel.com', 3), ('barclays.co.uk', 3), ('twitter.com', 3), ('issuu.com', 2), ('hubspot.com', 2), ('sapo.pt', 2), ('nu.nl', 2), ('publico.es', 2), ('oracle.com', 2), ('lenta.ru', 2), ('consultant.ru', 2), ('sdpnoticias.com', 2), ('lanacion.com.ar', 2), ('le360.ma', 2), ('allocine.fr', 2), ('ilgiornale.it', 2), ('sohu.com', 2), ('biomedcentral.com', 2), ('elpais.com', 2), ('laprensagrafica.com', 1), ('diariolibre.com', 1), ('drugs.com', 1), ('nseindia.com', 1), ('abc.es', 1), ('sozcu.com.tr', 1), ('pravda.com.ua', 1), ('weather.gov', 1), ('wa.gov', 1), ('gds.it', 1), ('eleconomista.es', 1), ('ltn.com.tw', 1), ('sports.ru', 1), ('zeit.de', 1), ('20minutes.fr', 1), ('codeproject.com', 1), ('microsoft.com', 1), ('sport.es', 1), ('mingpao.com', 1), ('gmanetwork.com', 1), ('oreilly.com', 1), ('globo.com', 1), ('aif.ru', 1), ('fandango.com', 1), ('mediaset.it', 1), ('google.com.br', 1), ('as.com', 1), ('lequipe.fr', 1), ('ura.news', 1), ('jawharafm.net', 1), ('cna.com.tw', 1), ('corriere.it', 1), ('princeton.edu', 1)]


