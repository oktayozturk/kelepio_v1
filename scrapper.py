# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag, element
import urllib2
from urllib import urlencode as urlencoder
import tensorflow as tf
import os


def FetchBikeAds(brand, model):

    prefix = (brand + "-" + model).replace(" ", "-")
    df = GetBikeDataframeFromWEB(prefix)
    WriteBikeListToCSV(df, prefix)
    return df


def GetBikeDataframeFromWEB(prefix):

    print("Veriler URL'den alınıyor..............")

    url = "https://www.sahibinden.com/motosiklet-" + prefix + "?pagingSize=50"
    print("URL: %s" % url)
    html, next_page = GetPagesList(url)

    df = GetBikeDatasetFromHTML(html)

    print("Total Record Count: %s" % len(df))

    while len(next_page) > 0:
        print("Next Page: %s" % next_page)
        html, next_page = GetPagesList(next_page)
        df = df.append(GetBikeDatasetFromHTML(html))
        print("Total Record Count: %s" % len(df))

    print("Verilerin URL'den alınması işi bitti...........")

    return df


def WriteBikeListToCSV(dataframe, prefix):
    try:
        print(dataframe.dtypes)
        print("Veriler CSV dosyasına yazılıyor............")
        dataframe.to_csv(path_or_buf="CSV/" + prefix + ".csv", encoding="utf-8", index=False)
        print("Veriler CSV dosyasına yazılması işi bitti............")
    except Exception as exc:
        print("CSV dosyasını yazamadım...bişeyler yanlış gitti.")
        print("Dataframe lenght: %s" % len(dataframe) )
        print("Prefix: %s" % prefix)
        print("Hata mesajı: %s" % exc)
        raise


def GetBikeDatasetFromHTML(html):

    # Parse HTML data into BeautifulSOup object
    parser = BeautifulSoup(html, "html5lib")

    motosiklet_listesi = pd.DataFrame()

    # Get first TABLE list of all bikes from sahibinden.com
    liste = parser.table.tbody

    # Go through every TR in TABLE which is a bike record
    for tr in liste.findAll("tr"):

        bike_id = 0

        try:
            bike_id = int(tr["data-id"])
        except:
            # some TR is hidden and for advertising purposes, we eleminate them
            pass

        if bike_id > 0:

            bike = {"bike_id" : bike_id}

            for td in tr.findAll("td"):

                attr = {}

                if "searchResultsAttributeValue" in td['class']:
                    try:
                        km_or_year = float(td.string.replace(".", "").strip())

                        if "year" in bike:
                            bike["km"] = float(km_or_year)
                        else:
                            bike["year"] = float(km_or_year)

                    except:
                        attr["color"] = td.string.replace(".", "").strip()

                elif "searchResultsDateValue" in td['class']:
                    attr["ad_date"] =""
                    for tag in td.contents:
                        if type(tag) == element.Tag and tag.name == "span":
                            attr["ad_date"] = attr["ad_date"] + tag.string.replace(" ", "").strip()

                    attr["ad_date"] = fix_date_month(unicode(attr["ad_date"]).encode("utf-8"))

                elif "searchResultsLocationValue" in td['class']:
                    attr["city"] = td.contents[0].string.replace("<br>", "").replace(".", "").strip()

                elif "searchResultsTitleValue" in td['class'] :
                    for tag in td.contents:
                        if type(tag) == element.Tag and tag.name == "a" and "classifiedTitle" in tag["class"]:
                            attr["ad_title"] = tag.string.replace(":", "").replace(".", "").strip()
                            attr["ad_url"] = tag["href"]
                            details = getBikeDetail("https://www.sahibinden.com" + str(tag["href"]))
                            for k, v in details.iteritems():
                                bike[k] = v

                elif "searchResultsPriceValue" in td['class']:
                    price = unicode(td.contents[1].string).encode("utf-8")


                    if "TL" in price:
                        bike["price"] = float(price.replace("TL", "").replace(".", "").strip())

                    elif "$" in price:
                        bike["price"] = fix_currency(price.replace("$", "").replace(".", "").strip(),"$")

                    elif "€" in price:
                        bike["price"] = fix_currency(price.replace("€", "").replace(".", "").strip(), "€")

                    print(bike["price"])

                if len(attr) > 0:
                    for k, v in attr.iteritems():
                        bike[unicode(k).encode("utf-8").lower().replace("ü","u").replace("ö","o").replace("ı","i").replace("ğ","g").replace("ş","s").replace("ç","c").replace("(","").replace(")","")] = unicode(v).encode("utf-8")

            motosiklet_listesi = motosiklet_listesi.append([bike], bike.keys())


    motosiklet_listesi.set_index("bike_id")

    return motosiklet_listesi


def GetBikeHTML(url):
    try:

        user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
        values = {'name': 'Michael',
                  'location': 'Northampton',
                  'language': 'English'}
        headers = {'User-Agent': user_agent}

        data = urlencoder(values)


        req = urllib2.Request(url, data = data, headers = headers)

        response = urllib2.urlopen(req)

        html = response.read()

        return html.decode("utf-8")

    except Exception as exc:
        print(exc)
        print("url de bir sıkıntı var herhalde")


def GetPagesList(url):


    html = GetBikeHTML(url)

    parser = BeautifulSoup(html, "html5lib")

    #pages = [url.encode("utf-8")]
    next_page = ""
    #liste = parser.select("ul.pageNaviButtons li a")
    liste = parser.select("a.prevNextBut")

    for a in liste:
        if "sonraki" in a.text.lower():
            next_page = "https://www.sahibinden.com/" + a["href"]

    return html, next_page


def fix_currency(bike_price, bike_currency):
    if bike_currency == "€": return float(bike_price) * 5
    elif bike_currency == "$": return float(bike_price) * 4
    else: return bike_price


def fix_date_month(ad_date):

    return ad_date.replace("Ocak",".1.").replace("Şubat",".2.").replace("Mart",".3.").replace("Nisan",".4.").replace("Mayıs",".5.").replace("Haziran",".6.").replace("Temmuz",".7.").replace("Ağustos",".8.").replace("Eylül",".9.").replace("Ekim",".10.").replace("Kasım",".11.").replace("Aralık",".12.").replace(" ",".")


def getBikeDetail(url):

    parser = BeautifulSoup(GetBikeHTML(url), "html5lib")

    list = parser.find_all(id = "classifiedProperties")

    print("Detaylar toplanıyor..........................")

    dict = {}
    for tag in list:
        for li in tag.find_all("li"):
            dict[unicode(li.text.strip().replace(" ", "_")).lower()] = "selected" in li["class"]

    # -------- IS SAHIBINDEN? -----------------
    dict["sahibinden"] = bool(parser.select("span.fromOwner"))

    return dict

#---------------------- TEST TEST TEST TEST----------------------------------------------
#print("Testing functions:...........................")
#print("Total record count : %s" % len(FetchBike("bmw", "r 1200 gs")))