# -*- coding: utf-8 -*-
# !/usr/bin/python

import pandas as pd
from bs4 import BeautifulSoup, element
import urllib2
from urllib import urlencode as urlencoder
import os


class AdScrapper(object):

    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
        self.prefix = str(self.brand + "-" + self.model).replace(" ", "-")

    def FetchBikeAds(self):

        df = self.GetBikeDataframeFromWEB()
        self.WriteBikeListToCSV(df)
        return df

    def GetBikeDataframeFromWEB(self):

        print("Veriler URL'den alınıyor..............")
        url = "https://www.sahibinden.com/motosiklet-" + self.prefix + "?pagingSize=50"
        print("URL: %s" % url)
        html, next_page = self.GetPagesList(url)

        df = self.GetBikeDatasetFromHTML(html)

        print("Total Record Count: %s" % len(df))

        while len(next_page) > 0:
            print("Next Page: %s" % next_page)
            html, next_page = self.GetPagesList(next_page)
            df = df.append(self.GetBikeDatasetFromHTML(html))
            print("Total Record Count: %s" % len(df))

        print("Verilerin URL'den alınması işi bitti...........")

        return df

    def WriteBikeListToCSV(self, dataframe):
        try:
            print("Veriler CSV dosyasına yazılıyor............")
            dataframe.to_csv(path_or_buf="CSV/" + self.prefix + ".csv", encoding="utf-8", index=False)
            print("Veriler CSV dosyasına yazılması işi bitti............")
        except Exception as exc:
            print("CSV dosyasını yazamadım...bişeyler yanlış gitti.")
            print("Dataframe lenght: %s" % len(dataframe))
            print("Prefix: %s" % prefix)
            print("Hata mesajı: %s" % exc)
            raise

    def GetBikeDatasetFromHTML(self, html):

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
                bike = {"bike_id": bike_id}
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

                        attr["ad_date"] = self.fix_date_month(unicode(attr["ad_date"]).encode("utf-8"))

                    elif "searchResultsLocationValue" in td['class']:
                        attr["city"] = td.contents[0].string.replace("<br>", "").replace(".", "").strip()

                    elif "searchResultsTitleValue" in td['class'] :
                        for tag in td.contents:
                            if type(tag) == element.Tag and tag.name == "a" and "classifiedTitle" in tag["class"]:
                                attr["ad_title"] = tag.string.replace(":", "").replace(".", "").strip()
                                attr["ad_url"] = tag["href"]
                                details = self.getBikeDetail("https://www.sahibinden.com" + str(tag["href"]))
                                for k, v in details.iteritems():
                                    bike[k] = v

                    elif "searchResultsPriceValue" in td['class']:
                        price = unicode(td.contents[1].string).encode("utf-8")


                        if "TL" in price:
                            bike["price"] = float(price.replace("TL", "").replace(".", "").strip())

                        elif "$" in price:
                            bike["price"] = self.fix_currency(price.replace("$", "").replace(".", "").strip(),"$")

                        elif "€" in price:
                            bike["price"] = self.fix_currency(price.replace("€", "").replace(".", "").strip(), "€")

                        print(bike["price"])

                    if len(attr) > 0:
                        for k, v in attr.iteritems():
                            bike[unicode(k).encode("utf-8").lower().replace("ü","u").replace("ö","o").replace("ı","i").replace("ğ","g").replace("ş","s").replace("ç","c").replace("(","").replace(")","")] = unicode(v).encode("utf-8")

                motosiklet_listesi = motosiklet_listesi.append([bike], bike.keys())


        motosiklet_listesi.set_index("bike_id")

        return motosiklet_listesi

    def GetBikeHTML(self, url):
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

    def GetPagesList(self, url):

        html = self.GetBikeHTML(url)
        parser = BeautifulSoup(html, "html5lib")
        #pages = [url.encode("utf-8")]
        next_page = ""
        #liste = parser.select("ul.pageNaviButtons li a")
        liste = parser.select("a.prevNextBut")

        for a in liste:
            if "sonraki" in a.text.lower():
                next_page = "https://www.sahibinden.com/" + a["href"]

        return html, next_page

    def fix_currency(self, bike_price, bike_currency):
        if bike_currency == "€": return float(bike_price) * 5
        elif bike_currency == "$": return float(bike_price) * 4
        else: return bike_price

    def fix_date_month(self, ad_date):

        return ad_date.replace("Ocak",".1.").replace("Şubat",".2.").replace("Mart",".3.").replace("Nisan",".4.").replace("Mayıs",".5.").replace("Haziran",".6.").replace("Temmuz",".7.").replace("Ağustos",".8.").replace("Eylül",".9.").replace("Ekim",".10.").replace("Kasım",".11.").replace("Aralık",".12.").replace(" ",".")

    def getBikeDetail(self, url):

        parser = BeautifulSoup(self.GetBikeHTML(url), "html5lib")

        list = parser.find_all(id="classifiedProperties")

        print("Detaylar toplanıyor..........................")

        dict = {}
        for tag in list:
            for li in tag.find_all("li"):
                dict[unicode(li.text.strip().replace(" ", "_")).lower()] = "selected" in li["class"]

        # -------- IS SAHIBINDEN? -----------------
        dict["sahibinden"] = bool(parser.select("span.fromOwner"))

        return dict


class BikeDataScrapper(object):

    def __init__(self, brand=""):

        self.base_Url = "http://www.motorcyclespecs.co.za"
        self.brands = {}
        self.load_brands()
        self.brand = ""
        self.brand_pages = {}
        self.model_search_results = []
        self.model_urls = {}
        self.model = ""
        self.model_specs = []
        self.model_images = []

        if str(brand):
            self.set_brand_by_name(brand)

        if str(brand):
            self.load_model_detail_pages()
            self.search_models()

    def write_specs_to_csv(self):
        assert self.model_specs, "Model specleri yüklenmemiş......"
        keys = dict(self.model_specs[0]).keys()

        #Find different columns in diferent bke models
        for i in range(len(self.model_specs)):
            diff = set(dict(self.model_specs[i]).keys()) - set(keys)
            if diff:
                print("Added columns : {}".format(list(diff)))
                keys.append(list(diff)[0])

        # Merge columns and values into DF
        df = pd.DataFrame(columns=keys)
        for i, model in enumerate(self.model_specs):
            df = df.append(pd.Series(dict(model)), ignore_index=True)

        # Write merged specs into CSV
        assert len(df) > 0, "Bike spec'leri yazacağım ama hiç spec gelmemiş..."
        import os
        path = "Specs/{}/{}/".format(self.searchable_text(self.brand), self.searchable_text(self.model))
        if not os.path.exists(path):
            os.makedirs(path + "pics")
        df.to_csv(path_or_buf=path+"specs.csv", encoding="utf-8", index=False)

    @staticmethod
    def searchable_text(text):
        return unicode('_'.join(str(text).lower().strip().split()))

    def search_models(self, model_name=""):
        import re

        lookup = unicode('_'.join(model_name.lower().strip().split()))
        self.model_search_results = []

        for key in self.model_urls.keys():
            if re.search(lookup, unicode('_'.join(key.lower().strip().split()))): self.model_search_results.append(key)

        print("Found {} models.....".format(len(self.model_search_results)))

        for idx, model in enumerate(self.model_search_results):
            print("({})- {}".format(idx, repr(model)))


    def ls_models(self):
        assert len(self.model_urls) > 0, "Models could't find: Load models first...."
        for idx, item in enumerate(self.model_urls.items()):
            print("({}) - {} :".format(idx, repr(item[0])))
            for i in list(item[1]):
                print(i)


    def set_model_by_name(self, model_name):

        assert isinstance(model_name, basestring), "Model adı gelmedi..."
        value = '_'.join(str(model_name).lower().strip().split())
        assert value in self.model_urls.keys(), "Model adı listede yok...."
        self.model = model_name


    def set_model_by_index(self, model_index=0):
        assert model_index < len(self.model_search_results), "Belirtilen index Search Resultların arasında yok...."
        self.model = self.model_search_results[model_index]
        print("Seçilen model : {}".format(self.model))
        print("{} adet varyasyon bulundu".format(len(self.model_urls[self.model])))
        for idx, link in enumerate(self.model_urls[self.model]):
            print("({}) - {}".format(idx, link))


    def ls_brands(self):
        assert len(self.brands) > 0, "Brands could't find: Load brands first...."
        for idx, k in enumerate(self.brands.items()):
            print("({}) - {} - {}".format(idx, repr(k[0]), k[1]))


    def set_brand_by_index(self, brand_index=0):
        assert brand_index < len(self.brands), "Belirtilen index brand'lerin içinde yok...."
        self.brand = self.brands.keys()[brand_index]
        self.base_Url = self.brands.values()[brand_index]
        print("Seçilen marka : {}".format(self.brand))
        self.load_brand_pages()
        self.load_model_detail_pages()
        self.search_models()
        print("{} adet model bulundu".format(len(self.model_urls)))


    def set_brand_by_name(self, brand_name):
        key = '_'.join(str(brand_name).lower().strip().split())
        if key in self.brands:
            self.brand = brand_name
            self.base_Url = self.brands[key]

        assert self.base_Url != "http://www.motorcyclespecs.co.za", "Bu modeli listede bulamadım..."


    def load_brands(self):
        from lxml.html.clean import Cleaner
        print("Loading brands.....")
        url = self.base_Url
        html_cleaner = Cleaner(scripts=True, javascript=True, page_structure=True, style=True, inline_style=True, meta=True, links=False, forms=True, annoying_tags=True, remove_unknown_tags=True)
        html = html_cleaner.clean_html(self.fetch_HTML(url))
        root = BeautifulSoup(html, 'html.parser')
        hyperlinks = root.find_all("a")
        for link in hyperlinks:
            href = link.get("href")
            if "/bikes/" in href:
                feat_key = '_'.join(link.get_text().lower().strip().split())
                self.brands[feat_key] = href.replace("../..", "http://www.motorcyclespecs.co.za")

        assert len(self.brands) > 0, "Markalar yüklenirken bir sıkıntı oldu herhalde..."
        print("{} Brands loaded........".format(len(self.brands)))


    def load_brand_pages(self):
        self.brand_pages = self.scrap_brand_pages()


    def load_model_detail_pages(self):
        for main_page_url, idx in self.brand_pages.items():
            detail_page_urls = self.scrap_model_page_urls(main_page_url)
            for detail_page_url, model in detail_page_urls.items():
                self.model_urls[detail_page_url] = model


    def load_model_specs(self):
        from lxml.html.clean import Cleaner
        assert self.model, "Model seçmemeişsin ki neyi yükleyeyim...."

        print("{} modeli için teknik detaylar yükleniyor".format(self.model))
        urls = self.model_urls[self.model]
        html_cleaner = Cleaner(scripts=True, javascript=True, page_structure=True, style=True, inline_style=True, meta=True, links=True, forms=True, annoying_tags=True, remove_unknown_tags=True)
        for url in urls:
            html = html_cleaner.clean_html(self.fetch_HTML(url))
            root = BeautifulSoup(html, 'html.parser')
            tds = root.find_all("td")

            feats = []
            for idx, td in enumerate(tds):
                for tag in td(['a', 'input']):
                    #  print(tag.get("class")) # Gereksiz tagleri siliyoruz !! Önemli!!
                    tag.decompose()

                feat = td.get_text().strip().replace("\r", "").replace("\n", "").replace("\t", " ").replace(" / ", " | ")
                if feat and len(feat) < 300:
                    feats.append(feat)

            model_specs = {}
            for i in range(0, len(feats)-1, 2):
                feat_key = '_'.join(feats[i].lower().strip().split())
                model_specs[feat_key] = feats[i + 1]

            self.model_specs.append(model_specs)

        self.model_images = []
        imgs = root.find_all("img")

        for img in imgs:
            if "/Gallery/" in img.get("src"): self.model_images.append(
                str(img.get("src")).replace("../..", "http://www.motorcyclespecs.co.za"))

        print("Modelin teknik bilgileri ve imajları yüklendi.")


    def scrap_model_page_urls(self, url):
        import re
        html = self.fetch_HTML(url)
        scrapper = BeautifulSoup(html, 'html.parser')
        links = scrapper.find_all(href=re.compile("model"))
        bike_spec_urls = {}
        for link in links:
            url = unicode(link.get("href")).replace("../", "http://www.motorcyclespecs.co.za/")
            bike_model = unicode(' '.join(link.get_text(strip=True).split()))
            if bike_model not in bike_spec_urls:
                bike_spec_urls[bike_model] = [url]
            else:
                bike_spec_urls[bike_model].append(url)

            # print(bike_spec_urls[bike_model])
            # bike_spec_urls[url] = bike_model

        return bike_spec_urls


    def scrap_brand_pages(self):
        import re
        current_url = self.base_Url
        bike_specs_pages = {current_url: 0}
        while current_url:
            html = self.fetch_HTML(current_url)
            scrapper = BeautifulSoup(html, 'html.parser')
            links = scrapper.find_all("a", string=re.compile("Next"))
            current_url = None
            for i, link in enumerate(links):
                current_url = "http://www.motorcyclespecs.co.za/bikes/" + unicode(link.get("href"))
                bike_specs_pages[current_url] = len(bike_specs_pages)-1
                # current_url = link.get("href")
                # print(' '.join(link.get_text(strip=True).split()))
                # print(link.get("href"))

        return bike_specs_pages

    @staticmethod
    def fetch_HTML(url):
        try:

            user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
            values = {'name': 'Michael',
                      'location': 'Northampton',
                      'language': 'English'}
            headers = {'User-Agent': user_agent}

            data = urlencoder(values)
            req = urllib2.Request(url, data=data, headers=headers)
            response = urllib2.urlopen(req)
            html = response.read()
            return html

        except Exception as exc:
            print(exc)
            print("url de bir sıkıntı var herhalde")
