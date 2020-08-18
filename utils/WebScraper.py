#Chris Metzler 2020

#Download all the hdr files

import urllib.request
from bs4 import BeautifulSoup

#EMPA wouldn't open

#HDREye can be downloaded over FTP

#Download the EXR files from Fairchild
FairChild=False
if FairChild:
    web_address="http://rit-mcsl.org/fairchild//HDRPS/HDRthumbs.html"
    out_dir='./HDR_dataset/Fairchild/'

    page = urllib.request.urlopen(web_address)
    soup = BeautifulSoup(page)
    #print(soup.prettify())

    HDR_page_urls=[]
    for a in soup.find_all('a', href=True):
        link=a.get('href')
        if link[0:6]=='Scenes':
           HDR_page_urls.append(web_address[0:-14]+link)

    #Remove duplicate URLs
    unique_HDR_page_urls=[]
    for i in HDR_page_urls:
      if i not in unique_HDR_page_urls:
        unique_HDR_page_urls.append(i)

    #Download and save all the .EXR files in the Final_HDR_urls list
    HDR_urls=[]
    for HDR_page_url in unique_HDR_page_urls:
        HDR_page = urllib.request.urlopen(HDR_page_url)
        HDR_soup = BeautifulSoup(HDR_page)
        for a in HDR_soup.find_all('a', href=True):
            link=a.get('href')
            if link[-3:]=='exr':
               HDR_urls.append(link)

    for HDR_url in HDR_urls:
        filedata = urllib.request.urlopen(HDR_url)
        datatowrite = filedata.read()
        im_name=HDR_url[44:]#Name from the last /
        with open(out_dir+im_name, 'wb') as f:
            f.write(datatowrite)

#Ward can be downloaded by clicking on each image

#Stanford can be downloaded as zip

#MCSL has bad link

#Funt can be downloaded as 1 ZIP

#Boitard can be downloaded by clicking on a few links

#MPI can be downloaded by clicking on a few links

#DML-HDR can be downloaded by clicking on a few links

#HDR book: Could not find data

#JPEG-XT: Could not find data (May have used Fairchild and HDREye Data)

#Stuttgart: Downloaded over SFTP

#LiU HDRV can be downloaded by clicking on a many links

