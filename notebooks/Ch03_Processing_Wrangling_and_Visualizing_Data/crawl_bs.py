# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:01:41 2017

@author: Raghav Bali
"""

"""

This script crawls apress.com's blog page to:
    + extract list of recent blog post titles and their URLS
    + extract content related to each blog post in plain text

using requests and BeautifulSoup packages

``Execute``
        $ python crawl_bs.py

"""


import requests
from time import sleep
from bs4 import BeautifulSoup


def get_post_mapping(content):
    """This function extracts blog post title and url from response object

    Args:
        content (request.content): String content returned from requests.get

    Returns:
        list: a list of dictionaries with keys title and url

    """
    post_detail_list = []
    post_soup = BeautifulSoup(content,"lxml")
    h3_content = post_soup.find_all("h3")
    
    for h3 in h3_content:
        post_detail_list.append(
            {'title':h3.a.get_text(),'url':h3.a.attrs.get('href')}
            )
    
    return post_detail_list


def get_post_content(content):
    """This function extracts blog post content from response object

    Args:
        content (request.content): String content returned from requests.get

    Returns:
        str: blog's content in plain text

    """
    plain_text = ""
    text_soup = BeautifulSoup(content,"lxml")
    para_list = text_soup.find_all("div",
                                   {'class':'cms-richtext'})
    
    for p in para_list[0]:
        plain_text += p.getText()
    
    return plain_text
    
    

if __name__ =='__main__':
    
    crawl_url = "http://www.apress.com/in/blog/all-blog-posts"
    post_url_prefix = "http://www.apress.com"
    
    print("Crawling Apress.com for recent blog posts...\n\n")    
    
    response = requests.get(crawl_url)
    
    if response.status_code == 200:
        blog_post_details = get_post_mapping(response.content)
    
    if blog_post_details:
        print("Blog posts found:{}".format(len(blog_post_details)))
        
        for post in blog_post_details:
            print("Crawling content for post titled:",post.get('title'))
            post_response = requests.get(post_url_prefix+post.get('url'))
            
            if post_response.status_code == 200:
                post['content'] = get_post_content(post_response.content)
            
            print("Waiting for 10 secs before crawling next post...\n\n")
            sleep(10)
    
        print("Content crawled for all posts")
        
        # print/write content to file
        for post in blog_post_details:
            print(post)
    