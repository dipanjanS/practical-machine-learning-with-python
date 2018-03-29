# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:39:39 2017

@author: Raghav Bali
"""

"""

This script crawls apress.com's blog post to:
    + extract content related to blog post using regex

``Execute``
        $ python crawl_basic.py

"""

import re
import requests

def extract_blog_content(content):
    """This function extracts blog post content using regex

    Args:
        content (request.content): String content returned from requests.get

    Returns:
        str: string content as per regex match

    """
    content_pattern = re.compile(r'<div class="cms-richtext">(.*?)</div>')
    result = re.findall(content_pattern, content)
    return result[0] if result else "None"
    

if __name__ =='__main__':
    
    base_url = "http://www.apress.com/in/blog/all-blog-posts"
    blog_suffix = "/wannacry-how-to-prepare/12302194"
    
    print("Crawling Apress.com for required blog post...\n\n")    
    
    response = requests.get(base_url+blog_suffix)
    
    if response.status_code == 200:
        content = response.text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        content = content.replace("\n", '')
        blog_post_content = extract_blog_content(content)
        